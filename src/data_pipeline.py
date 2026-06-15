"""
data_pipeline.py
================
Phase 1 – Data Processing Module
Project : Optimizing Delivery ETAs with Graph-Based Network Intelligence
Author  : IIT Guwahati Consulting & Analytics Club

Responsibilities
----------------
1. Load and validate the raw Delhivery trip-segment CSV.
2. Parse all timestamp columns and engineer time-of-day buckets.
3. Compute per-segment and per-trip delay ratios
   (actual_time / osrm_time — the core signal for edge weights).
4. Merge trip segments chronologically so the graph builder receives
   one clean, enriched row per unique source→destination corridor.
5. Persist the processed artefact to data/processed/ for downstream use.
"""

import logging
import os
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Project-level path constants ──────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR      = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# ── Column catalogue (keeps the rest of the code readable) ───────────────────
TIMESTAMP_COLS = ["trip_creation_time", "od_start_time", "od_end_time", "cutoff_timestamp"]

ID_COLS = ["trip_uuid", "route_schedule_uuid"]

NUMERIC_COLS = [
    "actual_distance_to_destination",
    "actual_time",
    "osrm_time",
    "osrm_distance",
    "start_scan_to_end_scan",
    "segment_actual_time",
    "segment_osrm_time",
    "segment_osrm_distance",
    "cutoff_factor",
    "factor",
    "segment_factor",
]

CATEGORICAL_COLS = ["data", "route_type", "source_center", "destination_center",
                    "source_name", "destination_name", "is_cutoff"]

# Time-of-day bucket boundaries (hour, inclusive lower bound)
TOD_BINS   = [0, 6, 12, 18, 24]
TOD_LABELS = ["Night(0-6)", "Morning(6-12)", "Afternoon(12-18)", "Evening(18-24)"]


# ─────────────────────────────────────────────────────────────────────────────
# 1. LOADER
# ─────────────────────────────────────────────────────────────────────────────
def load_raw(filename: str = "delhivery_data.csv") -> pd.DataFrame:
    """
    Load the raw Delhivery CSV from data/raw/.

    Parameters
    ----------
    filename : str
        Name of the CSV file placed in data/raw/.

    Returns
    -------
    pd.DataFrame
        Unmodified raw dataframe for traceability.
    """
    filepath = RAW_DIR / filename
    if not filepath.exists():
        raise FileNotFoundError(
            f"Raw data not found at {filepath}.\n"
            "Download the Delhivery dataset from Kaggle and place it in data/raw/."
        )
    log.info("Loading raw data from %s …", filepath)
    df = pd.read_csv(filepath, low_memory=False)
    log.info("Raw shape: %d rows × %d columns", *df.shape)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 2. VALIDATOR
# ─────────────────────────────────────────────────────────────────────────────
def validate_schema(df: pd.DataFrame) -> None:
    """
    Assert that every expected column is present before any transformation.
    Raises ValueError with a clear diff if columns are missing.
    """
    expected = set(
        TIMESTAMP_COLS + ID_COLS + NUMERIC_COLS + CATEGORICAL_COLS
        + ["cutoff_timestamp"]
    )
    # cutoff_timestamp already in TIMESTAMP_COLS; deduplicate silently
    expected = {
        "data", "trip_creation_time", "route_schedule_uuid", "route_type",
        "trip_uuid", "source_center", "source_name", "destination_center",
        "destination_name", "od_start_time", "od_end_time",
        "start_scan_to_end_scan", "is_cutoff", "cutoff_factor",
        "cutoff_timestamp", "actual_distance_to_destination",
        "actual_time", "osrm_time", "osrm_distance", "factor",
        "segment_actual_time", "segment_osrm_time",
        "segment_osrm_distance", "segment_factor",
    }
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in raw data: {missing}")
    log.info("Schema validation passed ✓")


# ─────────────────────────────────────────────────────────────────────────────
# 3. TIMESTAMP PARSER
# ─────────────────────────────────────────────────────────────────────────────
def parse_timestamps(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert all timestamp strings to pandas datetime (UTC-naive).
    Unparseable values become NaT and are logged as warnings.
    """
    df = df.copy()
    for col in TIMESTAMP_COLS:
        if col not in df.columns:
            continue
        before_nulls = df[col].isna().sum()
        df[col] = pd.to_datetime(df[col], errors="coerce", infer_datetime_format=True)
        after_nulls = df[col].isna().sum()
        new_nulls = after_nulls - before_nulls
        if new_nulls > 0:
            log.warning("Column '%s': %d values could not be parsed → NaT", col, new_nulls)

    # Derive trip-level temporal features from od_start_time
    if "od_start_time" in df.columns:
        df["trip_start_hour"] = df["od_start_time"].dt.hour
        df["trip_start_dayofweek"] = df["od_start_time"].dt.dayofweek  # Mon=0, Sun=6
        df["trip_start_date"] = df["od_start_time"].dt.date

        # ── Time-of-day bucket (key stratification dimension) ──────────────
        df["time_of_day"] = pd.cut(
            df["trip_start_hour"],
            bins=TOD_BINS,
            labels=TOD_LABELS,
            right=False,
            include_lowest=True,
        )
        log.info("Timestamp parsing complete; time_of_day buckets created ✓")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 4. CLEANER
# ─────────────────────────────────────────────────────────────────────────────
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Perform targeted cleaning:
    - Cast numeric columns; coerce non-numerics to NaN.
    - Drop rows where both actual_time and osrm_time are missing
      (these cannot contribute to delay ratio computation).
    - Strip whitespace from string columns; normalise route_type casing.
    - Flag and remove physically impossible records (e.g. actual_time ≤ 0).
    """
    df = df.copy()

    # ── Numeric coercion ──────────────────────────────────────────────────
    for col in NUMERIC_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # ── String normalisation ──────────────────────────────────────────────
    for col in ["route_type", "source_name", "destination_name", "data"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().str.upper()

    # route_type must be FTL or CARTING
    valid_route_types = {"FTL", "CARTING"}
    invalid_mask = ~df["route_type"].isin(valid_route_types)
    if invalid_mask.sum() > 0:
        log.warning(
            "%d rows have unrecognised route_type values → will be dropped.",
            invalid_mask.sum(),
        )
        df = df[~invalid_mask].copy()

    # ── Drop rows unusable for delay ratio ───────────────────────────────
    critical_null_mask = df["actual_time"].isna() | df["osrm_time"].isna()
    log.info("Dropping %d rows with null actual_time or osrm_time.", critical_null_mask.sum())
    df = df[~critical_null_mask].copy()

    # ── Remove physically impossible values ──────────────────────────────
    impossible = (df["actual_time"] <= 0) | (df["osrm_time"] <= 0)
    log.info("Dropping %d rows with non-positive time values.", impossible.sum())
    df = df[~impossible].copy()

    # ── Drop exact duplicate rows (same trip_uuid AND corridor) ──────────
    dup_mask = df.duplicated(subset=["trip_uuid", "source_center", "destination_center"])
    log.info("Dropping %d duplicate segment rows.", dup_mask.sum())
    df = df[~dup_mask].copy()

    log.info("Cleaned shape: %d rows × %d columns", *df.shape)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 5. DELAY RATIO ENGINEER
# ─────────────────────────────────────────────────────────────────────────────
def engineer_delay_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute delay ratios at both the trip-segment level and the cumulative
    trip level. These ratios become the primary edge weights in the graph.

    Delay Ratio Definition
    ----------------------
    delay_ratio = actual_time / osrm_time

    Interpretation (logistics context)
    ------------------------------------
    ratio = 1.0  → Delivery matched OSRM estimate exactly (baseline).
    ratio > 1.0  → Delivery took longer than estimated; every 0.1 above 1.0
                   represents a ~10 % SLA risk increase on that corridor.
    ratio < 1.0  → Faster than expected; possible for FTL on low-traffic nights.

    Segment vs. Cumulative
    -----------------------
    segment_delay_ratio : Uses segment_actual_time / segment_osrm_time.
                          Isolates performance of individual legs — critical
                          for pinpointing which leg within a multi-stop trip
                          causes the final delivery delay.
    cumulative_delay_ratio : Uses actual_time / osrm_time.
                              Reflects the total trip performance and is used
                              as the edge weight aggregated over all trips
                              sharing the same corridor.
    """
    df = df.copy()

    # ── Segment-level delay ratio ─────────────────────────────────────────
    df["segment_delay_ratio"] = df["segment_actual_time"] / df["segment_osrm_time"]

    # ── Cumulative (trip-level) delay ratio ───────────────────────────────
    df["cumulative_delay_ratio"] = df["actual_time"] / df["osrm_time"]

    # ── Distance efficiency ratio (actual distance vs OSRM distance) ──────
    # Highlights corridors where drivers deviate from optimal routes.
    with np.errstate(divide="ignore", invalid="ignore"):
        df["distance_efficiency_ratio"] = np.where(
            df["osrm_distance"] > 0,
            df["actual_distance_to_destination"] / df["osrm_distance"],
            np.nan,
        )

    # ── Binary SLA breach flag (>20 % over OSRM estimate = breach) ────────
    # Threshold of 1.20 is a standard logistics SLA heuristic.
    df["is_sla_breach"] = (df["segment_delay_ratio"] > 1.20).astype(int)

    # ── Log-transform of delay ratio (stabilises skewed distributions) ────
    df["log_segment_delay_ratio"] = np.log1p(df["segment_delay_ratio"])

    log.info(
        "Delay features engineered. Avg segment delay ratio: %.3f | SLA breach rate: %.1f%%",
        df["segment_delay_ratio"].mean(),
        df["is_sla_breach"].mean() * 100,
    )
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 6. CHRONOLOGICAL TRIP MERGER
# ─────────────────────────────────────────────────────────────────────────────
def merge_trip_segments(df: pd.DataFrame) -> pd.DataFrame:
    """
    Sort all segments within each trip chronologically by od_start_time,
    then aggregate to produce one summary row per unique corridor
    (source_center → destination_center) per route_type and time_of_day bucket.

    This is the graph-ready output: each row = one potential directed edge.

    Aggregation Strategy
    ---------------------
    Median is chosen over mean to suppress the influence of extreme outliers
    (e.g., weather events, vehicle breakdowns) that inflate mean delay ratios
    and would create misleading edge weights in the network graph.
    """
    # Sort within each trip chronologically
    df = df.sort_values(["trip_uuid", "od_start_time"]).reset_index(drop=True)
    log.info("Segments sorted chronologically within each trip ✓")

    # ── Corridor-level aggregation (graph edges) ──────────────────────────
    corridor_agg = (
        df.groupby(
            ["source_center", "source_name",
             "destination_center", "destination_name",
             "route_type"],
            observed=True,
            sort=False,
        )
        .agg(
            trip_count            = ("trip_uuid",             "nunique"),
            median_segment_delay  = ("segment_delay_ratio",   "median"),
            mean_segment_delay    = ("segment_delay_ratio",   "mean"),
            std_segment_delay     = ("segment_delay_ratio",   "std"),
            median_cum_delay      = ("cumulative_delay_ratio","median"),
            sla_breach_rate       = ("is_sla_breach",         "mean"),
            median_actual_dist_km = ("actual_distance_to_destination", "median"),
            median_segment_time_h = ("segment_actual_time",   "median"),
            median_osrm_time_h    = ("segment_osrm_time",     "median"),
            median_osrm_dist_km   = ("segment_osrm_distance", "median"),
        )
        .reset_index()
    )

    # ── Time-of-day stratified aggregation ───────────────────────────────
    tod_agg = (
        df.groupby(
            ["source_center", "destination_center", "route_type", "time_of_day"],
            observed=True,
            sort=False,
        )
        .agg(
            tod_trip_count       = ("trip_uuid",           "nunique"),
            tod_median_delay     = ("segment_delay_ratio", "median"),
            tod_sla_breach_rate  = ("is_sla_breach",       "mean"),
        )
        .reset_index()
    )

    log.info(
        "Corridor aggregation complete: %d unique directed corridors across %d route types.",
        len(corridor_agg),
        corridor_agg["route_type"].nunique(),
    )
    return corridor_agg, tod_agg


# ─────────────────────────────────────────────────────────────────────────────
# 7. PERSIST
# ─────────────────────────────────────────────────────────────────────────────
def save_processed(
    df_clean: pd.DataFrame,
    corridor_agg: pd.DataFrame,
    tod_agg: pd.DataFrame,
) -> None:
    """Save all processed artefacts to data/processed/ for downstream use."""
    paths = {
        "cleaned_trips.csv"   : df_clean,
        "corridor_edges.csv"  : corridor_agg,
        "tod_stratified.csv"  : tod_agg,
    }
    for fname, frame in paths.items():
        out = PROCESSED_DIR / fname
        frame.to_csv(out, index=False)
        log.info("Saved → %s  (%d rows)", out, len(frame))


# ─────────────────────────────────────────────────────────────────────────────
# 8. MASTER PIPELINE ENTRY-POINT
# ─────────────────────────────────────────────────────────────────────────────
def run_pipeline(filename: str = "delhivery_data.csv") -> dict:
    """
    Execute the full Phase 1 data pipeline in sequence.

    Returns
    -------
    dict with keys:
        'raw'         : Original dataframe (for audit trail).
        'cleaned'     : Row-level cleaned + feature-engineered dataframe.
        'corridors'   : Aggregated corridor-level dataframe (graph edges).
        'tod'         : Time-of-day stratified corridor dataframe.
    """
    log.info("=" * 60)
    log.info("  PHASE 1 — DATA PIPELINE STARTING")
    log.info("=" * 60)

    raw          = load_raw(filename)
    validate_schema(raw)
    ts_parsed    = parse_timestamps(raw)
    cleaned      = clean_data(ts_parsed)
    featured     = engineer_delay_features(cleaned)
    corridors, tod = merge_trip_segments(featured)
    save_processed(featured, corridors, tod)

    log.info("=" * 60)
    log.info("  PHASE 1 — PIPELINE COMPLETE")
    log.info("  Clean rows     : %d", len(featured))
    log.info("  Unique corridors: %d", len(corridors))
    log.info("  TOD records    : %d", len(tod))
    log.info("=" * 60)

    return {
        "raw"      : raw,
        "cleaned"  : featured,
        "corridors": corridors,
        "tod"      : tod,
    }


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    results = run_pipeline()
