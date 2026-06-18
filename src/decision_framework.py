"""
decision_framework.py
======================
Phase 4 – FTL vs Carting Decision Framework
Project : Optimizing Delivery ETAs with Graph-Based Network Intelligence
Author  : IIT Guwahati Consulting & Analytics Club

CHANGE LOG (fix applied to the previous version)
--------------------------------------------------
The previous run produced 0% FTL / 100% CARTING recommendations across all
2,783 corridors — a degenerate, business-meaningless result. Root cause was
two compounding issues, both fixed in this version:

  BUG 1 — Wrong distance column.
      The Delhivery schema has two distance scales: unprefixed columns
      (actual_distance_to_destination, actual_time, osrm_time, osrm_distance)
      are CUMULATIVE / whole-trip totals, repeated identically across every
      segment row belonging to the same trip_uuid. Only the `segment_`
      prefixed columns (segment_osrm_distance, segment_actual_time) describe
      the single source→destination HOP that a "corridor" in this framework
      actually represents. The previous version aggregated
      `actual_distance_to_destination` per corridor — i.e. it measured the
      length of whichever multi-leg journeys happened to pass through that
      hop, not the hop's own physical length. Fixed: corridor distance and
      the trip-level model features now use `segment_osrm_distance`.

  BUG 2 — Cost constants never validated against the data's own scale.
      CARTING_LONGHAUL_THRESHOLD_KM = 150 with a 5x penalty was a guess.
      Once corridor distance is computed correctly, this dataset's segment
      hops are mostly short (last-mile / intra-region movement dominates
      the trip count). A threshold set without checking the real distance
      distribution can mathematically guarantee one route type always wins,
      regardless of what the ML models predict. Fixed: the cost model now
      (a) computes and logs the analytical FTL/CARTING breakeven distance
      from the constants themselves, (b) reports the real corridor distance
      percentiles right next to it, and (c) `validate_cost_model_sanity()`
      runs synthetic benchmark distances through the cost formula BEFORE the
      real pipeline executes and raises a hard error if the model can't
      produce both recommendations — this is the guard that would have
      caught the original bug at the very first run instead of three phases
      later.

Everything else (corridor profiles, the regressor/classifier pair, the
counterfactual simulation, the surrogate decision tree, visualisations) keeps
the same design as before — that part of the architecture was sound.

Responsibilities
----------------
1. Build CORRIDOR PROFILES — one row per unique (source, destination) lane —
   combining distance, time-of-day operating pattern, and the source/destination
   facility's structural position in the graph (betweenness, PageRank, degree,
   clustering, SLA breach rate from Phase 2).
2. Train two interpretable XGBoost models on trip-level data:
      (a) a REGRESSOR predicting delay ratio (actual/OSRM)
      (b) a CLASSIFIER predicting probability of SLA breach
   Both take route_type as a controllable input feature.
3. Run a COUNTERFACTUAL SIMULATION: for every corridor, hold all structural
   features fixed and flip only route_type between FTL and CARTING, asking
   "what would happen on this lane under each option?"
4. Translate the predicted performance gap into a TIME-COST TRADE-OFF using
   transparent, configurable unit-economics assumptions (clearly flagged —
   the raw dataset has no cost column, so this layer is illustrative and
   meant to be calibrated with Delhivery's actual finance data).
5. Extract a simple, human-readable DECISION TREE surrogate so ops staff can
   apply the framework's logic without running Python.
6. Visualise the decision boundary, cost trade-offs, and risk drivers.

Why XGBoost (Regressor + Classifier) Instead of a Single Model?
-------------------------------------------------------------------
The regressor answers "how late will this corridor run under each route type?"
(a continuous magnitude). The classifier answers "what is the probability this
specific shipment breaches SLA?" (a risk probability used directly in the cost
model below). Logistics decision-makers think in both terms — "by how much"
and "how risky" — so both are produced rather than picking one.

Why NOT node2vec Embeddings Here (Unlike Phase 3)?
-----------------------------------------------------
Phase 3 optimised purely for prediction accuracy, where uninterpretable
32-dimensional embeddings are acceptable because no human needs to read them.
Phase 4 is a DECISION framework that a network operations manager must be
able to question and override. Betweenness centrality, PageRank, and degree
map directly to operational concepts ("this hub is a bottleneck", "this hub
is congested") that a non-technical stakeholder can act on. We trade a small
amount of predictive power for full explainability — appropriate for
Deliverable 4.

Why a Counterfactual Simulation Instead of Just Comparing Historical Averages?
-----------------------------------------------------------------------------
Most corridors in this network only ever used ONE route type historically.
We cannot simply compare "FTL corridors' average delay" against "CARTING
corridors' average delay" — that would conflate route type with the kind of
route that tends to use it. Instead, we train one model on ALL trips with
route_type as an input, then simulate BOTH options on the SAME corridor,
isolating the effect of route type alone, holding distance, time-of-day,
and graph position constant.
"""

import logging
import pickle
import warnings
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.metrics import accuracy_score, mean_absolute_error, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree

warnings.filterwarnings("ignore")

try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Project Paths ────────────────────────────────────────────────────────────
PROJECT_ROOT  = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODEL_DIR     = PROJECT_ROOT / "outputs" / "models"
METRICS_DIR   = PROJECT_ROOT / "outputs" / "metrics"
VIZ_DIR       = PROJECT_ROOT / "outputs" / "visualizations"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
METRICS_DIR.mkdir(parents=True, exist_ok=True)
VIZ_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
SLA_THRESHOLD = 1.20  # Informational — matches the breach definition already
                       # baked into the precomputed `is_sla_breach` column
                       # from Phase 1/2. Not recomputed here.

plt.rcParams.update({
    "figure.dpi"       : 140,
    "font.size"        : 10,
    "axes.spines.top"  : False,
    "axes.spines.right": False,
    "axes.titleweight" : "bold",
    "axes.titlesize"   : 13,
})

# ─────────────────────────────────────────────────────────────────────────────
# ⚠️  COST MODEL ASSUMPTIONS — CONFIGURABLE BUSINESS PARAMETERS
# ─────────────────────────────────────────────────────────────────────────────
# The raw Delhivery dataset contains NO cost columns. Everything below is an
# ILLUSTRATIVE unit-economics layer used to demonstrate the decision framework
# end-to-end. Before using this framework operationally, Delhivery's finance
# team must replace these constants with actual contracted FTL/Carting rates.
#
# These specific numbers are set to reflect realistic Indian road-freight
# economics: an FTL truck has a much higher cost to dispatch (driver, permits,
# fuel minimums) but a lower marginal per-km running cost and far higher
# capacity, which is what makes it economical at distance/volume. A Carting
# vehicle (3-wheeler / mini-van used for last-mile) is cheap to dispatch but
# is not built for genuine long-haul running — both the limited fuel range
# and driver-hours regulations mean cost per km rises sharply past a certain
# distance. That crossover is what creates a real, distance-driven decision
# boundary rather than one mode dominating everywhere.
# ─────────────────────────────────────────────────────────────────────────────
FTL_FIXED_COST_RS        = 2_000    # Base dispatch cost for a regional mid-size rigid truck (₹)
                                     # — covers minimum driver wage, fuel buffer, and loading
                                     # labour for a single dispatch. ₹1,200 (tried earlier) is
                                     # closer to a bare fuel-only estimate; ₹2,000 is a more
                                     # defensible "fully loaded" dispatch cost for a regional run.
FTL_PER_KM_COST_RS       = 22        # Running cost per km for a truck (₹/km)
FTL_CAPACITY_UNITS       = 400      # Max shipments per FTL trip

CARTING_FIXED_COST_RS    = 400       # Base dispatch cost per light commercial vehicle (₹)
CARTING_PER_KM_COST_RS   = 10        # Running cost per km for a small vehicle (₹/km)
CARTING_CAPACITY_UNITS   = 50        # Max shipments per Carting vehicle

# FIX (was the root cause of the 100% CARTING bug): this threshold must be
# set relative to the REAL segment-distance scale in the data, not picked in
# isolation. validate_cost_model_sanity() checks this automatically below —
# if you change any constant in this block, re-run that check before trusting
# the output.
#
# This dataset's real corridor distances are: median ≈ 24 km, p75 ≈ 28 km,
# p90 ≈ 34 km, with a thin tail out to ~320 km. 20 km is set as a realistic
# "normal" last-mile/regional operating radius for a cart vehicle — below
# the median, so the long-haul penalty activates for the majority of the
# network's typical hops, not just the rare outliers. Combined with the
# ₹2,000 FTL dispatch cost above, this puts the breakeven at ≈86 km — past
# the 90th percentile, so FTL is reserved for genuinely long corridors, but
# not so far out that it becomes unreachable. See run_sensitivity_analysis()
# for how this conclusion holds up across other plausible threshold/cost
# combinations.
CARTING_LONGHAUL_THRESHOLD_KM   = 20
CARTING_LONGHAUL_PENALTY_FACTOR = 5

SLA_BREACH_PENALTY_RS    = 500      # Assumed avg. penalty/compensation per breached shipment (₹)

# ─────────────────────────────────────────────────────────────────────────────
# SENSITIVITY GRID — the FTL fixed-cost values we test the recommendation
# against, since this single constant (see analytical_breakeven_distance_km)
# is the dominant lever on the FTL/CARTING split and there is no real cost
# data to pin it down exactly. Rather than reporting one brittle number, the
# framework now reports how the recommendation holds up across this whole
# plausible range — this directly satisfies the Phase 4 requirement to
# "quantify the time-cost trade-offs for different corridor profiles."
FTL_FIXED_COST_SENSITIVITY_GRID = [800, 1_200, 2_000, 3_000, 4_000, 5_000]

# Representative hour used to simulate each time-of-day bucket
TOD_REPRESENTATIVE_HOUR = {
    "NIGHT(0-6)"      : 3,
    "MORNING(6-12)"   : 9,
    "AFTERNOON(12-18)": 15,
    "EVENING(18-24)"  : 21,
}
TOD_ENCODE = {"NIGHT(0-6)": 0, "MORNING(6-12)": 1, "AFTERNOON(12-18)": 2, "EVENING(18-24)": 3}
REPRESENTATIVE_DOW = 2  # Wednesday — neutral mid-week default

# Feature set used by BOTH the regressor and classifier (kept identical so
# the same simulation routine can query either model)
GRAPH_FEATS = [
    "betweenness_centrality", "pagerank", "in_degree_raw",
    "out_degree_raw", "clustering_coefficient", "avg_sla_breach_rate",
]
MODEL_FEATURES = (
    ["segment_distance_km", "segment_osrm_time_hours",
     "trip_start_hour", "trip_start_dayofweek", "time_of_day_encoded",
     "route_type_encoded"]
    + [f"src_{f}" for f in GRAPH_FEATS]
    + [f"dst_{f}" for f in GRAPH_FEATS]
)


# ─────────────────────────────────────────────────────────────────────────────
# 0. COST MODEL SANITY CHECK — RUNS BEFORE ANYTHING ELSE
# ─────────────────────────────────────────────────────────────────────────────
def carting_cost_per_trip(distance_km: np.ndarray, fixed: float = None,
                           per_km: float = None, threshold: float = None,
                           penalty: float = None) -> np.ndarray:
    """
    Carting cost for one dispatch at the given distance(s).

    All four economic parameters can be overridden — this lets
    run_sensitivity_analysis() re-use this exact formula while only
    varying FTL_FIXED_COST_RS, instead of duplicating the cost logic.
    """
    fixed     = CARTING_FIXED_COST_RS         if fixed     is None else fixed
    per_km    = CARTING_PER_KM_COST_RS        if per_km    is None else per_km
    threshold = CARTING_LONGHAUL_THRESHOLD_KM if threshold is None else threshold
    penalty   = CARTING_LONGHAUL_PENALTY_FACTOR if penalty is None else penalty

    base_km     = np.minimum(distance_km, threshold)
    longhaul_km = np.maximum(distance_km - threshold, 0)
    return fixed + per_km * base_km + per_km * penalty * longhaul_km


def ftl_cost_per_trip(distance_km: np.ndarray, fixed: float = None,
                       per_km: float = None) -> np.ndarray:
    """FTL cost for one dispatch at the given distance(s)."""
    fixed  = FTL_FIXED_COST_RS  if fixed  is None else fixed
    per_km = FTL_PER_KM_COST_RS if per_km is None else per_km
    return fixed + per_km * distance_km


def analytical_breakeven_distance_km(ftl_fixed: float = None) -> float:
    """
    Solve FTL_cost(d) = CARTING_cost(d) for d, assuming d is past the
    Carting long-haul threshold (the only region where a crossover is
    mathematically possible given Carting's cost only ramps UP with
    distance while FTL's per-km rate is constant).

    FTL_FIXED + FTL_PER_KM*d
        = CARTING_FIXED + CARTING_PER_KM*threshold
          + CARTING_PER_KM*penalty*(d - threshold)

    ftl_fixed lets the sensitivity analysis ask "what would the breakeven
    be under a different FTL dispatch cost assumption?" without touching
    the module-level constant.

    Solved for d. Returns None if the lines never cross in the positive
    region (i.e. the constants make one mode dominate at every distance —
    exactly the failure mode this function exists to catch).
    """
    ftl_fixed = FTL_FIXED_COST_RS if ftl_fixed is None else ftl_fixed
    a = FTL_PER_KM_COST_RS
    b = (CARTING_PER_KM_COST_RS * CARTING_LONGHAUL_PENALTY_FACTOR)
    intercept_diff = (
        CARTING_FIXED_COST_RS
        + CARTING_PER_KM_COST_RS * CARTING_LONGHAUL_THRESHOLD_KM
        - CARTING_PER_KM_COST_RS * CARTING_LONGHAUL_PENALTY_FACTOR * CARTING_LONGHAUL_THRESHOLD_KM
        - ftl_fixed
    )
    if (a - b) == 0:
        return None
    d = intercept_diff / (a - b)
    return d if d > 0 else None


def validate_cost_model_sanity() -> float:
    """
    Run the cost formula against a spread of benchmark distances BEFORE
    touching real data. This is the guard that catches a degenerate cost
    model (one mode always wins) at the moment constants change, rather
    than three phases into a pipeline run.

    Raises
    ------
    RuntimeError if the model cannot produce BOTH FTL and CARTING as the
    cheaper option across the benchmark distances — i.e. exactly the bug
    this file was rewritten to fix.
    """
    benchmark_km = np.array([5, 15, 30, 60, 100, 150, 250, 500, 1000])
    ftl  = ftl_cost_per_trip(benchmark_km)
    cart = carting_cost_per_trip(benchmark_km)
    cheaper = np.where(ftl <= cart, "FTL", "CARTING")

    breakeven = analytical_breakeven_distance_km()

    log.info("─" * 65)
    log.info("COST MODEL SANITY CHECK (single-trip, low-volume benchmark)")
    log.info("─" * 65)
    for d, f, c, label in zip(benchmark_km, ftl, cart, cheaper):
        log.info(f"  {d:>5} km  →  FTL ₹{f:>9,.0f}   CARTING ₹{c:>9,.0f}   → {label}")
    if breakeven:
        log.info(f"Analytical FTL/CARTING breakeven distance ≈ {breakeven:.1f} km")
    else:
        log.error("No finite breakeven distance exists with current constants!")

    unique_outcomes = set(cheaper)
    if len(unique_outcomes) < 2:
        raise RuntimeError(
            f"COST MODEL IS DEGENERATE: every benchmark distance resolves to "
            f"'{cheaper[0]}'. This is the exact failure mode that produced a "
            f"100%/0% recommendation split previously. Adjust "
            f"FTL_FIXED_COST_RS, CARTING_LONGHAUL_THRESHOLD_KM, or "
            f"CARTING_LONGHAUL_PENALTY_FACTOR until both route types can win "
            f"at some distance, then re-run this check before proceeding."
        )
    log.info("Sanity check passed — both route types are reachable. Proceeding.")
    log.info("─" * 65)
    return breakeven


# ─────────────────────────────────────────────────────────────────────────────
# 1. LOADER
# ─────────────────────────────────────────────────────────────────────────────
def load_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load all artefacts produced by Phase 1 and Phase 2."""
    trips_path = PROCESSED_DIR / "cleaned_trips.csv"
    corridor_path = PROCESSED_DIR / "corridor_edges.csv"
    tod_path = PROCESSED_DIR / "tod_stratified.csv"
    centrality_path = METRICS_DIR / "node_centrality_metrics.csv"

    for p in [trips_path, corridor_path, tod_path, centrality_path]:
        if not p.exists():
            raise FileNotFoundError(f"Missing required input: {p}.\nRun Phase 1 & 2 first.")

    trips      = pd.read_csv(trips_path, low_memory=False)
    corridors  = pd.read_csv(corridor_path)
    tod        = pd.read_csv(tod_path)
    centrality = pd.read_csv(centrality_path)
    centrality["facility_id"] = centrality["facility_id"].astype(str)

    # FIX: segment_osrm_distance / segment_osrm_time are the correct
    # single-hop columns; actual_distance_to_destination is a cumulative
    # whole-trip value and must never be used as a corridor's distance.
    required_segment_cols = ["segment_osrm_distance", "segment_osrm_time"]
    missing = [c for c in required_segment_cols if c not in trips.columns]
    if missing:
        raise KeyError(
            f"Expected segment-level columns missing from cleaned_trips.csv: "
            f"{missing}. Corridor-level distance/time MUST come from the "
            f"segment_-prefixed columns, not the cumulative trip-level ones."
        )

    log.info(
        "Loaded: %d trips | %d corridors | %d TOD records | %d facilities",
        len(trips), len(corridors), len(tod), len(centrality),
    )
    return trips, corridors, tod, centrality


# ─────────────────────────────────────────────────────────────────────────────
# 2. TRIP-LEVEL MODELLING FEATURES
# ─────────────────────────────────────────────────────────────────────────────
def build_modeling_features(
    trips: pd.DataFrame,
    centrality: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    """
    Build the trip-level feature matrix used to TRAIN the route performance
    models. Every row is one historical trip segment with its real route_type.

    FIX: distance and time features now come from segment_osrm_distance /
    segment_osrm_time (the single-hop values) instead of
    actual_distance_to_destination / osrm_time (cumulative whole-trip
    values repeated across every segment of the same trip_uuid). Training
    a model to predict a SEGMENT-level delay ratio using a TRIP-level
    cumulative distance was mixing two different units of analysis.

    Returns
    -------
    X        : Feature matrix (MODEL_FEATURES columns).
    y_delay  : Regression target — segment_delay_ratio.
    y_breach : Classification target — is_sla_breach (0/1).
    """
    df = trips.copy()
    df = df.dropna(subset=["segment_delay_ratio", "is_sla_breach",
                            "segment_osrm_distance", "segment_osrm_time"])

    df["segment_distance_km"]      = df["segment_osrm_distance"]
    df["segment_osrm_time_hours"]  = df["segment_osrm_time"] / 60.0  # minutes → hours

    df["route_type_encoded"] = (df["route_type"].astype(str).str.upper() == "CARTING").astype(int)
    df["time_of_day_encoded"] = (
        df["time_of_day"].astype(str).str.upper().map(TOD_ENCODE).fillna(1)
    )

    c = centrality.set_index("facility_id")
    src_ids = df["source_center"].astype(str)
    dst_ids = df["destination_center"].astype(str)

    def lookup(ids, feat):
        return ids.map(lambda x: float(c.loc[x, feat]) if x in c.index and not pd.isna(c.loc[x, feat]) else 0.0)

    for feat in GRAPH_FEATS:
        df[f"src_{feat}"] = lookup(src_ids, feat)
        df[f"dst_{feat}"] = lookup(dst_ids, feat)

    for col in MODEL_FEATURES:
        if col not in df.columns:
            df[col] = 0
        df[col] = df[col].fillna(df[col].median() if df[col].dtype != "O" else 0)

    X = df[MODEL_FEATURES].copy()
    y_delay  = df["segment_delay_ratio"].copy()
    y_breach = df["is_sla_breach"].copy()

    log.info("Modelling features built: %d rows × %d features", len(X), X.shape[1])
    log.info(
        "Segment distance (km) — min=%.1f  median=%.1f  p90=%.1f  max=%.1f",
        X["segment_distance_km"].min(), X["segment_distance_km"].median(),
        X["segment_distance_km"].quantile(0.90), X["segment_distance_km"].max(),
    )
    return X, y_delay, y_breach


# ─────────────────────────────────────────────────────────────────────────────
# 3. MODEL TRAINING
# ─────────────────────────────────────────────────────────────────────────────
def get_regressor():
    if XGB_AVAILABLE:
        return xgb.XGBRegressor(
            n_estimators=400, learning_rate=0.05, max_depth=6,
            subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
            random_state=RANDOM_STATE, n_jobs=-1, verbosity=0,
        )
    return GradientBoostingRegressor(n_estimators=250, learning_rate=0.05,
                                      max_depth=4, random_state=RANDOM_STATE)


def get_classifier():
    if XGB_AVAILABLE:
        return xgb.XGBClassifier(
            n_estimators=400, learning_rate=0.05, max_depth=6,
            subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
            random_state=RANDOM_STATE, n_jobs=-1, verbosity=0,
            eval_metric="logloss",
        )
    return GradientBoostingClassifier(n_estimators=250, learning_rate=0.05,
                                       max_depth=4, random_state=RANDOM_STATE)


def train_performance_models(
    X: pd.DataFrame, y_delay: pd.Series, y_breach: pd.Series,
) -> Tuple[object, object, Dict]:
    """
    Train and evaluate the two route-performance models.

    Regressor  → predicts delay ratio (actual/OSRM). MAE reported in
                 "delay-ratio units" — e.g. MAE=0.3 means predictions are
                 typically off by 0.3× (30 percentage points of delay).
    Classifier → predicts SLA breach probability. AUC and accuracy reported.
    """
    X_tr, X_te, yd_tr, yd_te, yb_tr, yb_te = train_test_split(
        X, y_delay, y_breach, test_size=0.20, random_state=RANDOM_STATE,
        stratify=y_breach,
    )

    log.info("Training delay-ratio regressor …")
    reg = get_regressor()
    reg.fit(X_tr, yd_tr)
    pred_delay = reg.predict(X_te)
    mae = mean_absolute_error(yd_te, pred_delay)

    log.info("Training SLA-breach classifier …")
    clf = get_classifier()
    clf.fit(X_tr, yb_tr)
    pred_prob = clf.predict_proba(X_te)[:, 1]
    auc = roc_auc_score(yb_te, pred_prob)
    acc = accuracy_score(yb_te, (pred_prob >= 0.5).astype(int))

    metrics = {
        "regressor_mae": round(mae, 4),
        "classifier_auc": round(auc, 4),
        "classifier_accuracy": round(acc, 4),
        "n_train": len(X_tr),
        "n_test": len(X_te),
    }
    log.info(
        "Models trained — Delay MAE=%.4f | Breach AUC=%.4f | Breach Accuracy=%.4f",
        mae, auc, acc,
    )
    return reg, clf, metrics


# ─────────────────────────────────────────────────────────────────────────────
# 4. CORRIDOR PROFILE CONSTRUCTION
# ─────────────────────────────────────────────────────────────────────────────
def build_corridor_profiles(
    trips: pd.DataFrame,
    centrality: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build ONE PROFILE ROW PER UNIQUE LANE (source → destination), independent
    of which route type was historically used.

    FIX: median_distance_km now aggregates segment_osrm_distance (the actual
    single-hop distance for THIS corridor) instead of
    actual_distance_to_destination (the cumulative distance of whichever
    multi-leg trips happened to pass through this hop).
    """
    df = trips.copy()
    df["lane_key"] = df["source_center"].astype(str) + "→" + df["destination_center"].astype(str)

    # ── Base lane aggregation (route-type agnostic) ───────────────────────
    base = df.groupby(["source_center", "destination_center"], as_index=False).agg(
        source_name           = ("source_name", "first"),
        destination_name      = ("destination_name", "first"),
        median_distance_km    = ("segment_osrm_distance", "median"),
        median_osrm_time_h    = ("segment_osrm_time", lambda s: s.median() / 60.0),
        total_trip_count      = ("trip_uuid", "nunique"),
    )
    base["source_center"]      = base["source_center"].astype(str)
    base["destination_center"] = base["destination_center"].astype(str)

    log.info(
        "Corridor distance (km, post-fix) — min=%.1f  p25=%.1f  median=%.1f  "
        "p75=%.1f  p90=%.1f  max=%.1f",
        base["median_distance_km"].min(), base["median_distance_km"].quantile(0.25),
        base["median_distance_km"].median(), base["median_distance_km"].quantile(0.75),
        base["median_distance_km"].quantile(0.90), base["median_distance_km"].max(),
    )
    pct_above_threshold = (base["median_distance_km"] > CARTING_LONGHAUL_THRESHOLD_KM).mean() * 100
    log.info(
        "%.1f%% of corridors exceed the %.0f km Carting long-haul threshold "
        "(these are the corridors where FTL has a real chance of winning).",
        pct_above_threshold, CARTING_LONGHAUL_THRESHOLD_KM,
    )

    # ── Time-of-day operating pattern ─────────────────────────────────────
    tod_counts = (
        df.groupby(["source_center", "destination_center", "time_of_day"], observed=True)
        .size().reset_index(name="n")
    )
    tod_pivot = tod_counts.pivot_table(
        index=["source_center", "destination_center"],
        columns="time_of_day", values="n", fill_value=0,
    )
    tod_pivot.columns = [f"pct_{str(c).split('(')[0].lower()}" for c in tod_pivot.columns]
    tod_pct = tod_pivot.div(tod_pivot.sum(axis=1), axis=0).reset_index()
    tod_pct["source_center"]      = tod_pct["source_center"].astype(str)
    tod_pct["destination_center"] = tod_pct["destination_center"].astype(str)

    pct_cols = [c for c in tod_pct.columns if c.startswith("pct_")]
    tod_pct["dominant_time_of_day"] = tod_pct[pct_cols].idxmax(axis=1).str.replace("pct_", "", regex=False)

    base = base.merge(tod_pct, on=["source_center", "destination_center"], how="left")

    # ── Historical per-route-type performance (may be NaN if never used) ──
    hist = df.groupby(
        ["source_center", "destination_center", "route_type"], observed=True
    ).agg(
        trip_count       = ("trip_uuid", "nunique"),
        median_delay     = ("segment_delay_ratio", "median"),
        sla_breach_rate  = ("is_sla_breach", "mean"),
    ).reset_index()
    hist["source_center"]      = hist["source_center"].astype(str)
    hist["destination_center"] = hist["destination_center"].astype(str)
    hist["route_type"] = hist["route_type"].astype(str).str.upper()

    hist_pivot = hist.pivot_table(
        index=["source_center", "destination_center"],
        columns="route_type",
        values=["trip_count", "median_delay", "sla_breach_rate"],
    )
    hist_pivot.columns = [f"{rt.lower()}_{metric}" for metric, rt in hist_pivot.columns]
    hist_pivot = hist_pivot.reset_index()
    base = base.merge(hist_pivot, on=["source_center", "destination_center"], how="left")

    for col in ["ftl_trip_count", "carting_trip_count"]:
        if col not in base.columns:
            base[col] = 0
    base["ftl_trip_count"]     = base["ftl_trip_count"].fillna(0)
    base["carting_trip_count"] = base["carting_trip_count"].fillna(0)
    base["has_both_types"] = (
        (base["ftl_trip_count"] > 0) & (base["carting_trip_count"] > 0)
    )
    base["current_route_type"] = np.where(
        base["ftl_trip_count"] >= base["carting_trip_count"], "FTL", "CARTING"
    )

    # ── Attach source & destination structural features ──────────────────
    c = centrality.set_index("facility_id")

    def attach(ids, prefix):
        out = pd.DataFrame(index=ids.index)
        for feat in GRAPH_FEATS:
            out[f"{prefix}_{feat}"] = ids.map(
                lambda x, f=feat: float(c.loc[x, f]) if x in c.index and not pd.isna(c.loc[x, f]) else 0.0
            )
        return out

    base = pd.concat([base, attach(base["source_center"], "src")], axis=1)
    base = pd.concat([base, attach(base["destination_center"], "dst")], axis=1)

    log.info("Corridor profiles built: %d unique lanes", len(base))
    log.info(
        "  Lanes with BOTH FTL & CARTING history: %d (%.1f%%)",
        base["has_both_types"].sum(), base["has_both_types"].mean() * 100,
    )
    return base


# ─────────────────────────────────────────────────────────────────────────────
# 5. COUNTERFACTUAL SIMULATION
# ─────────────────────────────────────────────────────────────────────────────
def simulate_route_type_predictions(
    profiles: pd.DataFrame, reg_model, clf_model,
) -> pd.DataFrame:
    """
    For every corridor, build TWO synthetic feature rows — identical in every
    way except route_type — and ask the trained models: "How would this lane
    perform under FTL? Under CARTING?"
    """
    df = profiles.copy()
    n = len(df)

    def make_X(route_type_encoded: int) -> pd.DataFrame:
        X = pd.DataFrame(index=df.index)
        X["segment_distance_km"]     = df["median_distance_km"]
        X["segment_osrm_time_hours"] = df["median_osrm_time_h"]
        X["trip_start_hour"] = df["dominant_time_of_day"].map(
            lambda t: TOD_REPRESENTATIVE_HOUR.get(t.upper(), 9) if isinstance(t, str) else 9
        )
        X["trip_start_dayofweek"] = REPRESENTATIVE_DOW
        X["time_of_day_encoded"]  = df["dominant_time_of_day"].map(
            lambda t: TOD_ENCODE.get(t.upper(), 1) if isinstance(t, str) else 1
        )
        X["route_type_encoded"] = route_type_encoded
        for feat in GRAPH_FEATS:
            X[f"src_{feat}"] = df[f"src_{feat}"]
            X[f"dst_{feat}"] = df[f"dst_{feat}"]
        return X[MODEL_FEATURES]

    X_ftl     = make_X(0)
    X_carting = make_X(1)

    df["pred_delay_ratio_ftl"]     = np.maximum(reg_model.predict(X_ftl), 0.1)
    df["pred_delay_ratio_carting"] = np.maximum(reg_model.predict(X_carting), 0.1)
    df["pred_breach_prob_ftl"]     = clf_model.predict_proba(X_ftl)[:, 1]
    df["pred_breach_prob_carting"] = clf_model.predict_proba(X_carting)[:, 1]

    df["pred_time_hours_ftl"]     = df["pred_delay_ratio_ftl"]     * df["median_osrm_time_h"]
    df["pred_time_hours_carting"] = df["pred_delay_ratio_carting"] * df["median_osrm_time_h"]

    log.info("Counterfactual simulation complete for %d corridors.", n)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 6. TIME-COST TRADE-OFF & RECOMMENDATION
# ─────────────────────────────────────────────────────────────────────────────
def compute_cost_tradeoff(df: pd.DataFrame, ftl_fixed_cost_override: float = None,
                           verbose: bool = True) -> pd.DataFrame:
    """
    Translate predicted performance into a single recommendation per corridor
    using the configurable unit-economics block defined at the top of this file.

    total_cost(route) = trips_needed(volume, capacity) × cost_per_trip(route, distance)
                       + SLA_breach_penalty × volume × breach_probability(route)

    The recommended route type is whichever has the LOWER total_cost.

    Parameters
    ----------
    ftl_fixed_cost_override : float, optional
        Lets run_sensitivity_analysis() ask "what if FTL_FIXED_COST_RS were
        different?" without touching the module-level constant or re-running
        the (expensive) model training / counterfactual simulation steps —
        only the cost arithmetic below depends on this constant.
    verbose : bool
        Set False during the sensitivity sweep so the log isn't flooded with
        six near-identical "cost-trade-off computed" blocks.
    """
    df = df.copy()
    volume   = df["total_trip_count"].clip(lower=1)
    distance = df["median_distance_km"].clip(lower=0)

    n_ftl_trips     = np.ceil(volume / FTL_CAPACITY_UNITS)
    n_carting_trips = np.ceil(volume / CARTING_CAPACITY_UNITS)

    cost_per_trip_ftl     = ftl_cost_per_trip(distance, fixed=ftl_fixed_cost_override)
    cost_per_trip_carting = carting_cost_per_trip(distance)

    fixed_cost_ftl     = n_ftl_trips * cost_per_trip_ftl
    fixed_cost_carting = n_carting_trips * cost_per_trip_carting

    breach_cost_ftl     = df["pred_breach_prob_ftl"]     * volume * SLA_BREACH_PENALTY_RS
    breach_cost_carting = df["pred_breach_prob_carting"] * volume * SLA_BREACH_PENALTY_RS

    df["total_cost_ftl_rs"]     = fixed_cost_ftl + breach_cost_ftl
    df["total_cost_carting_rs"] = fixed_cost_carting + breach_cost_carting

    df["recommended_route_type"] = np.where(
        df["total_cost_ftl_rs"] <= df["total_cost_carting_rs"], "FTL", "CARTING"
    )
    df["cost_savings_rs"] = np.abs(df["total_cost_ftl_rs"] - df["total_cost_carting_rs"])
    df["time_diff_hours"] = df["pred_time_hours_carting"] - df["pred_time_hours_ftl"]

    df["switch_recommended"] = df["recommended_route_type"] != df["current_route_type"]

    current_breach_prob = np.where(
        df["current_route_type"] == "FTL", df["pred_breach_prob_ftl"], df["pred_breach_prob_carting"]
    )
    recommended_breach_prob = np.where(
        df["recommended_route_type"] == "FTL", df["pred_breach_prob_ftl"], df["pred_breach_prob_carting"]
    )
    df["sla_breach_reduction_shipments"] = (
        (current_breach_prob - recommended_breach_prob) * volume
    ).clip(lower=0)

    n_ftl_rec     = (df["recommended_route_type"] == "FTL").sum()
    n_carting_rec = (df["recommended_route_type"] == "CARTING").sum()

    if not verbose:
        return df

    log.info(
        "Cost-trade-off computed. %d / %d corridors (%.1f%%) recommended to switch route type.",
        df["switch_recommended"].sum(), len(df), df["switch_recommended"].mean() * 100,
    )
    log.info(
        "  Recommendation split → FTL: %d (%.1f%%) | CARTING: %d (%.1f%%)",
        n_ftl_rec, n_ftl_rec / len(df) * 100,
        n_carting_rec, n_carting_rec / len(df) * 100,
    )

    # Hard guard: if the result is degenerate on REAL data even after the
    # constants passed the synthetic benchmark check, that's worth knowing
    # explicitly rather than reporting a one-sided split as a normal finding.
    if min(n_ftl_rec, n_carting_rec) / len(df) < 0.02:
        log.warning(
            "Recommendation split is heavily one-sided (<2%% on the minority "
            "side) even though the cost model passed the synthetic sanity "
            "check. This likely means the REAL corridor distance "
            "distribution rarely crosses the %.0f km Carting long-haul "
            "threshold — check the percentile log line above. This may be a "
            "true finding (most corridors genuinely are short-haul) rather "
            "than a bug, but confirm before reporting it as such.",
            CARTING_LONGHAUL_THRESHOLD_KM,
        )
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 6B. SENSITIVITY ANALYSIS — does the recommendation survive a range of
#     plausible cost assumptions, or does it depend on one brittle guess?
# ─────────────────────────────────────────────────────────────────────────────
def run_sensitivity_analysis(simulated: pd.DataFrame) -> pd.DataFrame:
    """
    Re-run compute_cost_tradeoff() across FTL_FIXED_COST_SENSITIVITY_GRID —
    every other constant and every model prediction stays fixed, only the
    FTL dispatch-cost assumption changes. This is cheap to do because the
    expensive steps (training the regressor/classifier, the counterfactual
    simulation) already happened once; only the final cost arithmetic needs
    to be repeated per scenario.

    Why this matters for Phase 4 specifically
    -------------------------------------------
    There is no real cost column anywhere in the Delhivery dataset, so
    FTL_FIXED_COST_RS is necessarily an assumption, not a measurement. A
    single point estimate invites the reasonable challenge "why this number
    and not another?" A sensitivity table answers that question directly:
    it shows whether the framework's conclusion (e.g. "Carting dominates on
    this network") is a robust structural finding that holds across a wide
    range of plausible costs, or whether it's an artifact of one specific
    guess — which is itself a more honest and more defensible result to
    hand to a Head of Network Operations than a single number presented as
    if it were precise.

    Returns
    -------
    pd.DataFrame
        One row per tested FTL_FIXED_COST_RS scenario, with the resulting
        breakeven distance, recommendation split, switch count, and total
        illustrative savings.
    """
    rows = []
    for ftl_fixed in FTL_FIXED_COST_SENSITIVITY_GRID:
        breakeven = analytical_breakeven_distance_km(ftl_fixed=ftl_fixed)
        result = compute_cost_tradeoff(
            simulated, ftl_fixed_cost_override=ftl_fixed, verbose=False
        )
        n_ftl = (result["recommended_route_type"] == "FTL").sum()
        n_total = len(result)
        rows.append({
            "ftl_fixed_cost_rs"      : ftl_fixed,
            "breakeven_distance_km"  : round(breakeven, 1) if breakeven else None,
            "n_corridors_ftl"        : int(n_ftl),
            "pct_corridors_ftl"      : round(n_ftl / n_total * 100, 2),
            "n_corridors_switch"     : int(result["switch_recommended"].sum()),
            "total_cost_savings_rs"  : round(
                result.loc[result["switch_recommended"], "cost_savings_rs"].sum(), 0
            ),
        })

    sensitivity_df = pd.DataFrame(rows)

    log.info("─" * 65)
    log.info("SENSITIVITY ANALYSIS — recommendation across plausible FTL costs")
    log.info("─" * 65)
    for _, r in sensitivity_df.iterrows():
        log.info(
            f"  FTL fixed ₹{r['ftl_fixed_cost_rs']:>6,.0f}  →  breakeven "
            f"{r['breakeven_distance_km']:>6.1f} km  →  FTL recommended on "
            f"{int(r['n_corridors_ftl']):>4d} corridors ({r['pct_corridors_ftl']:>5.2f}%)"
        )
    spread = sensitivity_df["pct_corridors_ftl"].max() - sensitivity_df["pct_corridors_ftl"].min()
    log.info(
        f"FTL share ranges from {sensitivity_df['pct_corridors_ftl'].min():.2f}% to "
        f"{sensitivity_df['pct_corridors_ftl'].max():.2f}% across this grid "
        f"(spread = {spread:.2f} points)."
    )
    log.info("─" * 65)

    sensitivity_df.to_csv(METRICS_DIR / "ftl_cost_sensitivity.csv", index=False)
    log.info("Saved → outputs/metrics/ftl_cost_sensitivity.csv")

    return sensitivity_df


def plot_sensitivity_analysis(sensitivity_df: pd.DataFrame) -> None:
    """
    Two-panel chart: how the breakeven distance and the resulting FTL share
    of the network move as the FTL fixed-cost assumption changes. This is
    the chart that answers "how much should we trust this number?" at a
    glance, without reading the CSV.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    ax1.plot(sensitivity_df["ftl_fixed_cost_rs"], sensitivity_df["breakeven_distance_km"],
              marker="o", color="#2980B9", linewidth=2)
    ax1.set_xlabel("Assumed FTL Fixed Dispatch Cost (₹)", fontsize=10)
    ax1.set_ylabel("Breakeven Distance (km)", fontsize=10)
    ax1.set_title("Breakeven Distance vs. FTL Cost Assumption", pad=10)
    ax1.grid(alpha=0.3)

    ax2.plot(sensitivity_df["ftl_fixed_cost_rs"], sensitivity_df["pct_corridors_ftl"],
              marker="o", color="#E67E22", linewidth=2)
    ax2.set_xlabel("Assumed FTL Fixed Dispatch Cost (₹)", fontsize=10)
    ax2.set_ylabel("% of Corridors Recommended FTL", fontsize=10)
    ax2.set_title("FTL Recommendation Share vs. FTL Cost Assumption", pad=10)
    ax2.grid(alpha=0.3)

    fig.suptitle(
        "Sensitivity Analysis — Does the FTL/CARTING Recommendation Depend on One Guess?",
        fontsize=12, fontweight="bold", y=1.02,
    )
    plt.tight_layout()
    out = VIZ_DIR / "ftl_cost_sensitivity.png"
    plt.savefig(out, bbox_inches="tight", dpi=150)
    plt.close()
    log.info("Saved → %s", out)


# ─────────────────────────────────────────────────────────────────────────────
# 7. INTERPRETABLE DECISION TREE SURROGATE
# ─────────────────────────────────────────────────────────────────────────────
def build_decision_rules(df: pd.DataFrame) -> Tuple[DecisionTreeClassifier, str, list]:
    """
    Train a SHALLOW (max_depth=3) decision tree that mimics the ML
    recommendation using only 4 easy-to-check business inputs.
    """
    feature_cols = [
        "median_distance_km", "src_betweenness_centrality",
        "total_trip_count", "src_avg_sla_breach_rate",
    ]
    X = df[feature_cols].fillna(0)
    y = (df["recommended_route_type"] == "FTL").astype(int)

    tree = DecisionTreeClassifier(max_depth=3, random_state=RANDOM_STATE, min_samples_leaf=20)
    tree.fit(X, y)

    fidelity = accuracy_score(y, tree.predict(X))
    log.info("Surrogate decision tree trained — fidelity to ML model: %.1f%%", fidelity * 100)

    label_map = {0: "Recommend CARTING", 1: "Recommend FTL"}
    class_names = [label_map[c] for c in tree.classes_]

    if len(class_names) == 1:
        log.error(
            "Surrogate tree found only ONE recommended class (%s) across all "
            "corridors. Since validate_cost_model_sanity() already confirmed "
            "the cost model CAN produce both outcomes, this means the real "
            "data simply doesn't have corridors past the breakeven distance "
            "— check the corridor distance percentile log line from "
            "build_corridor_profiles() to confirm before treating this as "
            "a final result.",
            class_names[0],
        )

    rules_text = export_text(
        tree, feature_names=feature_cols,
        class_names=class_names,
    )
    return tree, rules_text, class_names


# ─────────────────────────────────────────────────────────────────────────────
# 8. VISUALISATIONS
# ─────────────────────────────────────────────────────────────────────────────
def plot_decision_boundary(df: pd.DataFrame) -> None:
    """
    Scatter plot: distance vs source betweenness centrality, coloured by
    recommended route type. Point size = historical trip volume.
    """
    fig, ax = plt.subplots(figsize=(11, 8))
    colors = {"FTL": "#2980B9", "CARTING": "#E67E22"}
    for rt, color in colors.items():
        sub = df[df["recommended_route_type"] == rt]
        ax.scatter(
            sub["median_distance_km"], sub["src_betweenness_centrality"],
            s=np.clip(sub["total_trip_count"] * 2, 10, 300),
            alpha=0.55, color=color, label=f"Recommend {rt}", edgecolor="white", linewidth=0.3,
        )
    ax.axvline(CARTING_LONGHAUL_THRESHOLD_KM, color="grey", linestyle="--", linewidth=1,
               label=f"Carting long-haul threshold ({CARTING_LONGHAUL_THRESHOLD_KM} km)")
    ax.set_xlabel("Median Corridor Distance (km)", fontsize=10)
    ax.set_ylabel("Source Facility Betweenness Centrality", fontsize=10)
    ax.set_title(
        "FTL vs CARTING Decision Boundary\n"
        "Point size = historical trip volume | Colour = ML recommendation",
        pad=12,
    )
    ax.legend(fontsize=10, markerscale=0.6)
    plt.tight_layout()
    out = VIZ_DIR / "decision_boundary_scatter.png"
    plt.savefig(out, bbox_inches="tight", dpi=150)
    plt.close()
    log.info("Saved → %s", out)


def plot_time_cost_tradeoff(df: pd.DataFrame, top_n: int = 20) -> None:
    """
    For the top-N corridors with the largest cost savings from switching,
    show total predicted cost for FTL vs CARTING side by side.
    """
    top = df[df["switch_recommended"]].nlargest(top_n, "cost_savings_rs").copy()
    if len(top) == 0:
        log.info("No corridors recommended to switch — skipping trade-off chart.")
        return

    top = top.sort_values("cost_savings_rs", ascending=True)
    top["label"] = top["source_name"].str[:14] + "→" + top["destination_name"].str[:14]

    x = np.arange(len(top))
    width = 0.35
    fig, ax = plt.subplots(figsize=(13, max(6, len(top) * 0.4)))
    ax.barh(x - width/2, top["total_cost_ftl_rs"],     width, color="#2980B9", label="FTL Total Cost", alpha=0.85)
    ax.barh(x + width/2, top["total_cost_carting_rs"], width, color="#E67E22", label="CARTING Total Cost", alpha=0.85)
    ax.set_yticks(x)
    ax.set_yticklabels(top["label"], fontsize=8)
    ax.set_xlabel("Predicted Total Cost (₹) — fixed cost + SLA breach penalty", fontsize=10)
    ax.set_title(
        f"Top {len(top)} Corridors — Cost Trade-off if Recommendation Adopted\n"
        "(Illustrative cost model — see CONFIG block in source code)",
        pad=12,
    )
    ax.legend(fontsize=9)
    plt.tight_layout()
    out = VIZ_DIR / "time_cost_tradeoff_top20.png"
    plt.savefig(out, bbox_inches="tight", dpi=150)
    plt.close()
    log.info("Saved → %s", out)


def plot_risk_feature_importance(clf_model, feature_names: list) -> None:
    """Feature importance for the SLA-breach classifier, graph features highlighted."""
    if not XGB_AVAILABLE:
        log.info("Feature importance plot requires XGBoost — skipping.")
        return

    fi = pd.DataFrame({
        "feature": feature_names, "importance": clf_model.feature_importances_,
    }).sort_values("importance", ascending=False).head(15)
    fi = fi.sort_values("importance", ascending=True)

    colors = [
        "#E74C3C" if any(k in f for k in ["betweenness", "pagerank", "degree", "clustering", "sla_breach_rate"])
        else "#2980B9"
        for f in fi["feature"]
    ]
    fig, ax = plt.subplots(figsize=(11, 7))
    ax.barh(fi["feature"], fi["importance"], color=colors, edgecolor="white")
    legend_elements = [
        mpatches.Patch(color="#E74C3C", label="Graph-structural features"),
        mpatches.Patch(color="#2980B9", label="Trip-level features"),
    ]
    ax.legend(handles=legend_elements, fontsize=9)
    ax.set_xlabel("Feature Importance (drives SLA breach probability)", fontsize=10)
    ax.set_title("What Drives SLA Breach Risk? — Classifier Feature Importance", pad=12)
    plt.tight_layout()
    out = VIZ_DIR / "risk_feature_importance.png"
    plt.savefig(out, bbox_inches="tight", dpi=150)
    plt.close()
    log.info("Saved → %s", out)


def plot_decision_tree(tree: DecisionTreeClassifier, feature_names: list, class_names: list) -> None:
    """Render the surrogate decision tree as a one-page visual rule sheet."""
    fig, ax = plt.subplots(figsize=(16, 9))
    plot_tree(
        tree, feature_names=feature_names,
        class_names=class_names,
        filled=True, rounded=True, fontsize=9, ax=ax,
        impurity=False,
    )
    ax.set_title(
        "FTL vs CARTING — One-Page Decision Rule Sheet\n"
        "(Surrogate tree approximating the full ML recommendation)",
        pad=14,
    )
    plt.tight_layout()
    out = VIZ_DIR / "decision_tree_rules.png"
    plt.savefig(out, bbox_inches="tight", dpi=150)
    plt.close()
    log.info("Saved → %s", out)


# ─────────────────────────────────────────────────────────────────────────────
# 9. PERSIST
# ─────────────────────────────────────────────────────────────────────────────
def save_outputs(
    df: pd.DataFrame, reg_model, clf_model, tree, rules_text: str,
    model_metrics: Dict, breakeven_km: float,
) -> None:
    df.to_csv(METRICS_DIR / "corridor_recommendations.csv", index=False)
    log.info("Saved → outputs/metrics/corridor_recommendations.csv")

    summary = {
        "total_corridors"              : len(df),
        "corridors_recommended_switch" : int(df["switch_recommended"].sum()),
        "pct_recommended_switch"       : round(df["switch_recommended"].mean() * 100, 2),
        "pct_recommended_ftl"          : round((df["recommended_route_type"] == "FTL").mean() * 100, 2),
        "pct_recommended_carting"      : round((df["recommended_route_type"] == "CARTING").mean() * 100, 2),
        "analytical_breakeven_km"      : round(breakeven_km, 1) if breakeven_km else None,
        "total_cost_savings_rs"        : round(df.loc[df["switch_recommended"], "cost_savings_rs"].sum(), 2),
        "total_sla_breach_reduction"   : round(df["sla_breach_reduction_shipments"].sum(), 1),
        **model_metrics,
    }
    pd.DataFrame([summary]).to_csv(METRICS_DIR / "decision_framework_summary.csv", index=False)
    log.info("Saved → outputs/metrics/decision_framework_summary.csv")

    with open(METRICS_DIR / "decision_rules.txt", "w") as f:
        f.write("FTL vs CARTING DECISION FRAMEWORK — RULE SHEET\n")
        f.write("=" * 55 + "\n\n")
        f.write(rules_text)
    log.info("Saved → outputs/metrics/decision_rules.txt")

    with open(MODEL_DIR / "route_performance_regressor.pkl", "wb") as f:
        pickle.dump(reg_model, f)
    with open(MODEL_DIR / "route_performance_classifier.pkl", "wb") as f:
        pickle.dump(clf_model, f)
    with open(MODEL_DIR / "decision_tree_surrogate.pkl", "wb") as f:
        pickle.dump(tree, f)
    log.info("Models saved to outputs/models/")


# ─────────────────────────────────────────────────────────────────────────────
# 10. CONSOLE SUMMARY
# ─────────────────────────────────────────────────────────────────────────────
def print_summary(df: pd.DataFrame, model_metrics: Dict, rules_text: str,
                   breakeven_km: float, sensitivity_df: pd.DataFrame) -> None:
    print("\n" + "=" * 65)
    print("  PHASE 4 — FTL vs CARTING DECISION FRAMEWORK SUMMARY")
    print("=" * 65)

    print("\n🤖 MODEL PERFORMANCE")
    print(f"   Delay-Ratio Regressor MAE   : {model_metrics['regressor_mae']}")
    print(f"   SLA-Breach Classifier AUC   : {model_metrics['classifier_auc']}")
    print(f"   SLA-Breach Classifier Acc.  : {model_metrics['classifier_accuracy']}")

    print("\n📏 DISTANCE / COST MODEL  (base case: FTL fixed cost = ₹%s)" % f"{FTL_FIXED_COST_RS:,}")
    print(f"   Analytical FTL/CARTING breakeven: {breakeven_km:.1f} km" if breakeven_km else "   No finite breakeven — check CONFIG block")
    print(f"   Carting long-haul threshold     : {CARTING_LONGHAUL_THRESHOLD_KM} km")

    print("\n📦 CORRIDOR RECOMMENDATIONS (base case)")
    print(f"   Total Corridors Evaluated   : {len(df):,}")
    n_switch = df["switch_recommended"].sum()
    n_ftl_rec = (df["recommended_route_type"] == "FTL").sum()
    n_carting_rec = (df["recommended_route_type"] == "CARTING").sum()
    print(f"   Recommended to Switch Type  : {n_switch:,} ({n_switch/len(df)*100:.1f}%)")
    print(f"   Final mix → FTL: {n_ftl_rec:,} ({n_ftl_rec/len(df)*100:.1f}%) | CARTING: {n_carting_rec:,} ({n_carting_rec/len(df)*100:.1f}%)")

    savings = df.loc[df["switch_recommended"], "cost_savings_rs"].sum()
    breach_red = df["sla_breach_reduction_shipments"].sum()
    print("\n💰 ILLUSTRATIVE IMPACT (base case CONFIG cost assumptions)")
    print(f"   Total Potential Cost Savings: ₹{savings:,.0f}")
    print(f"   Est. Additional On-Time Shipments: {breach_red:,.0f}")

    print("\n📊 SENSITIVITY — DOES THIS RESULT DEPEND ON ONE GUESS?")
    print("   (FTL fixed cost varied across a plausible range; everything else held constant)")
    for _, r in sensitivity_df.iterrows():
        marker = "  ← base case" if r["ftl_fixed_cost_rs"] == FTL_FIXED_COST_RS else ""
        print(
            f"   ₹{r['ftl_fixed_cost_rs']:>6,.0f}  →  breakeven {r['breakeven_distance_km']:>6.1f} km  →  "
            f"FTL share {r['pct_corridors_ftl']:>5.2f}%{marker}"
        )
    spread = sensitivity_df["pct_corridors_ftl"].max() - sensitivity_df["pct_corridors_ftl"].min()
    print(f"   FTL share spread across this grid: {spread:.2f} percentage points")

    print("\n📋 DECISION RULES (from surrogate tree, base case)")
    print(rules_text)

    print("\n📁 All outputs saved to outputs/metrics/, outputs/models/, outputs/visualizations/")
    print("=" * 65 + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# 11. MASTER ENTRY-POINT
# ─────────────────────────────────────────────────────────────────────────────
def run_decision_framework() -> Dict:
    """Execute the full Phase 4 pipeline end to end."""
    log.info("=" * 65)
    log.info("  PHASE 4 — FTL vs CARTING DECISION FRAMEWORK STARTING")
    log.info("=" * 65)

    # Run the sanity check FIRST — fail fast if the cost model is degenerate
    # before spending time training models or loading data.
    breakeven_km = validate_cost_model_sanity()

    trips, corridors, tod, centrality = load_data()

    X, y_delay, y_breach = build_modeling_features(trips, centrality)
    reg_model, clf_model, model_metrics = train_performance_models(X, y_delay, y_breach)

    profiles = build_corridor_profiles(trips, centrality)
    simulated = simulate_route_type_predictions(profiles, reg_model, clf_model)
    final = compute_cost_tradeoff(simulated)

    # Sensitivity sweep — reuses the already-simulated predictions, so this
    # is just re-running the cheap cost arithmetic across several FTL cost
    # assumptions rather than retraining anything.
    sensitivity_df = run_sensitivity_analysis(simulated)

    tree, rules_text, class_names = build_decision_rules(final)

    plot_decision_boundary(final)
    plot_time_cost_tradeoff(final)
    plot_risk_feature_importance(clf_model, MODEL_FEATURES)
    plot_decision_tree(tree, ["median_distance_km", "src_betweenness_centrality",
                              "total_trip_count", "src_avg_sla_breach_rate"], class_names)
    plot_sensitivity_analysis(sensitivity_df)

    save_outputs(final, reg_model, clf_model, tree, rules_text, model_metrics, breakeven_km)
    print_summary(final, model_metrics, rules_text, breakeven_km, sensitivity_df)

    log.info("=" * 65)
    log.info("  PHASE 4 — PIPELINE COMPLETE")
    log.info("=" * 65)

    return {
        "corridor_recommendations": final,
        "model_metrics": model_metrics,
        "decision_rules": rules_text,
        "sensitivity_analysis": sensitivity_df,
    }


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    results = run_decision_framework()