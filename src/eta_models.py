"""
eta_models.py
=============
Phase 3 – Graph-Enhanced ETA Prediction Model
Project : Optimizing Delivery ETAs with Graph-Based Network Intelligence
Author  : IIT Guwahati Consulting & Analytics Club

Responsibilities
----------------
1. Build a Baseline XGBoost regression model using only trip-level features.
2. Build a Graph-Enhanced XGBoost model that augments trip features with:
      (a) node2vec embeddings for source and destination facilities
      (b) Centrality metrics (betweenness, pagerank, degree) per facility
3. Evaluate both models on MAE and Within-15% Accuracy.
4. Quantify and visualise the "Graph Advantage" — the measurable improvement
   the graph-enhanced model delivers over the baseline.
5. Save model artefacts, benchmark report, and visualisations.

Why XGBoost as the base learner (not a neural network)?
---------------------------------------------------------
The dataset has ~140K trip segments — large enough for gradient boosting
but below the scale where deep learning adds reliable lift. XGBoost:
  - Handles mixed feature types (numeric + categorical embeddings) natively
  - Is explainable (SHAP values usable in Phase 5 memo)
  - Trains in seconds, not hours, enabling fast iteration
The graph intelligence is injected via node2vec EMBEDDINGS as input features,
not by replacing XGBoost with a GNN. This hybrid approach delivers the
structural awareness of graph learning with the training stability of boosting.

Why node2vec over GraphSAGE?
-----------------------------
GraphSAGE is optimal when new, unseen nodes appear at inference time
(inductive setting). Our logistics network is relatively STATIC —
the same 1,657 hubs operate day-to-day. In this transductive setting,
node2vec's random-walk-based embeddings capture structural equivalence
(two facilities with similar roles get similar embeddings) and are:
  - Faster to train (no GPU required)
  - Simpler to integrate with XGBoost
  - Equally expressive for our network size

GraphSAGE would be the right upgrade if Delhivery expands to new cities
(new nodes not seen at training time).

Evaluation Metrics
-------------------
MAE (Mean Absolute Error):
    Average absolute difference between predicted and actual time.
    Measures raw prediction accuracy in the same unit as delivery time.
    Lower = better.

Within-15% Accuracy:
    % of predictions where |predicted - actual| / actual ≤ 0.15
    The industry-standard SLA metric: "Did we predict ETA within 15%
    of what actually happened?" This directly translates to customer
    promise accuracy and SLA compliance.
    Higher = better.
"""

import logging
import pickle
import warnings
from pathlib import Path
from typing import Dict, Optional, Tuple

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

warnings.filterwarnings("ignore")

# ── Try importing optional heavy dependencies ─────────────────────────────────
try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False
    logging.warning("xgboost not found — falling back to GradientBoostingRegressor.")

try:
    from node2vec import Node2Vec
    N2V_AVAILABLE = True
except ImportError:
    N2V_AVAILABLE = False
    logging.warning(
        "node2vec not found. Install via: pip install node2vec\n"
        "Falling back to centrality-only graph features."
    )

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Project Paths ─────────────────────────────────────────────────────────────
PROJECT_ROOT  = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODEL_DIR     = PROJECT_ROOT / "outputs" / "models"
METRICS_DIR   = PROJECT_ROOT / "outputs" / "metrics"
VIZ_DIR       = PROJECT_ROOT / "outputs" / "visualizations"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
METRICS_DIR.mkdir(parents=True, exist_ok=True)
VIZ_DIR.mkdir(parents=True, exist_ok=True)

# ── Constants ─────────────────────────────────────────────────────────────────
RANDOM_STATE    = 42
TEST_SIZE       = 0.20        # 80/20 train-test split
TARGET_COL      = "segment_actual_time"
WITHIN_PCT      = 0.15        # 15% accuracy threshold
N2V_DIMENSIONS  = 32          # node2vec embedding size per node
N2V_WALK_LEN    = 20          # random walk length
N2V_NUM_WALKS   = 100         # walks per node
N2V_WORKERS     = 2           # parallel workers

plt.rcParams.update({
    "figure.dpi"       : 140,
    "font.size"        : 10,
    "axes.spines.top"  : False,
    "axes.spines.right": False,
    "axes.titleweight" : "bold",
    "axes.titlesize"   : 13,
})


# ─────────────────────────────────────────────────────────────────────────────
# 1. DATA LOADER
# ─────────────────────────────────────────────────────────────────────────────
def load_data() -> Tuple[pd.DataFrame, nx.MultiDiGraph, pd.DataFrame]:
    """
    Load all Phase 1 & Phase 2 artefacts needed for ETA modelling.

    Returns
    -------
    df           : Cleaned trip-level dataframe from Phase 1.
    G            : Directed graph from Phase 1 (graph_builder.py).
    centrality   : Node centrality metrics from Phase 2 (network_audit.py).
    """
    # ── Trip data ─────────────────────────────────────────────────────────
    trips_path = PROCESSED_DIR / "cleaned_trips.csv"
    if not trips_path.exists():
        raise FileNotFoundError(f"Run Phase 1 first. Not found: {trips_path}")
    df = pd.read_csv(trips_path, low_memory=False)
    log.info("Trip data loaded: %d rows", len(df))

    # ── Graph ─────────────────────────────────────────────────────────────
    graph_path = MODEL_DIR / "delhivery_graph.pkl"
    if not graph_path.exists():
        raise FileNotFoundError(f"Run Phase 1 first. Not found: {graph_path}")
    with open(graph_path, "rb") as f:
        G = pickle.load(f)
    log.info("Graph loaded: %d nodes | %d edges", G.number_of_nodes(), G.number_of_edges())

    # ── Centrality metrics ────────────────────────────────────────────────
    centrality_path = METRICS_DIR / "node_centrality_metrics.csv"
    if not centrality_path.exists():
        raise FileNotFoundError(f"Run Phase 2 first. Not found: {centrality_path}")
    centrality = pd.read_csv(centrality_path)
    log.info("Centrality metrics loaded: %d nodes", len(centrality))

    return df, G, centrality


# ─────────────────────────────────────────────────────────────────────────────
# 2. BASELINE FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────────────────────
def build_baseline_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Construct the feature matrix for the BASELINE model.
    Uses only features available from the raw trip data — NO graph structure.

    This represents what Delhivery's current OSRM-based system effectively knows:
    how long a route should take, how far it is, and what time of day it is.

    Features
    ---------
    segment_osrm_time        : OSRM's estimate for this leg (the current system's prediction)
    segment_osrm_distance    : OSRM distance estimate for this leg
    actual_distance_to_dest  : Real-world distance to final destination
    route_type_encoded       : FTL=0, CARTING=1
    trip_start_hour          : Hour of departure (0–23)
    trip_start_dayofweek     : Day of week (Mon=0, Sun=6)
    time_of_day_encoded      : Night=0, Morning=1, Afternoon=2, Evening=3
    cumulative_delay_ratio   : Historical delay ratio on this trip (proxy for traffic)

    Target
    -------
    segment_actual_time : Actual time taken for this leg (what we want to predict)

    Why These Features?
    --------------------
    The baseline must be honest — it should only use what a non-graph system
    would know. The performance gap between baseline and graph-enhanced model
    is then attributable purely to the graph structure, not better raw features.
    """
    df = df.copy()

    # ── Drop rows missing target or key features ──────────────────────────
    essential = [
        TARGET_COL, "segment_osrm_time", "segment_osrm_distance",
        "actual_distance_to_destination", "route_type",
    ]
    df = df.dropna(subset=essential)
    df = df[df[TARGET_COL] > 0].copy()

    # ── Encode categoricals ───────────────────────────────────────────────
    df["route_type_encoded"] = (df["route_type"] == "CARTING").astype(int)

    tod_map = {
        "NIGHT(0-6)"       : 0,
        "MORNING(6-12)"    : 1,
        "AFTERNOON(12-18)" : 2,
        "EVENING(18-24)"   : 3,
    }
    if "time_of_day" in df.columns:
        df["time_of_day_encoded"] = (
            df["time_of_day"].astype(str).str.upper().map(tod_map).fillna(1)
        )
    else:
        df["time_of_day_encoded"] = 1  # default to Morning if missing

    # ── Feature matrix ────────────────────────────────────────────────────
    feature_cols = [
        "segment_osrm_time",
        "segment_osrm_distance",
        "actual_distance_to_destination",
        "route_type_encoded",
        "trip_start_hour",
        "trip_start_dayofweek",
        "time_of_day_encoded",
        "cumulative_delay_ratio",
    ]

    # Fill any remaining numeric nulls with column median
    for col in feature_cols:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())
        else:
            df[col] = 0

    X = df[feature_cols].copy()
    y = df[TARGET_COL].copy()

    log.info(
        "Baseline features: %d rows × %d features | Target range: [%.2f, %.2f]",
        len(X), X.shape[1], y.min(), y.max(),
    )
    return X, y, df


# ─────────────────────────────────────────────────────────────────────────────
# 3. GRAPH FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────────────────────
def build_node2vec_embeddings(G: nx.MultiDiGraph) -> Optional[pd.DataFrame]:
    """
    Train node2vec on the logistics graph and return a DataFrame of
    node embeddings (one row per facility, N2V_DIMENSIONS columns).

    node2vec Algorithm (Plain English)
    ------------------------------------
    Imagine an ant walking randomly through the logistics network — sometimes
    it prefers to stay in the local neighbourhood (BFS-like, p < 1),
    sometimes it explores further (DFS-like, q < 1). After thousands of
    such walks, facilities that are visited together get similar embeddings.

    In logistics terms: two hubs that always appear on the same delivery
    paths (e.g., Delhi → Gurgaon → Jaipur) will get similar vector
    representations. The model then learns that "hubs with similar
    structural roles have similar delay patterns."

    Parameters Chosen
    ------------------
    dimensions = 32  : Balances expressiveness vs. overfitting.
                       128-dim helps for large graphs (>100K nodes);
                       32 is sufficient for our 1,657-node network.
    walk_length = 20 : Each ant takes 20 steps. Captures 2-hop neighbourhood.
    num_walks = 100  : 100 walks per node → statistically stable embeddings.
    p = 1, q = 0.5  : Biased toward DFS (exploration) — appropriate for
                       logistics where long-distance structural patterns
                       (national hubs vs regional hubs) matter more than
                       hyper-local clustering.

    Returns None if node2vec package is not installed.
    """
    if not N2V_AVAILABLE:
        log.warning("node2vec not available — skipping embeddings.")
        return None

    log.info("Training node2vec embeddings (this may take 1–2 minutes) …")

    # node2vec works on undirected or directed graphs
    # We use undirected projection to capture bidirectional corridor relationships
    G_undirected = G.to_undirected()

    n2v = Node2Vec(
        G_undirected,
        dimensions  = N2V_DIMENSIONS,
        walk_length = N2V_WALK_LEN,
        num_walks   = N2V_NUM_WALKS,
        p           = 1,
        q           = 0.5,
        workers     = N2V_WORKERS,
        quiet       = True,
    )
    model = n2v.fit(window=5, min_count=1, batch_words=4)

    # Extract embeddings into a DataFrame
    nodes      = list(G.nodes())
    embeddings = []
    for node in nodes:
        try:
            vec = model.wv[str(node)]
        except KeyError:
            vec = np.zeros(N2V_DIMENSIONS)
        embeddings.append(vec)

    emb_cols = [f"n2v_{i}" for i in range(N2V_DIMENSIONS)]
    emb_df   = pd.DataFrame(embeddings, columns=emb_cols)
    emb_df.insert(0, "facility_id", [str(n) for n in nodes])

    # Save embeddings for reuse
    emb_path = MODEL_DIR / "node2vec_embeddings.pkl"
    with open(emb_path, "wb") as f:
        pickle.dump((emb_df, model), f)

    log.info("node2vec embeddings trained and saved → %s", emb_path)
    return emb_df


def build_graph_enhanced_features(
    df_with_base: pd.DataFrame,
    centrality: pd.DataFrame,
    embeddings: Optional[pd.DataFrame],
    X_baseline: pd.DataFrame,
) -> pd.DataFrame:
    """
    Augment the baseline feature matrix with graph-derived features:

    (A) CENTRALITY FEATURES (always available, from Phase 2)
        For both source and destination facility:
        - betweenness_centrality : How critical this hub is structurally
        - pagerank               : How important this hub is network-wide
        - in_degree_raw          : Volume of inbound routes
        - out_degree_raw         : Volume of outbound routes
        - avg_incident_delay     : Historical average delay at this hub

    (B) node2vec EMBEDDING FEATURES (if node2vec is installed)
        32-dim embedding for source facility + 32-dim for destination
        = 64 additional features capturing structural neighbourhood context

    Why This Matters for ETA Prediction
    -------------------------------------
    The baseline model sees: "This OSRM leg is 2 hours, it's morning, it's FTL."
    The graph model adds: "The SOURCE hub has BC=0.27 (critical bottleneck)
    and the DESTINATION has in_degree=45 (highly congested receiver)."

    A 2-hour OSRM estimate on a critical-hub-to-congested-hub corridor
    should be inflated more aggressively than the same 2-hour estimate
    on a quiet local-to-local corridor. The graph features teach the
    model to make that distinction automatically.
    """
    log.info("Building graph-enhanced feature matrix …")

    # Centrality lookup maps
    c = centrality.set_index("facility_id")
    centality_features = [
        "betweenness_centrality", "pagerank",
        "in_degree_raw", "out_degree_raw",
        "avg_incident_delay", "avg_sla_breach_rate",
    ]

    def get_centrality(facility_id, feat):
        try:
            val = c.loc[str(facility_id), feat]
            return float(val) if not pd.isna(val) else 0.0
        except (KeyError, TypeError):
            return 0.0

    X = X_baseline.copy()

    # ── (A) Source centrality features ────────────────────────────────────
    src_ids = df_with_base["source_center"].astype(str)
    dst_ids = df_with_base["destination_center"].astype(str)

    for feat in centality_features:
        X[f"src_{feat}"]  = src_ids.map(lambda x, f=feat: get_centrality(x, f))
        X[f"dst_{feat}"]  = dst_ids.map(lambda x, f=feat: get_centrality(x, f))

    # ── Corridor-level graph features ─────────────────────────────────────
    # Difference in centrality: captures "direction" of the flow
    # (from critical hub to local = different pattern than local to critical)
    X["centrality_differential"] = (
        X["src_betweenness_centrality"] - X["dst_betweenness_centrality"]
    )
    X["pagerank_differential"] = (
        X["src_pagerank"] - X["dst_pagerank"]
    )
    X["degree_ratio"] = (
        X["src_out_degree_raw"] / (X["dst_in_degree_raw"] + 1)
    )

    # ── (B) node2vec embedding features ──────────────────────────────────
    if embeddings is not None:
        emb_map = embeddings.set_index("facility_id")
        emb_cols = [c for c in embeddings.columns if c.startswith("n2v_")]

        src_emb = src_ids.map(
            lambda x: emb_map.loc[x].values if x in emb_map.index else np.zeros(N2V_DIMENSIONS)
        )
        dst_emb = dst_ids.map(
            lambda x: emb_map.loc[x].values if x in emb_map.index else np.zeros(N2V_DIMENSIONS)
        )

        src_emb_df = pd.DataFrame(
            np.stack(src_emb.values),
            columns=[f"src_{c}" for c in emb_cols],
            index=X.index,
        )
        dst_emb_df = pd.DataFrame(
            np.stack(dst_emb.values),
            columns=[f"dst_{c}" for c in emb_cols],
            index=X.index,
        )
        X = pd.concat([X, src_emb_df, dst_emb_df], axis=1)
        log.info("node2vec embeddings added: %d additional features", len(emb_cols) * 2)
    else:
        log.info("Centrality-only graph features used (node2vec not available).")

    log.info(
        "Graph-enhanced features: %d rows × %d features (baseline had %d)",
        len(X), X.shape[1], X_baseline.shape[1],
    )
    return X


# ─────────────────────────────────────────────────────────────────────────────
# 4. MODEL TRAINING
# ─────────────────────────────────────────────────────────────────────────────
def get_model():
    """
    Return the best available regression model.
    XGBoost preferred; falls back to sklearn GradientBoostingRegressor.
    """
    if XGB_AVAILABLE:
        return xgb.XGBRegressor(
            n_estimators      = 500,
            learning_rate     = 0.05,
            max_depth         = 7,
            subsample         = 0.8,
            colsample_bytree  = 0.8,
            min_child_weight  = 5,
            reg_alpha         = 0.1,
            reg_lambda        = 1.0,
            random_state      = RANDOM_STATE,
            n_jobs            = -1,
            verbosity         = 0,
        )
    else:
        return GradientBoostingRegressor(
            n_estimators  = 300,
            learning_rate = 0.05,
            max_depth     = 5,
            subsample     = 0.8,
            random_state  = RANDOM_STATE,
        )


def train_model(X_train: pd.DataFrame, y_train: pd.Series, label: str):
    """Train the regression model and return the fitted model."""
    model = get_model()
    model_type = "XGBoost" if XGB_AVAILABLE else "GradientBoosting"
    log.info("Training %s [%s] on %d samples × %d features …",
             model_type, label, len(X_train), X_train.shape[1])
    model.fit(X_train, y_train)
    log.info("%s [%s] training complete ✓", model_type, label)
    return model


# ─────────────────────────────────────────────────────────────────────────────
# 5. EVALUATION METRICS
# ─────────────────────────────────────────────────────────────────────────────
def evaluate_model(
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    label: str,
) -> Dict:
    """
    Evaluate a trained model and return all benchmark metrics.

    Metrics Computed
    -----------------
    MAE (Mean Absolute Error):
        Average absolute gap between predicted and actual time.
        e.g. MAE=1.5 hours means predictions are off by 1.5 hours on average.

    Within-15% Accuracy:
        % of trips where the prediction is within ±15% of actual.
        This is the key SLA metric — each % point improvement here
        directly reduces customer-facing late delivery notifications.

    RMSE (Root Mean Squared Error):
        Penalises large errors more heavily than MAE.
        Useful for catching models that occasionally make catastrophic predictions.

    Bias (Mean Error):
        Average of (predicted - actual). Positive = over-estimating,
        negative = under-estimating. OSRM is known to under-estimate
        (negative bias), so a good model should have bias closer to 0.

    Within-25% Accuracy:
        Secondary threshold — shows if the model at least gets in the
        right ballpark even when missing the 15% threshold.
    """
    y_pred = model.predict(X_test)
    y_pred = np.maximum(y_pred, 0)  # predictions can't be negative time

    mae  = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(np.mean((y_pred - y_test) ** 2))
    bias = np.mean(y_pred - y_test)

    pct_error   = np.abs(y_pred - y_test) / (np.abs(y_test) + 1e-9)
    within_15   = (pct_error <= 0.15).mean() * 100
    within_25   = (pct_error <= 0.25).mean() * 100

    results = {
        "label"        : label,
        "n_test"       : len(y_test),
        "n_features"   : X_test.shape[1],
        "mae"          : round(mae, 4),
        "rmse"         : round(rmse, 4),
        "bias"         : round(bias, 4),
        "within_15_pct": round(within_15, 2),
        "within_25_pct": round(within_25, 2),
        "y_pred"       : y_pred,
        "y_test"       : y_test.values,
    }

    log.info(
        "[%s] MAE=%.4f | RMSE=%.4f | Within-15%%=%.2f%% | Within-25%%=%.2f%%",
        label, mae, rmse, within_15, within_25,
    )
    return results


# ─────────────────────────────────────────────────────────────────────────────
# 6. BENCHMARKING — GRAPH ADVANTAGE
# ─────────────────────────────────────────────────────────────────────────────
def compute_graph_advantage(
    baseline_results: Dict,
    enhanced_results: Dict,
) -> Dict:
    """
    Compute and log the 'Graph Advantage' — how much the graph-enhanced
    model improves over the baseline.

    Business Translation
    ---------------------
    Every 1% gain in Within-15% Accuracy = 1% more trips with accurate
    ETA promises = fewer customer complaints = fewer SLA penalty payments.

    For a network processing ~140K trips/month, a 5% Within-15% improvement
    = ~7,000 additional trips per month delivered with accurate ETA promises.
    """
    mae_reduction_pct = (
        (baseline_results["mae"] - enhanced_results["mae"])
        / baseline_results["mae"] * 100
    )
    within15_gain = enhanced_results["within_15_pct"] - baseline_results["within_15_pct"]
    within25_gain = enhanced_results["within_25_pct"] - baseline_results["within_25_pct"]
    bias_improvement = abs(baseline_results["bias"]) - abs(enhanced_results["bias"])
    graph_features_added = enhanced_results["n_features"] - baseline_results["n_features"]

    advantage = {
        "mae_reduction_pct"    : round(mae_reduction_pct, 2),
        "within15_gain_pts"    : round(within15_gain, 2),
        "within25_gain_pts"    : round(within25_gain, 2),
        "bias_improvement"     : round(bias_improvement, 4),
        "graph_features_added" : graph_features_added,
    }

    print("\n" + "=" * 65)
    print("  GRAPH ADVANTAGE — PHASE 3 BENCHMARK REPORT")
    print("=" * 65)
    print(f"\n  {'Metric':<35} {'Baseline':>12} {'Graph-Enhanced':>16} {'Δ':>10}")
    print("  " + "-" * 73)
    print(f"  {'MAE (hours)':<35} {baseline_results['mae']:>12.4f} {enhanced_results['mae']:>16.4f} {-mae_reduction_pct:>+9.2f}%")
    print(f"  {'RMSE (hours)':<35} {baseline_results['rmse']:>12.4f} {enhanced_results['rmse']:>16.4f}")
    print(f"  {'Prediction Bias (hours)':<35} {baseline_results['bias']:>12.4f} {enhanced_results['bias']:>16.4f} {bias_improvement:>+9.4f}")
    print(f"  {'Within-15% Accuracy':<35} {baseline_results['within_15_pct']:>11.2f}% {enhanced_results['within_15_pct']:>15.2f}% {within15_gain:>+9.2f}pp")
    print(f"  {'Within-25% Accuracy':<35} {baseline_results['within_25_pct']:>11.2f}% {enhanced_results['within_25_pct']:>15.2f}% {within25_gain:>+9.2f}pp")
    print(f"  {'Features Used':<35} {baseline_results['n_features']:>12} {enhanced_results['n_features']:>16} {graph_features_added:>+10}")
    print("\n  💡 KEY FINDING:")
    print(f"     Graph-enhanced model reduces MAE by {mae_reduction_pct:.1f}%")
    print(f"     and improves Within-15% accuracy by {within15_gain:+.1f} percentage points.")
    n_test = baseline_results["n_test"]
    extra_correct = int(n_test * within15_gain / 100)
    print(f"     On {n_test:,} test trips, this means {extra_correct:,} additional trips")
    print(f"     received an accurate ETA promise (within 15% of actual time).")
    print("=" * 65 + "\n")

    return advantage


# ─────────────────────────────────────────────────────────────────────────────
# 7. VISUALISATIONS
# ─────────────────────────────────────────────────────────────────────────────
def plot_prediction_comparison(
    baseline_results: Dict,
    enhanced_results: Dict,
) -> None:
    """
    Four-panel visualisation:
    Panel 1 — Predicted vs Actual scatter (Baseline)
    Panel 2 — Predicted vs Actual scatter (Graph-Enhanced)
    Panel 3 — Error distribution comparison
    Panel 4 — Metric benchmark bar chart
    """
    log.info("Generating prediction comparison visualisation …")

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(
        "Phase 3 — ETA Model Benchmark: Baseline vs Graph-Enhanced",
        fontsize=15, fontweight="bold", y=1.01,
    )

    clip_val = np.percentile(baseline_results["y_test"], 99)

    for ax, res, color, title in [
        (axes[0, 0], baseline_results,  "#2980B9", "Baseline Model\n(Trip Features Only)"),
        (axes[0, 1], enhanced_results,  "#E74C3C", "Graph-Enhanced Model\n(Trip + Graph Features)"),
    ]:
        y_t = np.clip(res["y_test"],  0, clip_val)
        y_p = np.clip(res["y_pred"],  0, clip_val)
        ax.scatter(y_t, y_p, alpha=0.08, s=3, color=color)
        lims = [0, clip_val]
        ax.plot(lims, lims,           "k--",  linewidth=1.2, label="Perfect prediction")
        ax.plot(lims, [l * 1.15 for l in lims], "--", color="orange",
                linewidth=1, alpha=0.8, label="+15% SLA boundary")
        ax.plot(lims, [l * 0.85 for l in lims], "--", color="orange",
                linewidth=1, alpha=0.8, label="-15% SLA boundary")
        ax.set_xlim(lims); ax.set_ylim(lims)
        ax.set_xlabel("Actual Time (hours)")
        ax.set_ylabel("Predicted Time (hours)")
        ax.set_title(
            f"{title}\nMAE={res['mae']:.3f}h | Within-15%={res['within_15_pct']:.1f}%"
        )
        ax.legend(fontsize=8)

    # ── Panel 3: Error distribution ───────────────────────────────────────
    ax = axes[1, 0]
    base_errors = baseline_results["y_pred"] - baseline_results["y_test"]
    enh_errors  = enhanced_results["y_pred"] - enhanced_results["y_test"]
    clip_e = np.percentile(np.abs(base_errors), 98)

    ax.hist(np.clip(base_errors, -clip_e, clip_e),
            bins=80, alpha=0.6, color="#2980B9", label="Baseline", density=True)
    ax.hist(np.clip(enh_errors,  -clip_e, clip_e),
            bins=80, alpha=0.6, color="#E74C3C", label="Graph-Enhanced", density=True)
    ax.axvline(0, color="black", linestyle="--", linewidth=1.5, label="Zero Error")
    ax.axvline(baseline_results["bias"], color="#2980B9",
               linestyle=":", linewidth=1.5, label=f"Baseline bias={baseline_results['bias']:.3f}h")
    ax.axvline(enhanced_results["bias"], color="#E74C3C",
               linestyle=":", linewidth=1.5, label=f"Enhanced bias={enhanced_results['bias']:.3f}h")
    ax.set_xlabel("Prediction Error (hours)  [Predicted − Actual]")
    ax.set_ylabel("Density")
    ax.set_title("Error Distribution\n(Narrower = More Consistent Predictions)")
    ax.legend(fontsize=8)

    # ── Panel 4: Metric comparison bars ──────────────────────────────────
    ax = axes[1, 1]
    metrics  = ["MAE (hours)", "RMSE (hours)", "Within-15% Acc (%)", "Within-25% Acc (%)"]
    base_vals = [
        baseline_results["mae"],
        baseline_results["rmse"],
        baseline_results["within_15_pct"],
        baseline_results["within_25_pct"],
    ]
    enh_vals = [
        enhanced_results["mae"],
        enhanced_results["rmse"],
        enhanced_results["within_15_pct"],
        enhanced_results["within_25_pct"],
    ]
    x     = np.arange(len(metrics))
    width = 0.35
    b1 = ax.bar(x - width/2, base_vals, width, label="Baseline",       color="#2980B9", alpha=0.85, edgecolor="white")
    b2 = ax.bar(x + width/2, enh_vals,  width, label="Graph-Enhanced", color="#E74C3C", alpha=0.85, edgecolor="white")

    for bar in list(b1) + list(b2):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f"{bar.get_height():.2f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=9)
    ax.set_title("Model Performance Comparison\n(For MAE/RMSE: lower is better | For accuracy: higher is better)")
    ax.legend(fontsize=9)

    plt.tight_layout()
    out = VIZ_DIR / "eta_model_benchmark.png"
    plt.savefig(out, bbox_inches="tight", dpi=150)
    plt.close()
    log.info("Saved → %s", out)


def plot_feature_importance(model, feature_names: list, label: str) -> None:
    """
    Plot top-20 feature importances for the graph-enhanced model.
    Highlights which graph features are contributing most to the improvement.
    """
    if not XGB_AVAILABLE:
        log.info("Feature importance plot requires XGBoost — skipping.")
        return

    importances = model.feature_importances_
    fi_df = pd.DataFrame({
        "feature"   : feature_names,
        "importance": importances,
    }).sort_values("importance", ascending=False).head(20)

    fig, ax = plt.subplots(figsize=(12, 8))
    colors = [
        "#E74C3C" if any(k in f for k in ["n2v_", "betweenness", "pagerank", "degree", "centrality", "sla_breach"])
        else "#2980B9"
        for f in fi_df["feature"]
    ]
    fi_df_plot = fi_df.sort_values("importance", ascending=True)
    ax.barh(fi_df_plot["feature"], fi_df_plot["importance"],
            color=[c for c in reversed(colors)], edgecolor="white", height=0.7)

    legend_elements = [
        mpatches.Patch(color="#E74C3C", label="Graph-derived features"),
        mpatches.Patch(color="#2980B9", label="Trip-level features (baseline)"),
    ]
    ax.legend(handles=legend_elements, fontsize=9)
    ax.set_xlabel("Feature Importance (XGBoost Gain)", fontsize=10)
    ax.set_title(
        f"Top 20 Feature Importances — {label}\n"
        "Red = graph features | Blue = trip-level features",
        pad=12,
    )
    plt.tight_layout()
    out = VIZ_DIR / f"feature_importance_{label.replace(' ', '_').lower()}.png"
    plt.savefig(out, bbox_inches="tight", dpi=150)
    plt.close()
    log.info("Saved → %s", out)


def plot_within15_by_route_type(
    df_test: pd.DataFrame,
    baseline_pred: np.ndarray,
    enhanced_pred: np.ndarray,
    y_test: np.ndarray,
) -> None:
    """
    Within-15% accuracy breakdown by route type (FTL vs CARTING).
    Shows whether the graph advantage is concentrated in one mode.
    """
    df_eval = df_test.copy()
    df_eval["y_test"]      = y_test
    df_eval["base_pred"]   = baseline_pred
    df_eval["enh_pred"]    = enhanced_pred

    df_eval["base_within15"] = (
        np.abs(df_eval["base_pred"] - df_eval["y_test"]) / (df_eval["y_test"] + 1e-9) <= 0.15
    ).astype(int)
    df_eval["enh_within15"] = (
        np.abs(df_eval["enh_pred"] - df_eval["y_test"]) / (df_eval["y_test"] + 1e-9) <= 0.15
    ).astype(int)

    if "route_type" not in df_eval.columns:
        log.info("route_type not in test set — skipping route-type breakdown.")
        return

    grouped = df_eval.groupby("route_type")[["base_within15", "enh_within15"]].mean() * 100

    x     = np.arange(len(grouped))
    width = 0.35
    fig, ax = plt.subplots(figsize=(9, 6))
    b1 = ax.bar(x - width/2, grouped["base_within15"], width,
                label="Baseline", color="#2980B9", alpha=0.85, edgecolor="white")
    b2 = ax.bar(x + width/2, grouped["enh_within15"],  width,
                label="Graph-Enhanced", color="#E74C3C", alpha=0.85, edgecolor="white")

    for bar in list(b1) + list(b2):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f"{bar.get_height():.1f}%", ha="center", va="bottom", fontsize=10)

    ax.set_xticks(x)
    ax.set_xticklabels(grouped.index, fontsize=12)
    ax.set_ylabel("Within-15% Accuracy (%)", fontsize=10)
    ax.set_ylim(0, 105)
    ax.axhline(100, color="green", linestyle="--", linewidth=1, alpha=0.4)
    ax.set_title(
        "Within-15% ETA Accuracy by Route Type\nBaseline vs Graph-Enhanced",
        pad=12,
    )
    ax.legend(fontsize=10)
    plt.tight_layout()
    out = VIZ_DIR / "within15_by_route_type.png"
    plt.savefig(out, bbox_inches="tight", dpi=150)
    plt.close()
    log.info("Saved → %s", out)


# ─────────────────────────────────────────────────────────────────────────────
# 8. PERSIST RESULTS
# ─────────────────────────────────────────────────────────────────────────────
def save_results(
    baseline_model,
    enhanced_model,
    baseline_results: Dict,
    enhanced_results: Dict,
    advantage: Dict,
) -> None:
    """Save models, benchmark table, and advantage summary."""

    # Models
    with open(MODEL_DIR / "eta_baseline_model.pkl", "wb") as f:
        pickle.dump(baseline_model, f)
    with open(MODEL_DIR / "eta_enhanced_model.pkl", "wb") as f:
        pickle.dump(enhanced_model, f)
    log.info("Models saved to outputs/models/")

    # Benchmark table
    bench = pd.DataFrame([
        {k: v for k, v in baseline_results.items() if k not in ("y_pred", "y_test")},
        {k: v for k, v in enhanced_results.items()  if k not in ("y_pred", "y_test")},
    ])
    bench.to_csv(METRICS_DIR / "eta_benchmark_report.csv", index=False)
    log.info("Benchmark report saved → outputs/metrics/eta_benchmark_report.csv")

    # Graph advantage
    adv_df = pd.DataFrame([advantage])
    adv_df.to_csv(METRICS_DIR / "graph_advantage_summary.csv", index=False)
    log.info("Graph advantage summary saved → outputs/metrics/graph_advantage_summary.csv")


# ─────────────────────────────────────────────────────────────────────────────
# 9. MASTER ENTRY-POINT
# ─────────────────────────────────────────────────────────────────────────────
def run_eta_pipeline() -> Dict:
    """
    Execute the full Phase 3 ETA prediction pipeline.

    Steps
    ------
    1.  Load all Phase 1 & 2 artefacts.
    2.  Build baseline feature matrix.
    3.  Train/test split (80/20, stratified by route_type).
    4.  Train Baseline XGBoost model.
    5.  Generate node2vec embeddings.
    6.  Build graph-enhanced feature matrix.
    7.  Train Graph-Enhanced XGBoost model.
    8.  Evaluate both models.
    9.  Compute and display Graph Advantage.
    10. Generate all visualisations.
    11. Save models and benchmark reports.

    Returns
    -------
    dict with keys: 'baseline', 'enhanced', 'advantage'
    """
    log.info("=" * 65)
    log.info("  PHASE 3 — ETA PREDICTION PIPELINE STARTING")
    log.info("=" * 65)

    # ── 1. Load data ──────────────────────────────────────────────────────
    df, G, centrality = load_data()

    # ── 2. Baseline features ──────────────────────────────────────────────
    X_base, y, df_feat = build_baseline_features(df)

    # ── 3. Train/test split ───────────────────────────────────────────────
    # Stratify by route_type to ensure both FTL and CARTING in both splits
    strat = df_feat["route_type_encoded"] if "route_type_encoded" in df_feat.columns else None
    X_tr_b, X_te_b, y_train, y_test, idx_train, idx_test = train_test_split(
        X_base, y, df_feat.index,
        test_size    = TEST_SIZE,
        random_state = RANDOM_STATE,
        stratify     = strat,
    )
    df_test = df_feat.loc[idx_test].copy()
    log.info("Train: %d | Test: %d", len(y_train), len(y_test))

    # ── 4. Baseline model ─────────────────────────────────────────────────
    baseline_model   = train_model(X_tr_b, y_train, "Baseline")
    baseline_results = evaluate_model(baseline_model, X_te_b, y_test, "Baseline")

    # ── 5. node2vec embeddings ────────────────────────────────────────────
    emb_pkl = MODEL_DIR / "node2vec_embeddings.pkl"
    if emb_pkl.exists():
        log.info("Loading cached node2vec embeddings …")
        with open(emb_pkl, "rb") as f:
            embeddings, _ = pickle.load(f)
    else:
        embeddings = build_node2vec_embeddings(G)

    # ── 6. Graph-enhanced features ────────────────────────────────────────
    X_enh_full = build_graph_enhanced_features(df_feat, centrality, embeddings, X_base)
    X_tr_e = X_enh_full.loc[idx_train]
    X_te_e = X_enh_full.loc[idx_test]

    # ── 7. Graph-enhanced model ───────────────────────────────────────────
    enhanced_model   = train_model(X_tr_e, y_train, "Graph-Enhanced")
    enhanced_results = evaluate_model(enhanced_model, X_te_e, y_test, "Graph-Enhanced")

    # ── 8. Graph advantage ────────────────────────────────────────────────
    advantage = compute_graph_advantage(baseline_results, enhanced_results)

    # ── 9. Visualisations ─────────────────────────────────────────────────
    plot_prediction_comparison(baseline_results, enhanced_results)
    plot_feature_importance(enhanced_model, list(X_te_e.columns), "Graph-Enhanced")
    plot_within15_by_route_type(
        df_test,
        baseline_results["y_pred"],
        enhanced_results["y_pred"],
        y_test.values,
    )

    # ── 10. Save ──────────────────────────────────────────────────────────
    save_results(baseline_model, enhanced_model, baseline_results, enhanced_results, advantage)

    log.info("=" * 65)
    log.info("  PHASE 3 — PIPELINE COMPLETE")
    log.info("=" * 65)

    return {
        "baseline" : baseline_results,
        "enhanced" : enhanced_results,
        "advantage": advantage,
    }


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    results = run_eta_pipeline()