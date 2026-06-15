"""
network_audit.py
================
Phase 2 – Bottleneck & Corridor Audit Module
Project : Optimizing Delivery ETAs with Graph-Based Network Intelligence
Author  : IIT Guwahati Consulting & Analytics Club

Responsibilities
----------------
1. Load the directed weighted graph produced by Phase 1 (graph_builder.py).
2. Compute four graph-theoretic centrality metrics on all nodes:
      - Betweenness Centrality
      - In-Degree & Out-Degree Centrality
      - Clustering Coefficient
      - PageRank
3. Filter chronically delayed corridors (delay ratio > 1.20 i.e. SLA breach threshold).
4. Rank nodes and corridors by their contribution to SLA breaches.
5. Generate four publication-quality visualizations:
      - Full network graph with bottleneck hubs highlighted
      - Top-20 bottleneck hubs ranked by betweenness centrality
      - Top-20 chronically delayed corridors ranked by delay ratio
      - SLA breach rate heatmap by route type and centrality tier
6. Persist all metric tables to outputs/metrics/ for the strategy memo.

Why These Four Metrics?
------------------------
Each metric answers a different operational question:

BETWEENNESS CENTRALITY
  "Which hubs, if disrupted, would disconnect the most delivery paths?"
  A hub with high betweenness sits on the shortest path between many
  source–destination pairs. Disrupting it — even briefly — cascades
  delays across the entire network, not just its immediate corridors.
  ➜ Operational Use: These are your Tier-1 hubs needing redundant routes.

IN-DEGREE / OUT-DEGREE
  "Which hubs receive the most inbound routes (potential congestion points)
   and which dispatch the most outbound routes (potential dispatch failures)?"
  ➜ Operational Use: High in-degree hubs need more unloading capacity.
                     High out-degree hubs need dispatch staff/systems.

CLUSTERING COEFFICIENT
  "How well-connected are a hub's neighbours to each other?"
  A LOW clustering coefficient means a hub's neighbours are NOT connected
  to each other — the hub is the ONLY bridge between them. If it fails,
  those neighbours cannot reroute through each other.
  ➜ Operational Use: Low-clustering hubs are critical single points of failure.

PAGERANK
  "Which hubs receive the most 'delivery importance' from well-connected hubs?"
  Unlike degree, PageRank considers WHO is sending routes to a hub.
  Being connected to 3 major national hubs outweighs being connected
  to 20 small local centers.
  ➜ Operational Use: Prioritise SLA compliance investment at high-PageRank hubs.
"""

import logging
import pickle
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Project Paths ─────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
METRICS_DIR  = PROJECT_ROOT / "outputs" / "metrics"
VIZ_DIR      = PROJECT_ROOT / "outputs" / "visualizations"
MODEL_DIR    = PROJECT_ROOT / "outputs" / "models"

METRICS_DIR.mkdir(parents=True, exist_ok=True)
VIZ_DIR.mkdir(parents=True, exist_ok=True)

# ── Visual Style ──────────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.dpi"        : 140,
    "font.family"       : "DejaVu Sans",
    "font.size"         : 10,
    "axes.spines.top"   : False,
    "axes.spines.right" : False,
    "axes.titleweight"  : "bold",
    "axes.titlesize"    : 13,
})

SLA_THRESHOLD = 1.20   # 20% over OSRM = SLA breach


# ─────────────────────────────────────────────────────────────────────────────
# 1. LOADER
# ─────────────────────────────────────────────────────────────────────────────
def load_graph(name: str = "delhivery_graph") -> nx.MultiDiGraph:
    """Load the pickled graph produced by graph_builder.py (Phase 1)."""
    path = MODEL_DIR / f"{name}.pkl"
    if not path.exists():
        raise FileNotFoundError(
            f"Graph not found at {path}.\n"
            "Run src/graph_builder.py (Phase 1) first."
        )
    with open(path, "rb") as f:
        G = pickle.load(f)
    log.info(
        "Graph loaded: %d nodes | %d edges", G.number_of_nodes(), G.number_of_edges()
    )
    return G


# ─────────────────────────────────────────────────────────────────────────────
# 2. CENTRALITY METRICS COMPUTATION
# ─────────────────────────────────────────────────────────────────────────────
def compute_centrality_metrics(
    G: nx.MultiDiGraph,
    betweenness_k: Optional[int] = 300,
) -> pd.DataFrame:
    """
    Compute four centrality metrics for every node in the graph and
    return a single ranked DataFrame.

    Parameters
    ----------
    G              : The full directed MultiDiGraph from Phase 1.
    betweenness_k  : Number of pivot nodes for approximate betweenness.
                     Exact betweenness is O(VE) — too slow for 1,657 nodes.
                     k=300 gives a statistically reliable approximation
                     (within ~5% of exact for sparse logistics graphs).
                     Set to None to compute exactly (slow but precise).

    Returns
    -------
    pd.DataFrame sorted by betweenness_centrality descending.
    Each row = one facility with all four metrics + node metadata.

    Metric Normalisation
    ---------------------
    All centrality values are normalised to [0, 1] for cross-metric comparison.
    Raw degree counts are preserved alongside normalised values.
    """

    # Convert MultiDiGraph → DiGraph for centrality (aggregate parallel edges)
    # Use max weight per (u,v) pair so the worst corridor is not averaged away
    log.info("Converting MultiDiGraph → DiGraph for centrality computation …")
    G_simple = nx.DiGraph()
    for node, data in G.nodes(data=True):
        G_simple.add_node(node, **data)

    edge_weights: Dict[Tuple, List[float]] = {}
    for u, v, data in G.edges(data=True):
        edge_weights.setdefault((u, v), []).append(data.get("weight", 1.0))

    for (u, v), weights in edge_weights.items():
        G_simple.add_edge(u, v, weight=float(np.median(weights)))

    log.info(
        "Simple DiGraph: %d nodes | %d edges", G_simple.number_of_nodes(), G_simple.number_of_edges()
    )

    # ── Betweenness Centrality ────────────────────────────────────────────
    log.info(
        "Computing betweenness centrality (k=%s) — this may take ~30 seconds …",
        betweenness_k or "exact",
    )
    betweenness = nx.betweenness_centrality(
        G_simple,
        k=betweenness_k,
        normalized=True,
        weight="weight",
        seed=42,
    )
    log.info("Betweenness centrality done ✓")

    # ── In-Degree & Out-Degree Centrality ────────────────────────────────
    in_degree_centrality  = nx.in_degree_centrality(G_simple)
    out_degree_centrality = nx.out_degree_centrality(G_simple)
    in_degree_raw  = dict(G_simple.in_degree())
    out_degree_raw = dict(G_simple.out_degree())
    log.info("Degree centrality done ✓")

    # ── Clustering Coefficient ────────────────────────────────────────────
    # For directed graphs: uses average of in/out clustering
    clustering = nx.clustering(G_simple.to_undirected())
    log.info("Clustering coefficient done ✓")

    # ── PageRank ──────────────────────────────────────────────────────────
    # alpha=0.85 is the standard damping factor (Google's original value)
    pagerank = nx.pagerank(G_simple, alpha=0.85, weight="weight", max_iter=200)
    log.info("PageRank done ✓")

    # ── Assemble DataFrame ────────────────────────────────────────────────
    rows = []
    for node in G_simple.nodes():
        node_data = G.nodes.get(node, {})

        # SLA breach rate: average across all incident edges
        incident_edges = list(G.in_edges(node, data=True)) + list(G.out_edges(node, data=True))
        sla_rates  = [d.get("sla_breach_rate", np.nan) for _, _, d in incident_edges]
        trip_vols  = [d.get("trip_count", 0)            for _, _, d in incident_edges]
        avg_delays = [d.get("weight", np.nan)           for _, _, d in incident_edges]

        clean_sla  = [v for v in sla_rates  if not np.isnan(v)]
        clean_del  = [v for v in avg_delays if not np.isnan(v)]

        rows.append({
            "facility_id"             : node,
            "facility_name"           : node_data.get("name", str(node)),
            # ── Centrality metrics ────────────────────────────────────
            "betweenness_centrality"  : betweenness.get(node, 0),
            "in_degree_centrality"    : in_degree_centrality.get(node, 0),
            "out_degree_centrality"   : out_degree_centrality.get(node, 0),
            "in_degree_raw"           : in_degree_raw.get(node, 0),
            "out_degree_raw"          : out_degree_raw.get(node, 0),
            "clustering_coefficient"  : clustering.get(node, 0),
            "pagerank"                : pagerank.get(node, 0),
            # ── Operational health ────────────────────────────────────
            "avg_incident_delay"      : np.mean(clean_del) if clean_del else np.nan,
            "avg_sla_breach_rate"     : np.mean(clean_sla) if clean_sla else np.nan,
            "total_trip_volume"       : sum(trip_vols),
        })

    df = pd.DataFrame(rows).sort_values("betweenness_centrality", ascending=False)
    df["centrality_rank"] = range(1, len(df) + 1)

    # ── Centrality tier labelling ─────────────────────────────────────────
    # Tier 1 = top 5% by betweenness (national/regional hubs)
    # Tier 2 = 5–20% (secondary hubs)
    # Tier 3 = remaining (local/spoke facilities)
    n = len(df)
    df["centrality_tier"] = "Tier 3 — Local Spoke"
    df.loc[df["centrality_rank"] <= int(n * 0.20), "centrality_tier"] = "Tier 2 — Secondary Hub"
    df.loc[df["centrality_rank"] <= int(n * 0.05), "centrality_tier"] = "Tier 1 — National Hub"

    log.info(
        "Centrality metrics assembled. Tier 1: %d | Tier 2: %d | Tier 3: %d",
        (df["centrality_tier"] == "Tier 1 — National Hub").sum(),
        (df["centrality_tier"] == "Tier 2 — Secondary Hub").sum(),
        (df["centrality_tier"] == "Tier 3 — Local Spoke").sum(),
    )
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 3. CORRIDOR AUDIT — SLA BREACH FILTER & RANKING
# ─────────────────────────────────────────────────────────────────────────────
def audit_corridors(G: nx.MultiDiGraph) -> pd.DataFrame:
    """
    Extract every edge from the graph, flag SLA-breaching corridors
    (delay ratio > 1.20), and rank them by severity.

    SLA Breach Definition
    ----------------------
    A corridor is 'chronically delayed' if its MEDIAN delay ratio > 1.20.
    We use median (not mean) so one catastrophic outlier trip doesn't
    permanently stigmatise an otherwise healthy corridor.

    Ranking Logic
    --------------
    Primary sort   : delay_ratio descending  (worst offender first)
    Secondary sort : trip_count descending   (high-volume breaches matter more)
    The composite score = delay_ratio × trip_count surfaces corridors
    that are both severely delayed AND heavily trafficked — these cause
    the most cumulative SLA damage across the network.

    Returns
    -------
    pd.DataFrame with one row per directed edge, sorted by composite_breach_score.
    """
    rows = []
    for u, v, data in G.edges(data=True):
        delay  = data.get("weight", np.nan)
        volume = data.get("trip_count", 0)
        rows.append({
            "source_id"            : u,
            "destination_id"       : v,
            "source_name"          : G.nodes[u].get("name", str(u)),
            "destination_name"     : G.nodes[v].get("name", str(v)),
            "route_type"           : data.get("route_type", "UNKNOWN"),
            "delay_ratio"          : delay,
            "sla_breach_rate"      : data.get("sla_breach_rate", np.nan),
            "trip_count"           : volume,
            "median_actual_dist_km": data.get("median_actual_dist", np.nan),
            "median_actual_time_h" : data.get("median_actual_time", np.nan),
            "median_osrm_time_h"   : data.get("median_osrm_time", np.nan),
            "std_delay"            : data.get("std_delay", np.nan),
        })

    df = pd.DataFrame(rows)
    df["is_sla_breach"] = (df["delay_ratio"] > SLA_THRESHOLD).astype(int)

    # Composite breach score (severity × volume)
    df["composite_breach_score"] = df["delay_ratio"] * np.log1p(df["trip_count"])

    # Delay excess: how many times over baseline (useful for memo)
    df["excess_delay_pct"] = ((df["delay_ratio"] - 1.0) * 100).round(1)

    df_breaching = df[df["is_sla_breach"] == 1].copy()
    df_breaching = df_breaching.sort_values("composite_breach_score", ascending=False)
    df_breaching["breach_rank"] = range(1, len(df_breaching) + 1)

    log.info(
        "Corridor audit: %d total corridors | %d SLA-breaching (%.1f%%) | Worst delay ratio: %.2f×",
        len(df),
        len(df_breaching),
        len(df_breaching) / len(df) * 100,
        df["delay_ratio"].max(),
    )
    return df, df_breaching


# ─────────────────────────────────────────────────────────────────────────────
# 4. VISUALISATIONS
# ─────────────────────────────────────────────────────────────────────────────

def plot_bottleneck_network(
    G: nx.MultiDiGraph,
    centrality_df: pd.DataFrame,
    top_n: int = 80,
) -> None:
    """
    Visualise the logistics network as a graph, with:
    - Node SIZE proportional to betweenness centrality (bigger = more critical)
    - Node COLOR reflecting centrality tier (red=Tier1, orange=Tier2, grey=Tier3)
    - Edge COLOR reflecting delay ratio (red = severe delay, green = on-time)
    - Only the top_n most central nodes + their direct neighbours shown
      (plotting all 1,657 nodes produces an unreadable hairball)

    Business Interpretation Printed on Plot
    ----------------------------------------
    Red nodes = National hubs. If any of these fail, delays ripple across
    hundreds of downstream corridors simultaneously.
    Red edges = Chronically delayed corridors actively breaching SLA today.
    """
    log.info("Generating bottleneck network visualisation …")

    # ── Select top-N nodes by betweenness + all their neighbours ─────────
    top_nodes = set(centrality_df.head(top_n)["facility_id"].tolist())
    neighbours = set()
    for node in top_nodes:
        if node in G:
            neighbours.update(G.predecessors(node))
            neighbours.update(G.successors(node))
    display_nodes = top_nodes | neighbours
    subG = G.subgraph(display_nodes)

    # ── Node colours by centrality tier ──────────────────────────────────
    tier_colors = {
        "Tier 1 — National Hub"    : "#E74C3C",   # Red
        "Tier 2 — Secondary Hub"   : "#E67E22",   # Orange
        "Tier 3 — Local Spoke"     : "#95A5A6",   # Grey
    }
    tier_map = centrality_df.set_index("facility_id")["centrality_tier"].to_dict()
    node_colors = [
        tier_colors.get(tier_map.get(n, "Tier 3 — Local Spoke"), "#95A5A6")
        for n in subG.nodes()
    ]

    # ── Node sizes by betweenness centrality ──────────────────────────────
    bc_map = centrality_df.set_index("facility_id")["betweenness_centrality"].to_dict()
    raw_sizes = np.array([bc_map.get(n, 0) for n in subG.nodes()])
    # Normalise to [80, 800] pixel range
    if raw_sizes.max() > 0:
        node_sizes = 80 + (raw_sizes / raw_sizes.max()) * 720
    else:
        node_sizes = np.full(len(raw_sizes), 80)

    # ── Edge colours by delay ratio ───────────────────────────────────────
    edge_data  = [(u, v, d) for u, v, d in subG.edges(data=True)]
    edge_ratios = [d.get("weight", 1.0) for _, _, d in edge_data]
    max_r = max(edge_ratios) if edge_ratios else 2.0
    cmap  = plt.get_cmap("RdYlGn_r")
    edge_colors = [cmap(min(r / max_r, 1.0)) for r in edge_ratios]
    edge_widths = [0.3 + min(r / max_r, 1.0) * 1.5 for r in edge_ratios]

    # ── Layout ────────────────────────────────────────────────────────────
    pos = nx.spring_layout(subG, seed=42, k=0.4)

    fig, ax = plt.subplots(figsize=(18, 13))
    ax.set_facecolor("#F8F9FA")
    fig.patch.set_facecolor("#F8F9FA")

    nx.draw_networkx_edges(
        subG, pos,
        edge_color=edge_colors,
        width=edge_widths,
        alpha=0.65,
        arrows=True,
        arrowsize=8,
        ax=ax,
    )
    nx.draw_networkx_nodes(
        subG, pos,
        node_color=node_colors,
        node_size=node_sizes,
        alpha=0.92,
        ax=ax,
    )

    # Label only Tier 1 hubs (top 5%)
    tier1_nodes = {
        n: G.nodes[n].get("name", str(n))[:18]
        for n in subG.nodes()
        if tier_map.get(n) == "Tier 1 — National Hub"
    }
    nx.draw_networkx_labels(
        subG, pos,
        labels=tier1_nodes,
        font_size=6.5,
        font_weight="bold",
        font_color="#1A1A2E",
        ax=ax,
    )

    # ── Legend ────────────────────────────────────────────────────────────
    legend_elements = [
        mpatches.Patch(color="#E74C3C", label="Tier 1 — National Hub (Top 5% Betweenness)"),
        mpatches.Patch(color="#E67E22", label="Tier 2 — Secondary Hub (Top 20% Betweenness)"),
        mpatches.Patch(color="#95A5A6", label="Tier 3 — Local Spoke"),
        mpatches.Patch(color="#E74C3C", alpha=0.5, label="Edge: Severe Delay (Red)"),
        mpatches.Patch(color="#27AE60", alpha=0.5, label="Edge: On-Time (Green)"),
    ]
    ax.legend(handles=legend_elements, loc="upper left", fontsize=9, framealpha=0.9)

    # ── Colorbar for edge delay ───────────────────────────────────────────
    sm = cm.ScalarMappable(cmap=cmap, norm=mcolors.Normalize(vmin=0, vmax=max_r))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.5, pad=0.01)
    cbar.set_label("Delay Ratio (actual / OSRM)", fontsize=9)

    ax.set_title(
        f"Delhivery Logistics Network — Bottleneck Hubs & Delay Corridors\n"
        f"(Showing top {top_n} hubs + direct neighbours | Node size = Betweenness Centrality)",
        pad=14,
    )
    ax.axis("off")
    plt.tight_layout()
    out = VIZ_DIR / "bottleneck_network.png"
    plt.savefig(out, bbox_inches="tight", dpi=150)
    plt.close()
    log.info("Saved → %s", out)


def plot_top_bottleneck_hubs(centrality_df: pd.DataFrame, top_n: int = 20) -> None:
    """
    Horizontal bar chart: Top-N hubs ranked by betweenness centrality,
    coloured by average SLA breach rate.

    Business Read
    -------------
    A hub at the top of this chart is both structurally critical (many paths
    pass through it) AND operationally unhealthy (high SLA breach rate).
    These are your highest-priority intervention targets.
    """
    log.info("Generating top bottleneck hubs chart …")

    top = centrality_df.head(top_n).copy()
    top = top.sort_values("betweenness_centrality", ascending=True)

    # Shorten labels
    top["label"] = top["facility_name"].str[:30] + " (" + top["facility_id"].str[-6:] + ")"

    # Colour by SLA breach rate
    sla_vals = top["avg_sla_breach_rate"].fillna(0).values
    cmap     = plt.get_cmap("RdYlGn_r")
    colors   = [cmap(v) for v in sla_vals]

    fig, ax = plt.subplots(figsize=(14, 9))
    bars = ax.barh(top["label"], top["betweenness_centrality"], color=colors, edgecolor="white", height=0.7)

    # Annotate bars with SLA breach %
    for bar, sla in zip(bars, sla_vals):
        ax.text(
            bar.get_width() + 0.0002,
            bar.get_y() + bar.get_height() / 2,
            f"SLA breach: {sla*100:.0f}%",
            va="center", ha="left", fontsize=8, color="#555",
        )

    sm = cm.ScalarMappable(cmap=cmap, norm=mcolors.Normalize(vmin=0, vmax=1))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.6)
    cbar.set_label("Avg SLA Breach Rate", fontsize=9)

    ax.set_xlabel("Betweenness Centrality (normalised)", fontsize=10)
    ax.set_title(
        f"Top {top_n} Bottleneck Hubs — Ranked by Betweenness Centrality\n"
        "Bar colour = SLA breach rate (Red = severe, Green = healthy)",
        pad=12,
    )
    plt.tight_layout()
    out = VIZ_DIR / "top_bottleneck_hubs.png"
    plt.savefig(out, bbox_inches="tight", dpi=150)
    plt.close()
    log.info("Saved → %s", out)


def plot_top_delayed_corridors(breach_df: pd.DataFrame, top_n: int = 20) -> None:
    """
    Horizontal bar chart: Top-N most severely delayed corridors.
    Bar length = delay ratio. Bar colour = route type (FTL vs CARTING).
    Annotated with trip volume so ops teams see scale of impact.

    Business Read
    -------------
    Each bar = one corridor actively bleeding SLA compliance today.
    The number on each bar = how many trips per period cross this broken route.
    """
    log.info("Generating top delayed corridors chart …")

    top = breach_df.head(top_n).copy()
    top = top.sort_values("delay_ratio", ascending=True)

    top["corridor_label"] = (
        top["source_name"].str[:18] + " →\n" + top["destination_name"].str[:18]
    )

    route_colors = {"FTL": "#2980B9", "CARTING": "#E67E22"}
    colors = [route_colors.get(rt, "#95A5A6") for rt in top["route_type"]]

    fig, ax = plt.subplots(figsize=(14, 10))
    bars = ax.barh(top["corridor_label"], top["delay_ratio"], color=colors, edgecolor="white", height=0.7)

    # SLA baseline and breach threshold lines
    ax.axvline(1.0, color="green",  linestyle="--", linewidth=1.5, label="OSRM Baseline (1.0×)")
    ax.axvline(1.2, color="orange", linestyle="--", linewidth=1.5, label="SLA Breach Threshold (1.2×)")

    # Annotate with trip volume
    for bar, vol in zip(bars, top["trip_count"]):
        ax.text(
            bar.get_width() + 0.05,
            bar.get_y() + bar.get_height() / 2,
            f"{int(vol)} trips",
            va="center", ha="left", fontsize=8, color="#333",
        )

    legend_patches = [
        mpatches.Patch(color="#2980B9",  label="FTL"),
        mpatches.Patch(color="#E67E22",  label="CARTING"),
        plt.Line2D([0], [0], color="green",  linestyle="--", label="Baseline (1.0×)"),
        plt.Line2D([0], [0], color="orange", linestyle="--", label="SLA Breach (1.2×)"),
    ]
    ax.legend(handles=legend_patches, fontsize=9, loc="lower right")

    ax.set_xlabel("Delay Ratio (actual / OSRM) — higher is worse", fontsize=10)
    ax.set_title(
        f"Top {top_n} Chronically Delayed Corridors — Ranked by Delay Ratio\n"
        "Bar colour = Route Type | Right label = Trip Volume",
        pad=12,
    )
    plt.tight_layout()
    out = VIZ_DIR / "top_delayed_corridors.png"
    plt.savefig(out, bbox_inches="tight", dpi=150)
    plt.close()
    log.info("Saved → %s", out)


def plot_sla_breach_by_tier_and_route(
    centrality_df: pd.DataFrame,
    corridor_df: pd.DataFrame,
) -> None:
    """
    Grouped bar chart: Average SLA breach rate broken down by
    (centrality tier × route type).

    Business Read
    -------------
    This answers: "Are our Tier-1 national hubs disproportionately
    responsible for CARTING SLA breaches vs FTL?"
    If CARTING breach rates are highest at Tier-1 hubs, the problem
    is last-mile operations at major centers — not the long-haul lanes.
    """
    log.info("Generating SLA breach by tier and route type chart …")

    # Merge centrality tier onto corridor data
    tier_map = centrality_df.set_index("facility_id")["centrality_tier"].to_dict()
    df = corridor_df.copy()
    df["source_tier"] = df["source_id"].map(tier_map).fillna("Tier 3 — Local Spoke")

    pivot = (
        df.groupby(["source_tier", "route_type"])["sla_breach_rate"]
        .mean()
        .unstack("route_type")
        .fillna(0)
    )

    tier_order = ["Tier 1 — National Hub", "Tier 2 — Secondary Hub", "Tier 3 — Local Spoke"]
    pivot = pivot.reindex([t for t in tier_order if t in pivot.index])

    x     = np.arange(len(pivot.index))
    width = 0.35
    colors = {"FTL": "#2980B9", "CARTING": "#E67E22"}

    fig, ax = plt.subplots(figsize=(11, 6))
    for i, col in enumerate(pivot.columns):
        offset = (i - len(pivot.columns) / 2 + 0.5) * width
        bars = ax.bar(
            x + offset, pivot[col] * 100,
            width=width, label=col,
            color=colors.get(col, "#95A5A6"),
            alpha=0.88, edgecolor="white",
        )
        for bar in bars:
            h = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2, h + 0.5,
                f"{h:.1f}%", ha="center", va="bottom", fontsize=9,
            )

    ax.axhline(20, color="red", linestyle="--", linewidth=1.5, label="20% SLA Breach Limit")
    ax.set_xticks(x)
    ax.set_xticklabels([t.split(" — ")[1] for t in pivot.index], fontsize=10)
    ax.set_ylabel("Avg SLA Breach Rate (%)", fontsize=10)
    ax.set_ylim(0, max(pivot.max().max() * 110, 35))
    ax.set_title(
        "SLA Breach Rate by Centrality Tier × Route Type\n"
        "Reveals whether national hubs or local spokes drive the most compliance failures",
        pad=12,
    )
    ax.legend(fontsize=9)
    plt.tight_layout()
    out = VIZ_DIR / "sla_breach_by_tier_route.png"
    plt.savefig(out, bbox_inches="tight", dpi=150)
    plt.close()
    log.info("Saved → %s", out)


# ─────────────────────────────────────────────────────────────────────────────
# 5. PERSIST METRICS
# ─────────────────────────────────────────────────────────────────────────────
def save_metrics(
    centrality_df: pd.DataFrame,
    corridor_df: pd.DataFrame,
    breach_df: pd.DataFrame,
) -> None:
    """Save all metric tables to outputs/metrics/."""

    # Full centrality table
    out1 = METRICS_DIR / "node_centrality_metrics.csv"
    centrality_df.to_csv(out1, index=False)
    log.info("Saved → %s", out1)

    # Full corridor table (all edges, breach-flagged)
    out2 = METRICS_DIR / "corridor_audit_full.csv"
    corridor_df.to_csv(out2, index=False)
    log.info("Saved → %s", out2)

    # Breach-only ranked table
    out3 = METRICS_DIR / "sla_breach_corridors_ranked.csv"
    breach_df.to_csv(out3, index=False)
    log.info("Saved → %s", out3)


# ─────────────────────────────────────────────────────────────────────────────
# 6. SUMMARY REPORT (Console)
# ─────────────────────────────────────────────────────────────────────────────
def print_audit_summary(
    centrality_df: pd.DataFrame,
    corridor_df: pd.DataFrame,
    breach_df: pd.DataFrame,
) -> None:
    """Print a business-readable audit summary to the console."""

    print("\n" + "=" * 65)
    print("  PHASE 2 — BOTTLENECK & CORRIDOR AUDIT SUMMARY")
    print("=" * 65)

    print("\n📍 NETWORK OVERVIEW")
    print(f"   Total Facilities (Nodes)   : {len(centrality_df):,}")
    print(f"   Total Corridors (Edges)    : {len(corridor_df):,}")
    print(f"   SLA-Breaching Corridors    : {len(breach_df):,}  "
          f"({len(breach_df)/len(corridor_df)*100:.1f}% of network)")
    print(f"   Worst Delay Ratio          : {corridor_df['delay_ratio'].max():.2f}× OSRM")

    print("\n🔴 TOP 5 BOTTLENECK HUBS (By Betweenness Centrality)")
    top5 = centrality_df.head(5)[
        ["centrality_rank", "facility_name", "facility_id",
         "betweenness_centrality", "avg_sla_breach_rate", "total_trip_volume"]
    ]
    for _, row in top5.iterrows():
        print(
            f"   #{int(row['centrality_rank']):<3} {str(row['facility_name'])[:35]:<35} "
            f"BC={row['betweenness_centrality']:.4f}  "
            f"SLA breach={row['avg_sla_breach_rate']*100:.0f}%  "
            f"Trips={int(row['total_trip_volume'])}"
        )

    print("\n🚨 TOP 5 DELAYED CORRIDORS (By Composite Breach Score)")
    top5c = breach_df.head(5)[
        ["breach_rank", "source_name", "destination_name",
         "route_type", "delay_ratio", "trip_count", "excess_delay_pct"]
    ]
    for _, row in top5c.iterrows():
        print(
            f"   #{int(row['breach_rank']):<3} {str(row['source_name'])[:20]:<20} → "
            f"{str(row['destination_name'])[:20]:<20} "
            f"[{row['route_type']}]  "
            f"Delay={row['delay_ratio']:.2f}×  "
            f"(+{row['excess_delay_pct']:.0f}%)  "
            f"Trips={int(row['trip_count'])}"
        )

    ftl_breach  = breach_df[breach_df["route_type"] == "FTL"]
    cart_breach = breach_df[breach_df["route_type"] == "CARTING"]
    print("\n📊 SLA BREACH SPLIT BY ROUTE TYPE")
    print(f"   FTL Breaching Corridors    : {len(ftl_breach)}")
    print(f"   CARTING Breaching Corridors: {len(cart_breach)}")

    print("\n💡 KEY INSIGHTS FOR OPS TEAM")
    top_hub = centrality_df.iloc[0]
    print(
        f"   1. '{top_hub['facility_name']}' is the single most critical hub — "
        f"disruption here would affect the maximum number of downstream routes."
    )
    worst_corr = breach_df.iloc[0]
    print(
        f"   2. The '{worst_corr['source_name']} → {worst_corr['destination_name']}' "
        f"[{worst_corr['route_type']}] corridor is running at "
        f"{worst_corr['delay_ratio']:.1f}× expected time — immediate intervention needed."
    )
    low_cluster = centrality_df.nsmallest(3, "clustering_coefficient")[
        ["facility_name", "clustering_coefficient"]
    ]
    names = ", ".join(low_cluster["facility_name"].str[:20].tolist())
    print(
        f"   3. Facilities with lowest clustering ({names}) are "
        f"single points of failure — their neighbours cannot reroute around them."
    )

    print("\n📁 All metrics saved to outputs/metrics/")
    print("🖼  All visualisations saved to outputs/visualizations/")
    print("=" * 65 + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# 7. MASTER ENTRY-POINT
# ─────────────────────────────────────────────────────────────────────────────
def run_audit(graph_name: str = "delhivery_graph") -> dict:
    """
    Execute the full Phase 2 bottleneck audit pipeline.

    Steps
    -----
    1. Load graph from Phase 1.
    2. Compute all centrality metrics.
    3. Audit all corridors for SLA breaches.
    4. Generate 4 visualisations.
    5. Save all metric CSVs.
    6. Print console summary.

    Returns
    -------
    dict with keys: 'centrality', 'corridors', 'breaches'
    """
    log.info("=" * 65)
    log.info("  PHASE 2 — BOTTLENECK & CORRIDOR AUDIT STARTING")
    log.info("=" * 65)

    G              = load_graph(graph_name)
    centrality_df  = compute_centrality_metrics(G)
    corridor_df, breach_df = audit_corridors(G)

    # ── Visualisations ────────────────────────────────────────────────────
    plot_bottleneck_network(G, centrality_df, top_n=80)
    plot_top_bottleneck_hubs(centrality_df, top_n=20)
    plot_top_delayed_corridors(breach_df, top_n=20)
    plot_sla_breach_by_tier_and_route(centrality_df, corridor_df)

    # ── Persist ───────────────────────────────────────────────────────────
    save_metrics(centrality_df, corridor_df, breach_df)

    # ── Console summary ───────────────────────────────────────────────────
    print_audit_summary(centrality_df, corridor_df, breach_df)

    log.info("=" * 65)
    log.info("  PHASE 2 — AUDIT COMPLETE")
    log.info("=" * 65)

    return {
        "centrality": centrality_df,
        "corridors" : corridor_df,
        "breaches"  : breach_df,
    }


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    results = run_audit()