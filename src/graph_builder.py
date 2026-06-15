"""
graph_builder.py
================
Phase 1 – Graph Construction Module
Project : Optimizing Delivery ETAs with Graph-Based Network Intelligence
Author  : IIT Guwahati Consulting & Analytics Club

Responsibilities
----------------
1. Construct a directed, weighted MultiDiGraph from the processed corridor CSV.
2. Assign meaningful edge attributes (delay ratio, SLA breach rate, distance,
   route type, time-of-day variants).
3. Attach node-level metadata (facility name, in/out degree, geographic label).
4. Provide helper functions to extract route-type subgraphs (FTL / CARTING)
   and time-of-day subgraphs for stratified analysis.
5. Export the graph to GraphML for Gephi visualisation and to a PyG-compatible
   format for the GNN models in Phase 3.

Graph Design Rationale
-----------------------
Node  = Logistics facility (warehouse / hub / dark store).
        Identified by source_center / destination_center (integer IDs).

Edge  = Directed corridor from source → destination.
        A MultiDiGraph is used because the same pair of facilities can have
        both an FTL edge and a CARTING edge — they carry fundamentally
        different traffic patterns and must not be collapsed.

Edge weight (primary) = median_segment_delay (actual / OSRM ratio)
        Chosen over raw time because it normalises for distance, making
        a 2-hour corridor comparable to a 10-hour corridor in the same graph.
        Values > 1.0 indicate a delay-prone corridor; used by network_audit.py
        to surface bottleneck edges.
"""

import logging
import pickle
from pathlib import Path
from typing import Dict, Optional, Tuple

import networkx as nx
import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

PROJECT_ROOT  = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
VIZ_DIR       = PROJECT_ROOT / "outputs" / "visualizations"
MODEL_DIR     = PROJECT_ROOT / "outputs" / "models"

VIZ_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# 1. CORE GRAPH CONSTRUCTOR
# ─────────────────────────────────────────────────────────────────────────────
def build_graph(
    corridor_df: pd.DataFrame,
    weight_col: str = "median_segment_delay",
) -> nx.MultiDiGraph:
    """
    Construct a directed weighted MultiDiGraph from the corridor-level dataframe
    produced by data_pipeline.merge_trip_segments().

    Parameters
    ----------
    corridor_df : pd.DataFrame
        One row per unique (source_center, destination_center, route_type) triple.
        Must contain columns produced by data_pipeline.run_pipeline().
    weight_col  : str
        Column to use as the primary edge weight. Default is median_segment_delay.

    Returns
    -------
    nx.MultiDiGraph
        Fully attributed directed graph ready for centrality analysis (Phase 2)
        and GNN feature extraction (Phase 3).

    Edge Attributes
    ---------------
    weight              : Primary edge weight = median delay ratio (actual/OSRM).
    route_type          : 'FTL' or 'CARTING'.
    trip_count          : Number of observed trips on this corridor.
    sla_breach_rate     : Fraction of trips that exceeded the 20% delay threshold.
    median_actual_dist  : Median real-world distance in km.
    median_osrm_dist    : Median OSRM-estimated distance in km.
    median_actual_time  : Median actual travel time (hours).
    median_osrm_time    : Median OSRM estimated travel time (hours).
    std_delay           : Std dev of segment delay ratio (reliability proxy).
    mean_delay          : Mean delay ratio (useful alongside median for skew).
    """
    G = nx.MultiDiGraph()
    G.graph["name"]        = "Delhivery Logistics Network"
    G.graph["weight_col"]  = weight_col
    G.graph["description"] = (
        "Directed weighted graph: nodes = fulfilment/hub facilities, "
        "edges = delivery corridors weighted by median delay ratio (actual/OSRM)."
    )

    required_cols = {
        "source_center", "destination_center", "source_name",
        "destination_name", "route_type", weight_col,
    }
    missing = required_cols - set(corridor_df.columns)
    if missing:
        raise ValueError(f"corridor_df is missing columns: {missing}")

    # ── Add nodes first (with facility name metadata) ─────────────────────
    src_nodes = corridor_df[["source_center", "source_name"]].drop_duplicates()
    dst_nodes = corridor_df[["destination_center", "destination_name"]].drop_duplicates()
    dst_nodes.columns = ["source_center", "source_name"]  # rename for concat
    all_nodes = pd.concat([src_nodes, dst_nodes]).drop_duplicates(subset=["source_center"])

    for _, row in all_nodes.iterrows():
        G.add_node(
            str(row["source_center"]),
            name=str(row["source_name"]),
            label=str(row["source_name"]),
    )

    log.info("Nodes added: %d facilities", G.number_of_nodes())

    # ── Add edges ─────────────────────────────────────────────────────────
    edge_count = 0
    for _, row in corridor_df.iterrows():
        src = str(row["source_center"])
        dst = str(row["destination_center"])
        

        # Skip self-loops (data anomaly: same source and destination)
        if src == dst:
            log.debug("Self-loop skipped: %d → %d", src, dst)
            continue

        weight = row.get(weight_col, np.nan)
        if pd.isna(weight) or weight <= 0:
            log.debug("Skipping edge %d→%d: invalid weight %.3f", src, dst, weight)
            continue

        G.add_edge(
            src, dst,
            # ── Primary weight ──────────────────────────────────────────
            weight              = float(weight),
            # ── Route classification ────────────────────────────────────
            route_type          = str(row["route_type"]),
            # ── Volume signal ───────────────────────────────────────────
            trip_count          = int(row.get("trip_count", 0)),
            # ── SLA health ──────────────────────────────────────────────
            sla_breach_rate     = float(row.get("sla_breach_rate", np.nan)),
            # ── Distance / time characteristics ─────────────────────────
            median_actual_dist  = float(row.get("median_actual_dist_km", np.nan)),
            median_osrm_dist    = float(row.get("median_osrm_dist_km", np.nan)),
            median_actual_time  = float(row.get("median_segment_time_h", np.nan)),
            median_osrm_time    = float(row.get("median_osrm_time_h", np.nan)),
            # ── Reliability (spread of delay) ────────────────────────────
            std_delay           = float(row.get("std_segment_delay", np.nan)),
            mean_delay          = float(row.get("mean_segment_delay", np.nan)),
        )
        edge_count += 1

    log.info(
        "Graph built: %d nodes | %d directed edges | %d isolated nodes",
        G.number_of_nodes(),
        G.number_of_edges(),
        sum(1 for n in G.nodes() if G.degree(n) == 0),
    )
    return G


# ─────────────────────────────────────────────────────────────────────────────
# 2. NODE FEATURE ENRICHMENT
# ─────────────────────────────────────────────────────────────────────────────
def enrich_node_features(G: nx.MultiDiGraph) -> nx.MultiDiGraph:
    """
    Attach node-level aggregate features derived from incident edges.
    These become node feature vectors for GraphSAGE in Phase 3.

    Node Attributes Added
    ----------------------
    in_degree          : Number of incoming corridor edges.
    out_degree         : Number of outgoing corridor edges.
    avg_inbound_delay  : Mean delay ratio of all inbound corridors.
    avg_outbound_delay : Mean delay ratio of all outbound corridors.
    max_inbound_delay  : Worst inbound delay ratio (identifies delay sinks).
    max_outbound_delay : Worst outbound delay ratio (identifies delay sources).
    inbound_sla_breach : Avg SLA breach rate across inbound corridors.
    outbound_sla_breach: Avg SLA breach rate across outbound corridors.
    total_trip_volume  : Sum of trip counts on all incident edges (hub importance).

    Practical Interpretation
    ------------------------
    A node with high avg_inbound_delay but low avg_outbound_delay is a
    DELAY SINK — packages arrive late but leave on time, suggesting the
    facility itself (loading/unloading operations) causes the bottleneck.
    The reverse pattern indicates a DELAY SOURCE — a hub dispatching to
    chronically congested downstream corridors.
    """
    G = G.copy()

    for node in G.nodes():
        # ── In-edge features ──────────────────────────────────────────────
        in_edges  = [(u, v, d) for u, v, d in G.in_edges(node, data=True)]
        out_edges = [(u, v, d) for u, v, d in G.out_edges(node, data=True)]

        def safe_mean(vals):
            clean = [v for v in vals if not np.isnan(v)]
            return float(np.mean(clean)) if clean else np.nan

        def safe_max(vals):
            clean = [v for v in vals if not np.isnan(v)]
            return float(np.max(clean)) if clean else np.nan

        in_delays  = [d.get("weight", np.nan)         for _, _, d in in_edges]
        out_delays = [d.get("weight", np.nan)         for _, _, d in out_edges]
        in_sla     = [d.get("sla_breach_rate", np.nan) for _, _, d in in_edges]
        out_sla    = [d.get("sla_breach_rate", np.nan) for _, _, d in out_edges]
        volumes    = [d.get("trip_count", 0)           for _, _, d in (in_edges + out_edges)]

        G.nodes[node]["in_degree"]           = G.in_degree(node)
        G.nodes[node]["out_degree"]          = G.out_degree(node)
        G.nodes[node]["avg_inbound_delay"]   = safe_mean(in_delays)
        G.nodes[node]["avg_outbound_delay"]  = safe_mean(out_delays)
        G.nodes[node]["max_inbound_delay"]   = safe_max(in_delays)
        G.nodes[node]["max_outbound_delay"]  = safe_max(out_delays)
        G.nodes[node]["inbound_sla_breach"]  = safe_mean(in_sla)
        G.nodes[node]["outbound_sla_breach"] = safe_mean(out_sla)
        G.nodes[node]["total_trip_volume"]   = int(sum(volumes))

    log.info("Node feature enrichment complete ✓")
    return G


# ─────────────────────────────────────────────────────────────────────────────
# 3. SUBGRAPH EXTRACTORS
# ─────────────────────────────────────────────────────────────────────────────
def get_route_type_subgraph(
    G: nx.MultiDiGraph,
    route_type: str,
) -> nx.DiGraph:
    """
    Extract a simple DiGraph containing only edges of the specified route_type.
    Collapses MultiDiGraph edges by averaging weights for the same (u, v) pair
    within that route type (only relevant if multiple parallel edges of the
    same type exist — rare but possible).

    Parameters
    ----------
    route_type : str
        'FTL' or 'CARTING' (case-insensitive).

    Returns
    -------
    nx.DiGraph
        Simple directed graph for that mode. Used by network_audit.py for
        mode-specific betweenness centrality and corridor ranking.

    Why separate subgraphs?
    -----------------------
    FTL and CARTING corridors serve completely different freight profiles:
    FTL = high-volume, time-critical, longer distances.
    CARTING = last-mile, smaller loads, urban density effects.
    Mixing them in one centrality analysis would mask mode-specific bottlenecks.
    """
    rt = route_type.upper()
    sub = nx.DiGraph()

    # Copy node attributes
    for node, data in G.nodes(data=True):
        sub.add_node(node, **data)

    # Aggregate edges of target route_type
    edge_weights: Dict[Tuple, list] = {}
    edge_attrs:   Dict[Tuple, dict] = {}

    for u, v, data in G.edges(data=True):
        if data.get("route_type", "").upper() != rt:
            continue
        key = (u, v)
        edge_weights.setdefault(key, []).append(data.get("weight", np.nan))
        if key not in edge_attrs:
            edge_attrs[key] = {k: v_ for k, v_ in data.items() if k != "weight"}

    for (u, v), weights in edge_weights.items():
        valid = [w for w in weights if not np.isnan(w)]
        avg_w = float(np.mean(valid)) if valid else np.nan
        sub.add_edge(u, v, weight=avg_w, **edge_attrs[(u, v)])

    log.info(
        "Subgraph [%s]: %d nodes | %d edges", rt, sub.number_of_nodes(), sub.number_of_edges()
    )
    return sub


def get_tod_subgraph(
    G: nx.MultiDiGraph,
    tod_df: pd.DataFrame,
    time_of_day: str,
    route_type: Optional[str] = None,
) -> nx.DiGraph:
    """
    Build a subgraph using time-of-day stratified delay weights.
    Each edge weight = median delay ratio during the specified TOD bucket.

    Parameters
    ----------
    G           : Full MultiDiGraph (for node metadata).
    tod_df      : Time-of-day stratified dataframe from data_pipeline.
    time_of_day : One of 'Night(0-6)', 'Morning(6-12)', 'Afternoon(12-18)', 'Evening(18-24)'.
    route_type  : Optional filter ('FTL' / 'CARTING').

    Returns
    -------
    nx.DiGraph filtered to the requested TOD bucket.

    Practical Use
    -------------
    Night corridors often show delay_ratio < 1.0 (less traffic) while
    Morning corridors spike above 1.3 in metro areas. This function lets
    the bottleneck audit isolate WHEN a corridor becomes critical, not just
    whether it is critical on average.
    """
    filtered = tod_df[tod_df["time_of_day"] == time_of_day].copy()
    if route_type:
        filtered = filtered[filtered["route_type"].str.upper() == route_type.upper()]

    sub = nx.DiGraph()
    for node, data in G.nodes(data=True):
        sub.add_node(node, **data)

    for _, row in filtered.iterrows():
        src = str(row["source_center"])
        dst = str(row["destination_center"])
        if src == dst:
            continue
        sub.add_edge(
            src, dst,
            weight          = float(row.get("tod_median_delay", np.nan)),
            time_of_day     = time_of_day,
            route_type      = str(row.get("route_type", "")),
            trip_count      = int(row.get("tod_trip_count", 0)),
            sla_breach_rate = float(row.get("tod_sla_breach_rate", np.nan)),
        )

    log.info(
        "TOD subgraph [%s | %s]: %d edges",
        time_of_day, route_type or "ALL", sub.number_of_edges(),
    )
    return sub


# ─────────────────────────────────────────────────────────────────────────────
# 4. GRAPH SUMMARY STATS
# ─────────────────────────────────────────────────────────────────────────────
def summarise_graph(G: nx.MultiDiGraph) -> pd.DataFrame:
    """
    Print and return a human-readable summary table of key graph properties.
    Useful as a quick sanity-check after construction and as a Notebook cell output.
    """
    all_weights = [d["weight"] for _, _, d in G.edges(data=True) if "weight" in d]

    summary = {
        "Total Nodes (Facilities)"     : G.number_of_nodes(),
        "Total Directed Edges (Corridors)": G.number_of_edges(),
        "Graph Density"                : round(nx.density(G), 6),
        "Avg Edge Weight (Delay Ratio)": round(np.mean(all_weights), 4) if all_weights else "N/A",
        "Max Edge Weight (Worst Corridor)": round(np.max(all_weights), 4) if all_weights else "N/A",
        "Min Edge Weight (Best Corridor)": round(np.min(all_weights), 4) if all_weights else "N/A",
        "FTL Edges"    : sum(1 for _, _, d in G.edges(data=True) if d.get("route_type") == "FTL"),
        "CARTING Edges": sum(1 for _, _, d in G.edges(data=True) if d.get("route_type") == "CARTING"),
        "Weakly Connected Components"  : nx.number_weakly_connected_components(G),
        "Strongly Connected Components": nx.number_strongly_connected_components(G),
    }

    df = pd.DataFrame(list(summary.items()), columns=["Metric", "Value"])
    print("\n" + "=" * 50)
    print("  DELHIVERY NETWORK — GRAPH SUMMARY")
    print("=" * 50)
    print(df.to_string(index=False))
    print("=" * 50 + "\n")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 5. EXPORT HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def save_graph(G: nx.MultiDiGraph, name: str = "delhivery_graph") -> None:
    """
    Persist the graph in two formats:
    1. GraphML (.graphml) — for Gephi / yEd visualisation.
    2. Pickle   (.pkl)    — for fast Python reload in downstream modules.
    """
    # GraphML (convert complex types to strings for compatibility)
    graphml_path = VIZ_DIR / f"{name}.graphml"
    G_export = G.copy()
    # GraphML does not support NaN; replace with sentinel
    for u, v, k, d in G_export.edges(data=True, keys=True):
        for attr, val in d.items():
            if isinstance(val, float) and np.isnan(val):
                G_export[u][v][k][attr] = -1.0
    nx.write_graphml(G_export, str(graphml_path))
    log.info("Graph saved → %s", graphml_path)

    # Pickle (preserves all Python objects faithfully)
    pkl_path = MODEL_DIR / f"{name}.pkl"
    with open(pkl_path, "wb") as f:
        pickle.dump(G, f, protocol=pickle.HIGHEST_PROTOCOL)
    log.info("Graph pickled → %s", pkl_path)


def load_graph(name: str = "delhivery_graph") -> nx.MultiDiGraph:
    """Load a previously pickled graph."""
    pkl_path = MODEL_DIR / f"{name}.pkl"
    with open(pkl_path, "rb") as f:
        G = pickle.load(f)
    log.info("Graph loaded from %s — %d nodes | %d edges",
             pkl_path, G.number_of_nodes(), G.number_of_edges())
    return G


def export_edge_list(G: nx.MultiDiGraph) -> pd.DataFrame:
    """
    Return the full edge list as a DataFrame for CSV export / EDA in notebooks.
    Each row is one directed edge with all attributes flattened.
    """
    rows = []
    for u, v, data in G.edges(data=True):
        row = {"source": u, "destination": v}
        row.update(data)
        rows.append(row)
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# 6. MASTER BUILDER ENTRY-POINT
# ─────────────────────────────────────────────────────────────────────────────
def build_full_network(
    corridor_csv: Optional[str] = None,
    corridor_df: Optional[pd.DataFrame] = None,
    tod_csv: Optional[str] = None,
    tod_df: Optional[pd.DataFrame] = None,
    save: bool = True,
) -> Dict:
    """
    End-to-end graph construction: load corridors → build graph →
    enrich nodes → summarise → (optionally) save.

    Accepts either a file path or a pre-loaded DataFrame to allow
    notebook-friendly usage without disk I/O.

    Returns
    -------
    dict with keys: 'graph', 'summary', 'edge_list', 'ftl_subgraph', 'carting_subgraph'
    """
    log.info("=" * 60)
    log.info("  PHASE 1 — GRAPH CONSTRUCTION STARTING")
    log.info("=" * 60)

    # ── Load corridors ────────────────────────────────────────────────────
    if corridor_df is None:
        path = corridor_csv or str(PROCESSED_DIR / "corridor_edges.csv")
        corridor_df = pd.read_csv(path)
        log.info("Loaded corridor_edges from %s", path)

    if tod_df is None:
        path = tod_csv or str(PROCESSED_DIR / "tod_stratified.csv")
        tod_df = pd.read_csv(path)
        log.info("Loaded tod_stratified from %s", path)

    # ── Build and enrich ──────────────────────────────────────────────────
    G = build_graph(corridor_df)
    G = enrich_node_features(G)

    # ── Subgraphs ─────────────────────────────────────────────────────────
    ftl_G    = get_route_type_subgraph(G, "FTL")
    carting_G = get_route_type_subgraph(G, "CARTING")

    # ── Summary ───────────────────────────────────────────────────────────
    summary = summarise_graph(G)
    edge_list = export_edge_list(G)

    if save:
        save_graph(G)
        edge_list.to_csv(PROCESSED_DIR / "graph_edge_list.csv", index=False)
        log.info("Edge list saved → data/processed/graph_edge_list.csv")

    log.info("=" * 60)
    log.info("  PHASE 1 — GRAPH CONSTRUCTION COMPLETE")
    log.info("=" * 60)

    return {
        "graph"           : G,
        "summary"         : summary,
        "edge_list"       : edge_list,
        "ftl_subgraph"    : ftl_G,
        "carting_subgraph": carting_G,
        "tod_df"          : tod_df,
    }


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    results = build_full_network()
