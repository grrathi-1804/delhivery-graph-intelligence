"""
dashboard/app.py
=================
Optional (Phase 6) — Live Network Risk Dashboard
Project : Optimizing Delivery ETAs with Graph-Based Network Intelligence
Author  : IIT Guwahati Consulting & Analytics Club

What this is
-------------
A Streamlit app that sits on top of the artefacts already produced by
Phases 1-4 — it trains nothing new. Four views:

1. Network Map        — interactive graph, node colour = SLA breach rate,
                         edge colour = delay severity (Phase 1 + 2 outputs).
2. Live Risk Scorer    — pick a source, destination, route type and time of
                         day; the Phase 4 regressor + classifier score that
                         exact lane on the spot, plus an FTL-vs-Carting
                         counterfactual comparison using the same cost model
                         validated in Phase 4. This is the "real-time delay
                         risk score" — computed live in the running session,
                         not pre-baked into a static chart.
3. Bottleneck Leaderboard — sortable view of Phase 2's centrality metrics.
4. Corridor Explorer   — filterable view of Phase 2's full corridor audit.

Run with:
    streamlit run dashboard/app.py
from the project root, after Phases 1-4 have been run at least once.

Design note: this file deliberately imports the cost functions and feature
schema directly from src/decision_framework.py instead of redefining them.
A dashboard that quietly drifts out of sync with the validated framework
(different cost constants, different feature order) would be worse than no
dashboard at all — importing the real module guarantees they can't diverge.
"""

import pickle
import sys
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ── Make src/ importable so we can re-use Phase 4's validated logic ──────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODEL_DIR = PROJECT_ROOT / "outputs" / "models"
METRICS_DIR = PROJECT_ROOT / "outputs" / "metrics"

st.set_page_config(
    page_title="Delhivery Network Risk Dashboard",
    layout="wide",
    page_icon="🚚",
)

FRAMEWORK_AVAILABLE = True
FRAMEWORK_IMPORT_ERROR = ""
try:
    from decision_framework import (  # noqa: E402
        GRAPH_FEATS, MODEL_FEATURES, REPRESENTATIVE_DOW, TOD_ENCODE,
        TOD_REPRESENTATIVE_HOUR, carting_cost_per_trip, ftl_cost_per_trip,
    )
except Exception as e:  # pragma: no cover - surfaced in the UI, not raised
    FRAMEWORK_AVAILABLE = False
    FRAMEWORK_IMPORT_ERROR = str(e)


# ─────────────────────────────────────────────────────────────────────────────
# CACHED LOADERS — each artefact is loaded once per session, not per rerun
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading network graph…")
def load_graph():
    path = MODEL_DIR / "delhivery_graph.pkl"
    if not path.exists():
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


@st.cache_data(show_spinner="Loading facility metrics…")
def load_centrality():
    path = METRICS_DIR / "node_centrality_metrics.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    df["facility_id"] = df["facility_id"].astype(str)
    return df


@st.cache_data(show_spinner="Loading corridor audit…")
def load_corridor_audit():
    path = METRICS_DIR / "corridor_audit_full.csv"
    if not path.exists():
        return None
    return pd.read_csv(path)


@st.cache_data(show_spinner="Loading corridor edge data…")
def load_corridor_edges():
    path = PROCESSED_DIR / "corridor_edges.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    df["source_center"] = df["source_center"].astype(str)
    df["destination_center"] = df["destination_center"].astype(str)
    return df


@st.cache_resource(show_spinner="Loading risk models…")
def load_models():
    reg_path = MODEL_DIR / "route_performance_regressor.pkl"
    clf_path = MODEL_DIR / "route_performance_classifier.pkl"
    if not reg_path.exists() or not clf_path.exists():
        return None, None
    with open(reg_path, "rb") as f:
        reg = pickle.load(f)
    with open(clf_path, "rb") as f:
        clf = pickle.load(f)
    return reg, clf


# ─────────────────────────────────────────────────────────────────────────────
# LIVE FEATURE BUILDER — mirrors decision_framework.simulate_route_type_predictions
# ─────────────────────────────────────────────────────────────────────────────
def get_centrality_feats(centrality_df: pd.DataFrame, facility_id: str, prefix: str) -> dict:
    row = centrality_df[centrality_df["facility_id"] == facility_id]
    feats = {}
    for feat in GRAPH_FEATS:
        if len(row) and not pd.isna(row.iloc[0].get(feat, np.nan)):
            feats[f"{prefix}_{feat}"] = float(row.iloc[0][feat])
        else:
            feats[f"{prefix}_{feat}"] = 0.0
    return feats


def build_feature_row(
    distance_km: float, osrm_time_min: float, tod_label: str,
    route_type_encoded: int, src_id: str, dst_id: str, centrality_df: pd.DataFrame,
) -> pd.DataFrame:
    """Build a single-row feature matrix matching MODEL_FEATURES exactly."""
    row = {
        "segment_distance_km": distance_km,
        "segment_osrm_time_hours": osrm_time_min / 60.0,
        "trip_start_hour": TOD_REPRESENTATIVE_HOUR.get(tod_label.upper(), 9),
        "trip_start_dayofweek": REPRESENTATIVE_DOW,
        "time_of_day_encoded": TOD_ENCODE.get(tod_label.upper(), 1),
        "route_type_encoded": route_type_encoded,
    }
    row.update(get_centrality_feats(centrality_df, src_id, "src"))
    row.update(get_centrality_feats(centrality_df, dst_id, "dst"))
    return pd.DataFrame([row])[MODEL_FEATURES]


def risk_tier(prob: float) -> tuple:
    if prob < 0.50:
        return "🟢 Low Risk", "#2ECC71"
    elif prob < 0.80:
        return "🟡 Moderate Risk", "#F1C40F"
    else:
        return "🔴 High Risk", "#E74C3C"


# ─────────────────────────────────────────────────────────────────────────────
# NETWORK MAP — cached on top_n only; G / centrality_df are loaded once via
# their own cache_resource/cache_data functions and referenced via closure,
# which is the recommended Streamlit pattern (avoids re-hashing large objects)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Computing network layout…")
def build_network_figure(top_n: int, _graph_token: int, _centrality_token: int):
    G = load_graph()
    centrality_df = load_centrality()

    top_nodes = set(
        centrality_df.sort_values("betweenness_centrality", ascending=False)
        .head(top_n)["facility_id"]
    )
    neighbours = set()
    for n in top_nodes:
        if n in G:
            neighbours.update(G.predecessors(n))
            neighbours.update(G.successors(n))
    display_nodes = top_nodes | neighbours
    subG = G.subgraph(display_nodes)
    pos = nx.spring_layout(subG, seed=42, k=0.45)

    buckets = {
        "Healthy (<1.2×)":  {"x": [], "y": [], "color": "#2ECC71"},
        "Moderate (1.2-2×)": {"x": [], "y": [], "color": "#F1C40F"},
        "Severe (>2×)":      {"x": [], "y": [], "color": "#E74C3C"},
    }
    for u, v, d in subG.edges(data=True):
        w = d.get("weight", 1.0)
        key = "Healthy (<1.2×)" if w < 1.2 else "Moderate (1.2-2×)" if w < 2.0 else "Severe (>2×)"
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        buckets[key]["x"].extend([x0, x1, None])
        buckets[key]["y"].extend([y0, y1, None])

    edge_traces = [
        go.Scatter(
            x=b["x"], y=b["y"], mode="lines",
            line=dict(width=1, color=b["color"]), opacity=0.45,
            name=label, hoverinfo="none",
        )
        for label, b in buckets.items() if b["x"]
    ]

    cmap_lookup = centrality_df.set_index("facility_id")
    node_x, node_y, node_size, node_color, node_text = [], [], [], [], []
    for n in subG.nodes():
        x, y = pos[n]
        node_x.append(x)
        node_y.append(y)
        if n in cmap_lookup.index:
            row = cmap_lookup.loc[n]
            bc = float(row["betweenness_centrality"])
            breach = float(row["avg_sla_breach_rate"]) if not pd.isna(
                row["avg_sla_breach_rate"]) else 0.0
            name = row["facility_name"]
            volume = int(row["total_trip_volume"])
        else:
            bc, breach, name, volume = 0.0, 0.0, str(n), 0
        node_size.append(8 + bc * 600)
        node_color.append(breach)
        node_text.append(
            f"{name}<br>Betweenness: {bc:.4f}<br>SLA Breach Rate: {breach*100:.0f}%<br>Trip Volume: {volume}"
        )

    node_trace = go.Scatter(
        x=node_x, y=node_y, mode="markers", hoverinfo="text", text=node_text,
        marker=dict(
            size=node_size, color=node_color, colorscale="RdYlGn_r",
            cmin=0, cmax=1, showscale=True,
            colorbar=dict(title="SLA Breach Rate"),
            line=dict(width=0.5, color="white"),
        ),
        name="Facilities",
    )

    fig = go.Figure(data=edge_traces + [node_trace])
    fig.update_layout(
        showlegend=True, height=650,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        xaxis=dict(showgrid=False, zeroline=False, visible=False),
        yaxis=dict(showgrid=False, zeroline=False, visible=False),
        margin=dict(l=10, r=10, t=40, b=10),
        plot_bgcolor="white",
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# LOAD EVERYTHING ONCE
# ─────────────────────────────────────────────────────────────────────────────
G = load_graph()
centrality_df = load_centrality()
corridor_audit_df = load_corridor_audit()
corridor_edges_df = load_corridor_edges()
reg_model, clf_model = load_models()

st.title("🚚 Delhivery Network Risk Intelligence Dashboard")
st.caption(
    "Live view of network structure, bottleneck risk, and FTL vs Carting "
    "recommendations — built on top of the Phase 1–4 analytics pipeline."
)

if G is None or centrality_df is None:
    st.error(
        "Required artefacts not found. From the project root, run:\n\n"
        "`python src/data_pipeline.py`\n`python src/graph_builder.py`\n"
        "`python src/network_audit.py`\n\nthen relaunch this dashboard."
    )
    st.stop()

with st.sidebar:
    st.header("About This Dashboard")
    st.markdown(
        "- **Network Map** — Phase 1 + 2 outputs, rendered interactively\n"
        "- **Live Risk Scorer** — Phase 4's trained models, queried live\n"
        "- **Bottleneck Leaderboard** — Phase 2's centrality audit\n"
        "- **Corridor Explorer** — Phase 2's full corridor audit, filterable"
    )
    st.markdown("---")
    st.caption(
        "All cost figures shown are the illustrative unit-economics "
        "assumptions documented in `src/decision_framework.py`, not actual "
        "Delhivery finance data."
    )

# ── KPI row ───────────────────────────────────────────────────────────────
total_facilities = G.number_of_nodes()
total_corridors = G.number_of_edges()
breach_rate = corridor_audit_df["is_sla_breach"].mean(
) * 100 if corridor_audit_df is not None else None
top_hub = centrality_df.sort_values(
    "betweenness_centrality", ascending=False).iloc[0]

c1, c2, c3, c4 = st.columns(4)
c1.metric("Facilities", f"{total_facilities:,}")
c2.metric("Active Corridors", f"{total_corridors:,}")
c3.metric("Corridors Breaching SLA",
          f"{breach_rate:.1f}%" if breach_rate is not None else "—")
c4.metric(
    "Top Bottleneck Hub", str(top_hub["facility_name"])[:24],
    f"{top_hub['avg_sla_breach_rate']*100:.0f}% breach rate",
)

st.markdown("---")

tab1, tab2, tab3, tab4 = st.tabs(
    ["🗺️ Network Map", "⚡ Live Risk Scorer",
        "🏆 Bottleneck Leaderboard", "📦 Corridor Explorer"]
)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 — NETWORK MAP
# ─────────────────────────────────────────────────────────────────────────────
with tab1:
    st.subheader("Bottleneck Network Map")
    top_n = st.slider(
        "Number of top hubs to display (plus their direct neighbours)", 20, 200, 80, step=10)
    fig = build_network_figure(top_n, hash(id(G)), hash(id(centrality_df)))
    st.plotly_chart(fig, width="stretch")
    st.caption(
        "Node size = structural criticality (betweenness centrality). "
        "Node colour = SLA breach rate (red = severe). "
        "Edge colour = corridor delay severity. Hover any node for details."
    )

# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 — LIVE RISK SCORER
# ─────────────────────────────────────────────────────────────────────────────
with tab2:
    st.subheader("Live Corridor Risk Scorer")
    st.caption(
        "Pick any two facilities and a route type — the Phase 4 models score "
        "that exact lane live, right now, in this session."
    )

    if not FRAMEWORK_AVAILABLE:
        st.error(
            f"Could not import src/decision_framework.py: {FRAMEWORK_IMPORT_ERROR}")
    elif reg_model is None or clf_model is None:
        st.warning(
            "Run `python src/decision_framework.py` first to enable live scoring.")
    else:
        name_to_id = (
            centrality_df.dropna(subset=["facility_name"])
            .drop_duplicates(subset=["facility_name"])
            .set_index("facility_name")["facility_id"].to_dict()
        )
        facility_names = sorted(name_to_id.keys())

        col1, col2 = st.columns(2)
        with col1:
            src_name = st.selectbox("Source Facility", facility_names, index=0)
        with col2:
            default_dst_idx = 1 if len(facility_names) > 1 else 0
            dst_name = st.selectbox(
                "Destination Facility", facility_names, index=default_dst_idx)

        route_type = st.radio("Route Type to Evaluate", [
                              "FTL", "CARTING"], horizontal=True)
        tod_label = st.selectbox(
            "Time of Day", ["Night(0-6)", "Morning(6-12)", "Afternoon(12-18)", "Evening(18-24)"], index=1,
        )

        src_id = name_to_id.get(src_name)
        dst_id = name_to_id.get(dst_name)

        default_dist, default_time_min = 25.0, 45.0
        if corridor_edges_df is not None and src_id and dst_id:
            match = corridor_edges_df[
                (corridor_edges_df["source_center"] == src_id)
                & (corridor_edges_df["destination_center"] == dst_id)
            ]
            if len(match):
                default_dist = float(match.iloc[0].get(
                    "median_osrm_dist_km", default_dist))
                default_time_min = float(match.iloc[0].get(
                    "median_osrm_time_h", default_time_min / 60)) * 60
                st.success(
                    f"Historical data found for this lane — auto-filled below ({len(match)} record(s)).")
            else:
                st.info(
                    "No historical data for this exact lane — using network defaults. Adjust below if you have better estimates.")

        col3, col4 = st.columns(2)
        with col3:
            distance_km = st.number_input(
                "Estimated Distance (km)", min_value=1.0, value=round(default_dist, 1), step=1.0)
        with col4:
            osrm_time_min = st.number_input(
                "OSRM Estimated Time (minutes)", min_value=1.0, value=round(default_time_min, 1), step=1.0)

        if st.button("🔍 Compute Risk Score", type="primary"):
            if src_id == dst_id:
                st.error("Source and destination must be different facilities.")
            else:
                rt_encoded = 1 if route_type == "CARTING" else 0
                X = build_feature_row(
                    distance_km, osrm_time_min, tod_label, rt_encoded, src_id, dst_id, centrality_df)
                pred_delay = max(float(reg_model.predict(X)[0]), 0.1)
                pred_breach_prob = float(clf_model.predict_proba(X)[0][1])
                tier_label, tier_color = risk_tier(pred_breach_prob)

                m1, m2, m3 = st.columns(3)
                m1.metric(
                    f"Predicted Delay Ratio ({route_type})", f"{pred_delay:.2f}×")
                m2.metric("SLA Breach Probability",
                          f"{pred_breach_prob*100:.1f}%")
                with m3:
                    st.markdown(f"**Risk Tier**")
                    st.markdown(
                        f"<span style='font-size:22px; color:{tier_color}; font-weight:700'>{tier_label}</span>",
                        unsafe_allow_html=True,
                    )

                st.markdown("#### FTL vs CARTING Comparison for This Lane")
                X_ftl = build_feature_row(
                    distance_km, osrm_time_min, tod_label, 0, src_id, dst_id, centrality_df)
                X_cart = build_feature_row(
                    distance_km, osrm_time_min, tod_label, 1, src_id, dst_id, centrality_df)
                delay_ftl = max(float(reg_model.predict(X_ftl)[0]), 0.1)
                delay_cart = max(float(reg_model.predict(X_cart)[0]), 0.1)
                breach_ftl = float(clf_model.predict_proba(X_ftl)[0][1])
                breach_cart = float(clf_model.predict_proba(X_cart)[0][1])
                cost_ftl = float(ftl_cost_per_trip(np.array([distance_km]))[0])
                cost_cart = float(carting_cost_per_trip(
                    np.array([distance_km]))[0])
                recommended = "FTL" if cost_ftl <= cost_cart else "CARTING"

                comp_df = pd.DataFrame({
                    "Route Type": ["FTL", "CARTING"],
                    "Predicted Delay Ratio": [f"{delay_ftl:.2f}×", f"{delay_cart:.2f}×"],
                    "SLA Breach Probability": [f"{breach_ftl*100:.1f}%", f"{breach_cart*100:.1f}%"],
                    "Per-Trip Cost (illustrative)": [f"₹{cost_ftl:,.0f}", f"₹{cost_cart:,.0f}"],
                })
                st.dataframe(comp_df, width="stretch", hide_index=True)
                st.success(
                    f"📌 Framework recommendation for this lane: **{recommended}** "
                    f"— based on the distance-aware cost model validated in Phase 4."
                )

# ─────────────────────────────────────────────────────────────────────────────
# TAB 3 — BOTTLENECK LEADERBOARD
# ─────────────────────────────────────────────────────────────────────────────
with tab3:
    st.subheader("Facility Bottleneck Leaderboard")
    sort_col = st.selectbox(
        "Sort by", ["betweenness_centrality", "pagerank", "avg_sla_breach_rate", "total_trip_volume"], index=0,
    )
    top_k = st.slider("Show top K facilities", 5, 50, 15)

    display_df = (
        centrality_df.sort_values(sort_col, ascending=False).head(top_k)[
            ["facility_name", "centrality_tier", "betweenness_centrality", "pagerank",
             "in_degree_raw", "out_degree_raw", "clustering_coefficient",
             "avg_sla_breach_rate", "total_trip_volume"]
        ].copy()
    )
    display_df["avg_sla_breach_rate"] = (
        display_df["avg_sla_breach_rate"] * 100).round(1)
    display_df.columns = [
        "Facility", "Tier", "Betweenness", "PageRank", "In-Degree",
        "Out-Degree", "Clustering", "SLA Breach %", "Trip Volume",
    ]
    st.dataframe(display_df, width="stretch", hide_index=True)

    chart_df = display_df.iloc[::-1]
    fig = go.Figure(go.Bar(
        x=chart_df["Betweenness"], y=chart_df["Facility"], orientation="h",
        marker=dict(color=chart_df["SLA Breach %"], colorscale="RdYlGn_r", colorbar=dict(
            title="SLA Breach %")),
    ))
    fig.update_layout(
        height=max(400, top_k * 28), margin=dict(l=10, r=10, t=40, b=10),
        title="Betweenness Centrality (bar length) vs SLA Breach Rate (colour)",
    )
    st.plotly_chart(fig, width="stretch")

# ─────────────────────────────────────────────────────────────────────────────
# TAB 4 — CORRIDOR EXPLORER
# ─────────────────────────────────────────────────────────────────────────────
with tab4:
    st.subheader("Corridor Risk Explorer")
    if corridor_audit_df is None:
        st.warning(
            "Run `python src/network_audit.py` first to populate corridor audit data.")
    else:
        rt_filter = st.multiselect(
            "Route Type", ["FTL", "CARTING"], default=["FTL", "CARTING"])
        max_delay = float(corridor_audit_df["delay_ratio"].max())
        min_delay = st.slider("Minimum delay ratio", 0.0,
                              max_delay, 1.2, step=0.1)

        filtered = corridor_audit_df[
            corridor_audit_df["route_type"].isin(rt_filter)
            & (corridor_audit_df["delay_ratio"] >= min_delay)
        ].sort_values("delay_ratio", ascending=False)

        st.write(f"**{len(filtered):,}** corridors match these filters")
        show_cols = ["source_name", "destination_name", "route_type", "delay_ratio",
                     "sla_breach_rate", "trip_count", "excess_delay_pct"]
        st.dataframe(filtered[show_cols].head(200),
                     width="stretch", hide_index=True)

        fig = go.Figure(go.Histogram(
            x=filtered["delay_ratio"], nbinsx=40, marker_color="#E74C3C"))
        fig.update_layout(
            title="Delay Ratio Distribution (filtered corridors)", height=350,
            xaxis_title="Delay Ratio", yaxis_title="Corridor Count",
            margin=dict(l=10, r=10, t=40, b=10),
        )
        st.plotly_chart(fig, width="stretch")
