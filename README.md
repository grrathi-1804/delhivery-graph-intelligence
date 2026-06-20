# Optimizing Delivery ETAs with Graph-Based Network Intelligence

**IIT Guwahati Consulting & Analytics Club — Case Study**

A five-phase analytics pipeline that re-frames Delhivery's logistics network as a graph — not a flat table of trips — to find structural bottlenecks, predict delivery times more accurately, and decide between Full Truckload (FTL) and Carting service at the corridor level. The project ends in a one-page strategy memo quantifying the business impact for a Head of Network Operations.

---

## Why a Graph?

Treating each delivery as an independent row hides the real story: a small number of facilities sit on a disproportionate share of every delivery path in the network. When one of them slows down, the damage doesn't stay local — it radiates into every corridor connected to it. Modelling the network as a directed, weighted graph is what makes that visible.

---

## Headline Findings

| Phase | Finding |
|---|---|
| **1 — Network Map** | 1,657 facilities, 2,806 active corridors, but the network is split into **64 disconnected islands** — most corridors have no backup route if they fail |
| **2 — Bottleneck Audit** | **89.4%** of all corridors are breaching SLA (>20% over OSRM's time estimate); just **5 hubs** are linked to roughly **1 in 5** late deliveries nationwide |
| **3 — ETA Prediction** | Adding graph-structural features (centrality, embeddings) to the prediction model cuts average error by **7.9%** and lifts on-time-promise accuracy by **+0.36 points** over a baseline that only sees OSRM's own estimate |
| **4 — FTL vs Carting** | A distance-aware cost model puts the FTL/Carting breakeven at **~86 km**; only **24 corridors** network-wide are long-haul enough to justify switching to FTL — validated as robust across a 6x range of cost assumptions |
| **5 — Strategy Memo** | Upgrading just the **top 3 bottleneck hubs** (0.2% of all facilities) is projected to cut network-wide late deliveries by **6%** and recover **₹6.8 lakh** in at-risk revenue |

---

## Project Structure

```
delhivery-graph-intelligence/
│
├── data/
│   ├── raw/                              # Original Delhivery CSV (not committed — see Setup)
│   └── processed/                        # Cleaned trips, corridor edges, TOD-stratified data
│
├── notebooks/
│   ├── 1.0_graph_exploration.ipynb       # EDA, delay-ratio distributions, degree plots
│   ├── 2.0_eta_model_benchmarking.ipynb  # (optional) interactive Phase 3 walkthrough
│   └── 3.0_ftl_vs_carting_framework.ipynb# (optional) interactive Phase 4 walkthrough
│
├── src/
│   ├── data_pipeline.py                  # Phase 1 — clean, merge, engineer delay features
│   ├── graph_builder.py                  # Phase 1 — construct the directed weighted graph
│   ├── network_audit.py                  # Phase 2 — centrality metrics, SLA breach audit
│   ├── eta_models.py                     # Phase 3 — baseline vs graph-enhanced ETA models
│   └── decision_framework.py             # Phase 4 — FTL vs Carting decision framework
│
├── outputs/
│   ├── models/                           # Pickled graph, trained models, embeddings
│   ├── metrics/                          # Benchmark tables, audit CSVs, recommendations
│   └── visualizations/                   # All charts generated across phases
│
├── delivery_memo/
│   ├── network_operations_strategy_memo.pdf
│   └── network_operations_strategy_memo.docx
│
├── requirements.txt
└── README.md
```

---

## Setup

```bash
git clone https://github.com/grrathi-1804
cd delhivery-graph-intelligence

python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # macOS/Linux

pip install -r requirements.txt
```

**Dataset:** Download the *Delhivery Logistics Dataset* from Kaggle and place it at:
```
data/raw/delhivery_data.csv
```

---

## Running the Pipeline

Run phases **in order** — each one depends on the previous phase's output:

```bash
python src/data_pipeline.py        # Phase 1a — cleans data, engineers delay ratios
python src/graph_builder.py        # Phase 1b — builds the directed weighted graph
python src/network_audit.py        # Phase 2  — centrality + bottleneck audit
python src/eta_models.py           # Phase 3  — baseline vs graph-enhanced ETA models
python src/decision_framework.py   # Phase 4  — FTL vs Carting recommendation engine
```

Phase 5 (the strategy memo) is a written deliverable in `delivery_memo/`, built from the outputs of Phases 1–4.

---

## Phase-by-Phase Summary

### Phase 1 — Data Pipeline & Graph Construction
Parses raw trip segments, computes the **delay ratio** (`actual_time ÷ OSRM_time`) as the core signal, and builds a `MultiDiGraph` where nodes are facilities and edges are corridors — keeping FTL and Carting as separate edges even on the same route, since their delay profiles differ completely.

**Key design decision:** segment-level columns (`segment_actual_time`, `segment_osrm_distance`) are used throughout, never the cumulative whole-trip columns (`actual_time`, `actual_distance_to_destination`) — the two are easy to confuse but describe different things in this schema.

### Phase 2 — Bottleneck & Corridor Audit
Computes four centrality metrics — betweenness, in/out-degree, clustering coefficient, PageRank — to rank facilities by structural risk, then audits every corridor for SLA breaches. Surfaces the network's Tier-1 hubs, its single points of failure (zero-clustering facilities with no alternate route), and its most severely delayed corridors.

### Phase 3 — Graph-Enhanced ETA Prediction
Benchmarks an XGBoost model using only trip-level features (OSRM estimate, distance, time-of-day) against a second model enriched with Phase 2's centrality features and node2vec structural embeddings. Reports MAE and **within-15%-of-actual accuracy** — the metric that maps directly to customer-facing delivery promises.

### Phase 4 — FTL vs Carting Decision Framework
Trains a delay-ratio regressor and an SLA-breach classifier, then runs a **counterfactual simulation**: for every corridor, asks "what would happen under FTL vs under Carting?", holding distance and structural position fixed. Converts the answer into a cost-time trade-off using a transparent, explicitly-flagged unit-economics model (the raw dataset has no cost column), validated with a sensitivity analysis across a 6x range of cost assumptions before trusting the result.

### Phase 5 — Network Operations Strategy Memo
A 2-page memo for the Head of Network Operations — no raw model output, no technical jargon. Names the top 5 bottleneck hubs with their estimated SLA-breach contribution, recommends three categorized interventions (facility upgrade / parallel route / route-type shift), and quantifies the projected % reduction in late deliveries and revenue-at-risk recovered from upgrading the top 3 hubs.

---

## Tech Stack

| Category | Tools |
|---|---|
| Data & Graph | pandas, NumPy, NetworkX |
| Machine Learning | XGBoost, scikit-learn, node2vec |
| Visualization | Matplotlib |
| Document Generation | python-docx (strategy memo) |
| Environment | Python 3.10+, virtualenv |

---

## Key Engineering Lessons from This Project

- **Segment vs. cumulative columns** look similar but measure different things — using the wrong one quietly corrupts every downstream distance-based feature.
- **Time columns are in minutes**, not hours — easy to mislabel when reporting MAE in a business-friendly unit.
- **A cost model with unchecked constants can mathematically guarantee one outcome** regardless of what the ML layer predicts — worth a sanity check *before* trusting a 100%/0% split.
- **Median over mean** for delay-ratio aggregation, throughout — a single catastrophic outlier trip shouldn't define a corridor's reputation.
- **Illustrative assumptions (unit costs, SLA penalties) are clearly isolated and labeled**, not buried in code, since the raw dataset has no financial columns of its own.

---

## Author

Gaurav Rathi|Nit Trichy
