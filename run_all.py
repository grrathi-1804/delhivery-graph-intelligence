# run_all.py
from src.data_pipeline  import run_pipeline
from src.graph_builder  import build_full_network
from src.network_audit  import run_audit
from src.eta_models     import run_eta_pipeline

if __name__ == "__main__":
    pipeline = run_pipeline()
    network  = build_full_network(
                   corridor_df = pipeline["corridors"],
                   tod_df      = pipeline["tod"]
               )
    audit    = run_audit()
    eta      = run_eta_pipeline()