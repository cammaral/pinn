import os
import pandas as pd

RESULTS_DIR = "experimentos_pinn"

SUMMARY_CLASSIC_PATH = os.path.join(RESULTS_DIR, "sumario_classico.csv")
SUMMARY_HYBRID_PATH = os.path.join(RESULTS_DIR, "sumario_hibrido.csv")
SUMMARY_CHYBRID_PATH = os.path.join(RESULTS_DIR, "sumario_chibrido.csv")
SUMMARY_QUANTUM_PATH = os.path.join(RESULTS_DIR, "sumario_quantico.csv")
SUMMARY_CQUANTUM_PATH = os.path.join(RESULTS_DIR, "sumario_cquantico.csv")  # Placeholder

def resolve_summary_path(model_type):
    # mapeia para o CSV correto (ajuste se tiver mais tipos)
    if model_type == "HCQNN":
        return SUMMARY_CHYBRID_PATH
    elif model_type == "CQNN_nonlinear":
        return SUMMARY_CQUANTUM_PATH
    elif model_type == "QNN":
        return SUMMARY_QUANTUM_PATH
    else:
        return SUMMARY_CLASSIC_PATH

def run_already_done(run_id, summary_path=None, model_dir=None, loss_dir=None):
    # 1) checa se já consta no CSV (sem carregar tudo na memória)
    if summary_path and os.path.exists(summary_path):
        try:
            for chunk in pd.read_csv(summary_path, usecols=["run_id"], chunksize=10000):
                if (chunk["run_id"] == run_id).any():
                    return True
        except Exception:
            pass
    # 2) checa artefatos gravados
    if model_dir:
        if os.path.exists(os.path.join(model_dir, f"modelo_{run_id}.pth")):
            return True
    if loss_dir:
        if os.path.exists(os.path.join(loss_dir, f"loss_{run_id}.json")):
            return True
    return False
