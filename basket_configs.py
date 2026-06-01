from __future__ import annotations

import numpy as np
import torch.nn as nn

# ============================================================
# Diretórios
# ============================================================
RESULTS_DIR = "experimentos_pinn_basket"
DEVICE = "cpu"  # quantum simulator usually safer on CPU

# ============================================================
# Configurações realistas/controladas
# ============================================================
# 2D baseline: equity-like vols and moderate positive correlation
BASKET_2D_PARAMS = {
    "n_assets": 2,
    "S_max": [160.0, 160.0],
    "T": 1.0,
    "K": 40.0,
    "r": 0.05,
    "sigmas": [0.20, 0.25],
    "rho": [[1.0, 0.50], [0.50, 1.0]],
    "weights": [0.5, 0.5],
    "V_max": 120.0,
    "gh_order": 13,
}

# 3D baseline: heterogeneous vols and moderate positive correlation
BASKET_3D_PARAMS = {
    "n_assets": 3,
    "S_max": [160.0, 160.0, 160.0],
    "T": 1.0,
    "K": 40.0,
    "r": 0.05,
    "sigmas": [0.20, 0.25, 0.30],
    "rho": [[1.0, 0.50, 0.50], [0.50, 1.0, 0.50], [0.50, 0.50, 1.0]],
    "weights": [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0],
    "V_max": 120.0,
    "gh_order": 9,
}

# ============================================================
# Tamanhos de dados
# ============================================================
DATA_2D = {
    "N_domain": 1000,
    "N_terminal": 500,
    "N_boundary": 600,
    "train_seed": 1924,
    "test_seed": 2025,
}

DATA_3D = {
    "N_domain": 1500,
    "N_terminal": 700,
    "N_boundary": 900,
    "train_seed": 1924,
    "test_seed": 2025,
}

# ============================================================
# Sweeps pequenos para exemplo inicial
# Aumente depois para o artigo.
# ============================================================
SEEDS = [1924, 1925, 1926]

CLASSIC_BASE = {
    "model_type": "MLP",
    "model_class": "MLPND",
    "run_id_prefix": "basket_classic",
    "lr": 2e-3,
    "epochs": 3000,
    "activation": nn.Tanh(),
}

CLASSIC_SWEEP = {
    "hidden": [5, 10],
    "blocks": [2, 4],
    "seed": SEEDS,
}

QNN_BASE = {
    "model_type": "QNN",
    "run_id_prefix": "basket_qnn",
    "lr": 2e-3,
    "epochs": 3000,
    "entangler": "strong",
}

QNN_SWEEP_2D = {
    "n_qubits": [3, 4, 5],
    "n_layers": [1, 2],
    "seed": SEEDS,
}

QNN_SWEEP_3D = {
    "n_qubits": [4, 5, 6],
    "n_layers": [1, 2],
    "seed": SEEDS,
}

HQNN_BASE = {
    "model_type": "HQNN",
    "model_class": "FeatureMLPND",
    "run_id_prefix": "basket_hqnn",
    "lr": 2e-3,
    "epochs": 3000,
    "activation": nn.Tanh(),
    "entangler": "strong",
}

HQNN_SWEEP_2D = {
    "hidden": [3, 5],
    "blocks": [1, 2],
    "n_qubits": [3, 5],
    "n_layers": [1, 2],
    "seed": SEEDS,
}

HQNN_SWEEP_3D = {
    "hidden": [3, 5],
    "blocks": [1, 2],
    "n_qubits": [4, 6],
    "n_layers": [1, 2],
    "seed": SEEDS,
}
