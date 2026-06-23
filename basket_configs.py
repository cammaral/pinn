from __future__ import annotations

import torch.nn as nn

# ============================================================
# Diretórios / dispositivo
# ============================================================

RESULTS_DIR = "experimentos_pinn_basket"
DEVICE = "cpu"

# ============================================================
# Protocolo experimental
# ============================================================

EPOCHS_MAIN = 5000
LR_BASKET = 1e-3

SEEDS = [1924, 1925, 1926, 1973, 2025, 2024, 2012, 1958, 1962, 1997]

# ============================================================
# Basket 2D — média ponderada
# ============================================================
# B = 0.5*S1 + 0.5*S2
# S_i em [0, 160]  =>  B em [0, 160]
# K = 80 coloca o kink no centro do domínio
# V_max = max(B-K) = 160 - 80 = 80
# ============================================================

BASKET_2D_PARAMS = {
    "n_assets": 2,
    "S_max": [160.0, 160.0],
    "T": 1.0,
    "K": 80.0,
    "r": 0.05,
    "sigmas": [0.20, 0.25],
    "rho": [
        [1.00, 0.50],
        [0.50, 1.00],
    ],
    "weights": [0.5, 0.5],
    "V_max": 80.0,
    "gh_order": 25,
    "option_type": "call",
}

# ============================================================
# Basket 3D — média ponderada
# ============================================================
# B = (S1 + S2 + S3)/3
# S_i em [0, 160]  =>  B em [0, 160]
# K = 80 coloca o kink no centro do domínio
# V_max = max(B-K) = 160 - 80 = 80
# ============================================================

BASKET_3D_PARAMS = {
    "n_assets": 3,
    "S_max": [160.0, 160.0, 160.0],
    "T": 1.0,
    "K": 80.0,
    "r": 0.05,
    "sigmas": [0.20, 0.25, 0.30],
    "rho": [
        [1.00, 0.50, 0.50],
        [0.50, 1.00, 0.50],
        [0.50, 0.50, 1.00],
    ],
    "weights": [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0],
    "V_max": 80.0,
    "gh_order": 15,
    "option_type": "call",
}

# ============================================================
# Dados de treino/teste
# ============================================================
# Para 5000 épocas:
# 2D fica bem com 8000 pontos de domínio.
# 3D usa mais boundary por causa das faces do cubo.
# ============================================================

DATA_2D = {
    "N_domain": 8000,
    "N_terminal": 3000,
    "N_boundary": 3000,
    "train_seed": 42,
    "test_seed": 45,
}

DATA_3D = {
    "N_domain": 10000,
    "N_terminal": 3500,
    "N_boundary": 5000,
    "train_seed": 42,
    "test_seed": 45,
}

# ============================================================
# Bases dos modelos
# ============================================================

CLASSIC_BASE = {
    "model_type": "MLP",
    "model_class": "MLPND",
    "run_id_prefix": "basket_classic",
    "lr": LR_BASKET,
    "epochs": EPOCHS_MAIN,
    "activation": nn.Tanh(),
}

QNN_BASE = {
    "model_type": "QNN",
    "run_id_prefix": "basket_qnn",
    "lr": LR_BASKET,
    "epochs": EPOCHS_MAIN,
    "entangler": "strong",
}

HQNN_BASE = {
    "model_type": "HQNN",
    "model_class": "FeatureMLPND",
    "run_id_prefix": "basket_hqnn",
    "lr": LR_BASKET,
    "epochs": EPOCHS_MAIN,
    "activation": nn.Tanh(),
    "entangler": "strong",
}

# ============================================================
# Sweeps 2D
# Input 2D = [S1, S2, t], então QNN precisa de pelo menos 3 qubits.
# ============================================================

CLASSIC_SWEEP_2D = {
    "hidden": [5, 10],
    "blocks": [2, 4, 6],
    "seed": SEEDS,
}

QNN_SWEEP_2D = {
    "n_qubits": [3, 4, 5],
    "n_layers": [1, 2],
    "seed": SEEDS,
}

HQNN_SWEEP_2D = {
    "hidden": [5],
    "blocks": [2, 4],
    "n_qubits": [3, 4],
    "n_layers": [1, 2],
    "seed": SEEDS,
}

# ============================================================
# Sweeps 3D
# Input 3D = [S1, S2, S3, t], então QNN precisa de pelo menos 4 qubits.
# ============================================================

CLASSIC_SWEEP_3D = {
    "hidden": [8, 12],
    "blocks": [2, 4],
    "seed": SEEDS,
}

QNN_SWEEP_3D = {
    "n_qubits": [4, 5, 6],
    "n_layers": [1, 2],
    "seed": SEEDS,
}

HQNN_SWEEP_3D = {
    "hidden": [6],
    "blocks": [2, 4],
    "n_qubits": [4, 5],
    "n_layers": [1, 2],
    "seed": SEEDS,
}

# ============================================================
# Compatibilidade com scripts que usam nomes genéricos
# ============================================================

CLASSIC_SWEEP = CLASSIC_SWEEP_2D
QNN_SWEEP = QNN_SWEEP_2D
HQNN_SWEEP = HQNN_SWEEP_2D

# ============================================================
# Configurações opcionais para sensibilidade posterior
# Não são usadas automaticamente nos generate atuais.
# ============================================================

RHO_LEVELS = [0.25, 0.50, 0.75]

SIGMAS_2D_LOW = [0.15, 0.20]
SIGMAS_2D_BASE = [0.20, 0.25]
SIGMAS_2D_HIGH = [0.30, 0.35]

SIGMAS_3D_LOW = [0.15, 0.18, 0.22]
SIGMAS_3D_BASE = [0.20, 0.25, 0.30]
SIGMAS_3D_HIGH = [0.30, 0.35, 0.40]
