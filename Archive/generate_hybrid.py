from __future__ import annotations

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import json
import time
import itertools
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import torch as tc
import torch.nn as nn
from tqdm import tqdm

from equation.option_pricing import BlackScholes
from optimize.option_princing import BlackScholeOptimizer

from method.nn import MLP, ResNet
from method.hnn import HybridCQN
from method.qnn import QuantumNeuralNetwork


# ============================================================
# CONFIG
# ============================================================

DEVICE = "cpu"

RESULTS_DIR = Path("fase_0_triagem")
RUNS_DIR = RESULTS_DIR / "runs"

SUMMARY_CLASSIC_PATH = RESULTS_DIR / "sumario_classico.csv"
SUMMARY_HYBRID_PATH = RESULTS_DIR / "sumario_hibrido.csv"
SUMMARY_QUANTUM_PATH = RESULTS_DIR / "sumario_quantico.csv"

for d in [RESULTS_DIR, RUNS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

HEADERS_WRITTEN = {
    str(SUMMARY_CLASSIC_PATH): SUMMARY_CLASSIC_PATH.exists(),
    str(SUMMARY_HYBRID_PATH): SUMMARY_HYBRID_PATH.exists(),
    str(SUMMARY_QUANTUM_PATH): SUMMARY_QUANTUM_PATH.exists(),
}


# ============================================================
# UTILS
# ============================================================

def sanitize_for_json(obj):
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, nn.Module):
        return obj.__class__.__name__
    if isinstance(obj, dict):
        return {k: sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [sanitize_for_json(v) for v in obj]
    return str(obj)


def save_json(data, path):
    with open(path, "w") as f:
        json.dump(sanitize_for_json(data), f, indent=2)


def set_seed(seed):
    np.random.seed(seed)
    tc.manual_seed(seed)


def generate_runs(base_config, sweep_params):
    experiments = []

    keys = list(sweep_params.keys())
    values = list(sweep_params.values())

    for combo in itertools.product(*values):
        cfg = base_config.copy()
        parts = [cfg["run_id_prefix"]]

        for k, v in zip(keys, combo):
            cfg[k] = v

            short = (
                k.replace("learning_rate", "lr")
                 .replace("hidden", "h")
                 .replace("layers", "l")
                 .replace("blocks", "b")
                 .replace("epochs", "ep")
            )
            parts.append(f"{short}_{v}")

        cfg["run_id"] = "_".join(parts)
        experiments.append(cfg)

    return experiments


def resolve_summary_path(model_type):
    if model_type in ["MLP", "ResNet"]:
        return str(SUMMARY_CLASSIC_PATH)
    if model_type == "QNN":
        return str(SUMMARY_QUANTUM_PATH)
    if model_type == "HQNN":
        return str(SUMMARY_HYBRID_PATH)


def get_run_dirs(model_type, run_id):
    root = RUNS_DIR / model_type / run_id
    model_dir = root / "model"
    loss_dir = root / "losses"
    meta_dir = root / "metadata"

    for d in [root, model_dir, loss_dir, meta_dir]:
        d.mkdir(parents=True, exist_ok=True)

    return model_dir, loss_dir, meta_dir


# ============================================================
# MODELOS
# ============================================================

def build_classical_block(cfg):
    if cfg["model_class"] == "MLP":
        return MLP(
            hidden=cfg["hidden"],
            blocks=cfg["blocks"],
            device=DEVICE,
            activation=cfg["activation"],
        )

    if cfg["model_class"] == "ResNet":
        return ResNet(
            hidden=cfg["hidden"],
            blocks=cfg["blocks"],
            device=DEVICE,
            activation=cfg["activation"],
        )

    raise ValueError(f"model_class '{cfg['model_class']}' não reconhecida.")

def build_model(cfg):
    model_type = cfg["model_type"]

    if model_type == "MLP":
        return build_classical_block(cfg), str(SUMMARY_CLASSIC_PATH)

    if model_type == "ResNet":
        return build_classical_block(cfg), str(SUMMARY_CLASSIC_PATH)

    if model_type == "QNN":
        qnn = QuantumNeuralNetwork(
            n_qubits=cfg["n_qubits"],
            n_layers=cfg["n_layers"],
            device=DEVICE,
            circuit_type=cfg.get("entangler"),
        )
        model = HybridCQN(
            classical_pre=None,
            qnn_block=qnn,
            classical_post=None,
            device=DEVICE,
        )
        return model, str(SUMMARY_QUANTUM_PATH)

    if model_type == "HQNN":
        qnn = QuantumNeuralNetwork(
            n_qubits=cfg["n_qubits"],
            n_layers=cfg["n_layers"],
            device=DEVICE,
            circuit_type=cfg.get("entangler"),
        )

        classical = build_classical_block(cfg)

        model = HybridCQN(
            classical_pre=classical,
            qnn_block=qnn,
            classical_post=None,
            device=DEVICE,
        )
        return model, str(SUMMARY_HYBRID_PATH)

    raise ValueError(f"model_type '{model_type}' não reconhecido.")

# ============================================================
# TREINO
# ============================================================

def train_and_evaluate(cfg, data_train, data_test):

    set_seed(cfg["seed"])

    model, summary_path = build_model(cfg)
    model_dir, loss_dir, meta_dir = get_run_dirs(cfg["model_type"], cfg["run_id"])

    opt = BlackScholeOptimizer(
        data_train,
        model,
        epochs=cfg["epochs"],
        lr=cfg["lr"],
        device=DEVICE,
    )

    t0 = time.time()
    loss = opt.train(return_loss=True, return_all=True)
    t1 = time.time()

    mse_n, mse_u, _ = opt.test(data_test, return_unormalized=True)

    # save
    tc.save(model.state_dict(), model_dir / "model.pth")
    save_json(loss, loss_dir / "loss.json")

    result = {
        "run_id": cfg["run_id"],
        "model_type": cfg["model_type"],
        "epochs": cfg["epochs"],
        "lr": cfg["lr"],
        "num_params": opt.num_params,
        "training_time_sec": t1 - t0,
        "mse_teste_normalizado": mse_n,
        "mse_teste_desnormalizado": mse_u,
        "loss_history_path": str(loss_dir / "loss.json"),
    }

    save_json(result, meta_dir / "results.json")

    row = cfg.copy()
    row.update(result)

    return row, summary_path


def append_summary(row, path):
    df = pd.DataFrame([row])
    header = not HEADERS_WRITTEN.get(path, False)
    df.to_csv(path, mode="a", header=header, index=False)
    HEADERS_WRITTEN[path] = True


# ============================================================
# EXPERIMENTOS
# ============================================================

experiment_grid = []

# MLP
experiment_grid += generate_runs(
    {
        "model_type": "MLP",
        "run_id_prefix": "mlp",
        "model_class": "MLP",
        "activation": nn.Tanh(),
    },
    {
        "hidden": [2, 3, 4],
        "blocks": [1, 2,3],
        "epochs": [1000, 2500, 5000],
        "lr": [1e-3, 2e-3, 1e-4, 1e-1, 1e-2, 5e-3, 1e-4, 1],
        "seed": [1924, 1925, 1926, 1927, 1928],
    },
)

# QNN
experiment_grid += generate_runs(
    {
        "model_type": "QNN",
        "run_id_prefix": "qnn",
        "entangler": "strong",
    },
    {
        "n_qubits": [2, 3, 4],
        "n_layers": [1, 2,3],
        "epochs": [1000, 2500, 5000],
        "lr": [1e-3, 2e-3, 1e-4, 1e-1, 1e-2, 5e-3, 1e-4],
        "seed": [1924, 1925, 1926, 1927, 1928],
    },
)

# HQNN
experiment_grid += generate_runs(
    {
        "model_type": "HQNN",
        "run_id_prefix": "hqnn",
        "model_class": "MLP",
        "activation": nn.Tanh(),
        "entangler": "strong",
    },
    {
        "hidden": [2, 4, 8],
        "blocks": [1, 2, 3],
        "n_qubits": [2, 3, 4],
        "n_layers": [1, 2,3],
        "epochs": [1000, 2500, 5000],
        "lr": [1e-3, 2e-3, 1e-4, 1e-1, 1e-2, 5e-3, 1e-4],
        "seed": [1924, 1925, 1926, 1927, 1928],
    },
)

print(f"Total experiments: {len(experiment_grid)}")


# ============================================================
# DADOS
# ============================================================

bse = BlackScholes()
data_train = bse.generate_data(seed=123)
data_test = bse.generate_data(seed=456)


# ============================================================
# LOOP
# ============================================================

for cfg in tqdm(experiment_grid):

    try:
        row, path = train_and_evaluate(cfg, data_train, data_test)
        append_summary(row, path)

        print(
            f"{cfg['run_id']} | mse={row['mse_teste_desnormalizado']:.4f}"
        )

    except Exception as e:
        print("ERRO:", e)