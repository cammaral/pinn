from __future__ import annotations

import os
import json
import time
import itertools
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import torch as tc
import torch.nn as nn
from tqdm import tqdm

# =========================
# IMPORTS DO PROJETO
# =========================
from equation.option_pricing import BlackScholes
from optimize.option_princing import BlackScholeOptimizer

from method.nn import MLP, ResNet
from method.hnn import HybridCQN
from method.qnn import QuantumNeuralNetwork, CorrelatorQuantumNeuralNetwork

from utils.save import (
    RESULTS_DIR,
    SUMMARY_HYBRID_PATH,
    SUMMARY_CQUANTUM_PATH,
    run_already_done,
    resolve_summary_path,
)

# =============================================================================
# CONFIGURAÇÃO GLOBAL
# =============================================================================

DEVICE = "cuda" if tc.cuda.is_available() else "cpu"

RESULTS_DIR = Path(RESULTS_DIR)

RUNS_DIR = RESULTS_DIR / "runs"
MODELS_DIR = RESULTS_DIR / "models"
LOSSES_DIR = RESULTS_DIR / "losses"
METADATA_DIR = RESULTS_DIR / "metadata"

for d in [RUNS_DIR, MODELS_DIR, LOSSES_DIR, METADATA_DIR]:
    d.mkdir(parents=True, exist_ok=True)

HEADERS_WRITTEN = {
    str(SUMMARY_HYBRID_PATH): os.path.exists(SUMMARY_HYBRID_PATH),
    str(SUMMARY_CQUANTUM_PATH): os.path.exists(SUMMARY_CQUANTUM_PATH),
}


# =============================================================================
# FUNÇÕES AUXILIARES
# =============================================================================

def sanitize_for_json(obj: Any) -> Any:
    """Converte objetos não serializáveis para formatos seguros."""
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


def generate_runs(base_config: Dict[str, Any], sweep_params: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
    """
    Gera a grade de experimentos a partir de uma configuração base + sweep.
    """
    experiments = []

    param_keys = list(sweep_params.keys())
    param_values = list(sweep_params.values())

    for combination in itertools.product(*param_values):
        cfg = base_config.copy()
        run_parts = [base_config.get("run_id_prefix", base_config.get("model_type", "run"))]

        for key, value in zip(param_keys, combination):
            cfg[key] = value

            if isinstance(value, nn.Module):
                val_str = value.__class__.__name__
            else:
                val_str = str(value)

            short_key = (
                key.replace("learning_rate", "lr")
                   .replace("hidden", "h")
                   .replace("layers", "l")
                   .replace("blocks", "b")
            )
            run_parts.append(f"{short_key}_{val_str}")

        cfg["run_id"] = "_".join(run_parts)
        experiments.append(cfg)

    return experiments


def pretty_print(config_list: List[Dict[str, Any]], num_to_show: int = 5) -> None:
    for cfg in config_list[:num_to_show]:
        print(json.dumps(sanitize_for_json(cfg), indent=2, ensure_ascii=False))
    if len(config_list) > num_to_show:
        print(f"... e mais {len(config_list) - num_to_show} outros.")


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    tc.manual_seed(seed)
    if tc.cuda.is_available():
        tc.cuda.manual_seed_all(seed)


def get_run_dirs(model_type: str, run_id: str) -> Dict[str, Path]:
    """
    Estrutura organizada por run:
      resultados/
        runs/
          HQNN/
            run_id/
              model/
              losses/
              metadata/
    """
    run_root = RUNS_DIR / model_type / run_id
    model_dir = run_root / "model"
    losses_dir = run_root / "losses"
    metadata_dir = run_root / "metadata"

    for d in [run_root, model_dir, losses_dir, metadata_dir]:
        d.mkdir(parents=True, exist_ok=True)

    return {
        "root": run_root,
        "model_dir": model_dir,
        "losses_dir": losses_dir,
        "metadata_dir": metadata_dir,
    }


def build_classical_block(config: Dict[str, Any]):
    model_class = config.get("model_class", "MLP")

    if model_class == "MLP":
        return MLP(
            hidden=config["hidden"],
            blocks=config["blocks"],
            device=DEVICE,
            activation=config["activation"],
        )

    if model_class == "ResNet":
        return ResNet(
            hidden=config["hidden"],
            blocks=config["blocks"],
            device=DEVICE,
            activation=config["activation"],
        )

    raise ValueError(f"model_class '{model_class}' não reconhecida.")


def build_model(config: Dict[str, Any]):
    """
    Mantém apenas modelos híbridos:
      - HQNN
      - CQNN / CQNN_nonlinear
    """
    model_type = config["model_type"]

    if model_type == "HQNN":
        qnn = QuantumNeuralNetwork(
            n_qubits=config["n_qubits"],
            n_layers=config["n_layers"],
            device=DEVICE,
            circuit_type=config.get("entangler"),
        )

        classical_pre = build_classical_block(config)

        model = HybridCQN(
            classical_pre=classical_pre,
            qnn_block=qnn,
            classical_post=None,
            device=DEVICE,
        )
        summary_path = str(SUMMARY_HYBRID_PATH)
        return model, summary_path

    if model_type in ["CQNN", "CQNN_nonlinear"]:
        qnn = CorrelatorQuantumNeuralNetwork(
            n_qubits=config["n_qubits"],
            n_layers=config["n_layers"],
            k=config["k"],
            n_vertex=config["n_vertex"],
            nonlinear=(model_type == "CQNN_nonlinear"),
            device=DEVICE,
            circuit_type=config.get("entangler"),
        )

        classical_pre = None
        if config.get("use_classical_pre", False):
            classical_pre = build_classical_block(config)

        model = HybridCQN(
            classical_pre=classical_pre,
            qnn_block=qnn,
            classical_post=None,
            device=DEVICE,
        )
        summary_path = str(SUMMARY_CQUANTUM_PATH)
        return model, summary_path

    raise ValueError(f"model_type '{model_type}' não suportado neste script.")


def save_json(data: Dict[str, Any], path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(sanitize_for_json(data), f, indent=2, ensure_ascii=False)


def extract_loss_metrics(loss_history: Any) -> Dict[str, Any]:
    """
    Calcula métricas úteis a partir do histórico completo.
    Espera um dict, por exemplo:
      {
        "Total": [...],
        "PDE": [...],
        "BC_left": [...],
        "BC_right": [...],
        ...
      }
    """
    metrics = {}

    if isinstance(loss_history, dict):
        for key, values in loss_history.items():
            if isinstance(values, list) and len(values) > 0:
                metrics[f"final_loss_{key.lower()}"] = values[-1]
                metrics[f"mean_last_100_loss_{key.lower()}"] = float(np.mean(values[-100:]))

    return metrics


def append_to_summary(log_row: Dict[str, Any], summary_path: str) -> None:
    df_run = pd.DataFrame([sanitize_for_json(log_row)])
    write_header = not HEADERS_WRITTEN.get(summary_path, False)
    df_run.to_csv(summary_path, mode="a", header=write_header, index=False)
    HEADERS_WRITTEN[summary_path] = True


def train_and_evaluate(config: Dict[str, Any], data_train, data_test) -> Dict[str, Any]:
    """
    Executa uma run completa:
      1) cria modelo
      2) treina
      3) testa
      4) salva artefatos
      5) retorna log consolidado
    """
    run_id = config["run_id"]
    model_type = config["model_type"]
    seed = config.get("seed", 42)

    set_seed(seed)

    run_dirs = get_run_dirs(model_type, run_id)

    model, summary_path = build_model(config)

    optimizer = BlackScholeOptimizer(
        data=data_train,
        model=model,
        epochs=config["epochs"],
        lr=config["lr"],
        device=DEVICE,
        weights=config.get("weights", [1, 1, 1, 1]),
    )

    start_time = time.time()

    # IMPORTANTE:
    # aqui assumimos que return_all=True faz o optimizer retornar TODAS as losses
    loss_history = optimizer.train(return_loss=True, return_all=True)

    end_time = time.time()
    training_time_sec = end_time - start_time

    mse_norm, mse_unorm, _ = optimizer.test(data_test, return_unormalized=True)
    num_params = optimizer.num_params

    # =========================
    # SALVAMENTO FINAL DA RUN
    # =========================

    model_path = run_dirs["model_dir"] / "model_state_dict.pth"
    tc.save(model.state_dict(), model_path)

    loss_path = run_dirs["losses_dir"] / "loss_history_full.json"
    save_json(loss_history, loss_path)

    config_path = run_dirs["metadata_dir"] / "config.json"
    save_json(config, config_path)

    results_path = run_dirs["metadata_dir"] / "results.json"

    results_payload = {
        "run_id": run_id,
        "model_type": model_type,
        "seed": seed,
        "training_time_sec": training_time_sec,
        "num_params": num_params,
        "mse_teste_normalizado": mse_norm,
        "mse_teste_desnormalizado": mse_unorm,
        "model_path": str(model_path),
        "loss_history_path": str(loss_path),
        "config_path": str(config_path),
    }
    results_payload.update(extract_loss_metrics(loss_history))
    save_json(results_payload, results_path)

    # linha para CSV de sumário
    log_row = config.copy()
    log_row.update(results_payload)

    return {
        "log_row": log_row,
        "summary_path": summary_path,
    }


# =============================================================================
# DEFINIÇÃO DOS EXPERIMENTOS
# =============================================================================

experiment_grid: List[Dict[str, Any]] = []

# -------------------------------------------------------------------------
# EXEMPLO 1: HQNN
# -------------------------------------------------------------------------
base_hqnn = {
    "model_type": "HQNN",
    "run_id_prefix": "hqnn_strong_mlp",
    "model_class": "MLP",
    "activation": nn.Tanh(),
    "lr": 2e-3,
    "epochs": 15000,
    "entangler": "strong",
}

sweep_hqnn = {
    "hidden": [2, 3, 5],
    "blocks": [1, 5, 10],
    "n_qubits": [2, 3, 4, 5, 7, 10],
    "n_layers": [1, 3, 5, 10],
    "seed": [1924, 1925, 1926, 1973, 2025, 2024, 2012, 1958, 1962, 1997]
}

experiment_grid.extend(generate_runs(base_hqnn, sweep_hqnn))

# -------------------------------------------------------------------------
# EXEMPLO 2: CQNN
# descomente se quiser usar
# -------------------------------------------------------------------------
# base_cqnn = {
#     "model_type": "CQNN_nonlinear",
#     "run_id_prefix": "cqnn_nonlinear",
#     "lr": 2e-3,
#     "epochs": 15000,
#     "entangler": "strong",
#     "use_classical_pre": False,
# }
#
# sweep_cqnn = {
#     "n_qubits": [5],
#     "n_layers": [4],
#     "k": [2],
#     "n_vertex": [5],
#     "seed": [1924, 1925, 1926],
# }
#
# experiment_grid.extend(generate_runs(base_cqnn, sweep_cqnn))

print(f"Total de {len(experiment_grid)} experimentos gerados.")
pretty_print(experiment_grid, num_to_show=3)


# =============================================================================
# DADOS
# =============================================================================

print("Gerando dados de treino e teste...")
bse = BlackScholes(eps=1e-10)
data_train = bse.generate_data(seed=1234)
data_test = bse.generate_data(seed=4321)


# =============================================================================
# LOOP PRINCIPAL
# =============================================================================

print(f"Iniciando {len(experiment_grid)} experimentos...")

for config in tqdm(experiment_grid, desc="Total de Experimentos"):
    run_id = config["run_id"]
    model_type = config["model_type"]

    summary_path = resolve_summary_path(model_type)

    # verificador
    if run_already_done(
        run_id,
        summary_path=summary_path,
        model_dir=str(MODELS_DIR),
        loss_dir=str(LOSSES_DIR),
    ):
        print(f"Pulando run '{run_id}' — já encontrada em sumário/artefatos.")
        continue

    try:
        output = train_and_evaluate(config, data_train, data_test)
        log_row = output["log_row"]
        summary_path = output["summary_path"]

        append_to_summary(log_row, summary_path)

        print(
            f"Run '{run_id}' concluída | "
            f"Tempo: {log_row['training_time_sec']:.2f}s | "
            f"MSE (desnorm): {log_row['mse_teste_desnormalizado']:.6f} | "
            f"Parâmetros: {log_row['num_params']}"
        )

    except Exception as e:
        print(f"ERRO na run '{run_id}': {e}")
        continue

print("\n--- TODOS OS EXPERIMENTOS CONCLUÍDOS ---")
print("Os modelos, losses e metadados foram salvos por run.")