from __future__ import annotations

import json
import time
import itertools
from math import comb
from pathlib import Path
from typing import Any, Dict, List, Tuple

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


# =============================================================================
# CONFIGURAÇÃO GLOBAL
# =============================================================================

DEVICE = "cuda" if tc.cuda.is_available() else "cpu"

RESULTS_DIR = Path("experimentos_pinn")
RUNS_DIR = RESULTS_DIR / "runs"

SUMMARY_CLASSIC_PATH = RESULTS_DIR / "sumario_classico.csv"
SUMMARY_HYBRID_PATH = RESULTS_DIR / "sumario_hibrido.csv"
SUMMARY_QUANTUM_PATH = RESULTS_DIR / "sumario_quantico.csv"
SUMMARY_CQUANTUM_PATH = RESULTS_DIR / "sumario_cquantico.csv"

for d in [RESULTS_DIR, RUNS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

HEADERS_WRITTEN = {
    str(SUMMARY_CLASSIC_PATH): SUMMARY_CLASSIC_PATH.exists(),
    str(SUMMARY_HYBRID_PATH): SUMMARY_HYBRID_PATH.exists(),
    str(SUMMARY_QUANTUM_PATH): SUMMARY_QUANTUM_PATH.exists(),
    str(SUMMARY_CQUANTUM_PATH): SUMMARY_CQUANTUM_PATH.exists(),
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


def save_json(data: Any, path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(sanitize_for_json(data), f, indent=2, ensure_ascii=False)


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    tc.manual_seed(seed)
    if tc.cuda.is_available():
        tc.cuda.manual_seed_all(seed)


def generate_runs(base_config: Dict[str, Any], sweep_params: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
    """
    Gera a grade de experimentos a partir de uma configuração base + sweep.
    """
    generated_experiments = []
    param_keys = list(sweep_params.keys())
    param_values = list(sweep_params.values())

    for combination in itertools.product(*param_values):
        new_config = base_config.copy()
        run_id_parts = [base_config.get("run_id_prefix", base_config.get("model_type", "run"))]

        for key, value in zip(param_keys, combination):
            new_config[key] = value

            if isinstance(value, nn.Module):
                val_str = value.__class__.__name__
            else:
                val_str = str(value)

            key_str = (
                key.replace("learning_rate", "lr")
                   .replace("hidden", "h")
                   .replace("layers", "l")
                   .replace("blocks", "b")
            )
            run_id_parts.append(f"{key_str}_{val_str}")

        new_config["run_id"] = "_".join(run_id_parts)
        generated_experiments.append(new_config)

    return generated_experiments


def pretty_print(config_list: List[Dict[str, Any]], num_to_show: int = 5) -> None:
    for cfg in config_list[:num_to_show]:
        print(json.dumps(sanitize_for_json(cfg), indent=2, ensure_ascii=False))
    if len(config_list) > num_to_show:
        print(f"... e mais {len(config_list) - num_to_show} outros.")


def resolve_summary_path(model_type: str) -> str:
    if model_type in ["MLP", "ResNet"]:
        return str(SUMMARY_CLASSIC_PATH)
    if model_type in ["QPINN"]:
        return str(SUMMARY_HYBRID_PATH)
    if model_type in ["QNN"]:
        return str(SUMMARY_QUANTUM_PATH)
    if model_type in ["CQNN", "CQNN_nonlinear"]:
        return str(SUMMARY_CQUANTUM_PATH)
    raise ValueError(f"model_type '{model_type}' não reconhecido.")


def get_run_dirs(model_type: str, run_id: str) -> Dict[str, Path]:
    """
    Estrutura por run:
      expermientos_pinn/
        runs/
          QNN/
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


def validate_config(config: Dict[str, Any]) -> Tuple[bool, str]:
    """
    Valida configurações inválidas, principalmente para CQNN.
    """
    model_type = config["model_type"]

    if model_type in ["CQNN", "CQNN_nonlinear"]:
        k = config["k"]
        n_qubits = config["n_qubits"]
        n_vertex = config["n_vertex"]

        if k > n_qubits:
            return False, f"k={k} > n_qubits={n_qubits}"

        max_vertices = 3 * comb(n_qubits, k)
        if n_vertex > max_vertices:
            return (
                False,
                f"n_vertex={n_vertex} > máximo possível {max_vertices} para n={n_qubits}, k={k}",
            )

    return True, ""


def run_already_done(run_id: str, model_type: str, summary_path: str) -> bool:
    """
    Verifica se a run já foi executada:
      1) se já existe no CSV
      2) se já existem os artefatos principais da run
    """
    try:
        csv_path = Path(summary_path)
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            if "run_id" in df.columns and run_id in df["run_id"].astype(str).values:
                return True
    except Exception:
        pass

    run_root = RUNS_DIR / model_type / run_id
    model_path = run_root / "model" / "model_state_dict.pth"
    loss_path = run_root / "losses" / "loss_history_full.json"
    results_path = run_root / "metadata" / "results.json"

    return model_path.exists() and loss_path.exists() and results_path.exists()


def build_model(config: Dict[str, Any]):
    model_type = config["model_type"]

    if model_type == "MLP":
        model = MLP(
            hidden=config["hidden"],
            blocks=config["blocks"],
            device=DEVICE,
            activation=config["activation"],
        )
        return model, str(SUMMARY_CLASSIC_PATH)

    if model_type == "ResNet":
        model = ResNet(
            hidden=config["hidden"],
            blocks=config["blocks"],
            device=DEVICE,
            activation=config["activation"],
        )
        return model, str(SUMMARY_CLASSIC_PATH)

    if model_type == "QPINN":
        qnn = QuantumNeuralNetwork(
            n_qubits=config["n_qubits"],
            n_layers=config["n_layers"],
            device=DEVICE,
            circuit_type=config.get("entangler"),
        )
        model = HybridCQN(
            classical_pre=None,
            qnn_block=qnn,
            classical_post=None,
            device=DEVICE,
        )
        return model, str(SUMMARY_HYBRID_PATH)

    if model_type == "QNN":
        qnn = QuantumNeuralNetwork(
            n_qubits=config["n_qubits"],
            n_layers=config["n_layers"],
            device=DEVICE,
            circuit_type=config.get("entangler"),
        )
        model = HybridCQN(
            classical_pre=None,
            qnn_block=qnn,
            classical_post=None,
            device=DEVICE,
        )
        return model, str(SUMMARY_QUANTUM_PATH)

    if model_type == "CQNN":
        qnn = CorrelatorQuantumNeuralNetwork(
            n_qubits=config["n_qubits"],
            n_layers=config["n_layers"],
            k=config["k"],
            n_vertex=config["n_vertex"],
            nonlinear=False,
            device=DEVICE,
            circuit_type=config.get("entangler"),
        )
        model = HybridCQN(
            classical_pre=None,
            qnn_block=qnn,
            classical_post=None,
            device=DEVICE,
        )
        return model, str(SUMMARY_CQUANTUM_PATH)

    if model_type == "CQNN_nonlinear":
        qnn = CorrelatorQuantumNeuralNetwork(
            n_qubits=config["n_qubits"],
            n_layers=config["n_layers"],
            k=config["k"],
            n_vertex=config["n_vertex"],
            nonlinear=True,
            device=DEVICE,
            circuit_type=config.get("entangler"),
        )
        model = HybridCQN(
            classical_pre=None,
            qnn_block=qnn,
            classical_post=None,
            device=DEVICE,
        )
        return model, str(SUMMARY_CQUANTUM_PATH)

    raise ValueError(f"Tipo de modelo '{model_type}' não reconhecido.")


def extract_loss_metrics(loss_history: Any) -> Dict[str, Any]:
    """
    Extrai métricas finais do histórico completo de loss.
    Funciona se loss_history for dict ou lista.
    """
    metrics = {}

    if isinstance(loss_history, dict):
        for key, values in loss_history.items():
            if isinstance(values, list) and len(values) > 0:
                safe_key = str(key).lower().replace(" ", "_")
                metrics[f"final_loss_{safe_key}"] = values[-1]
                metrics[f"mean_last_100_loss_{safe_key}"] = float(np.mean(values[-100:]))
    elif isinstance(loss_history, list) and len(loss_history) > 0:
        metrics["final_loss"] = loss_history[-1]
        metrics["mean_last_100_loss"] = float(np.mean(loss_history[-100:]))

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

    opt = BlackScholeOptimizer(
        data_train,
        model,
        epochs=config["epochs"],
        lr=config["lr"],
        device=DEVICE,
        weights=config.get("weights", [1, 1, 1, 1]),
    )

    start_time = time.time()

    # IMPORTANTE:
    # aqui estamos pedindo o histórico completo de losses
    loss_history = opt.train(return_loss=True, return_all=True)

    end_time = time.time()
    training_time_sec = end_time - start_time

    mse_norm, mse_unorm, _ = opt.test(data_test, return_unormalized=True)
    num_params = opt.num_params

    # =========================
    # SALVAMENTO FINAL
    # =========================

    model_path = run_dirs["model_dir"] / "model_state_dict.pth"
    tc.save(model.state_dict(), model_path)

    loss_path = run_dirs["losses_dir"] / "loss_history_full.json"
    save_json(loss_history, loss_path)

    config_path = run_dirs["metadata_dir"] / "config.json"
    save_json(config, config_path)

    results_payload = {
        "run_id": run_id,
        "model_type": model_type,
        "seed": seed,
        "device": DEVICE,
        "training_time_sec": training_time_sec,
        "num_params": num_params,
        "mse_teste_normalizado": mse_norm,
        "mse_teste_desnormalizado": mse_unorm,
        "model_path": str(model_path),
        "loss_history_path": str(loss_path),
        "config_path": str(config_path),
    }
    results_payload.update(extract_loss_metrics(loss_history))

    results_path = run_dirs["metadata_dir"] / "results.json"
    save_json(results_payload, results_path)

    log_row = config.copy()
    log_row.update(results_payload)

    if "activation" in log_row:
        log_row["activation"] = str(log_row["activation"])

    return {
        "log_row": log_row,
        "summary_path": summary_path,
    }


# =============================================================================
# 1. DEFINIÇÃO DOS EXPERIMENTOS
# =============================================================================

experiment_grid: List[Dict[str, Any]] = []

base_seed_test = {
    "model_type": "QNN",
    "run_id_prefix": "qnn_strong",
    "lr": 2e-3,
    "epochs": 15000,
    "activation": None,
    "entangler": "strong",
}

sweep_seed = {
    "n_qubits": [2, 3, 4, 5, 7, 10],
    "n_layers": [1, 3, 5, 10],
    "seed": [1924, 1925, 1926, 1973, 2025, 2024, 2012, 1958, 1962, 1997],
}

experiment_grid.extend(generate_runs(base_seed_test, sweep_seed))

print(f"Total de {len(experiment_grid)} experimentos gerados para a grade.")
pretty_print(experiment_grid, num_to_show=3)


# =============================================================================
# 2. GERAÇÃO DE DADOS
# =============================================================================

print("Gerando dados de treino e teste...")
bse = BlackScholes(eps=1e-10)
data_treino = bse.generate_data(seed=2025)
data_teste = bse.generate_data(seed=42)


# =============================================================================
# 3. LOOP PRINCIPAL
# =============================================================================

print(f"Iniciando {len(experiment_grid)} experimentos...")
print(f"Resultados clássicos serão salvos em: {SUMMARY_CLASSIC_PATH}")
print(f"Resultados híbridos serão salvos em: {SUMMARY_HYBRID_PATH}")
print(f"Resultados quânticos serão salvos em: {SUMMARY_QUANTUM_PATH}")
print(f"Resultados correlator quânticos serão salvos em: {SUMMARY_CQUANTUM_PATH}")

for config in tqdm(experiment_grid, desc="Total de Experimentos"):
    run_id = config["run_id"]
    model_type = config["model_type"]

    print(f"\n--- Iniciando Run: {run_id} ---")

    summary_path = resolve_summary_path(model_type)

    is_valid, reason = validate_config(config)
    if not is_valid:
        print(f"Pulando run '{run_id}': {reason}")
        continue

    if run_already_done(run_id, model_type=model_type, summary_path=summary_path):
        print(f"Pulando run '{run_id}' — já encontrada em sumário/artefatos.")
        continue

    try:
        output = train_and_evaluate(config, data_treino, data_teste)
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


# =============================================================================
# 4. CONCLUSÃO
# =============================================================================

print("\n--- TODOS OS EXPERIMENTOS CONCLUÍDOS ---")
print("Os modelos, losses, metadados e sumários foram salvos por run.")