from __future__ import annotations

import itertools
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import torch as tc
import torch.nn as nn

from equation.basket_option import ArithmeticBasketOption
from optimize.basket_option import BasketOptionOptimizerND
from method.nn_nd import MLPND, ResNetND, FeatureMLPND
from method.hnn import HybridCQN
from method.qnn import QuantumNeuralNetwork


def sanitize_for_json(obj: Any) -> Any:
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, nn.Module):
        return obj.__class__.__name__
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, dict):
        return {k: sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [sanitize_for_json(v) for v in obj]
    return str(obj)


def save_json(data: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(sanitize_for_json(data), f, indent=2, ensure_ascii=False)


def load_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    tc.manual_seed(seed)
    if tc.cuda.is_available():
        tc.cuda.manual_seed_all(seed)


def generate_runs(base_config: Dict[str, Any], sweep_params: Dict[str, List[Any]], dimension: int) -> List[Dict[str, Any]]:
    runs = []
    keys = list(sweep_params.keys())
    values = list(sweep_params.values())

    for combo in itertools.product(*values):
        cfg = base_config.copy()
        cfg["dimension"] = int(dimension)
        parts = [base_config.get("run_id_prefix", cfg.get("model_type", "run")), f"{dimension}d"]

        for key, value in zip(keys, combo):
            cfg[key] = value
            val_str = value.__class__.__name__ if isinstance(value, nn.Module) else str(value)
            key_str = key.replace("hidden", "h").replace("blocks", "b").replace("n_qubits", "q").replace("n_layers", "l")
            parts.append(f"{key_str}_{val_str}")

        cfg["run_id"] = "_".join(parts)
        runs.append(cfg)
    return runs


def activation_from_config(name: Any) -> nn.Module:
    if isinstance(name, nn.Module):
        return name
    if name is None:
        return nn.Tanh()
    key = str(name).replace("()", "").lower()
    mapping = {
        "tanh": nn.Tanh,
        "relu": nn.ReLU,
        "sigmoid": nn.Sigmoid,
        "gelu": nn.GELU,
        "softplus": nn.Softplus,
    }
    return mapping.get(key, nn.Tanh)()


def get_run_dirs(results_dir: Path, model_type: str, run_id: str) -> Dict[str, Path]:
    root = Path(results_dir) / "runs" / model_type / run_id
    model_dir = root / "model"
    losses_dir = root / "losses"
    metadata_dir = root / "metadata"
    for d in [root, model_dir, losses_dir, metadata_dir]:
        d.mkdir(parents=True, exist_ok=True)
    return {"root": root, "model_dir": model_dir, "losses_dir": losses_dir, "metadata_dir": metadata_dir}


def summary_path(results_dir: Path, dimension: int, model_type: str) -> Path:
    family = "classico" if model_type in ["MLP", "ResNet"] else model_type.lower()
    return Path(results_dir) / f"sumario_basket_{dimension}d_{family}.csv"


def run_already_done(results_dir: Path, model_type: str, run_id: str, summary_csv: Path) -> bool:
    if summary_csv.exists():
        try:
            df = pd.read_csv(summary_csv, usecols=["run_id"])
            if run_id in df["run_id"].astype(str).values:
                return True
        except Exception:
            pass
    run_root = Path(results_dir) / "runs" / model_type / run_id
    return (
        (run_root / "model" / "model_state_dict.pth").exists()
        and (run_root / "losses" / "loss_history_full.json").exists()
        and (run_root / "metadata" / "results.json").exists()
    )


def build_basket_model(config: Dict[str, Any], basket_params: Dict[str, Any], device: str = "cpu"):
    model_type = config["model_type"]
    n_assets = int(basket_params["n_assets"])
    input_dim = n_assets + 1

    if model_type == "MLP":
        return MLPND(
            input_dim=input_dim,
            hidden=config["hidden"],
            blocks=config["blocks"],
            activation=activation_from_config(config.get("activation")),
            output_dim=1,
            device=device,
        )

    if model_type == "ResNet":
        return ResNetND(
            input_dim=input_dim,
            hidden=config["hidden"],
            blocks=config["blocks"],
            activation=activation_from_config(config.get("activation")),
            output_dim=1,
            device=device,
        )

    if model_type == "QNN":
        qnn = QuantumNeuralNetwork(
            n_qubits=config["n_qubits"],
            n_layers=config["n_layers"],
            device=device,
            circuit_type=config.get("entangler", "strong"),
        )
        return HybridCQN(
            classical_pre=None,
            qnn_block=qnn,
            classical_post=None,
            input_dim=input_dim,
            output_dim=1,
            device=device,
        )

    if model_type == "HQNN":
        pre = FeatureMLPND(
            input_dim=input_dim,
            hidden=config["hidden"],
            blocks=config["blocks"],
            activation=activation_from_config(config.get("activation")),
            feature_dim=config.get("feature_dim", config["hidden"]),
            device=device,
        )
        qnn = QuantumNeuralNetwork(
            n_qubits=config["n_qubits"],
            n_layers=config["n_layers"],
            device=device,
            circuit_type=config.get("entangler", "strong"),
        )
        return HybridCQN(
            classical_pre=pre,
            qnn_block=qnn,
            classical_post=None,
            input_dim=input_dim,
            output_dim=1,
            device=device,
        )

    raise ValueError(f"Unsupported model_type={model_type}")


def extract_loss_metrics(loss_history: Dict[str, List[float]]) -> Dict[str, float]:
    out = {}
    for key, values in loss_history.items():
        if isinstance(values, list) and len(values):
            safe = key.lower().replace(" ", "_")
            out[f"final_loss_{safe}"] = values[-1]
            out[f"mean_last_100_loss_{safe}"] = float(np.mean(values[-100:]))
    return out


def make_basket_problem(basket_params: Dict[str, Any], results_dir: Path) -> ArithmeticBasketOption:
    p = dict(basket_params)
    p["cache_dir"] = Path(results_dir) / "benchmarks"
    return ArithmeticBasketOption(**p)


def train_and_evaluate_basket_run(
    config: Dict[str, Any],
    basket_params: Dict[str, Any],
    data_params: Dict[str, Any],
    results_dir: str | Path,
    device: str = "cpu",
) -> Dict[str, Any]:
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    seed = int(config.get("seed", 42))
    set_seed(seed)

    model_type = config["model_type"]
    run_id = config["run_id"]
    dimension = int(basket_params["n_assets"])
    spath = summary_path(results_dir, dimension, model_type)

    if run_already_done(results_dir, model_type, run_id, spath):
        print(f"[skip] {run_id}")
        return {"run_id": run_id, "skipped": True}

    problem = make_basket_problem(basket_params, results_dir)
    data_train = problem.generate_data(
        N_domain=data_params["N_domain"],
        N_terminal=data_params["N_terminal"],
        N_boundary=data_params["N_boundary"],
        seed=data_params.get("train_seed", seed),
        tag="train",
        cache=True,
    )
    data_test = problem.generate_data(
        N_domain=data_params["N_domain"],
        N_terminal=data_params["N_terminal"],
        N_boundary=data_params["N_boundary"],
        seed=data_params.get("test_seed", seed + 1000),
        tag="test",
        cache=True,
    )

    model = build_basket_model(config, basket_params, device=device)
    optimizer = BasketOptionOptimizerND(
        data=data_train,
        model=model,
        epochs=config["epochs"],
        lr=config["lr"],
        sigmas=basket_params["sigmas"],
        rho=basket_params["rho"],
        r=basket_params["r"],
        S_max=basket_params["S_max"],
        T=basket_params["T"],
        V_max=basket_params["V_max"],
        device=device,
    )

    run_dirs = get_run_dirs(results_dir, model_type, run_id)
    start = time.time()
    loss_history = optimizer.train(return_loss=True, return_all=True)
    training_time_sec = time.time() - start

    mse_norm, mse_unorm, pred = optimizer.test(data_test, return_unormalized=True)
    num_params = optimizer.num_params

    model_path = run_dirs["model_dir"] / "model_state_dict.pth"
    tc.save(model.state_dict(), model_path)

    loss_path = run_dirs["losses_dir"] / "loss_history_full.json"
    save_json(loss_history, loss_path)

    config_payload = dict(config)
    config_payload["basket_params"] = basket_params
    config_payload["data_params"] = data_params
    config_payload["results_dir"] = str(results_dir)
    config_path = run_dirs["metadata_dir"] / "config.json"
    save_json(config_payload, config_path)

    results_payload = {
        "run_id": run_id,
        "model_type": model_type,
        "dimension": dimension,
        "seed": seed,
        "training_time_sec": training_time_sec,
        "num_params": num_params,
        "mse_teste_normalizado": mse_norm,
        "mse_teste_desnormalizado": mse_unorm,
        "rmse_teste_desnormalizado": float(np.sqrt(mse_unorm)),
        "model_path": str(model_path),
        "loss_history_path": str(loss_path),
        "config_path": str(config_path),
        "benchmark_cache_dir": str(Path(results_dir) / "benchmarks"),
    }
    results_payload.update(extract_loss_metrics(loss_history))

    results_path = run_dirs["metadata_dir"] / "results.json"
    save_json(results_payload, results_path)

    row = dict(config_payload)
    row.update(results_payload)
    pd.DataFrame([sanitize_for_json(row)]).to_csv(spath, mode="a", header=not spath.exists(), index=False)

    print(f"[ok] {run_id} | mse_unorm={mse_unorm:.6e} | params={num_params}")
    return row


def run_experiment_grid(
    base_config: Dict[str, Any],
    sweep: Dict[str, List[Any]],
    basket_params: Dict[str, Any],
    data_params: Dict[str, Any],
    results_dir: str | Path,
    device: str = "cpu",
) -> pd.DataFrame:
    runs = generate_runs(base_config, sweep, dimension=int(basket_params["n_assets"]))
    rows = []
    print(f"Total runs: {len(runs)}")
    for cfg in runs:
        rows.append(train_and_evaluate_basket_run(cfg, basket_params, data_params, results_dir, device=device))
    return pd.DataFrame(rows)
