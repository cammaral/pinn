from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import torch as tc

from basket_configs import RESULTS_DIR, DEVICE
from equation.basket_option import ArithmeticBasketOption
from basket_experiment_utils import build_basket_model, load_json, save_json


def discover_basket_runs(results_dir: str | Path = RESULTS_DIR, families: Optional[List[str]] = None) -> List[Path]:
    families = families or ["MLP", "ResNet", "QNN", "HQNN"]
    runs_root = Path(results_dir) / "runs"
    out = []
    if not runs_root.exists():
        return out
    for family_dir in sorted(runs_root.iterdir()):
        if not family_dir.is_dir() or family_dir.name not in families:
            continue
        for run_dir in sorted(family_dir.iterdir()):
            if (run_dir / "metadata" / "config.json").exists() and (run_dir / "model" / "model_state_dict.pth").exists():
                out.append(run_dir)
    return out


def load_basket_model(run_dir: Path, device: str = DEVICE):
    cfg = load_json(run_dir / "metadata" / "config.json")
    basket_params = cfg["basket_params"]
    model = build_basket_model(cfg, basket_params, device=device)
    state = tc.load(run_dir / "model" / "model_state_dict.pth", map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model, cfg


def eval_points(problem: ArithmeticBasketOption, n_points: int = 512, seed: int = 777):
    rng = np.random.default_rng(seed)
    S = rng.uniform(0.05 * problem.S_max, 0.95 * problem.S_max, size=(n_points, problem.n_assets))
    t = rng.uniform(0.0, 0.95 * problem.T, size=(n_points, 1))
    return S, t


def benchmark_cache_path(problem: ArithmeticBasketOption, n_points: int, seed: int, h_rel: float, h_t: float, results_dir: Path):
    extra = {"eval_greeks_n": n_points, "eval_seed": seed, "h_rel": h_rel, "h_t": h_t}
    return results_dir / "basket_greeks" / "benchmark_cache" / f"basket_greeks_N{problem.n_assets}_{problem.config_hash(extra)}.npz"


def get_benchmark_greeks(problem: ArithmeticBasketOption, results_dir: Path, n_points: int = 512, seed: int = 777, h_rel: float = 1e-3, h_t: float = 1e-4):
    path = benchmark_cache_path(problem, n_points, seed, h_rel, h_t, results_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        z = np.load(path)
        return {
            "S": z["S"],
            "t": z["t"],
            "V": z["V"],
            "delta": z["delta"],
            "gamma": z["gamma"],
            "theta": z["theta"],
            "cache_path": str(path),
        }
    S, t = eval_points(problem, n_points=n_points, seed=seed)
    g = problem.finite_difference_greeks(S, t, h_rel=h_rel, h_t=h_t)
    np.savez_compressed(path, S=S, t=t, V=g["V"], delta=g["delta"], gamma=g["gamma"], theta=g["theta"])
    g.update({"S": S, "t": t, "cache_path": str(path)})
    return g


def model_greeks(model, S: np.ndarray, t: np.ndarray, basket_params: Dict[str, Any], device: str = DEVICE):
    n_assets = int(basket_params["n_assets"])
    S_max = np.asarray(basket_params["S_max"], dtype=float).reshape(n_assets)
    T = float(basket_params["T"])
    V_max = float(basket_params["V_max"])

    S_norm_np = S / S_max.reshape(1, -1)
    t_norm_np = t / T

    S_norm = tc.tensor(S_norm_np, dtype=tc.float32, requires_grad=True, device=device)
    t_norm = tc.tensor(t_norm_np, dtype=tc.float32, requires_grad=True, device=device)
    x = tc.cat([S_norm, t_norm], dim=1)

    V_norm = model(x).reshape(-1, 1)
    ones = tc.ones_like(V_norm)

    dV_dS_norm = tc.autograd.grad(V_norm, S_norm, grad_outputs=ones, create_graph=True, retain_graph=True)[0]
    dV_dt_norm = tc.autograd.grad(V_norm, t_norm, grad_outputs=ones, create_graph=True, retain_graph=True)[0]

    H_cols = []
    for i in range(n_assets):
        gi = dV_dS_norm[:, i:i+1]
        hi = tc.autograd.grad(gi, S_norm, grad_outputs=tc.ones_like(gi), create_graph=False, retain_graph=True)[0]
        H_cols.append(hi)
    H_norm = tc.stack(H_cols, dim=1)

    V = (V_norm * V_max).detach().cpu().numpy()
    delta = dV_dS_norm.detach().cpu().numpy() * (V_max / S_max.reshape(1, -1))

    H = H_norm.detach().cpu().numpy()
    gamma = np.zeros_like(H)
    for i in range(n_assets):
        for j in range(n_assets):
            gamma[:, i, j] = H[:, i, j] * V_max / (S_max[i] * S_max[j])

    theta = dV_dt_norm.detach().cpu().numpy() * (V_max / T)
    return {"V": V, "delta": delta, "gamma": gamma, "theta": theta}


def metric_block(y_true, y_pred, prefix: str):
    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=float).reshape(-1)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y = y_true[mask]
    p = y_pred[mask]
    e = p - y
    if len(e) == 0:
        return {f"MAE_{prefix}": np.nan, f"RMSE_{prefix}": np.nan, f"Bias_{prefix}": np.nan}
    return {
        f"MAE_{prefix}": float(np.mean(np.abs(e))),
        f"RMSE_{prefix}": float(np.sqrt(np.mean(e ** 2))),
        f"Bias_{prefix}": float(np.mean(e)),
    }


def process_run(run_dir: Path, results_dir: Path, n_points: int = 512, seed: int = 777):
    model, cfg = load_basket_model(run_dir, device=DEVICE)
    basket_params = cfg["basket_params"]
    problem = ArithmeticBasketOption(**{**basket_params, "cache_dir": results_dir / "benchmarks"})

    bench = get_benchmark_greeks(problem, results_dir, n_points=n_points, seed=seed)
    pred = model_greeks(model, bench["S"], bench["t"], basket_params, device=DEVICE)

    n_assets = int(basket_params["n_assets"])
    row = {
        "run_id": cfg["run_id"],
        "model_type": cfg["model_type"],
        "dimension": n_assets,
        "seed": cfg.get("seed"),
        "num_params": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "benchmark_cache_path": bench["cache_path"],
    }
    for key in ["hidden", "blocks", "n_qubits", "n_layers", "entangler"]:
        if key in cfg:
            row[key] = cfg[key]

    row.update(metric_block(bench["V"], pred["V"], "V_global"))
    row.update(metric_block(bench["theta"], pred["theta"], "theta_global"))

    for i in range(n_assets):
        row.update(metric_block(bench["delta"][:, i], pred["delta"][:, i], f"delta_{i+1}_global"))
        row.update(metric_block(bench["gamma"][:, i, i], pred["gamma"][:, i, i], f"gamma_{i+1}{i+1}_global"))
    for i in range(n_assets):
        for j in range(i + 1, n_assets):
            row.update(metric_block(bench["gamma"][:, i, j], pred["gamma"][:, i, j], f"gamma_{i+1}{j+1}_global"))

    out_dir = results_dir / "basket_greeks" / cfg["model_type"] / cfg["run_id"]
    out_dir.mkdir(parents=True, exist_ok=True)

    pointwise = pd.DataFrame({
        **{f"S{i+1}": bench["S"][:, i] for i in range(n_assets)},
        "t": bench["t"].reshape(-1),
        "V_true": bench["V"].reshape(-1),
        "V_pred": pred["V"].reshape(-1),
        "theta_true": bench["theta"].reshape(-1),
        "theta_pred": pred["theta"].reshape(-1),
    })
    for i in range(n_assets):
        pointwise[f"delta{i+1}_true"] = bench["delta"][:, i]
        pointwise[f"delta{i+1}_pred"] = pred["delta"][:, i]
        pointwise[f"gamma{i+1}{i+1}_true"] = bench["gamma"][:, i, i]
        pointwise[f"gamma{i+1}{i+1}_pred"] = pred["gamma"][:, i, i]
    for i in range(n_assets):
        for j in range(i + 1, n_assets):
            pointwise[f"gamma{i+1}{j+1}_true"] = bench["gamma"][:, i, j]
            pointwise[f"gamma{i+1}{j+1}_pred"] = pred["gamma"][:, i, j]

    pointwise.to_csv(out_dir / "basket_greeks_pointwise.csv", index=False)
    save_json(row, out_dir / "basket_greeks_metrics.json")
    return row


def main():
    results_dir = Path(RESULTS_DIR)
    runs = discover_basket_runs(results_dir)
    print(f"Runs encontradas: {len(runs)}")
    rows = []
    failed = []
    for i, run_dir in enumerate(runs, start=1):
        print(f"[{i}/{len(runs)}] {run_dir}")
        try:
            row = process_run(run_dir, results_dir)
            rows.append(row)
            print(f"  OK | RMSE_V={row.get('RMSE_V_global'):.6e}")
        except Exception as e:
            print(f"  FALHOU: {e}")
            failed.append({"run_dir": str(run_dir), "error": str(e)})

    out_dir = results_dir / "basket_greeks"
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_dir / "basket_greeks_summary.csv", index=False)
    pd.DataFrame(failed).to_csv(out_dir / "basket_greeks_failed_runs.csv", index=False)
    print("Resumo:", out_dir / "basket_greeks_summary.csv")
    print("Falhas:", out_dir / "basket_greeks_failed_runs.csv")


if __name__ == "__main__":
    main()
