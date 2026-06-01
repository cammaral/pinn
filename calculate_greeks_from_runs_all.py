from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch as tc
import torch.nn as nn
from scipy.stats import norm

# ============================================================
# Projeto
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from method.nn import MLP, ResNet
from method.hnn import HybridCQN
from method.qnn import QuantumNeuralNetwork, CorrelatorQuantumNeuralNetwork
from utils.device import pick_torch_device

# ============================================================
# Defaults
# ============================================================
DEFAULT_RESULTS_DIR = Path("experimentos_pinn")
DEFAULT_DEVICE = "cpu"
DEFAULT_DTYPE = tc.float32

BS_PARAMS = {
    "S_max": 160.0,
    "T": 1.0,
    "K": 40.0,
    "r": 0.05,
    "sigma": 0.2,
    "option_type": "call",
    "V_max": 120.0,
    "eps": 1e-10,
}

EVAL_PARAMS = {
    "eval_mode": "grid",      # "grid" ou "random_domain"
    "Ns": 81,
    "Nt": 41,
    "N_domain": 2048,
    "test_seed": 42,
    "batch_size": 128,
    "analytic_eps": 1e-5,
}

ALLOWED_FAMILIES = ["MLP", "ResNet", "QNN", "QPINN", "CQNN", "CQNN_nonlinear", "HQNN"]

# ============================================================
# Helpers
# ============================================================

def sanitize_for_json(obj: Any) -> Any:
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, tc.Tensor):
        return obj.detach().cpu().tolist()
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


def activation_from_config(name: Any) -> nn.Module:
    if isinstance(name, nn.Module):
        return name

    if name is None:
        return nn.Tanh()

    key = str(name).strip().lower()
    key = key.replace("()", "")
    key = key.replace("torch.nn.modules.activation.", "")

    if key in {"none", "identity"}:
        return nn.Identity()

    mapping = {
        "tanh": nn.Tanh,
        "sigmoid": nn.Sigmoid,
        "relu": nn.ReLU,
        "gelu": nn.GELU,
        "softplus": nn.Softplus,
        "leakyrelu": nn.LeakyReLU,
        "elu": nn.ELU,
        "selu": nn.SELU,
        "silu": nn.SiLU,
    }

    if key not in mapping:
        print(f"[WARN] activation '{name}' não reconhecida. Usando Tanh().")
        return nn.Tanh()

    return mapping[key]()


def model_file(run_dir: Path) -> Optional[Path]:
    for p in [run_dir / "model" / "model_state_dict.pth", run_dir / "model" / "model.pth"]:
        if p.exists():
            return p
    return None


def discover_all_runs(results_dir: Path, families: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    """
    Encontra automaticamente todas as runs em:
        experimentos_pinn/runs/<MODEL_TYPE>/<RUN_ID>/metadata/config.json
    """
    families = families or ALLOWED_FAMILIES
    runs_root = Path(results_dir) / "runs"

    targets: List[Dict[str, Any]] = []
    if not runs_root.exists():
        return targets

    for family_dir in sorted(runs_root.iterdir()):
        if not family_dir.is_dir():
            continue

        family = family_dir.name
        if family not in families:
            continue

        for run_dir in sorted(family_dir.iterdir()):
            cfg_path = run_dir / "metadata" / "config.json"
            if not cfg_path.exists():
                continue
            if model_file(run_dir) is None:
                continue

            targets.append({
                "model_type": family,
                "run_id": run_dir.name,
                "run_dir": str(run_dir),
                "config_path": str(cfg_path),
                "model_path": str(model_file(run_dir)),
            })

    return targets


def read_run_metadata(run_dir: Path) -> Dict[str, Any]:
    meta = {}
    for name in ["config.json", "results.json"]:
        p = run_dir / "metadata" / name
        if p.exists():
            try:
                meta.update(load_json(p))
            except Exception:
                pass
    return meta

# ============================================================
# Reconstrução dos modelos
# ============================================================

def build_classical_block_from_config(cfg: Dict[str, Any], device: str = DEFAULT_DEVICE):
    model_class = cfg.get("model_class", cfg.get("model_type", "MLP"))
    activation = activation_from_config(cfg.get("activation", "Tanh"))

    if model_class == "MLP":
        return MLP(
            hidden=int(cfg["hidden"]),
            blocks=int(cfg["blocks"]),
            activation=activation,
            device=device,
        )

    if model_class == "ResNet":
        return ResNet(
            hidden=int(cfg["hidden"]),
            blocks=int(cfg["blocks"]),
            activation=activation,
            device=device,
        )

    raise ValueError(f"Bloco clássico '{model_class}' não suportado.")


def build_model_from_config(cfg: Dict[str, Any], device: str = DEFAULT_DEVICE):
    model_type = cfg["model_type"]

    if model_type in {"MLP", "ResNet"}:
        return build_classical_block_from_config(cfg, device=device)

    if model_type in {"QNN", "QPINN"}:
        qnn = QuantumNeuralNetwork(
            n_qubits=int(cfg["n_qubits"]),
            n_layers=int(cfg["n_layers"]),
            device=device,
            circuit_type=cfg.get("entangler", "basic"),
        )
        return HybridCQN(classical_pre=None, qnn_block=qnn, classical_post=None, device=device)

    if model_type == "HQNN":
        qnn = QuantumNeuralNetwork(
            n_qubits=int(cfg["n_qubits"]),
            n_layers=int(cfg["n_layers"]),
            device=device,
            circuit_type=cfg.get("entangler", "basic"),
        )
        classical_pre = build_classical_block_from_config(cfg, device=device)
        return HybridCQN(classical_pre=classical_pre, qnn_block=qnn, classical_post=None, device=device)

    if model_type in {"CQNN", "CQNN_nonlinear"}:
        qnn = CorrelatorQuantumNeuralNetwork(
            n_qubits=int(cfg["n_qubits"]),
            n_layers=int(cfg["n_layers"]),
            k=int(cfg["k"]),
            n_vertex=int(cfg["n_vertex"]),
            nonlinear=bool(cfg.get("nonlinear", model_type == "CQNN_nonlinear")),
            device=device,
            circuit_type=cfg.get("entangler", "basic"),
        )
        classical_pre = None
        if cfg.get("use_classical_pre", False):
            classical_pre = build_classical_block_from_config(cfg, device=device)
        return HybridCQN(classical_pre=classical_pre, qnn_block=qnn, classical_post=None, device=device)

    raise ValueError(f"model_type '{model_type}' não suportado.")


def load_model(run_dir: Path, device: str = DEFAULT_DEVICE):
    cfg_path = run_dir / "metadata" / "config.json"
    cfg = load_json(cfg_path)
    model = build_model_from_config(cfg, device=device)
    state_path = model_file(run_dir)
    if state_path is None:
        raise FileNotFoundError(f"Modelo não encontrado em {run_dir / 'model'}")
    state = tc.load(state_path, map_location=pick_torch_device(device))
    model.load_state_dict(state)
    model.eval()
    return model, cfg, state_path

# ============================================================
# Black-Scholes analítico: preço + gregas
# ============================================================

def bs_price_delta_gamma_theta(
    S: np.ndarray,
    t: np.ndarray,
    *,
    S_max: float,
    T: float,
    K: float,
    r: float,
    sigma: float,
    option_type: str = "call",
    eps: float = 1e-10,
) -> Dict[str, np.ndarray]:
    S = np.asarray(S, dtype=float).reshape(-1, 1)
    t = np.asarray(t, dtype=float).reshape(-1, 1)

    tau = np.maximum(T - t, 0.0)
    S_safe = np.maximum(S, eps)
    tau_safe = np.maximum(tau, eps)
    sigma_safe = max(float(sigma), eps)

    sqrt_tau = np.sqrt(tau_safe)
    d1 = (np.log(S_safe / K) + (r + 0.5 * sigma_safe**2) * tau_safe) / (sigma_safe * sqrt_tau)
    d2 = d1 - sigma_safe * sqrt_tau

    pdf_d1 = norm.pdf(d1)

    if option_type == "call":
        V = S_safe * norm.cdf(d1) - K * np.exp(-r * tau_safe) * norm.cdf(d2)
        delta = norm.cdf(d1)
        theta = -(S_safe * pdf_d1 * sigma_safe) / (2.0 * sqrt_tau) - r * K * np.exp(-r * tau_safe) * norm.cdf(d2)
        payoff = np.maximum(S - K, 0.0)
    elif option_type == "put":
        V = K * np.exp(-r * tau_safe) * norm.cdf(-d2) - S_safe * norm.cdf(-d1)
        delta = norm.cdf(d1) - 1.0
        theta = -(S_safe * pdf_d1 * sigma_safe) / (2.0 * sqrt_tau) + r * K * np.exp(-r * tau_safe) * norm.cdf(-d2)
        payoff = np.maximum(K - S, 0.0)
    else:
        raise ValueError("option_type must be 'call' or 'put'")

    gamma = pdf_d1 / (S_safe * sigma_safe * sqrt_tau)

    V = np.where(tau <= eps, payoff, V)
    gamma = np.where(tau <= eps, np.nan, gamma)
    theta = np.where(tau <= eps, np.nan, theta)

    return {
        "V": V.reshape(-1, 1),
        "delta": delta.reshape(-1, 1),
        "gamma": gamma.reshape(-1, 1),
        "theta": theta.reshape(-1, 1),
    }


def normalize_bs_outputs(true_un: Dict[str, np.ndarray], *, S_max: float, T: float, V_max: float) -> Dict[str, np.ndarray]:
    return {
        "V": true_un["V"] / V_max,
        "delta": true_un["delta"] * (S_max / V_max),
        "gamma": true_un["gamma"] * (S_max**2 / V_max),
        "theta": true_un["theta"] * (T / V_max),
    }

# ============================================================
# Predição + autodiff das gregas
# ============================================================

def predict_price_and_greeks_autodiff(
    model: nn.Module,
    S: np.ndarray,
    t: np.ndarray,
    *,
    S_max: float,
    T: float,
    V_max: float,
    batch_size: int,
    device: str,
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    torch_device = pick_torch_device(device)
    S = np.asarray(S, dtype=np.float32).reshape(-1, 1)
    t = np.asarray(t, dtype=np.float32).reshape(-1, 1)

    pred_norm = {"V": [], "delta": [], "gamma": [], "theta": []}
    pred_un = {"V": [], "delta": [], "gamma": [], "theta": []}

    model.eval()
    for start in range(0, len(S), batch_size):
        end = min(start + batch_size, len(S))

        S_b = tc.tensor(S[start:end] / S_max, device=torch_device, dtype=DEFAULT_DTYPE, requires_grad=True)
        t_b = tc.tensor(t[start:end] / T, device=torch_device, dtype=DEFAULT_DTYPE, requires_grad=True)
        x_b = tc.cat([S_b, t_b], dim=1)

        V_b = model(x_b).reshape(-1, 1)
        ones = tc.ones_like(V_b)

        delta_b = tc.autograd.grad(V_b, S_b, grad_outputs=ones, create_graph=True, retain_graph=True)[0]
        theta_b = tc.autograd.grad(V_b, t_b, grad_outputs=ones, create_graph=True, retain_graph=True)[0]
        gamma_b = tc.autograd.grad(delta_b, S_b, grad_outputs=tc.ones_like(delta_b), create_graph=False, retain_graph=False)[0]

        pred_norm["V"].append(V_b.detach().cpu().numpy())
        pred_norm["delta"].append(delta_b.detach().cpu().numpy())
        pred_norm["gamma"].append(gamma_b.detach().cpu().numpy())
        pred_norm["theta"].append(theta_b.detach().cpu().numpy())

        pred_un["V"].append((V_b * V_max).detach().cpu().numpy())
        pred_un["delta"].append((delta_b * (V_max / S_max)).detach().cpu().numpy())
        pred_un["gamma"].append((gamma_b * (V_max / (S_max**2))).detach().cpu().numpy())
        pred_un["theta"].append((theta_b * (V_max / T)).detach().cpu().numpy())

    pred_norm = {k: np.vstack(v) for k, v in pred_norm.items()}
    pred_un = {k: np.vstack(v) for k, v in pred_un.items()}
    return pred_norm, pred_un

# ============================================================
# Pontos de avaliação
# ============================================================

def make_eval_points(bs_params: Dict[str, Any], eval_params: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, str]:
    S_max = float(bs_params["S_max"])
    T = float(bs_params["T"])
    eps = float(eval_params.get("analytic_eps", 1e-5))

    if eval_params["eval_mode"] == "grid":
        Ns = int(eval_params["Ns"])
        Nt = int(eval_params["Nt"])
        S_lin = np.linspace(eps * S_max, S_max, Ns)
        t_lin = np.linspace(0.0, T * (1.0 - eps), Nt)
        S_mesh, t_mesh = np.meshgrid(S_lin, t_lin, indexing="ij")
        return S_mesh.reshape(-1, 1), t_mesh.reshape(-1, 1), "grid"

    if eval_params["eval_mode"] == "random_domain":
        rng = np.random.default_rng(int(eval_params["test_seed"]))
        n = int(eval_params["N_domain"])
        S = rng.uniform(eps * S_max, S_max, size=(n, 1))
        t = rng.uniform(0.0, T * (1.0 - eps), size=(n, 1))
        return S, t, "random_domain"

    raise ValueError("eval_mode precisa ser 'grid' ou 'random_domain'.")

# ============================================================
# Métricas
# ============================================================

def metric_dict(pred: np.ndarray, true: np.ndarray) -> Dict[str, float]:
    pred = np.asarray(pred, dtype=float).reshape(-1)
    true = np.asarray(true, dtype=float).reshape(-1)
    mask = np.isfinite(pred) & np.isfinite(true)
    if mask.sum() == 0:
        return {"N": 0, "MSE": np.nan, "RMSE": np.nan, "MAE": np.nan, "Bias": np.nan, "Corr": np.nan}

    p = pred[mask]
    y = true[mask]
    err = p - y
    corr = np.nan
    if len(p) > 1 and np.std(p) > 0 and np.std(y) > 0:
        corr = float(np.corrcoef(y, p)[0, 1])

    return {
        "N": int(len(p)),
        "MSE": float(np.mean(err**2)),
        "RMSE": float(np.sqrt(np.mean(err**2))),
        "MAE": float(np.mean(np.abs(err))),
        "Bias": float(np.mean(err)),
        "Corr": corr,
    }


def add_regions(df: pd.DataFrame, *, K: float, T: float) -> pd.DataFrame:
    out = df.copy()
    out["moneyness"] = out["S"] / K
    out["tau"] = T - out["t"]

    out["region_moneyness"] = pd.cut(
        out["moneyness"],
        bins=[-np.inf, 0.97, 1.03, np.inf],
        labels=["OTM", "ATM", "ITM"],
    )

    out["region_maturity"] = pd.cut(
        out["tau"],
        bins=[-np.inf, 0.10 * T, 0.50 * T, np.inf],
        labels=["Near maturity", "Mid maturity", "Far maturity"],
    )

    out["region_hard"] = np.where(
        (out["region_moneyness"].astype(str) == "ATM") & (out["tau"] <= 0.10 * T),
        "ATM + near maturity",
        "Other",
    )
    return out


def build_comparison_dataframe(
    S: np.ndarray,
    t: np.ndarray,
    pred_norm: Dict[str, np.ndarray],
    pred_un: Dict[str, np.ndarray],
    true_norm: Dict[str, np.ndarray],
    true_un: Dict[str, np.ndarray],
    *,
    K: float,
    T: float,
) -> pd.DataFrame:
    df = pd.DataFrame({"S": S.reshape(-1), "t": t.reshape(-1)})

    for key in ["V", "delta", "gamma", "theta"]:
        df[f"{key}_pred_norm"] = pred_norm[key].reshape(-1)
        df[f"{key}_true_norm"] = true_norm[key].reshape(-1)
        df[f"{key}_pred_un"] = pred_un[key].reshape(-1)
        df[f"{key}_true_un"] = true_un[key].reshape(-1)
        df[f"{key}_err_un"] = df[f"{key}_pred_un"] - df[f"{key}_true_un"]
        df[f"{key}_abs_err_un"] = np.abs(df[f"{key}_err_un"])

    return add_regions(df, K=K, T=T)


def compute_all_metrics(df: pd.DataFrame) -> Dict[str, Any]:
    out: Dict[str, Any] = {}

    for key in ["V", "delta", "gamma", "theta"]:
        out[f"global_{key}"] = metric_dict(df[f"{key}_pred_un"], df[f"{key}_true_un"])

    for region_col in ["region_moneyness", "region_maturity", "region_hard"]:
        out[region_col] = {}
        for region_name, g in df.groupby(region_col, dropna=True):
            out[region_col][str(region_name)] = {}
            for key in ["V", "delta", "gamma", "theta"]:
                out[region_col][str(region_name)][key] = metric_dict(g[f"{key}_pred_un"], g[f"{key}_true_un"])

    return out


def flatten_metrics(metrics: Dict[str, Any]) -> Dict[str, Any]:
    row: Dict[str, Any] = {}

    for key in ["V", "delta", "gamma", "theta"]:
        m = metrics.get(f"global_{key}", {})
        for stat, value in m.items():
            row[f"{stat}_{key}_global"] = value

    for region_col, prefix in [
        ("region_moneyness", "moneyness"),
        ("region_maturity", "maturity"),
        ("region_hard", "hard"),
    ]:
        if region_col not in metrics:
            continue
        for region_name, block in metrics[region_col].items():
            safe_region = str(region_name).lower().replace(" ", "_").replace("+", "plus")
            for key, m in block.items():
                for stat, value in m.items():
                    row[f"{stat}_{key}_{prefix}_{safe_region}"] = value

    return row

# ============================================================
# Processamento
# ============================================================

def save_run_outputs(
    run_dir: Path,
    *,
    cfg: Dict[str, Any],
    state_path: Path,
    bs_params: Dict[str, Any],
    eval_params: Dict[str, Any],
    metrics: Dict[str, Any],
    df: pd.DataFrame,
) -> Path:
    greeks_dir = run_dir.parents[2] / "greeks" / run_dir.parent.name / run_dir.name
    greeks_dir.mkdir(parents=True, exist_ok=True)

    csv_path = greeks_dir / "greeks_comparison.csv"
    json_path = greeks_dir / "greeks_results.json"
    metrics_path = greeks_dir / "greeks_metrics_flat.json"

    df.to_csv(csv_path, index=False)
    save_json(flatten_metrics(metrics), metrics_path)

    payload = {
        "run_id": run_dir.name,
        "model_type": run_dir.parent.name,
        "run_dir": str(run_dir),
        "model_state_path": str(state_path),
        "output_csv": str(csv_path),
        "saved_config": cfg,
        "black_scholes_params": bs_params,
        "evaluation_params": eval_params,
        "metrics": metrics,
        "metrics_flat": flatten_metrics(metrics),
    }
    save_json(payload, json_path)
    return greeks_dir


def process_run(run_dir: Path, *, results_dir: Path, device: str, bs_params: Dict[str, Any], eval_params: Dict[str, Any], force: bool = False) -> Dict[str, Any]:
    existing_csv = Path(results_dir) / "greeks" / run_dir.parent.name / run_dir.name / "greeks_comparison.csv"
    existing_json = Path(results_dir) / "greeks" / run_dir.parent.name / run_dir.name / "greeks_results.json"

    if existing_csv.exists() and existing_json.exists() and not force:
        payload = load_json(existing_json)
        row = {
            "run_id": run_dir.name,
            "model_type": run_dir.parent.name,
            "greeks_dir": str(existing_csv.parent),
            **read_run_metadata(run_dir),
            **payload.get("metrics_flat", {}),
        }
        row["status"] = "cached"
        return row

    model, cfg, state_path = load_model(run_dir, device=device)
    S, t, eval_mode_used = make_eval_points(bs_params, eval_params)

    pred_norm, pred_un = predict_price_and_greeks_autodiff(
        model,
        S,
        t,
        S_max=float(bs_params["S_max"]),
        T=float(bs_params["T"]),
        V_max=float(bs_params["V_max"]),
        batch_size=int(eval_params["batch_size"]),
        device=device,
    )

    true_un = bs_price_delta_gamma_theta(
        S,
        t,
        S_max=float(bs_params["S_max"]),
        T=float(bs_params["T"]),
        K=float(bs_params["K"]),
        r=float(bs_params["r"]),
        sigma=float(bs_params["sigma"]),
        option_type=str(bs_params["option_type"]),
        eps=float(eval_params.get("analytic_eps", bs_params["eps"])),
    )
    true_norm = normalize_bs_outputs(
        true_un,
        S_max=float(bs_params["S_max"]),
        T=float(bs_params["T"]),
        V_max=float(bs_params["V_max"]),
    )

    df = build_comparison_dataframe(
        S,
        t,
        pred_norm,
        pred_un,
        true_norm,
        true_un,
        K=float(bs_params["K"]),
        T=float(bs_params["T"]),
    )

    eval_params_used = dict(eval_params)
    eval_params_used["eval_mode"] = eval_mode_used
    metrics = compute_all_metrics(df)

    greeks_dir = save_run_outputs(
        run_dir,
        cfg=cfg,
        state_path=state_path,
        bs_params=bs_params,
        eval_params=eval_params_used,
        metrics=metrics,
        df=df,
    )

    row = {
        "run_id": run_dir.name,
        "model_type": run_dir.parent.name,
        "greeks_dir": str(greeks_dir),
        **read_run_metadata(run_dir),
        **flatten_metrics(metrics),
    }
    row["status"] = "computed"
    return row


def run_batch(
    targets: List[Dict[str, Any]],
    *,
    results_dir: Path = DEFAULT_RESULTS_DIR,
    device: str = DEFAULT_DEVICE,
    bs_params: Dict[str, Any] = BS_PARAMS,
    eval_params: Dict[str, Any] = EVAL_PARAMS,
    force: bool = False,
) -> pd.DataFrame:
    if len(targets) == 0:
        raise ValueError("Nenhuma run encontrada para calcular Greeks.")

    rows = []
    failed = []

    for i, target in enumerate(targets, start=1):
        run_dir = Path(target["run_dir"])
        print(f"\n[{i}/{len(targets)}] {run_dir.parent.name}/{run_dir.name}")

        try:
            row = process_run(
                run_dir,
                results_dir=Path(results_dir),
                device=device,
                bs_params=dict(bs_params),
                eval_params=dict(eval_params),
                force=force,
            )
            rows.append(row)
            print(
                f"  -> {row['status']} | "
                f"RMSE V={row.get('RMSE_V_global', np.nan):.6g} | "
                f"Delta={row.get('RMSE_delta_global', np.nan):.6g} | "
                f"Gamma={row.get('RMSE_gamma_global', np.nan):.6g} | "
                f"Theta={row.get('RMSE_theta_global', np.nan):.6g}"
            )
        except Exception as e:
            print(f"  -> FALHOU: {e}")
            failed.append({
                "model_type": run_dir.parent.name,
                "run_id": run_dir.name,
                "run_dir": str(run_dir),
                "error": str(e),
            })

    out_dir = Path(results_dir) / "greeks"
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = pd.DataFrame(rows)
    failed_df = pd.DataFrame(failed)

    summary_path = out_dir / "greeks_summary.csv"
    failed_path = out_dir / "greeks_failed_runs.csv"

    summary.to_csv(summary_path, index=False)
    failed_df.to_csv(failed_path, index=False)

    # Resumo por família, estilo Trahan: erro vs parâmetros por método.
    if not summary.empty and "model_type" in summary.columns:
        agg_cols = {}
        for col in ["RMSE_V_global", "RMSE_delta_global", "RMSE_gamma_global", "RMSE_theta_global", "num_params"]:
            if col in summary.columns:
                agg_cols[col] = ["count", "min", "median", "mean", "std"] if col.startswith("RMSE") else ["median"]
        if agg_cols:
            family_summary = summary.groupby("model_type").agg(agg_cols)
            family_summary.to_csv(out_dir / "greeks_family_summary.csv")

    print("\n" + "=" * 72)
    print("CÁLCULO DE GREEKS FINALIZADO")
    print("=" * 72)
    print("Runs encontradas:", len(targets))
    print("Runs com sucesso:", len(summary))
    print("Runs com falha:", len(failed_df))
    print("Resumo:", summary_path)
    print("Falhas:", failed_path)
    print("=" * 72)

    return summary

# ============================================================
# CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Calcula V, Delta, Gamma e Theta para todas as runs salvas.")
    parser.add_argument("--results-dir", type=str, default=str(DEFAULT_RESULTS_DIR))
    parser.add_argument("--device", type=str, default=DEFAULT_DEVICE)
    parser.add_argument("--eval-mode", type=str, default=EVAL_PARAMS["eval_mode"], choices=["grid", "random_domain"])
    parser.add_argument("--Ns", type=int, default=EVAL_PARAMS["Ns"])
    parser.add_argument("--Nt", type=int, default=EVAL_PARAMS["Nt"])
    parser.add_argument("--N-domain", type=int, default=EVAL_PARAMS["N_domain"])
    parser.add_argument("--batch-size", type=int, default=EVAL_PARAMS["batch_size"])
    parser.add_argument("--families", type=str, default=",".join(ALLOWED_FAMILIES), help="Ex: MLP,ResNet,QNN,CQNN,HQNN")
    parser.add_argument("--force", action="store_true", help="Recalcula mesmo se já existir greeks_comparison.csv")

    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    families = [x.strip() for x in args.families.split(",") if x.strip()]

    eval_params = dict(EVAL_PARAMS)
    eval_params.update({
        "eval_mode": args.eval_mode,
        "Ns": args.Ns,
        "Nt": args.Nt,
        "N_domain": args.N_domain,
        "batch_size": args.batch_size,
    })

    targets = discover_all_runs(results_dir, families=families)

    print(f"[OK] Runs descobertas: {len(targets)}")
    for t in targets[:10]:
        print("  ", t["model_type"], t["run_id"])
    if len(targets) > 10:
        print(f"  ... e mais {len(targets) - 10}")

    run_batch(
        targets,
        results_dir=results_dir,
        device=args.device,
        bs_params=dict(BS_PARAMS),
        eval_params=eval_params,
        force=args.force,
    )


if __name__ == "__main__":
    main()
