import warnings
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# imports from your project (do not change other files)
from equation.option_pricing import BlackScholes
from optimize.option_princing import BlackScholeOptimizer
from method.nn import MLP, ResNet
from method.hnn import HybridCQN
from method.qnn import QuantumNeuralNetwork, CorrelatorQuantumNeuralNetwork
from utils.adapters import ResNetFeatures, MLPFeatures
import torch.nn as nn
import torch as tc


def _unwrap_train_return(ret, model_ref):
    """Return (trained_model, loss_dict). BlackScholeOptimizer.train may return either
    a loss dict or (model, loss_dict). If only loss dict, assume model_ref was trained in-place."""
    if isinstance(ret, tuple) and len(ret) >= 2:
        return ret[0] or model_ref, ret[1]
    else:
        return model_ref, ret


def _num_params(model):
    return int(sum(p.numel() for p in model.parameters()))


def _final_avg_loss_from_history(loss_obj):
    """Try to extract 'Total' loss history or use directly if list/array."""
    hist = None
    if isinstance(loss_obj, dict):
        # common pattern in notebook: loss['Total']
        for key in ("Total", "total", "loss", "Loss"):
            if key in loss_obj:
                hist = loss_obj[key]
                break
    if hist is None:
        # maybe train returned a plain list/array
        hist = loss_obj
    hist = np.array(hist)
    if hist.ndim == 0:
        return float(hist)
    last_k = min(100, hist.shape[0])
    return float(np.mean(hist[-last_k:]))


def _get_analytic_fn(bse):
    # try several possible method names for analytic Black-Scholes price
    candidates = [
        "analytic", "price", "bs_price", "black_scholes", "exact", "exact_price",
        "option_price", "call_price", "solution"
    ]
    for name in candidates:
        if hasattr(bse, name):
            fn = getattr(bse, name)
            if callable(fn):
                return fn
    # fallback: try attribute 'price_grid' or 'V' or 'solution' as arrays
    raise AttributeError("Could not find analytic price method on BlackScholes instance. "
                         "Expected one of common names (analytic, price, bs_price, ...).")


def _build_eval_grid_from_bse(bse, n_s=80, n_t=80):
    """Try to build a (S, T) mesh based on bse attributes or generated data."""
    # try common attributes
    Smin, Smax = None, None
    Tmin, Tmax = None, None
    # common attr names
    for attr in ("S_min", "Smin", "s_min", "Smin_", "S0",):
        if hasattr(bse, attr):
            v = getattr(bse, attr)
            try:
                Smin = float(v)
            except Exception:
                pass
    for attr in ("S_max", "Smax", "s_max", "Smax_"):
        if hasattr(bse, attr):
            v = getattr(bse, attr)
            try:
                Smax = float(v)
            except Exception:
                pass
    for attr in ("T", "Tmax", "T_max", "time_max"):
        if hasattr(bse, attr):
            try:
                Tmax = float(getattr(bse, attr))
            except Exception:
                pass

    # fallback: use generate_data() to infer ranges
    try:
        data = bse.generate_data()
        # Try to detect arrays in returned data
        if isinstance(data, dict):
            # common keys might be 'X' with shape [N,2] where columns are (S,t)
            if "X" in data:
                X = np.array(data["X"])
                Scol = X[:, 0]
                Tcol = X[:, 1]
                Smin, Smax = np.min(Scol), np.max(Scol)
                Tmin, Tmax = np.min(Tcol), np.max(Tcol)
            elif "s" in data and "t" in data:
                Scol = np.array(data["s"]).ravel()
                Tcol = np.array(data["t"]).ravel()
                Smin, Smax = np.min(Scol), np.max(Scol)
                Tmin, Tmax = np.min(Tcol), np.max(Tcol)
        elif isinstance(data, (list, tuple, np.ndarray)):
            # try tuple (X, y)
            if len(data) >= 1:
                X = np.array(data[0])
                if X.ndim == 2 and X.shape[1] >= 2:
                    Scol = X[:, 0]
                    Tcol = X[:, 1]
                    Smin, Smax = np.min(Scol), np.max(Scol)
                    Tmin, Tmax = np.min(Tcol), np.max(Tcol)
    except Exception:
        pass

    # sensible defaults
    if Smin is None or Smax is None:
        Smin, Smax = 0.0, 2.0  # normalized stock range; user may adjust
    if Tmin is None or Tmax is None:
        Tmin, Tmax = 0.0, 1.0  # time to maturity range

    S = np.linspace(Smin, Smax, n_s)
    T = np.linspace(Tmin, Tmax, n_t)
    Sg, Tg = np.meshgrid(S, T, indexing="xy")
    pts = np.column_stack([Sg.ravel(), Tg.ravel()])  # shape (n_s*n_t, 2)
    return Sg, Tg, pts


def _model_predict_on_grid(model, pts):
    """Return model predictions for pts (N,2). Works for classical and hybrid models.
    Assumes model accepts torch tensors shape [N,2] and returns [N,1] or [N,k]."""
    model.eval()
    with tc.no_grad():
        x = tc.tensor(pts, dtype=tc.float32)
        out = model(x)
        out = out.detach().cpu().numpy()
        # if output has more than 1 channel, reduce by mean across last dim
        if out.ndim == 2 and out.shape[1] > 1:
            out = out.mean(axis=1, keepdims=True)
        return out.ravel()


def grid_search_and_plot(
    *,
    epochs=200,
    lr=1e-2,
    n_runs_per_config=1,
    classical_grid=None,
    quantum_grid=None,
    hybrid_with_features=True,
    plot_show=True
):
    """
    Performs a simple grid search over small sets of architectures for classical and hybrid models,
    finds the best model by final average loss (mean of last 100 epochs), then evaluates the best
    classical and best hybrid models over a 2D grid and plots:
      - 3D surface of model prediction
      - 3D surface of analytic Black-Scholes
      - 3D surface of absolute difference (model - analytic)

    Only this file is modified; other code is used as-is.
    """

    bse = BlackScholes(eps=1e-10)
    data = bse.generate_data()
    analytic_fn = None
    try:
        analytic_fn = _get_analytic_fn(bse)
    except Exception as e:
        warnings.warn(str(e) + " -- analytic surface plotting may fail.")

    # default grids if not provided
    if classical_grid is None:
        classical_grid = [
            {"type": "MLP", "neurons": 4, "M": 3},
            {"type": "MLP", "neurons": 8, "M": 4},
            {"type": "ResNet", "hidden": 4, "blocks": 2},
        ]
    if quantum_grid is None:
        quantum_grid = [
            {"n_qubits": 3, "n_layers": 2, "n_vertex": 9},
            {"n_qubits": 4, "n_layers": 2, "n_vertex": 9},
        ]

    results = {"classical": [], "hybrid": []}

    # TRAIN classical models
    for cfg in classical_grid:
        for run in range(n_runs_per_config):
            if cfg["type"] == "MLP":
                model = MLP(neurons=cfg.get("neurons", 4), M=cfg.get("M", 3), activation=nn.Tanh())
            else:
                model = ResNet(hidden=cfg.get("hidden", 4), blocks=cfg.get("blocks", 2), activation=nn.Tanh())

            opt = BlackScholeOptimizer(data, model, epochs=epochs, lr=lr)
            ret = opt.train(return_loss=True)
            trained_model, loss_obj = _unwrap_train_return(ret, model)
            final_loss = _final_avg_loss_from_history(loss_obj)
            results["classical"].append({
                "cfg": cfg,
                "model": trained_model,
                "loss_obj": loss_obj,
                "final_loss": final_loss,
                "num_params": _num_params(trained_model),
            })

    # TRAIN hybrid models (classical pre + correlator qnn)
    for qcfg in quantum_grid:
        for c_cfg in classical_grid:
            for run in range(n_runs_per_config):
                # classical preprocessor can be MLPFeatures or ResNetFeatures depending on classical cfg
                if c_cfg["type"] == "MLP":
                    classical_pre = MLP(neurons=c_cfg.get("neurons", 4), M=c_cfg.get("M", 1), activation=nn.Tanh())
                    classical_adapter = MLPFeatures(classical_pre)
                else:
                    classical_pre = ResNet(hidden=c_cfg.get("hidden", 4), blocks=c_cfg.get("blocks", 1), activation=nn.Tanh())
                    classical_adapter = ResNetFeatures(classical_pre)

                # correlator quantum block
                cqnn = CorrelatorQuantumNeuralNetwork(
                    n_qubits=qcfg.get("n_qubits", 4),
                    n_layers=qcfg.get("n_layers", 2),
                    k=qcfg.get("k", 2),
                    n_vertex=qcfg.get("n_vertex", 9),
                    nonlinear=True
                )

                hybrid = HybridCQN(classical_pre=classical_adapter, qnn_block=cqnn, classical_post=None)
                opt = BlackScholeOptimizer(data, hybrid, epochs=epochs, lr=lr)
                ret = opt.train(return_loss=True)
                trained_model, loss_obj = _unwrap_train_return(ret, hybrid)
                final_loss = _final_avg_loss_from_history(loss_obj)
                results["hybrid"].append({
                    "cfg_classical": c_cfg,
                    "cfg_quantum": qcfg,
                    "model": trained_model,
                    "loss_obj": loss_obj,
                    "final_loss": final_loss,
                    "num_params": _num_params(trained_model),
                })

    # pick best classical and best hybrid by final_loss
    best_classical = min(results["classical"], key=lambda r: r["final_loss"]) if results["classical"] else None
    best_hybrid = min(results["hybrid"], key=lambda r: r["final_loss"]) if results["hybrid"] else None

    print("Best classical:", best_classical["cfg"] if best_classical else None, 
          "loss:", best_classical["final_loss"] if best_classical else None,
          "params:", best_classical["num_params"] if best_classical else None)
    print("Best hybrid:", 
          {"classical": best_hybrid.get("cfg_classical"), "quantum": best_hybrid.get("cfg_quantum")} if best_hybrid else None,
          "loss:", best_hybrid["final_loss"] if best_hybrid else None,
          "params:", best_hybrid["num_params"] if best_hybrid else None)

    # Build evaluation grid
    Sg, Tg, pts = _build_eval_grid_from_bse(bse, n_s=80, n_t=80)

    # analytic values
    analytic_vals = None
    if analytic_fn is not None:
        try:
            # try vectorized call
            analytic_vals = analytic_fn(pts)
            analytic_vals = np.array(analytic_vals).ravel()
        except Exception:
            # try calling with S and T grids separately
            try:
                analytic_vals = analytic_fn(Sg, Tg).ravel()
            except Exception:
                analytic_vals = None
                warnings.warn("Could not evaluate analytic_fn on grid. Skipping analytic surface.")

    # evaluate best models on grid
    if best_classical:
        pred_classical = _model_predict_on_grid(best_classical["model"], pts)
    else:
        pred_classical = None
    if best_hybrid:
        pred_hybrid = _model_predict_on_grid(best_hybrid["model"], pts)
    else:
        pred_hybrid = None

    # reshape to grid
    def _to_grid(arr):
        return None if arr is None else arr.reshape(Sg.shape)

    Z_analytic = _to_grid(analytic_vals)
    Z_classical = _to_grid(pred_classical)
    Z_hybrid = _to_grid(pred_hybrid)

    # plotting
    fig = plt.figure(figsize=(18, 12))

    # analytic
    ax1 = fig.add_subplot(2, 3, 1, projection='3d')
    if Z_analytic is not None:
        ax1.plot_surface(Sg, Tg, Z_analytic, cmap='viridis', edgecolor='none')
        ax1.set_title("Black-Scholes analytic")
    else:
        ax1.text(0.5, 0.5, 0.5, "analytic unavailable", transform=ax1.transAxes)

    # classical
    ax2 = fig.add_subplot(2, 3, 2, projection='3d')
    if Z_classical is not None:
        ax2.plot_surface(Sg, Tg, Z_classical, cmap='viridis', edgecolor='none')
        ax2.set_title("Best classical model")
    else:
        ax2.text(0.5, 0.5, 0.5, "no classical result", transform=ax2.transAxes)

    # hybrid
    ax3 = fig.add_subplot(2, 3, 3, projection='3d')
    if Z_hybrid is not None:
        ax3.plot_surface(Sg, Tg, Z_hybrid, cmap='viridis', edgecolor='none')
        ax3.set_title("Best hybrid model")
    else:
        ax3.text(0.5, 0.5, 0.5, "no hybrid result", transform=ax3.transAxes)

    # differences: classical - analytic
    ax4 = fig.add_subplot(2, 3, 4, projection='3d')
    if Z_analytic is not None and Z_classical is not None:
        D = np.abs(Z_classical - Z_analytic)
        ax4.plot_surface(Sg, Tg, D, cmap='inferno', edgecolor='none')
        ax4.set_title("Abs diff: classical vs analytic")
    else:
        ax4.text(0.5, 0.5, 0.5, "insufficient data", transform=ax4.transAxes)

    # differences: hybrid - analytic
    ax5 = fig.add_subplot(2, 3, 5, projection='3d')
    if Z_analytic is not None and Z_hybrid is not None:
        D = np.abs(Z_hybrid - Z_analytic)
        ax5.plot_surface(Sg, Tg, D, cmap='inferno', edgecolor='none')
        ax5.set_title("Abs diff: hybrid vs analytic")
    else:
        ax5.text(0.5, 0.5, 0.5, "insufficient data", transform=ax5.transAxes)

    # summary scatter: best loss vs num params
    ax6 = fig.add_subplot(2, 3, 6)
    xs = []
    ys = []
    labels = []
    if best_classical:
        xs.append(best_classical["num_params"])
        ys.append(best_classical["final_loss"])
        labels.append("best_classical")
    if best_hybrid:
        xs.append(best_hybrid["num_params"])
        ys.append(best_hybrid["final_loss"])
        labels.append("best_hybrid")
    if xs:
        ax6.scatter(xs, ys, s=80)
        for x, y, lab in zip(xs, ys, labels):
            ax6.annotate(lab, (x, y), textcoords="offset points", xytext=(5,5))
        ax6.set_xscale("log")
        ax6.set_yscale("log")
        ax6.set_xlabel("num params (log)")
        ax6.set_ylabel("final loss (log)")
        ax6.set_title("Best final loss vs #params")
        ax6.grid(True, which='both', ls='--', lw=0.4)
    else:
        ax6.text(0.5, 0.5, "no models", transform=ax6.transAxes)

    plt.tight_layout()
    if plot_show:
        plt.show()

    return {"best_classical": best_classical, "best_hybrid": best_hybrid, "grids": (Sg, Tg), "Z": (Z_analytic, Z_classical, Z_hybrid)}


if __name__ == "__main__":
    # small default run; adjust epochs and grids if you want heavier search
    out = grid_search_and_plot(epochs=200, lr=2e-2, n_runs_per_config=1)
    # results returned contain best models and grids for further inspection