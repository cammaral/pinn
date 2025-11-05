import numpy as np
import pandas as pd
import torch as tc
import tensorflow as tf
import yfinance as yf
from pathlib import Path
from scipy.stats import norm
import json

# ======================================================
# 📈 Black-Scholes Formula
# ======================================================

def black_scholes_call_price(S, t, T, K, r, sigma):
    """
    Computes the Black-Scholes call price.

    Parameters:
        S (array): Spot prices.
        t (array): Current times.
        T (float): Maturity time.
        K (float): Strike price.
        r (float): Risk-free rate.
        sigma (float): Volatility.

    Returns:
        array: Call option prices.
    """
    S = np.array(S)
    t = np.array(t)
    tau = T - t
    tau[tau == 0] = 1e-10  # avoid division by zero

    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * tau) / (sigma * np.sqrt(tau))
    d2 = d1 - sigma * np.sqrt(tau)

    call_price = S * norm.cdf(d1) - K * np.exp(-r * tau) * norm.cdf(d2)
    call_price[S <= 0] = 0.0  # boundary condition
    return call_price

# ======================================================
# 🛠️ Data Generation (Synthetic)
# ======================================================

def generate_data(r=0.05, K=40, T=1.0, sigma=0.2, S_max=160, N_domain=100, N_boundary=100, N_terminal=100, seed=98):
    """
    Generates synthetic data to train PINNs for Black-Scholes PDE.

    Returns:
        dict: domain, terminal, and boundary condition samples.
    """
    np.random.seed(seed)

    # Domain samples
    S_domain = np.random.uniform(0, S_max, (int(N_domain), 1))
    t_domain = np.random.uniform(0, T, (int(N_domain), 1))
    V_domain = black_scholes_call_price(S_domain, t_domain, T, K, r, sigma)
    # Terminal condition (payoff)
    S_terminal = np.random.uniform(0, S_max, (N_terminal, 1))
    t_terminal = T * np.ones((N_terminal, 1))
    V_terminal = np.maximum(S_terminal - K, 0)

    # Boundary conditions
    S_boundary_0 = np.zeros((N_boundary // 2, 1))
    t_boundary_0 = np.random.uniform(0, T, (N_boundary // 2, 1))
    V_boundary_0 = np.zeros_like(S_boundary_0)

    S_boundary_max = S_max * np.ones((N_boundary // 2, 1))
    t_boundary_max = np.random.uniform(0, T, (N_boundary // 2, 1))
    V_boundary_max = S_max - K * np.exp(-r * (T - t_boundary_max))

    return {
        'domain': (S_domain, t_domain, V_domain),
        'terminal': (S_terminal, t_terminal, V_terminal),
        'boundary_0': (S_boundary_0, t_boundary_0, V_boundary_0),
        'boundary_max': (S_boundary_max, t_boundary_max, V_boundary_max),
    }
"""
def convert_to_tensor(data, requires_grad=True, dtype="float32", backend="torch"):

    Converts numpy data to PyTorch or TensorFlow tensor.

    Parameters:
        data (array): Input data (NumPy array or convertible).
        requires_grad (bool): If gradients are needed (only used for PyTorch).
        dtype (str or torch/tf dtype): Desired data type.
        backend (str): "torch" or "tf"

    Returns:
        torch.Tensor or tf.Tensor

    data = np.array(data)

    if backend == "torch":
        tensor_dtype = getattr(tc, dtype) if isinstance(dtype, str) else dtype
        return tc.tensor(data, dtype=tensor_dtype, requires_grad=requires_grad).reshape(-1, 1)
    
    elif backend == "tf":
        tensor_dtype = dtype if isinstance(dtype, tf.DType) else getattr(tf, dtype)
        return tf.convert_to_tensor(data.reshape(-1, 1), dtype=tensor_dtype)

    else:
        raise ValueError("Backend must be either 'torch' or 'tf'")
"""

def convert_to_tensor(data, requires_grad=True, dtype="float32", backend="torch", device=None):
    """
    Converts numpy data to PyTorch or TensorFlow tensor.

    Parameters:
        data (array): Input data (NumPy array or convertible).
        requires_grad (bool): If gradients are needed (only used for PyTorch).
        dtype (str or torch/tf dtype): Desired data type.
        backend (str): "torch" or "tf"
        device (str or torch.device or None): Target device for PyTorch tensors (e.g., "cuda", "cpu", "mps").

    Returns:
        torch.Tensor or tf.Tensor
    """
    if backend == "torch":
        # dtype
        tensor_dtype = getattr(tc, dtype) if isinstance(dtype, str) else dtype

        # Se já for tensor Torch, só ajusta dtype/device/grad sem copiar à toa
        if isinstance(data, tc.Tensor):
            t = data.to(dtype=tensor_dtype)
            if device is not None:
                t = t.to(device)
            t.requires_grad_(bool(requires_grad))
            return t.reshape(-1, 1)

        # Caso geral: cria tensor a partir de array
        data = np.array(data)
        return tc.tensor(
            data,
            dtype=tensor_dtype,
            device=device,
            requires_grad=requires_grad
        ).reshape(-1, 1)

    elif backend == "tf":
        data = np.array(data)
        tensor_dtype = dtype if isinstance(dtype, tf.DType) else getattr(tf, dtype)
        return tf.convert_to_tensor(data.reshape(-1, 1), dtype=tensor_dtype)

    else:
        raise ValueError("Backend must be either 'torch' or 'tf'")
# ======================================================
# 💾 Data Saving and Loading (Synthetic)
# ======================================================

def save_data_sim(data_sim, r, K, T, S_max, N_domain, N_boundary, N_terminal, seed):
    """
    Saves generated synthetic data into JSON file.
    """
    root_path = Path(__file__).resolve().parent
    data_path = root_path / "../data/data_sim"
    data_path.mkdir(exist_ok=True)

    filename = f"data_sim_r{r}_K{K}_T{T}_Smax{S_max}_Nd{N_domain}_Nb{N_boundary}_Nt{N_terminal}_seed{seed}.json"
    filepath = data_path / filename

    data_serializable = {key: [v.tolist() for v in value] for key, value in data_sim.items()}

    with open(filepath, "w") as f:
        json.dump(data_serializable, f, indent=4)

def load_data_sim(r, K, T, S_max, N_domain, N_boundary, N_terminal, seed):
    """
    Loads synthetic data from JSON file.

    Returns:
        dict: Loaded data organized as domain, terminal, boundary.
    """
    filename = f"data_sim_r{r}_K{K}_T{T}_Smax{S_max}_Nd{N_domain}_Nb{N_boundary}_Nt{N_terminal}_seed{seed}.json"
    base_path = Path(__file__).resolve().parent
    full_path = base_path.parent / "data" / "data_sim" / filename

    if not full_path.exists():
        raise FileNotFoundError(f"[✗] File not found: {full_path}")

    with open(full_path, "r") as f:
        data = json.load(f)

    return {
        'domain': tuple(np.array(arr) for arr in data['domain']),
        'terminal': tuple(np.array(arr) for arr in data['terminal']),
        'boundary_0': tuple(np.array(arr) for arr in data['boundary_0']),
        'boundary_max': tuple(np.array(arr) for arr in data['boundary_max']),
    }

# ======================================================
# 📈 Real Market Data (Yahoo Finance + Option Quotes)
# ======================================================

def get_data(market):
    """
    Loads real market index prices and options from CSVs.

    Parameters:
        market (str): 'NDX' or 'SPXC'

    Returns:
        tuple: (historical prices, option quotes)
    """
    data_dir = Path().resolve() / "data"

    if market == 'NDX':
        opt = pd.read_csv(data_dir / "df_eur_call_NDX_jun2021.csv")
    elif market == 'SPXC':
        opt = pd.read_csv(data_dir / "df_amer_put_SPXC_feb2021.csv")
    else:
        print("=================\n Wrong Market \n=================")
        return None, None

    start = opt['date'].min()
    end = opt['date'].max()
    print(start, end)

    data = yf.download(market, start=start, end=end)
    data.columns = data.columns.droplevel(1) if isinstance(data.columns, pd.MultiIndex) else data.columns

    return data, opt

def adjust_data(market, data, opt):
    """
    Adjusts and merges market and option data.

    Parameters:
        market (str): Market ticker.
        data (DataFrame): Price history.
        opt (DataFrame): Option quotes.

    Returns:
        DataFrame: Final dataset for modeling.
    """
    opt["date"] = pd.to_datetime(opt["date"])
    merged_data = opt.merge(data.reset_index(), left_on="date", right_on="Date", suffixes=("", f"_{market}"))

    option_data = merged_data[[
        "Close", "current_time", "strike_price", "best_bid",
        "best_offer", "ticker", "impl_volatility"
    ]].copy()

    option_data["strike_price"] = option_data["strike_price"] / 1000
    option_data["market_price"] = (option_data["best_bid"] + option_data["best_offer"]) / 2
    option_data = option_data.rename(columns={"Close": "spot_price"})

    return option_data
