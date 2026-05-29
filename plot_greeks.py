from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
from itertools import product

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch as tc
import torch.nn as nn
from scipy.stats import norm, pearsonr, spearmanr, ttest_1samp, wilcoxon, mannwhitneyu, ks_2samp

# =============================================================================
# AJUSTE ESTE CAMINHO PARA A RAIZ DO PROJETO, SE NECESSÁRIO.
# O script tenta importar as classes originais do projeto e tem fallback mínimo.
# =============================================================================
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# =============================================================================
# IMPORTS DO PROJETO (com fallback mínimo, caso necessário)
# =============================================================================

def pick_torch_device(device: str = "auto") -> tc.device:
    if device == "auto":
        return tc.device("cuda" if tc.cuda.is_available() else "cpu")
    return tc.device(device)


def pick_pl_backend(device: str = "auto") -> str:
    return "default.qubit"


def pick_diff_method(_backend: str) -> str:
    return "best"


try:
    from equation.option_pricing import BlackScholes  # type: ignore
    from method.nn import MLP, ResNet  # type: ignore
    from method.hnn import HybridCQN  # type: ignore
    from method.qnn import QuantumNeuralNetwork, CorrelatorQuantumNeuralNetwork  # type: ignore
except Exception:
    import pennylane as qml
    from itertools import combinations
    from pennylane.qnn import TorchLayer

    class BlackScholes:
        def __init__(self, S_max=160, T=1.0, K=40, r=0.05, sigma=0.2, eps=1e-10):
            self.S_max = S_max
            self.T = T
            self.K = K
            self.r = r
            self.sigma = sigma
            self.eps = eps

        def V(self, S, t, option_type='call'):
            S = np.array(S)
            t = np.array(t)
            tau = np.array(self.T - t)

            if np.all(tau <= self.eps):
                if option_type == 'call':
                    return np.maximum(S - self.K, 0.0)
                elif option_type == 'put':
                    return np.maximum(self.K - S, 0.0)
                raise ValueError("option_type must be 'call' or 'put'")

            tau = np.maximum(tau, 0.0)
            S_safe = np.maximum(S, self.eps)
            sqrt_tau = np.sqrt(np.maximum(tau, self.eps))
            d1 = (np.log(S_safe / self.K) + (self.r + 0.5 * self.sigma**2) * tau) / (self.sigma * sqrt_tau)
            d2 = d1 - self.sigma * sqrt_tau
            discK = self.K * np.exp(-self.r * tau)

            if option_type == 'call':
                price = S_safe * norm.cdf(d1) - discK * norm.cdf(d2)
                price = np.where(S <= self.eps, 0.0, price)
                price = np.where(tau <= self.eps, np.maximum(S - self.K, 0.0), price)
                return price
            elif option_type == 'put':
                price = discK * norm.cdf(-d2) - S_safe * norm.cdf(-d1)
                price = np.where(S <= self.eps, discK, price)
                price = np.where(tau <= self.eps, np.maximum(self.K - S, 0.0), price)
                return price
            raise ValueError("option_type must be 'call' or 'put'")

    class MLP(nn.Module):
        def __init__(self, hidden, blocks, activation=nn.Tanh(), output=1, device: str = "auto", dtype=tc.float32):
            super().__init__()
            self.hidden_layers = nn.ModuleList([nn.Linear(2, hidden)])
            self.hidden_layers.extend([nn.Linear(hidden, hidden) for _ in range(blocks - 1)])
            self.output_layer = nn.Linear(hidden, output)
            self.activation = activation
            self._torch_device = pick_torch_device(device)
            self._dtype = dtype
            self.to(self._torch_device, dtype=self._dtype)

        def forward(self, x):
            x = x.to(self._torch_device, dtype=self._dtype)
            for layer in self.hidden_layers:
                x = self.activation(layer(x))
            return self.output_layer(x)

    class ResidualBlock(nn.Module):
        def __init__(self, dim, activation=nn.Tanh(), dropout=0.0):
            super().__init__()
            self.lin1 = nn.Linear(dim, dim)
            self.lin2 = nn.Linear(dim, dim)
            self.act = activation
            self.do = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        def forward(self, x):
            z = self.act(self.lin1(x))
            z = self.do(z)
            z = self.lin2(z)
            return self.act(z + x)

    class ResNet(nn.Module):
        def __init__(self, hidden=64, blocks=4, activation=nn.Tanh(), output=1, device: str = "auto", dtype=tc.float32):
            super().__init__()
            self.inp = nn.Linear(2, hidden)
            self.act = activation
            self.blocks = nn.ModuleList([ResidualBlock(hidden, activation=self.act) for _ in range(blocks)])
            self.out = nn.Linear(hidden, output)
            self._torch_device = pick_torch_device(device)
            self._dtype = dtype
            self.to(self._torch_device, dtype=self._dtype)

        def forward(self, x):
            x = x.to(self._torch_device, dtype=self._dtype)
            h = self.act(self.inp(x))
            for b in self.blocks:
                h = b(h)
            return self.out(h)

    def make_ansatz(circuit_type="basic"):
        if circuit_type == "basic":
            def ansatz(weights, n_qubits):
                qml.BasicEntanglerLayers(weights, wires=range(n_qubits))
            return ansatz
        if circuit_type == "strong":
            def ansatz(weights, n_qubits):
                qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
            return ansatz
        raise ValueError(f"circuit_type desconhecido: {circuit_type}")

    def get_weight_shapes(circuit_type, n_layers, n_qubits):
        if circuit_type == "basic":
            return {"weights": (n_layers, n_qubits)}
        if circuit_type == "strong":
            return {"weights": (n_layers, n_qubits, 3)}
        raise ValueError(f"circuit_type desconhecido: {circuit_type}")

    class QuantumNeuralNetwork(nn.Module):
        def __init__(self, n_qubits=4, n_layers=2, output_dim=1, ansatz_fn=None, circuit_type="basic",
                     device: str = "auto", diff_method=None, dtype=tc.float32):
            super().__init__()
            self.n_qubits = n_qubits
            self.n_layers = n_layers
            self.n_output = n_qubits
            self.output_dim = output_dim
            self._torch_device = pick_torch_device(device)
            self._dtype = dtype
            self.pl_backend = pick_pl_backend(device)
            self.diff_method = diff_method if diff_method is not None else pick_diff_method(self.pl_backend)
            self.circuit_type = circuit_type
            self.ansatz_fn = ansatz_fn if ansatz_fn is not None else make_ansatz(circuit_type)

            dev = qml.device(self.pl_backend, wires=n_qubits)

            @qml.qnode(dev, interface="torch", diff_method=self.diff_method)
            def circuit(inputs, weights):
                qml.AngleEmbedding(inputs, wires=range(n_qubits))
                self.ansatz_fn(weights, n_qubits)
                return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

            self.q_layer = TorchLayer(circuit, get_weight_shapes(self.circuit_type, self.n_layers, self.n_qubits))
            self.to(self._torch_device, dtype=self._dtype)

        def forward(self, x):
            x = tc.pi * tc.tanh(x)
            return self.q_layer(x)

    def generate_pauli_obs_list(n_qubits, k, n_vertex):
        pauli_list = []
        positions = list(range(n_qubits))
        for pauli_char in ["Z", "Y", "X"]:
            for combo in combinations(positions, k):
                pauli_str = ["I"] * n_qubits
                for idx in combo:
                    pauli_str[idx] = pauli_char
                obs = None
                for idx, p in enumerate(pauli_str):
                    if p == "I":
                        continue
                    current = {"X": qml.PauliX(idx), "Y": qml.PauliY(idx), "Z": qml.PauliZ(idx)}[p]
                    obs = current if obs is None else obs @ current
                pauli_list.append(obs if obs is not None else qml.Identity(0))
                if len(pauli_list) == n_vertex:
                    return pauli_list
        return pauli_list

    class CorrelatorQuantumNeuralNetwork(nn.Module):
        def __init__(self, n_qubits=4, n_layers=2, k=2, n_vertex=9, nonlinear=None, ansatz_fn=None,
                     circuit_type="basic", device: str = "auto", diff_method=None, dtype=tc.float32):
            super().__init__()
            self.n_qubits = n_qubits
            self.n_layers = n_layers
            self.k = k
            self.n_vertex = n_vertex
            self.n_output = n_vertex
            self.obs_list = generate_pauli_obs_list(n_qubits, k, n_vertex)
            self.alpha = 1.5 * self.n_qubits
            self.nonlinear = nn.Tanh() if nonlinear else None
            self._torch_device = pick_torch_device(device)
            self._dtype = dtype
            self.pl_backend = pick_pl_backend(device)
            self.diff_method = diff_method if diff_method is not None else pick_diff_method(self.pl_backend)
            self.circuit_type = circuit_type
            self.ansatz_fn = ansatz_fn if ansatz_fn is not None else make_ansatz(circuit_type)

            dev = qml.device(self.pl_backend, wires=n_qubits)

            @qml.qnode(dev, interface="torch", diff_method=self.diff_method)
            def circuit(inputs, weights):
                qml.AngleEmbedding(inputs, wires=range(n_qubits))
                self.ansatz_fn(weights, n_qubits)
                return [qml.expval(obs) for obs in self.obs_list]

            self.q_layer = TorchLayer(circuit, get_weight_shapes(self.circuit_type, self.n_layers, self.n_qubits))
            self.to(self._torch_device, dtype=self._dtype)

        def forward(self, x):
            x = tc.pi * tc.tanh(x)
            x = self.q_layer(x)
            if self.nonlinear:
                x = self.nonlinear(self.alpha * x)
            return x

    class HybridCQN(nn.Module):
        def __init__(self, classical_pre: Optional[nn.Module], qnn_block: nn.Module, classical_post: Optional[nn.Module] = None,
                     input_dim: int = 2, output_dim: int = 1, post_in_dim: Optional[int] = None,
                     device: str = "auto", dtype: tc.dtype = tc.float32):
            super().__init__()
            self.pre = classical_pre
            self.qnn = qnn_block
            self.post = classical_post
            self.n_qubits = qnn_block.n_qubits
            self.n_output = qnn_block.n_output
            self.input_dim = input_dim
            self._torch_device = pick_torch_device(device)
            self._dtype = dtype
            F = self._infer_out_dim(self.pre, input_dim)
            self.to_qubits = nn.Linear(F, self.n_qubits, bias=True)
            if self.post is None:
                self.q_out = nn.Linear(self.n_output, output_dim, bias=True)
            else:
                Dpost = post_in_dim or self._infer_first_linear_in(self.post) or input_dim
                self.decode_to_post = nn.Linear(self.n_qubits, Dpost, bias=True)
            self.to(self._torch_device, dtype=self._dtype)

        @tc.no_grad()
        def _infer_out_dim(self, module: Optional[nn.Module], in_dim: int) -> int:
            if module is None:
                return in_dim
            y = module(tc.zeros(1, in_dim, device=self._torch_device, dtype=self._dtype))
            return y.view(1, -1).shape[-1]

        def _infer_first_linear_in(self, module: nn.Module):
            for m in module.modules():
                if isinstance(m, nn.Linear):
                    return m.in_features
            return None

        def forward(self, x):
            x = x.to(self._torch_device, dtype=self._dtype)
            h = x if self.pre is None else self.pre(x)
            q_in = self.to_qubits(h)
            q_out = self.qnn(q_in)
            if self.post is None:
                return self.q_out(q_out)
            r_in = self.decode_to_post(q_out)
            return self.post(r_in)


# =============================================================================
# CONFIGURAÇÃO PADRÃO
# =============================================================================
DEFAULT_RESULTS_DIR = Path("experimentos_pinn")
DEFAULT_OUTPUT_DIRNAME = "greeks_grid_analysis"
DEFAULT_DEVICE = "cpu"
DEFAULT_DTYPE = tc.float32
CONTROL_KEYS = {"model_type", "run_id", "run_id_prefix", "group_name", "label"}
METRIC_KEYS = ["V", "delta", "gamma", "theta"]
REGION_ORDER = [
    "global",
    "near_strike",
    "away_strike",
    "near_maturity",
    "away_maturity",
    "problem_region",
    "outside_problem_region",
]

# -----------------------------------------------------------------------------
# Parâmetros do problema analítico / desnormalização.
# IMPORTANTE: no seu projeto, o treino/teste usa por padrão S/160, t/T e V/120.
# Se isso tiver sido mudado nas runs reais, ajuste aqui explicitamente.
# -----------------------------------------------------------------------------
BS_PARAMS = {
    "S_max": 160.0,
    "T": 1.0,
    "K": 40.0,
    "r": 0.05,
    "sigma": 0.2,
    "V_max": 120.0,
    "option_type": "call",
    "eps": 1e-10,
}

# -----------------------------------------------------------------------------
# Avaliação em GRID estruturado. Tudo salvo e analisado em escala desnormalizada.
# tau_min evita singularidade exatamente no vencimento para Gamma/Theta.
# -----------------------------------------------------------------------------
GRID_PARAMS = {
    "Ns": 1000,
    "Nt": 1000,
    "S_min": 0.0,
    "tau_min": 1e-4,
    "batch_size": 4096,
}

# -----------------------------------------------------------------------------
# Regiões. Por padrão, a banda problemática perto do strike usa porcentagem de K.
# Ex.: strike_band_frac = 0.10 -> região |S-K| <= 0.10*K.
# near_maturity_frac = 0.10 -> últimos 10% da maturidade.
# -----------------------------------------------------------------------------
REGION_PARAMS = {
    "strike_band_frac": 0.25,
    "near_maturity_frac": 0.25,
    "relative_error_eps": 1e-10,
}

# -----------------------------------------------------------------------------
# Geração de grids de experimentos.
# Você pode montar suas runs como no exemplo abaixo e depois passar para
# TARGET_GROUPS. O restante do script continua igual.
# -----------------------------------------------------------------------------
def generate_runs(base_cfg: Dict[str, Any], sweep_cfg: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
    """
    Expande um base_cfg por produto cartesiano dos campos em sweep_cfg.

    Exemplo:
        base = {
            "model_type": "QNN",
            "run_id_prefix": "qnn_strong",
            "lr": 2e-3,
            "epochs": 10000,
            "activation": None,
            "entangler": "strong",
        }
        sweep = {
            "n_qubits": [2, 3, 4],
            "n_layers": [1, 2],
            "seed": [1924, 1925],
        }
        runs = generate_runs(base, sweep)
    """
    if not isinstance(base_cfg, dict):
        raise TypeError("base_cfg deve ser dict.")
    if not isinstance(sweep_cfg, dict):
        raise TypeError("sweep_cfg deve ser dict[str, list].")

    if len(sweep_cfg) == 0:
        return [dict(base_cfg)]

    keys = list(sweep_cfg.keys())
    values_product = []
    for k in keys:
        vals = sweep_cfg[k]
        if not isinstance(vals, (list, tuple)) or len(vals) == 0:
            raise ValueError(f"sweep_cfg['{k}'] deve ser uma lista/tupla não vazia.")
        values_product.append(list(vals))

    runs: List[Dict[str, Any]] = []
    for combo in product(*values_product):
        item = dict(base_cfg)
        item.update({k: v for k, v in zip(keys, combo)})
        runs.append(item)
    return runs


# -----------------------------------------------------------------------------
# Monte seus grupos aqui. Cada grupo sai em uma pasta própria.
# A forma mais prática é criar um experiment_grid com generate_runs(...).
# -----------------------------------------------------------------------------
experiment_grid: List[Dict[str, Any]] = []

base_seed_test = {
    "model_type": "QNN",
    "run_id_prefix": "qnn_strong",
    "lr": 2e-3,
    "epochs": 10000,
    "activation": None,
    "entangler": "strong",
}

sweep_seed = {
    "n_qubits": [2, 3, 4],
    "n_layers": [1, 2, 3],
    "seed": [1924, 1925, 1926, 1973, 2025, 2024, 2012, 1958, 1962, 1997],
}

# Descomente para usar:
experiment_grid.extend(generate_runs(base_seed_test, sweep_seed))

TARGET_GROUPS: Dict[str, List[Dict[str, Any]]] = {
    "seed_test_qnn": experiment_grid,

    # Você também pode declarar outros grupos:
    # "misto": [
    #     {"model_type": "MLP", "run_id": "mlp_h_4_b_2_ep_2500_lr_0.001_seed_1924"},
    #     {"model_type": "QNN", "run_id": "qnn_strong_n_qubits_3_n_l_2_lr_0.001_seed_1924"},
    # ],
}


# =============================================================================
# UTILIDADES BÁSICAS
# =============================================================================

def sanitize_for_json(obj: Any) -> Any:
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {str(k): sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [sanitize_for_json(v) for v in obj]
    return str(obj)


def save_json(data: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(sanitize_for_json(data), f, indent=2, ensure_ascii=False)


def load_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def set_global_seed(seed: int = 1234) -> None:
    np.random.seed(seed)
    tc.manual_seed(seed)
    if tc.cuda.is_available():
        tc.cuda.manual_seed_all(seed)


def values_match(a: Any, b: Any) -> bool:
    if isinstance(a, float) or isinstance(b, float):
        try:
            return math.isclose(float(a), float(b), rel_tol=1e-12, abs_tol=1e-12)
        except Exception:
            return False
    return a == b


def safe_slug(s: str) -> str:
    out = []
    for ch in str(s):
        if ch.isalnum() or ch in {"-", "_"}:
            out.append(ch)
        else:
            out.append("_")
    return "".join(out)


def safe_filename(s: str) -> str:
    return safe_slug(s)[:180]


def activation_from_config(name: Any) -> nn.Module:
    if isinstance(name, nn.Module):
        return name
    if name is None or str(name).lower() in {"none", "identity"}:
        return nn.Identity()
    key = str(name).lower()
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
        raise ValueError(f"Activation '{name}' não suportada para reconstrução.")
    return mapping[key]()


def short_run_label(run_id: str, max_len: int = 48) -> str:
    if len(run_id) <= max_len:
        return run_id
    return run_id[: max_len - 3] + "..."


# =============================================================================
# LOCALIZAÇÃO DAS RUNS
# =============================================================================

def iter_saved_configs(results_dir: Path, model_type: Optional[str] = None) -> Iterable[Tuple[Path, Dict[str, Any]]]:
    runs_root = results_dir / "runs"
    if not runs_root.exists():
        return

    model_roots = [runs_root / model_type] if model_type else [p for p in runs_root.iterdir() if p.is_dir()]
    for model_root in model_roots:
        if not model_root.exists():
            continue
        for run_dir in model_root.iterdir():
            cfg_path = run_dir / "metadata" / "config.json"
            if cfg_path.exists():
                yield run_dir, load_json(cfg_path)


def find_run_dir(results_dir: Path, target: Dict[str, Any]) -> Path:
    model_type = target.get("model_type")
    if not model_type:
        raise ValueError("Cada target precisa de 'model_type' ou 'run_id'.")

    run_id = target.get("run_id")
    if run_id:
        run_dir = results_dir / "runs" / model_type / run_id
        if not run_dir.exists():
            raise FileNotFoundError(f"Run não encontrada: {run_dir}")
        return run_dir

    run_id_prefix = target.get("run_id_prefix")
    matches: List[Path] = []
    target_items = {k: v for k, v in target.items() if k not in CONTROL_KEYS}
    for run_dir, cfg in iter_saved_configs(results_dir, model_type=model_type):
        if run_id_prefix and not str(run_dir.name).startswith(str(run_id_prefix)):
            continue
        ok = True
        for k, v in target_items.items():
            if k not in cfg or not values_match(cfg[k], v):
                ok = False
                break
        if ok:
            matches.append(run_dir)

    if len(matches) == 0:
        raise FileNotFoundError(
            f"Nenhuma run encontrada para os parâmetros: {target_items} "
            f"com run_id_prefix={run_id_prefix}. Confira o results_dir e os campos do config salvo."
        )
    if len(matches) > 1:
        msg = "\n".join(str(p) for p in matches[:10])
        raise RuntimeError(
            f"Mais de uma run encontrada para {target_items} com run_id_prefix={run_id_prefix}. "
            f"Seja mais específico ou informe run_id.\n{msg}"
        )
    return matches[0]


def find_model_file(run_dir: Path) -> Path:
    candidates = [
        run_dir / "model" / "model_state_dict.pth",
        run_dir / "model" / "model.pth",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f"Nenhum arquivo de modelo encontrado em {run_dir / 'model'}")


# =============================================================================
# RECONSTRUÇÃO DO MODELO
# =============================================================================

def build_classical_block_from_config(cfg: Dict[str, Any], device: str = DEFAULT_DEVICE):
    model_class = cfg.get("model_class", cfg.get("model_type"))
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

    if model_type == "MLP":
        return build_classical_block_from_config(cfg, device=device)

    if model_type == "ResNet":
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
        classical = build_classical_block_from_config(cfg, device=device)
        return HybridCQN(classical_pre=classical, qnn_block=qnn, classical_post=None, device=device)

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
        return HybridCQN(classical_pre=None, qnn_block=qnn, classical_post=None, device=device)

    raise ValueError(f"model_type '{model_type}' não suportado para reconstrução.")


def load_model(run_dir: Path, device: str = DEFAULT_DEVICE):
    cfg = load_json(run_dir / "metadata" / "config.json")
    model = build_model_from_config(cfg, device=device)
    state_path = find_model_file(run_dir)
    state = tc.load(state_path, map_location=pick_torch_device(device))
    model.load_state_dict(state)
    model.eval()
    return model, cfg, state_path


# =============================================================================
# ASSINATURAS / GRUPOS DE MODELO
# =============================================================================

def model_family_from_cfg(cfg: Dict[str, Any]) -> str:
    return str(cfg.get("model_type", "UNKNOWN"))


def model_signature_from_cfg(cfg: Dict[str, Any]) -> str:
    model_type = str(cfg.get("model_type", "UNKNOWN"))
    if model_type in {"MLP", "ResNet"}:
        parts = [
            model_type,
            f"h={cfg.get('hidden')}",
            f"b={cfg.get('blocks')}",
            f"act={str(cfg.get('activation', ''))}",
            f"lr={cfg.get('lr')}",
            f"ep={cfg.get('epochs')}",
        ]
        return " | ".join(parts)
    if model_type in {"QNN", "QPINN"}:
        parts = [
            model_type,
            f"q={cfg.get('n_qubits')}",
            f"l={cfg.get('n_layers')}",
            f"ent={cfg.get('entangler', 'basic')}",
            f"lr={cfg.get('lr')}",
            f"ep={cfg.get('epochs')}",
        ]
        return " | ".join(parts)
    if model_type == "HQNN":
        parts = [
            model_type,
            f"pre={cfg.get('model_class', 'MLP')}",
            f"h={cfg.get('hidden')}",
            f"b={cfg.get('blocks')}",
            f"q={cfg.get('n_qubits')}",
            f"l={cfg.get('n_layers')}",
            f"ent={cfg.get('entangler', 'basic')}",
            f"lr={cfg.get('lr')}",
            f"ep={cfg.get('epochs')}",
        ]
        return " | ".join(parts)
    if model_type in {"CQNN", "CQNN_nonlinear"}:
        parts = [
            model_type,
            f"q={cfg.get('n_qubits')}",
            f"l={cfg.get('n_layers')}",
            f"k={cfg.get('k')}",
            f"v={cfg.get('n_vertex')}",
            f"ent={cfg.get('entangler', 'basic')}",
            f"lr={cfg.get('lr')}",
            f"ep={cfg.get('epochs')}",
        ]
        return " | ".join(parts)
    return model_type


# =============================================================================
# GRID DE AVALIAÇÃO E ANALÍTICO BS
# =============================================================================

def build_eval_grid(bs_params: Dict[str, Any], grid_params: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    S_min = float(grid_params.get("S_min", 0.0))
    S_max = float(bs_params["S_max"])
    T = float(bs_params["T"])
    tau_min = float(grid_params.get("tau_min", 1e-4))
    Ns = int(grid_params["Ns"])
    Nt = int(grid_params["Nt"])

    S_lin = np.linspace(S_min, S_max, Ns)
    t_lin = np.linspace(0.0, max(T - tau_min, 0.0), Nt)
    S_mesh, t_mesh = np.meshgrid(S_lin, t_lin, indexing="ij")
    tau_mesh = T - t_mesh
    return S_lin, t_lin, S_mesh.reshape(-1, 1), t_mesh.reshape(-1, 1)


def bs_price_delta_gamma_theta(
    S: np.ndarray,
    t: np.ndarray,
    *,
    S_max: float,
    T: float,
    K: float,
    r: float,
    sigma: float,
    option_type: str,
    eps: float,
) -> Dict[str, np.ndarray]:
    S = np.asarray(S, dtype=float).reshape(-1, 1)
    t = np.asarray(t, dtype=float).reshape(-1, 1)
    tau = np.maximum(T - t, 0.0)
    S_safe = np.maximum(S, eps)
    sqrt_tau = np.sqrt(np.maximum(tau, eps))

    d1 = (np.log(S_safe / K) + (r + 0.5 * sigma**2) * tau) / (sigma * sqrt_tau)
    d2 = d1 - sigma * sqrt_tau
    Nd1 = norm.cdf(d1)
    Nd2 = norm.cdf(d2)
    pdf_d1 = norm.pdf(d1)
    discK = K * np.exp(-r * tau)

    if option_type == "call":
        price = S_safe * Nd1 - discK * Nd2
        delta = Nd1
        # Aqui theta é interpretado como derivada em relação ao tempo t do modelo,
        # consistente com a PDE usada no treino e com a autodiff em relação à entrada t.
        theta = -(S_safe * pdf_d1 * sigma) / (2.0 * sqrt_tau) - r * discK * Nd2
        price = np.where(S <= eps, 0.0, price)
        delta = np.where(S <= eps, 0.0, delta)
        price = np.where(tau <= eps, np.maximum(S - K, 0.0), price)
        delta = np.where(tau <= eps, (S > K).astype(float), delta)
    elif option_type == "put":
        price = discK * norm.cdf(-d2) - S_safe * norm.cdf(-d1)
        delta = Nd1 - 1.0
        theta = -(S_safe * pdf_d1 * sigma) / (2.0 * sqrt_tau) + r * discK * norm.cdf(-d2)
        price = np.where(S <= eps, discK, price)
        delta = np.where(S <= eps, -1.0, delta)
        price = np.where(tau <= eps, np.maximum(K - S, 0.0), price)
        delta = np.where(tau <= eps, -(S < K).astype(float), delta)
    else:
        raise ValueError("option_type must be 'call' or 'put'")

    gamma = pdf_d1 / (S_safe * sigma * sqrt_tau)

    invalid_greeks = (tau <= eps) | (S <= eps)
    gamma = np.where(invalid_greeks, np.nan, gamma)
    theta = np.where(tau <= eps, np.nan, theta)

    return {"V": price, "delta": delta, "gamma": gamma, "theta": theta}


# =============================================================================
# PREDIÇÃO DO MODELO + AUTODIFF DAS GREGAS
# =============================================================================

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
) -> Dict[str, np.ndarray]:
    torch_device = pick_torch_device(device)
    S = np.asarray(S, dtype=np.float32).reshape(-1, 1)
    t = np.asarray(t, dtype=np.float32).reshape(-1, 1)

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

        pred_un["V"].append((V_b * V_max).detach().cpu().numpy())
        pred_un["delta"].append((delta_b * (V_max / S_max)).detach().cpu().numpy())
        pred_un["gamma"].append((gamma_b * (V_max / (S_max ** 2))).detach().cpu().numpy())
        pred_un["theta"].append((theta_b * (V_max / T)).detach().cpu().numpy())

    return {k: np.vstack(v) for k, v in pred_un.items()}


# =============================================================================
# REGIÕES / ERROS PONTO A PONTO
# =============================================================================

def build_region_masks(df: pd.DataFrame, bs_params: Dict[str, Any], region_params: Dict[str, Any]) -> Dict[str, np.ndarray]:
    K = float(bs_params["K"])
    T = float(bs_params["T"])
    strike_band = float(region_params["strike_band_frac"]) * K
    near_maturity_tau = float(region_params["near_maturity_frac"]) * T

    near_strike = np.abs(df["S"].to_numpy() - K) <= strike_band
    near_maturity = df["tau"].to_numpy() <= near_maturity_tau

    masks = {
        "global": np.ones(len(df), dtype=bool),
        "near_strike": near_strike,
        "away_strike": ~near_strike,
        "near_maturity": near_maturity,
        "away_maturity": ~near_maturity,
        "problem_region": near_strike & near_maturity,
        "outside_problem_region": ~(near_strike & near_maturity),
    }
    return masks


def add_error_columns(df: pd.DataFrame, metric_key: str, relative_error_eps: float) -> pd.DataFrame:
    pred = df[f"{metric_key}_pred"].to_numpy(dtype=float)
    true = df[f"{metric_key}_true"].to_numpy(dtype=float)

    err = pred - true
    abs_err = np.abs(err)
    sq_err = err ** 2
    denom = np.maximum(np.abs(true), relative_error_eps)
    rel_abs_err = abs_err / denom
    smape = 2.0 * abs_err / np.maximum(np.abs(pred) + np.abs(true), relative_error_eps)

    df[f"{metric_key}_err"] = err
    df[f"{metric_key}_abs_err"] = abs_err
    df[f"{metric_key}_sq_err"] = sq_err
    df[f"{metric_key}_rel_abs_err"] = rel_abs_err
    df[f"{metric_key}_smape"] = smape
    return df


def build_pointwise_dataframe(
    S: np.ndarray,
    t: np.ndarray,
    pred_un: Dict[str, np.ndarray],
    true_un: Dict[str, np.ndarray],
    bs_params: Dict[str, Any],
    region_params: Dict[str, Any],
) -> pd.DataFrame:
    df = pd.DataFrame({
        "S": S.reshape(-1),
        "t": t.reshape(-1),
    })
    df["tau"] = float(bs_params["T"]) - df["t"]
    df["abs_moneyness"] = np.abs(df["S"] - float(bs_params["K"]))
    df["rel_moneyness"] = df["abs_moneyness"] / max(float(bs_params["K"]), 1e-12)

    for key in METRIC_KEYS:
        df[f"{key}_pred"] = pred_un[key].reshape(-1)
        df[f"{key}_true"] = true_un[key].reshape(-1)
        df = add_error_columns(df, key, relative_error_eps=float(region_params["relative_error_eps"]))

    masks = build_region_masks(df, bs_params, region_params)
    for region_name, mask in masks.items():
        df[f"region_{region_name}"] = mask.astype(int)

    return df


# =============================================================================
# MÉTRICAS E TESTES ESTATÍSTICOS
# =============================================================================

def finite_xy(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=float).reshape(-1)
    y = np.asarray(y, dtype=float).reshape(-1)
    mask = np.isfinite(x) & np.isfinite(y)
    return x[mask], y[mask]


def finite_x(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float).reshape(-1)
    return x[np.isfinite(x)]


def safe_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true, y_pred = finite_xy(y_true, y_pred)
    if y_true.size < 2:
        return float("nan")
    sst = np.sum((y_true - np.mean(y_true)) ** 2)
    if sst <= 1e-20:
        return float("nan")
    sse = np.sum((y_true - y_pred) ** 2)
    return float(1.0 - sse / sst)


def sample_for_test(x: np.ndarray, max_n: int = 5000, seed: int = 1234) -> np.ndarray:
    x = finite_x(x)
    if x.size <= max_n:
        return x
    rng = np.random.default_rng(seed)
    idx = rng.choice(x.size, size=max_n, replace=False)
    return x[idx]


def safe_pearson(x: np.ndarray, y: np.ndarray) -> float:
    x, y = finite_xy(x, y)
    if x.size < 3:
        return float("nan")
    if np.allclose(x, x[0]) or np.allclose(y, y[0]):
        return float("nan")
    try:
        return float(pearsonr(x, y).statistic)
    except Exception:
        return float("nan")


def safe_spearman(x: np.ndarray, y: np.ndarray) -> float:
    x, y = finite_xy(x, y)
    if x.size < 3:
        return float("nan")
    if np.allclose(x, x[0]) or np.allclose(y, y[0]):
        return float("nan")
    try:
        return float(spearmanr(x, y).statistic)
    except Exception:
        return float("nan")


def safe_ttest_bias(err: np.ndarray) -> Tuple[float, float]:
    err = sample_for_test(err)
    if err.size < 3:
        return float("nan"), float("nan")
    try:
        res = ttest_1samp(err, popmean=0.0, nan_policy="omit")
        return float(res.statistic), float(res.pvalue)
    except Exception:
        return float("nan"), float("nan")


def safe_wilcoxon_bias(err: np.ndarray) -> Tuple[float, float]:
    err = sample_for_test(err)
    if err.size < 5:
        return float("nan"), float("nan")
    err = err[np.abs(err) > 1e-16]
    if err.size < 5:
        return float("nan"), float("nan")
    try:
        res = wilcoxon(err, zero_method="wilcox", alternative="two-sided")
        return float(res.statistic), float(res.pvalue)
    except Exception:
        return float("nan"), float("nan")


def cliffs_delta(x: np.ndarray, y: np.ndarray, max_n: int = 3000) -> float:
    x = sample_for_test(x, max_n=max_n, seed=123)
    y = sample_for_test(y, max_n=max_n, seed=456)
    if x.size == 0 or y.size == 0:
        return float("nan")
    gt = 0
    lt = 0
    for xi in x:
        gt += np.sum(xi > y)
        lt += np.sum(xi < y)
    return float((gt - lt) / (x.size * y.size))


def safe_region_distribution_tests(x: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    x = sample_for_test(x, max_n=4000, seed=11)
    y = sample_for_test(y, max_n=4000, seed=22)
    if x.size < 5 or y.size < 5:
        return {
            "mannwhitney_u": float("nan"),
            "mannwhitney_pvalue": float("nan"),
            "ks_stat": float("nan"),
            "ks_pvalue": float("nan"),
            "cliffs_delta": float("nan"),
        }
    out = {}
    try:
        res_u = mannwhitneyu(x, y, alternative="two-sided")
        out["mannwhitney_u"] = float(res_u.statistic)
        out["mannwhitney_pvalue"] = float(res_u.pvalue)
    except Exception:
        out["mannwhitney_u"] = float("nan")
        out["mannwhitney_pvalue"] = float("nan")
    try:
        res_ks = ks_2samp(x, y, alternative="two-sided", method="auto")
        out["ks_stat"] = float(res_ks.statistic)
        out["ks_pvalue"] = float(res_ks.pvalue)
    except Exception:
        out["ks_stat"] = float("nan")
        out["ks_pvalue"] = float("nan")
    out["cliffs_delta"] = cliffs_delta(x, y)
    return out


def bootstrap_ci(values: np.ndarray, func=np.median, n_boot: int = 2000, ci: float = 0.95, seed: int = 1234) -> Tuple[float, float]:
    values = finite_x(values)
    if values.size == 0:
        return float("nan"), float("nan")
    if values.size == 1:
        v = float(func(values))
        return v, v
    rng = np.random.default_rng(seed)
    boots = np.empty(n_boot, dtype=float)
    n = values.size
    for i in range(n_boot):
        sample = values[rng.integers(0, n, size=n)]
        boots[i] = float(func(sample))
    alpha = (1.0 - ci) / 2.0
    return float(np.quantile(boots, alpha)), float(np.quantile(boots, 1.0 - alpha))


def compute_point_metrics(df: pd.DataFrame, metric_key: str, region_name: str, region_mask: np.ndarray) -> Dict[str, Any]:
    work = df.loc[region_mask, [f"{metric_key}_pred", f"{metric_key}_true", f"{metric_key}_err", f"{metric_key}_abs_err", f"{metric_key}_sq_err", f"{metric_key}_rel_abs_err", f"{metric_key}_smape"]].copy()
    work = work.replace([np.inf, -np.inf], np.nan).dropna()

    n = int(len(work))
    if n == 0:
        return {
            "metric": metric_key,
            "region": region_name,
            "n_points": 0,
        }

    pred = work[f"{metric_key}_pred"].to_numpy()
    true = work[f"{metric_key}_true"].to_numpy()
    err = work[f"{metric_key}_err"].to_numpy()
    abs_err = work[f"{metric_key}_abs_err"].to_numpy()
    sq_err = work[f"{metric_key}_sq_err"].to_numpy()
    rel_abs_err = work[f"{metric_key}_rel_abs_err"].to_numpy()
    smape = work[f"{metric_key}_smape"].to_numpy()

    t_stat, t_p = safe_ttest_bias(err)
    w_stat, w_p = safe_wilcoxon_bias(err)

    sign_acc = float(np.mean(np.sign(pred) == np.sign(true))) if n > 0 else float("nan")

    out = {
        "metric": metric_key,
        "region": region_name,
        "n_points": n,
        "mse": float(np.mean(sq_err)),
        "rmse": float(np.sqrt(np.mean(sq_err))),
        "mae": float(np.mean(abs_err)),
        "medae": float(np.median(abs_err)),
        "bias": float(np.mean(err)),
        "mean_pred": float(np.mean(pred)),
        "mean_true": float(np.mean(true)),
        "std_signed_error": float(np.std(err, ddof=1)) if n > 1 else float("nan"),
        "q90_abs_err": float(np.quantile(abs_err, 0.90)),
        "q95_abs_err": float(np.quantile(abs_err, 0.95)),
        "q99_abs_err": float(np.quantile(abs_err, 0.99)),
        "max_abs_err": float(np.max(abs_err)),
        "mean_rel_abs_err": float(np.mean(rel_abs_err)),
        "median_rel_abs_err": float(np.median(rel_abs_err)),
        "mean_smape": float(np.mean(smape)),
        "median_smape": float(np.median(smape)),
        "pearson_r": safe_pearson(true, pred),
        "spearman_r": safe_spearman(true, pred),
        "r2": safe_r2(true, pred),
        "sign_accuracy": sign_acc,
        "bias_ttest_stat": t_stat,
        "bias_ttest_pvalue": t_p,
        "bias_wilcoxon_stat": w_stat,
        "bias_wilcoxon_pvalue": w_p,
    }
    return out


def compute_region_contrast(df: pd.DataFrame, metric_key: str, region_a: str, region_b: str) -> Dict[str, Any]:
    a = df.loc[df[f"region_{region_a}"] == 1, f"{metric_key}_abs_err"].replace([np.inf, -np.inf], np.nan).dropna().to_numpy()
    b = df.loc[df[f"region_{region_b}"] == 1, f"{metric_key}_abs_err"].replace([np.inf, -np.inf], np.nan).dropna().to_numpy()
    tests = safe_region_distribution_tests(a, b)

    rmse_a = float(np.sqrt(np.mean(a ** 2))) if a.size > 0 else float("nan")
    rmse_b = float(np.sqrt(np.mean(b ** 2))) if b.size > 0 else float("nan")
    med_a = float(np.median(a)) if a.size > 0 else float("nan")
    med_b = float(np.median(b)) if b.size > 0 else float("nan")

    return {
        "metric": metric_key,
        "region_a": region_a,
        "region_b": region_b,
        "n_a": int(a.size),
        "n_b": int(b.size),
        "rmse_a": rmse_a,
        "rmse_b": rmse_b,
        "rmse_ratio_a_over_b": float(rmse_a / rmse_b) if np.isfinite(rmse_a) and np.isfinite(rmse_b) and rmse_b > 0 else float("nan"),
        "median_abs_err_a": med_a,
        "median_abs_err_b": med_b,
        "median_ratio_a_over_b": float(med_a / med_b) if np.isfinite(med_a) and np.isfinite(med_b) and med_b > 0 else float("nan"),
        **tests,
    }


# =============================================================================
# PLOTS
# =============================================================================

def grouped_boxplot_with_points(
    df: pd.DataFrame,
    *,
    x_col: str,
    y_col: str,
    color_col: str,
    title: str,
    ylabel: str,
    out_path: Path,
    order_by_median: bool = True,
) -> None:
    plot_df = df[[x_col, y_col, color_col]].replace([np.inf, -np.inf], np.nan).dropna()
    if plot_df.empty:
        return

    if order_by_median:
        order = plot_df.groupby(x_col)[y_col].median().sort_values().index.tolist()
    else:
        order = sorted(plot_df[x_col].astype(str).unique().tolist())

    groups = [plot_df.loc[plot_df[x_col] == g, y_col].to_numpy() for g in order]
    families = plot_df[color_col].astype(str).unique().tolist()
    cmap = plt.cm.get_cmap("tab10", max(len(families), 1))
    family_to_color = {fam: cmap(i) for i, fam in enumerate(families)}

    fig_h = max(6.0, 0.42 * len(order))
    plt.figure(figsize=(14, fig_h))
    positions = np.arange(1, len(order) + 1)
    bp = plt.boxplot(groups, vert=False, positions=positions, patch_artist=True, showfliers=False)

    for box in bp["boxes"]:
        box.set(facecolor="white", alpha=0.95)

    rng = np.random.default_rng(1234)
    for i, group in enumerate(order, start=1):
        sub = plot_df.loc[plot_df[x_col] == group]
        jitter = rng.normal(loc=0.0, scale=0.06, size=len(sub))
        for fam, fam_sub in sub.groupby(color_col):
            idx = fam_sub.index
            plt.scatter(
                fam_sub[y_col],
                np.full(len(fam_sub), i, dtype=float) + jitter[: len(fam_sub)],
                s=26,
                alpha=0.80,
                color=family_to_color[str(fam)],
                label=str(fam),
            )
            jitter = jitter[len(fam_sub):]

    handles, labels = plt.gca().get_legend_handles_labels()
    uniq = {}
    for h, l in zip(handles, labels):
        uniq[l] = h
    plt.legend(uniq.values(), uniq.keys(), fontsize=8, loc="best")
    plt.yticks(positions, [short_run_label(str(x), 72) for x in order], fontsize=8)
    plt.xlabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close()


def scatter_error_vs_params(
    df: pd.DataFrame,
    *,
    x_col: str,
    y_col: str,
    family_col: str,
    annot_col: str,
    title: str,
    out_path: Path,
) -> None:
    plot_df = df[[x_col, y_col, family_col, annot_col]].replace([np.inf, -np.inf], np.nan).dropna()
    if plot_df.empty:
        return
    families = plot_df[family_col].astype(str).unique().tolist()
    cmap = plt.cm.get_cmap("tab10", max(len(families), 1))
    family_to_color = {fam: cmap(i) for i, fam in enumerate(families)}

    plt.figure(figsize=(9, 6))
    for fam in families:
        sub = plot_df.loc[plot_df[family_col] == fam]
        plt.scatter(sub[x_col], sub[y_col], s=42, alpha=0.85, color=family_to_color[fam], label=fam)
    for _, row in plot_df.iterrows():
        plt.annotate(str(row[annot_col]), (row[x_col], row[y_col]), fontsize=7, alpha=0.8)
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.title(title)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close()


def ecdf_plot(abs_err_df: pd.DataFrame, *, metric_key: str, region_name: str, family_col: str, out_path: Path, title: str) -> None:
    sub = abs_err_df[(abs_err_df["metric"] == metric_key) & (abs_err_df["region"] == region_name)].copy()
    sub = sub.replace([np.inf, -np.inf], np.nan).dropna(subset=["abs_err"])
    if sub.empty:
        return
    families = sub[family_col].astype(str).unique().tolist()
    cmap = plt.cm.get_cmap("tab10", max(len(families), 1))

    plt.figure(figsize=(8, 6))
    for i, fam in enumerate(families):
        vals = np.sort(sub.loc[sub[family_col] == fam, "abs_err"].to_numpy())
        if vals.size == 0:
            continue
        y = np.arange(1, vals.size + 1) / vals.size
        plt.plot(vals, y, linewidth=2.0, color=cmap(i), label=fam)
    plt.xlabel("erro absoluto")
    plt.ylabel("ECDF")
    plt.title(title)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close()


def line_plot_error_by_axis(
    point_df: pd.DataFrame,
    *,
    metric_key: str,
    family_col: str,
    axis_col: str,
    region_filter_col: Optional[str],
    reducer: str,
    out_path: Path,
    title: str,
) -> None:
    work = point_df.copy()
    if region_filter_col is not None:
        work = work.loc[work[region_filter_col] == 1]
    work = work.replace([np.inf, -np.inf], np.nan).dropna(subset=[f"{metric_key}_abs_err", axis_col])
    if work.empty:
        return

    families = work[family_col].astype(str).unique().tolist()
    cmap = plt.cm.get_cmap("tab10", max(len(families), 1))
    plt.figure(figsize=(9, 6))

    for i, fam in enumerate(families):
        sub = work.loc[work[family_col] == fam]
        grp = sub.groupby(axis_col)[f"{metric_key}_abs_err"]
        if reducer == "median":
            s = grp.median()
        else:
            s = grp.mean()
        plt.plot(s.index.to_numpy(), s.to_numpy(), linewidth=2.0, color=cmap(i), label=fam)

    plt.xlabel(axis_col)
    plt.ylabel(f"{reducer}(|erro|)")
    plt.title(title)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close()


def heatmap_median_abs_error(
    point_df: pd.DataFrame,
    *,
    metric_key: str,
    family_value: str,
    family_col: str,
    out_path: Path,
    title: str,
) -> None:
    sub = point_df.loc[point_df[family_col] == family_value, ["S", "t", f"{metric_key}_abs_err"]].copy()
    sub = sub.replace([np.inf, -np.inf], np.nan).dropna()
    if sub.empty:
        return
    pivot = sub.pivot_table(index="S", columns="t", values=f"{metric_key}_abs_err", aggfunc="median")
    if pivot.empty:
        return

    S_vals = pivot.index.to_numpy(dtype=float)
    t_vals = pivot.columns.to_numpy(dtype=float)
    Z = pivot.to_numpy(dtype=float)

    plt.figure(figsize=(8, 6))
    plt.imshow(
        Z,
        aspect="auto",
        origin="lower",
        extent=[t_vals.min(), t_vals.max(), S_vals.min(), S_vals.max()],
    )
    plt.xlabel("t")
    plt.ylabel("S")
    plt.title(title)
    plt.colorbar(label="mediana do erro absoluto")
    plt.tight_layout()
    plt.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close()


def pred_vs_true_scatter_best_runs(
    point_df: pd.DataFrame,
    run_metric_df: pd.DataFrame,
    *,
    metric_key: str,
    region_name: str,
    signature_col: str,
    title: str,
    out_path: Path,
) -> None:
    sub = run_metric_df[(run_metric_df["metric"] == metric_key) & (run_metric_df["region"] == region_name)].copy()
    if sub.empty:
        return
    best_runs = sub.sort_values("rmse").groupby(signature_col, as_index=False).first()
    if best_runs.empty:
        return

    families = best_runs["model_family"].astype(str).unique().tolist()
    cmap = plt.cm.get_cmap("tab10", max(len(families), 1))
    fam2color = {fam: cmap(i) for i, fam in enumerate(families)}

    plt.figure(figsize=(7, 7))
    all_min = []
    all_max = []
    for _, row in best_runs.iterrows():
        run_id = row["run_id"]
        data = point_df[point_df["run_id"] == run_id].copy()
        data = data[data[f"region_{region_name}"] == 1]
        data = data.replace([np.inf, -np.inf], np.nan).dropna(subset=[f"{metric_key}_pred", f"{metric_key}_true"])
        if data.empty:
            continue
        x = data[f"{metric_key}_true"].to_numpy()
        y = data[f"{metric_key}_pred"].to_numpy()
        all_min += [np.min(x), np.min(y)]
        all_max += [np.max(x), np.max(y)]
        plt.scatter(x, y, s=10, alpha=0.30, color=fam2color[str(row['model_family'])], label=str(row['model_family']))

    if not all_min:
        plt.close()
        return
    mn = min(all_min)
    mx = max(all_max)
    plt.plot([mn, mx], [mn, mx], linestyle="--", linewidth=1.5, color="black")
    handles, labels = plt.gca().get_legend_handles_labels()
    uniq = {}
    for h, l in zip(handles, labels):
        uniq[l] = h
    plt.legend(uniq.values(), uniq.keys(), fontsize=8)
    plt.xlabel("analítico")
    plt.ylabel("predito")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close()


# =============================================================================
# PROCESSAMENTO DE UMA RUN
# =============================================================================

def process_single_run(
    target: Dict[str, Any],
    *,
    group_name: str,
    results_dir: Path,
    output_group_dir: Path,
    device: str,
    bs_params: Dict[str, Any],
    grid_params: Dict[str, Any],
    region_params: Dict[str, Any],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    run_dir = find_run_dir(results_dir, target)
    model, saved_cfg, state_path = load_model(run_dir, device=device)

    run_id = run_dir.name
    model_family = model_family_from_cfg(saved_cfg)
    model_signature = model_signature_from_cfg(saved_cfg)
    seed = saved_cfg.get("seed", np.nan)

    _, _, S, t = build_eval_grid(bs_params, grid_params)
    pred_un = predict_price_and_greeks_autodiff(
        model,
        S,
        t,
        S_max=float(bs_params["S_max"]),
        T=float(bs_params["T"]),
        V_max=float(bs_params["V_max"]),
        batch_size=int(grid_params["batch_size"]),
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
        eps=float(max(bs_params["eps"], grid_params["tau_min"] * 0.5)),
    )

    point_df = build_pointwise_dataframe(S, t, pred_un, true_un, bs_params, region_params)
    point_df["group_name"] = group_name
    point_df["run_id"] = run_id
    point_df["model_family"] = model_family
    point_df["model_signature"] = model_signature
    point_df["seed"] = seed

    run_rows: List[Dict[str, Any]] = []
    contrast_rows: List[Dict[str, Any]] = []
    region_masks = build_region_masks(point_df, bs_params, region_params)

    num_params = int(sum(p.numel() for p in model.parameters() if p.requires_grad))

    for metric_key in METRIC_KEYS:
        for region_name in REGION_ORDER:
            stats_row = compute_point_metrics(point_df, metric_key, region_name, region_masks[region_name])
            stats_row.update({
                "group_name": group_name,
                "run_id": run_id,
                "model_family": model_family,
                "model_signature": model_signature,
                "seed": seed,
                "num_params": num_params,
                "model_state_path": str(state_path),
            })
            run_rows.append(stats_row)

        for region_a, region_b in [
            ("near_strike", "away_strike"),
            ("near_maturity", "away_maturity"),
            ("problem_region", "outside_problem_region"),
        ]:
            cmp_row = compute_region_contrast(point_df, metric_key, region_a, region_b)
            cmp_row.update({
                "group_name": group_name,
                "run_id": run_id,
                "model_family": model_family,
                "model_signature": model_signature,
                "seed": seed,
                "num_params": num_params,
            })
            contrast_rows.append(cmp_row)

    run_metric_df = pd.DataFrame(run_rows)
    contrast_df = pd.DataFrame(contrast_rows)

    run_out_dir = ensure_dir(output_group_dir / "per_run" / model_family / run_id)
    point_df.to_csv(run_out_dir / "grid_pointwise_errors.csv", index=False)
    run_metric_df.to_csv(run_out_dir / "run_metrics.csv", index=False)
    contrast_df.to_csv(run_out_dir / "region_contrasts.csv", index=False)

    metadata_payload = {
        "group_name": group_name,
        "run_id": run_id,
        "run_dir": str(run_dir),
        "model_family": model_family,
        "model_signature": model_signature,
        "seed": seed,
        "num_params": num_params,
        "model_state_path": str(state_path),
        "saved_config": saved_cfg,
        "target_request": target,
        "bs_params": bs_params,
        "grid_params": grid_params,
        "region_params": region_params,
        "outputs": {
            "pointwise_csv": str(run_out_dir / "grid_pointwise_errors.csv"),
            "run_metrics_csv": str(run_out_dir / "run_metrics.csv"),
            "region_contrasts_csv": str(run_out_dir / "region_contrasts.csv"),
        },
    }
    save_json(metadata_payload, run_out_dir / "run_metadata.json")

    return point_df, run_metric_df, contrast_df, metadata_payload


# =============================================================================
# AGREGAÇÃO POR ASSINATURA / FAMÍLIA
# =============================================================================

def summarize_metric_table(metric_df: pd.DataFrame, group_cols: List[str]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for keys, sub in metric_df.groupby(group_cols, dropna=False):
        row: Dict[str, Any] = {}
        if not isinstance(keys, tuple):
            keys = (keys,)
        for col, value in zip(group_cols, keys):
            row[col] = value

        rmse = finite_x(sub["rmse"].to_numpy())
        mae = finite_x(sub["mae"].to_numpy())
        medae = finite_x(sub["medae"].to_numpy())
        bias = finite_x(sub["bias"].to_numpy())
        pear = finite_x(sub["pearson_r"].to_numpy())
        spear = finite_x(sub["spearman_r"].to_numpy())
        r2 = finite_x(sub["r2"].to_numpy())

        rmse_ci_lo, rmse_ci_hi = bootstrap_ci(rmse, func=np.median, seed=123)
        mae_ci_lo, mae_ci_hi = bootstrap_ci(mae, func=np.median, seed=456)

        row.update({
            "n_runs": int(len(sub)),
            "n_unique_seeds": int(pd.Series(sub["seed"]).nunique(dropna=True)),
            "mean_rmse": float(np.mean(rmse)) if rmse.size else float("nan"),
            "median_rmse": float(np.median(rmse)) if rmse.size else float("nan"),
            "std_rmse": float(np.std(rmse, ddof=1)) if rmse.size > 1 else float("nan"),
            "iqr_rmse": float(np.quantile(rmse, 0.75) - np.quantile(rmse, 0.25)) if rmse.size else float("nan"),
            "median_rmse_ci_lo": rmse_ci_lo,
            "median_rmse_ci_hi": rmse_ci_hi,
            "mean_mae": float(np.mean(mae)) if mae.size else float("nan"),
            "median_mae": float(np.median(mae)) if mae.size else float("nan"),
            "median_mae_ci_lo": mae_ci_lo,
            "median_mae_ci_hi": mae_ci_hi,
            "mean_medae": float(np.mean(medae)) if medae.size else float("nan"),
            "median_medae": float(np.median(medae)) if medae.size else float("nan"),
            "mean_abs_bias": float(np.mean(np.abs(bias))) if bias.size else float("nan"),
            "median_abs_bias": float(np.median(np.abs(bias))) if bias.size else float("nan"),
            "mean_pearson_r": float(np.mean(pear)) if pear.size else float("nan"),
            "mean_spearman_r": float(np.mean(spear)) if spear.size else float("nan"),
            "mean_r2": float(np.mean(r2)) if r2.size else float("nan"),
            "median_r2": float(np.median(r2)) if r2.size else float("nan"),
        })
        rows.append(row)
    return pd.DataFrame(rows)


def build_long_abs_error_table(point_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    region_cols = [f"region_{name}" for name in REGION_ORDER]
    base_cols = ["group_name", "run_id", "model_family", "model_signature", "seed", "S", "t", "tau", "abs_moneyness"] + region_cols
    for key in METRIC_KEYS:
        cols = base_cols + [f"{key}_abs_err"]
        sub = point_df[cols].copy()
        sub = sub.rename(columns={f"{key}_abs_err": "abs_err"})
        for region_name in REGION_ORDER:
            rows.append(pd.DataFrame({
                "group_name": sub["group_name"],
                "run_id": sub["run_id"],
                "model_family": sub["model_family"],
                "model_signature": sub["model_signature"],
                "seed": sub["seed"],
                "S": sub["S"],
                "t": sub["t"],
                "tau": sub["tau"],
                "abs_moneyness": sub["abs_moneyness"],
                "metric": key,
                "region": region_name,
                "abs_err": sub["abs_err"],
                "in_region": sub[f"region_{region_name}"],
            }))
    long_df = pd.concat(rows, ignore_index=True)
    long_df = long_df[long_df["in_region"] == 1].drop(columns=["in_region"])
    return long_df


# =============================================================================
# PLOTS E TABELAS DO GRUPO
# =============================================================================

def save_group_outputs(
    output_group_dir: Path,
    point_df: pd.DataFrame,
    run_metric_df: pd.DataFrame,
    contrast_df: pd.DataFrame,
    metadata_rows: List[Dict[str, Any]],
) -> None:
    tables_dir = ensure_dir(output_group_dir / "tables")
    plots_dir = ensure_dir(output_group_dir / "plots")
    diagnostics_dir = ensure_dir(output_group_dir / "diagnostics")

    point_df.to_csv(tables_dir / "all_runs_pointwise_errors.csv", index=False)
    run_metric_df.to_csv(tables_dir / "all_runs_metrics.csv", index=False)
    contrast_df.to_csv(tables_dir / "all_runs_region_contrasts.csv", index=False)
    save_json(metadata_rows, tables_dir / "all_runs_metadata.json")

    signature_summary = summarize_metric_table(run_metric_df, ["group_name", "model_family", "model_signature", "metric", "region"])
    family_summary = summarize_metric_table(run_metric_df, ["group_name", "model_family", "metric", "region"])
    signature_summary.to_csv(tables_dir / "signature_summary.csv", index=False)
    family_summary.to_csv(tables_dir / "family_summary.csv", index=False)

    abs_err_long = build_long_abs_error_table(point_df)
    abs_err_long.to_csv(tables_dir / "abs_error_long.csv", index=False)

    # --------------------------
    # Boxplots por assinatura
    # --------------------------
    for metric_key in METRIC_KEYS:
        for region_name in ["global", "near_strike", "problem_region"]:
            sub = run_metric_df[(run_metric_df["metric"] == metric_key) & (run_metric_df["region"] == region_name)].copy()
            if sub.empty:
                continue
            grouped_boxplot_with_points(
                sub,
                x_col="model_signature",
                y_col="rmse",
                color_col="model_family",
                title=f"RMSE desnormalizado por assinatura | {metric_key} | {region_name}",
                ylabel="RMSE",
                out_path=plots_dir / f"boxplot_rmse_{metric_key}_{region_name}.png",
            )

    # --------------------------
    # RMSE vs número de parâmetros
    # --------------------------
    for metric_key in METRIC_KEYS:
        sub = run_metric_df[(run_metric_df["metric"] == metric_key) & (run_metric_df["region"] == "global")].copy()
        if sub.empty:
            continue
        scatter_error_vs_params(
            sub,
            x_col="num_params",
            y_col="rmse",
            family_col="model_family",
            annot_col="seed",
            title=f"RMSE global vs número de parâmetros | {metric_key}",
            out_path=plots_dir / f"scatter_rmse_vs_params_{metric_key}.png",
        )

    # --------------------------
    # ECDF dos erros absolutos por família
    # --------------------------
    for metric_key in METRIC_KEYS:
        for region_name in ["global", "problem_region"]:
            ecdf_plot(
                abs_err_long,
                metric_key=metric_key,
                region_name=region_name,
                family_col="model_family",
                out_path=plots_dir / f"ecdf_abs_error_{metric_key}_{region_name}.png",
                title=f"ECDF do erro absoluto | {metric_key} | {region_name}",
            )

    # --------------------------
    # Erro por S e por tau
    # --------------------------
    for metric_key in METRIC_KEYS:
        line_plot_error_by_axis(
            point_df,
            metric_key=metric_key,
            family_col="model_family",
            axis_col="S",
            region_filter_col=None,
            reducer="median",
            out_path=plots_dir / f"median_abs_error_by_S_{metric_key}.png",
            title=f"Mediana do erro absoluto por S | {metric_key}",
        )
        line_plot_error_by_axis(
            point_df,
            metric_key=metric_key,
            family_col="model_family",
            axis_col="tau",
            region_filter_col=None,
            reducer="median",
            out_path=plots_dir / f"median_abs_error_by_tau_{metric_key}.png",
            title=f"Mediana do erro absoluto por tau | {metric_key}",
        )
        line_plot_error_by_axis(
            point_df,
            metric_key=metric_key,
            family_col="model_family",
            axis_col="S",
            region_filter_col="region_problem_region",
            reducer="median",
            out_path=plots_dir / f"median_abs_error_by_S_{metric_key}_problem_region.png",
            title=f"Mediana do erro absoluto por S | {metric_key} | região problemática",
        )

    # --------------------------
    # Heatmap do erro absoluto mediano por família
    # --------------------------
    for metric_key in METRIC_KEYS:
        for family_value in point_df["model_family"].astype(str).unique().tolist():
            heatmap_median_abs_error(
                point_df,
                metric_key=metric_key,
                family_value=family_value,
                family_col="model_family",
                out_path=plots_dir / f"heatmap_median_abs_error_{metric_key}_{safe_filename(family_value)}.png",
                title=f"Heatmap do erro absoluto mediano | {metric_key} | {family_value}",
            )

    # --------------------------
    # Predito vs analítico para a melhor run de cada assinatura
    # --------------------------
    for metric_key in METRIC_KEYS:
        for region_name in ["global", "problem_region"]:
            pred_vs_true_scatter_best_runs(
                point_df,
                run_metric_df,
                metric_key=metric_key,
                region_name=region_name,
                signature_col="model_signature",
                title=f"Predito vs analítico | melhor run por assinatura | {metric_key} | {region_name}",
                out_path=plots_dir / f"pred_vs_true_best_by_signature_{metric_key}_{region_name}.png",
            )

    # --------------------------
    # Tabelas-resumo de diagnóstico textual simples
    # --------------------------
    notes = []
    for metric_key in METRIC_KEYS:
        sub = family_summary[(family_summary["metric"] == metric_key) & (family_summary["region"] == "global")].copy()
        if sub.empty:
            continue
        best = sub.sort_values("median_rmse").head(3)
        worst_problem = family_summary[(family_summary["metric"] == metric_key) & (family_summary["region"] == "problem_region")].sort_values("median_rmse", ascending=False).head(3)
        notes.append({
            "metric": metric_key,
            "best_global_families": best[["model_family", "median_rmse", "median_mae", "mean_r2"]].to_dict(orient="records"),
            "worst_problem_families": worst_problem[["model_family", "median_rmse", "median_mae", "mean_r2"]].to_dict(orient="records"),
        })
    save_json(notes, diagnostics_dir / "quick_diagnostics.json")


# =============================================================================
# EXECUÇÃO POR GRUPO
# =============================================================================

def run_group(
    group_name: str,
    targets: List[Dict[str, Any]],
    *,
    results_dir: Path,
    device: str,
    bs_params: Dict[str, Any],
    grid_params: Dict[str, Any],
    region_params: Dict[str, Any],
) -> None:
    if len(targets) == 0:
        print(f"[aviso] Grupo '{group_name}' vazio. Pulando.")
        return

    output_group_dir = ensure_dir(results_dir / DEFAULT_OUTPUT_DIRNAME / safe_slug(group_name))
    all_point = []
    all_run_metrics = []
    all_contrasts = []
    metadata_rows = []

    for idx, target in enumerate(targets, start=1):
        print(f"[{group_name}] {idx}/{len(targets)} -> {target}")
        point_df, run_metric_df, contrast_df, meta = process_single_run(
            target,
            group_name=group_name,
            results_dir=results_dir,
            output_group_dir=output_group_dir,
            device=device,
            bs_params=bs_params,
            grid_params=grid_params,
            region_params=region_params,
        )
        all_point.append(point_df)
        all_run_metrics.append(run_metric_df)
        all_contrasts.append(contrast_df)
        metadata_rows.append(meta)

    point_df = pd.concat(all_point, ignore_index=True)
    run_metric_df = pd.concat(all_run_metrics, ignore_index=True)
    contrast_df = pd.concat(all_contrasts, ignore_index=True)

    save_group_outputs(output_group_dir, point_df, run_metric_df, contrast_df, metadata_rows)
    print(f"Grupo '{group_name}' finalizado. Saída em: {output_group_dir}")


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    set_global_seed(1234)

    results_dir = DEFAULT_RESULTS_DIR
    if not results_dir.exists():
        raise FileNotFoundError(
            f"Pasta de resultados não encontrada: {results_dir}. Ajuste DEFAULT_RESULTS_DIR."
        )

    for group_name, targets in TARGET_GROUPS.items():
        run_group(
            group_name,
            targets,
            results_dir=results_dir,
            device=DEFAULT_DEVICE,
            bs_params=BS_PARAMS,
            grid_params=GRID_PARAMS,
            region_params=REGION_PARAMS,
        )


if __name__ == "__main__":
    main()

