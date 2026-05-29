from __future__ import annotations

import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch as tc
import torch.nn as nn
from scipy.stats import norm

# =============================================================================
# AJUSTE ESTE CAMINHO PARA A RAIZ DO PROJETO, SE NECESSÁRIO.
# O script tenta importar as classes originais do projeto.
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
    # Mantém algo simples e compatível para avaliação.
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
    from pennylane.qnn import TorchLayer
    from itertools import combinations

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

        def generate_data(self, N_domain=1000, N_boundary=1000, N_terminal=1000, seed=1924):
            np.random.seed(seed)
            S_domain = np.random.uniform(0, self.S_max, (int(N_domain), 1))
            t_domain = np.random.uniform(0, self.T, (int(N_domain), 1))
            V_domain = self.V(S_domain, t_domain)

            S_terminal = np.random.uniform(0, self.S_max, (N_terminal, 1))
            t_terminal = self.T * np.ones((N_terminal, 1))
            V_terminal = self.V(S_terminal, t_terminal)

            S_boundary_0 = np.zeros((N_boundary // 2, 1))
            t_boundary_0 = np.random.uniform(0, self.T, (N_boundary // 2, 1))
            V_boundary_0 = self.V(S_boundary_0, t_boundary_0)

            S_boundary_max = self.S_max * np.ones((N_boundary // 2, 1))
            t_boundary_max = np.random.uniform(0, self.T, (N_boundary // 2, 1))
            V_boundary_max = self.V(S_boundary_max, t_boundary_max)

            return {
                'domain': (S_domain, t_domain, V_domain),
                'terminal': (S_terminal, t_terminal, V_terminal),
                'bmax': (S_boundary_max, t_boundary_max, V_boundary_max),
                'b0': (S_boundary_0, t_boundary_0, V_boundary_0),
            }

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
DEFAULT_DEVICE = "cpu"
DEFAULT_DTYPE = tc.float32

# Estes valores precisam bater com o que você quer comparar no analítico.
# ATENÇÃO: nos arquivos enviados, há inconsistência entre defaults do solver e do gerador.
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
    "eval_mode": "random_domain",  # "random_domain" ou "grid"
    "test_seed": 42,
    "N_domain": 2048,
    "N_boundary": 128,
    "N_terminal": 128,
    "Ns": 128,
    "Nt": 128,
    "batch_size": 64,
    "analytic_eps": 1e-6,
}

# Você pode usar run_id direto, ou passar um subconjunto dos campos do config.json salvo.
TARGETS: List[Dict[str, Any]] = [
    # Exemplo 1: localizar por parâmetros salvos no config
    # {
    #     "model_type": "QNN",
    #     "run_id_prefix": "qnn_strong",
    #     "n_qubits": 4,
    #     "n_layers": 2,
    #     "seed": 1924,
    #     "entangler": "strong",
    # },

    # Exemplo 2: localizar por run_id diretamente
    # {
    #     "model_type": "QNN",
    #     "run_id": "qnn_strong_n_qubits_4_n_l_2_seed_1924",
    # },
]
TARGETS = []

for n_qubits in [2, 3, 4]:
    for n_layers in [1, 2, 3]:
        for seed in [1924, 1925, 1926]:
            TARGETS.append(
                {
                    "model_type": "QNN",
                    "run_id_prefix": "qnn_strong",
                    "n_qubits": n_qubits,
                    "n_layers": n_layers,
                    "seed": seed,
                    "entangler": "strong",
                }
            )


# =============================================================================
# HELPERS GERAIS
# =============================================================================
CONTROL_KEYS = {
    "run_id", "results_dir", "greeks_dir", "device", "dtype",
    "eval_mode", "test_seed", "N_domain", "N_boundary", "N_terminal",
    "Ns", "Nt", "batch_size", "analytic_eps",
    "S_max", "T", "K", "r", "sigma", "V_max", "option_type", "eps",
}


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


def values_match(a: Any, b: Any) -> bool:
    if isinstance(a, float) or isinstance(b, float):
        try:
            return math.isclose(float(a), float(b), rel_tol=1e-12, abs_tol=1e-12)
        except Exception:
            return False
    return a == b


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

    matches: List[Path] = []
    target_items = {k: v for k, v in target.items() if k not in CONTROL_KEYS}
    for run_dir, cfg in iter_saved_configs(results_dir, model_type=model_type):
        ok = True
        for k, v in target_items.items():
            if k not in cfg or not values_match(cfg[k], v):
                ok = False
                break
        if ok:
            matches.append(run_dir)

    if len(matches) == 0:
        raise FileNotFoundError(
            f"Nenhuma run encontrada para os parâmetros: {target_items}. "
            f"Confira o results_dir e os campos do config salvo."
        )
    if len(matches) > 1:
        msg = "\n".join(str(p) for p in matches[:10])
        raise RuntimeError(
            f"Mais de uma run encontrada para {target_items}. Seja mais específico ou informe run_id.\n{msg}"
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
# ANALÍTICO BLACK-SCHOLES
# =============================================================================

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

    # Para evitar explodir MSE por pontos essencialmente singulares.
    invalid_greeks = (tau <= eps) | (S <= eps)
    gamma = np.where(invalid_greeks, np.nan, gamma)
    theta = np.where(tau <= eps, np.nan, theta)

    return {
        "V": price,
        "delta": delta,
        "gamma": gamma,
        "theta": theta,
    }


def normalize_bs_outputs(outputs_un: Dict[str, np.ndarray], *, S_max: float, T: float, V_max: float) -> Dict[str, np.ndarray]:
    return {
        "V": outputs_un["V"] / V_max,
        "delta": outputs_un["delta"] * (S_max / V_max),
        "gamma": outputs_un["gamma"] * ((S_max ** 2) / V_max),
        "theta": outputs_un["theta"] * (T / V_max),
    }


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
        theta_b = tc.autograd.grad(V_b, t_b, grad_outputs=ones, create_graph=False, retain_graph=True)[0]
        gamma_b = tc.autograd.grad(delta_b, S_b, grad_outputs=tc.ones_like(delta_b), create_graph=False, retain_graph=False)[0]

        pred_norm["V"].append(V_b.detach().cpu().numpy())
        pred_norm["delta"].append(delta_b.detach().cpu().numpy())
        pred_norm["gamma"].append(gamma_b.detach().cpu().numpy())
        pred_norm["theta"].append(theta_b.detach().cpu().numpy())

        pred_un["V"].append((V_b * V_max).detach().cpu().numpy())
        pred_un["delta"].append((delta_b * (V_max / S_max)).detach().cpu().numpy())
        pred_un["gamma"].append((gamma_b * (V_max / (S_max ** 2))).detach().cpu().numpy())
        pred_un["theta"].append((theta_b * (V_max / T)).detach().cpu().numpy())

    pred_norm = {k: np.vstack(v) for k, v in pred_norm.items()}
    pred_un = {k: np.vstack(v) for k, v in pred_un.items()}
    return pred_norm, pred_un


# =============================================================================
# DADOS DE AVALIAÇÃO
# =============================================================================

def make_eval_points(bs: BlackScholes, eval_params: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, str]:
    eval_mode = eval_params["eval_mode"]
    if eval_mode == "random_domain":
        data = bs.generate_data(
            N_domain=int(eval_params["N_domain"]),
            N_boundary=int(eval_params.get("N_boundary", 128)),
            N_terminal=int(eval_params.get("N_terminal", 128)),
            seed=int(eval_params["test_seed"]),
        )
        S, t, _ = data["domain"]
        return S.reshape(-1, 1), t.reshape(-1, 1), "random_domain"

    if eval_mode == "grid":
        Ns = int(eval_params["Ns"])
        Nt = int(eval_params["Nt"])
        analytic_eps = float(eval_params.get("analytic_eps", 1e-6))
        S_lin = np.linspace(analytic_eps * bs.S_max, bs.S_max, Ns)
        t_lin = np.linspace(0.0, bs.T * (1.0 - analytic_eps), Nt)
        S_mesh, t_mesh = np.meshgrid(S_lin, t_lin, indexing="ij")
        return S_mesh.reshape(-1, 1), t_mesh.reshape(-1, 1), "grid"

    raise ValueError(f"eval_mode '{eval_mode}' não suportado.")


# =============================================================================
# MÉTRICAS E SALVAMENTO
# =============================================================================

def mse_with_mask(pred: np.ndarray, true: np.ndarray) -> Tuple[float, int]:
    pred = np.asarray(pred, dtype=float).reshape(-1, 1)
    true = np.asarray(true, dtype=float).reshape(-1, 1)
    mask = np.isfinite(pred) & np.isfinite(true)
    n = int(mask.sum())
    if n == 0:
        return float("nan"), 0
    return float(np.mean((pred[mask] - true[mask]) ** 2)), n


def build_comparison_dataframe(
    S: np.ndarray,
    t: np.ndarray,
    pred_norm: Dict[str, np.ndarray],
    pred_un: Dict[str, np.ndarray],
    true_norm: Dict[str, np.ndarray],
    true_un: Dict[str, np.ndarray],
) -> pd.DataFrame:
    df = pd.DataFrame({
        "S": S.reshape(-1),
        "t": t.reshape(-1),
    })
    for key in ["V", "delta", "gamma", "theta"]:
        df[f"{key}_pred_norm"] = pred_norm[key].reshape(-1)
        df[f"{key}_true_norm"] = true_norm[key].reshape(-1)
        df[f"{key}_pred_un"] = pred_un[key].reshape(-1)
        df[f"{key}_true_un"] = true_un[key].reshape(-1)
    return df


def compute_metrics(pred_norm, pred_un, true_norm, true_un):
    metrics = {"mse_normalizado": {}, "mse_desnormalizado": {}, "n_validos": {}}
    for key in ["V", "delta", "gamma", "theta"]:
        mse_n, n_n = mse_with_mask(pred_norm[key], true_norm[key])
        mse_u, n_u = mse_with_mask(pred_un[key], true_un[key])
        metrics["mse_normalizado"][key] = mse_n
        metrics["mse_desnormalizado"][key] = mse_u
        metrics["n_validos"][f"{key}_normalizado"] = n_n
        metrics["n_validos"][f"{key}_desnormalizado"] = n_u
    return metrics


def save_run_outputs(
    run_dir: Path,
    *,
    target: Dict[str, Any],
    saved_cfg: Dict[str, Any],
    state_path: Path,
    bs_params: Dict[str, Any],
    eval_params: Dict[str, Any],
    metrics: Dict[str, Any],
    df: pd.DataFrame,
) -> Path:
    greeks_run_dir = run_dir.parents[2] / "greeks" / run_dir.parent.name / run_dir.name
    greeks_run_dir.mkdir(parents=True, exist_ok=True)

    csv_path = greeks_run_dir / "greeks_comparison.csv"
    json_path = greeks_run_dir / "greeks_results.json"

    df.to_csv(csv_path, index=False)
    payload = {
        "run_id": run_dir.name,
        "model_type": run_dir.parent.name,
        "run_dir": str(run_dir),
        "model_state_path": str(state_path),
        "output_csv": str(csv_path),
        "saved_config": saved_cfg,
        "target_request": target,
        "black_scholes_params": bs_params,
        "evaluation_params": eval_params,
        **metrics,
    }
    save_json(payload, json_path)
    return greeks_run_dir


# =============================================================================
# PIPELINE PRINCIPAL
# =============================================================================

def process_target(
    target: Dict[str, Any],
    *,
    results_dir: Path,
    device: str,
    bs_params: Dict[str, Any],
    eval_params: Dict[str, Any],
) -> Dict[str, Any]:
    run_dir = find_run_dir(results_dir, target)
    model, saved_cfg, state_path = load_model(run_dir, device=device)

    bs = BlackScholes(
        S_max=float(bs_params["S_max"]),
        T=float(bs_params["T"]),
        K=float(bs_params["K"]),
        r=float(bs_params["r"]),
        sigma=float(bs_params["sigma"]),
        eps=float(bs_params["eps"]),
    )

    S, t, eval_mode_used = make_eval_points(bs, eval_params)

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

    df = build_comparison_dataframe(S, t, pred_norm, pred_un, true_norm, true_un)
    metrics = compute_metrics(pred_norm, pred_un, true_norm, true_un)

    eval_params_used = dict(eval_params)
    eval_params_used["eval_mode"] = eval_mode_used
    greeks_dir = save_run_outputs(
        run_dir,
        target=target,
        saved_cfg=saved_cfg,
        state_path=state_path,
        bs_params=bs_params,
        eval_params=eval_params_used,
        metrics=metrics,
        df=df,
    )

    return {
        "run_id": run_dir.name,
        "model_type": run_dir.parent.name,
        "greeks_dir": str(greeks_dir),
        **metrics,
    }


def run_batch(
    targets: List[Dict[str, Any]],
    *,
    results_dir: Path = DEFAULT_RESULTS_DIR,
    device: str = DEFAULT_DEVICE,
    bs_params: Dict[str, Any] = BS_PARAMS,
    eval_params: Dict[str, Any] = EVAL_PARAMS,
) -> pd.DataFrame:
    if len(targets) == 0:
        raise ValueError("A lista TARGETS está vazia.")

    rows = []
    for i, target in enumerate(targets, start=1):
        print(f"\n[{i}/{len(targets)}] Processando target: {target}")
        out = process_target(
            target,
            results_dir=Path(results_dir),
            device=device,
            bs_params=dict(bs_params),
            eval_params=dict(eval_params),
        )
        print(
            f"  -> run_id={out['run_id']} | "
            f"MSE un delta={out['mse_desnormalizado']['delta']:.6e} | "
            f"MSE un gamma={out['mse_desnormalizado']['gamma']:.6e} | "
            f"MSE un theta={out['mse_desnormalizado']['theta']:.6e}"
        )
        rows.append(out)

    summary = pd.DataFrame(rows)
    summary_path = Path(results_dir) / "greeks" / "greeks_summary.csv"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(summary_path, index=False)
    print(f"\nResumo salvo em: {summary_path}")
    return summary


if __name__ == "__main__":
    # Edite TARGETS, BS_PARAMS, EVAL_PARAMS e DEFAULT_RESULTS_DIR no topo.
    summary = run_batch(
        TARGETS,
        results_dir=DEFAULT_RESULTS_DIR,
        device=DEFAULT_DEVICE,
        bs_params=BS_PARAMS,
        eval_params=EVAL_PARAMS,
    )
    print(summary)

