import pennylane as qml
import torch as tc
import torch.nn as nn
from pennylane.qnn import TorchLayer
from itertools import combinations
from circuits.generate import make_ansatz
from utils.device import *


# =========================================================
# FUNÇÃO EXTERNA: GERA APENAS A ESTRUTURA DO CIRCUITO
# =========================================================

def get_weight_shapes(circuit_type, n_layers, n_qubits):
    if circuit_type == "basic":
        return {"weights": (n_layers, n_qubits)}
    elif circuit_type == "strong":
        return {"weights": (n_layers, n_qubits, 3)}
    else:
        raise ValueError(f"circuit_type desconhecido: {circuit_type}")


# =========================================================
# QNN PADRÃO
# =========================================================

class QuantumNeuralNetwork(nn.Module):
    def __init__(
        self,
        n_qubits=4,
        n_layers=2,
        output_dim=1,
        ansatz_fn=None,
        circuit_type="basic",
        device: str = "auto",
        diff_method=None,
        dtype=tc.float32,
    ):
        super().__init__()

        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.n_output = n_qubits   # compatibilidade
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

        weight_shapes = get_weight_shapes(self.circuit_type, self.n_layers, self.n_qubits)
        self.q_layer = TorchLayer(circuit, weight_shapes)

        self.to(self._torch_device, dtype=self._dtype)

    def forward(self, x):
        x = self.q_layer(x)
        return x


# =========================================================
# QNN SEQUENCIAL
# =========================================================

class QuantumSequentialNetwork(nn.Module):
    def __init__(
        self,
        n_qubits=4,
        n_layers=2,
        MK=1,
        ansatz_fn=None,
        circuit_type="basic",
        device: str = "auto",
        dtype=tc.float32,
    ):
        super().__init__()

        self._torch_device = pick_torch_device(device)
        self._dtype = dtype

        self.blocks = nn.ModuleList([
            QuantumNeuralNetwork(
                n_qubits=n_qubits,
                n_layers=n_layers,
                ansatz_fn=ansatz_fn,
                circuit_type=circuit_type,
                device=device,
                dtype=dtype,
            )
            for _ in range(MK)
        ])

        self.to(self._torch_device, dtype=self._dtype)

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x


# =========================================================
# GERADOR DOS OBSERVÁVEIS CORRELACIONADOS
# =========================================================

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
                elif p == "X":
                    current = qml.PauliX(idx)
                elif p == "Y":
                    current = qml.PauliY(idx)
                elif p == "Z":
                    current = qml.PauliZ(idx)

                if obs is None:
                    obs = current
                else:
                    obs = obs @ current

            if obs is None:
                obs = qml.Identity(0)

            pauli_list.append(obs)

            if len(pauli_list) == n_vertex:
                return pauli_list

    return pauli_list


# =========================================================
# QNN COM CORRELATORES
# =========================================================

class CorrelatorQuantumNeuralNetwork(nn.Module):
    def __init__(
        self,
        n_qubits=4,
        n_layers=2,
        k=2,
        n_vertex=9,
        nonlinear=None,
        ansatz_fn=None,
        circuit_type="basic",
        device: str = "auto",
        diff_method=None,
        dtype=tc.float32,
    ):
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

        weight_shapes = get_weight_shapes(self.circuit_type, self.n_layers, self.n_qubits)
        self.q_layer = TorchLayer(circuit, weight_shapes)

        self.to(self._torch_device, dtype=self._dtype)

    def forward(self, x):
        x = self.q_layer(x)
        if self.nonlinear:
            x = self.nonlinear(self.alpha * x)
        return x


# =========================================================
# QNN SEQUENCIAL COM CORRELATORES
# =========================================================

class CorrelatorQuantumSequentialNetwork(nn.Module):
    def __init__(
        self,
        n_qubits=4,
        n_layers=2,
        k=2,
        n_vertex=9,
        MK=1,
        ansatz_fn=None,
        circuit_type="basic",
        device: str = "auto",
        dtype=tc.float32,
    ):
        super().__init__()

        self._torch_device = pick_torch_device(device)
        self._dtype = dtype

        self.blocks = nn.ModuleList([
            CorrelatorQuantumNeuralNetwork(
                n_qubits=n_qubits,
                n_layers=n_layers,
                k=k,
                n_vertex=n_vertex,
                ansatz_fn=ansatz_fn,
                circuit_type=circuit_type,
                device=device,
                dtype=dtype,
            )
            for _ in range(MK)
        ])

        self.to(self._torch_device, dtype=self._dtype)

    def forward(self, x):
        outs = []
        for block in self.blocks:
            outs.append(block(x))

        stacked = tc.stack(outs, dim=0)              # [MK, batch, n_vertex]
        mean_over_blocks = tc.mean(stacked, dim=0)  # [batch, n_vertex]
        return mean_over_blocks.mean(dim=1, keepdim=True)  # [batch, 1]