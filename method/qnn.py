import pennylane as qml
import torch.nn as nn
from pennylane.qnn import TorchLayer
import torch as tc
from itertools import combinations
#from circuits.ansatz import ms_brickwall

class QuantumNeuralNetwork(nn.Module):
    def __init__(self, n_qubits=4, n_layers=2, output_dim=1, entangler='basic'):
        super().__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.n_vertex = n_qubits #Just to keep compatibility
        self.entanger = entangler
        dev = qml.device("default.qubit", wires=n_qubits)

        @qml.qnode(dev, interface="torch")
        def circuit(inputs, weights):
            qml.AngleEmbedding(inputs, wires=range(n_qubits))
            if self.entangler == 'basic':
                qml.BasicEntanglerLayers(weights, wires=range(n_qubits))
            elif self.entangler == 'strong':
                qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
            return [qml.expval(qml.PauliZ(wires=i)) for i in range(n_qubits)]

        weight_shapes = {"weights": (n_layers, n_qubits)}
        self.q_layer = TorchLayer(circuit, weight_shapes)
        #self.linear = nn.Linear(n_qubits, output_dim)

    def forward(self, x):
        x = self.q_layer(x)
        return x



class QuantumSequentialNetwork(nn.Module):
    def __init__(self, n_qubits=4, n_layers=2, MK=1):
        super().__init__()
        self.blocks = nn.ModuleList([
            QuantumNeuralNetwork(n_qubits=n_qubits, n_layers=n_layers)
            for _ in range(MK)
        ])

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        # x.shape = [batch_size, n_qubits]
        # Faz a média sobre os qubits (dimensão 1)
        return x # output shape: [batch_size, 1]

class CorrelatorQuantumNeuralNetwork(nn.Module):
    def __init__(self, n_qubits=4, n_layers=2, k=2, n_vertex=9, nonlinear=True, entangler='basic'):
        super().__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.k = k
        self.n_vertex = n_vertex
        self.obs_list = self._generate_obs_list()
        self.alpha = 1.5*self.n_qubits
        self.nonlinear = nn.Tanh()
        self.entangler = entangler

        # build qnode and TorchLayer
        dev = qml.device("default.qubit", wires=self.n_qubits)

        @qml.qnode(dev, interface="torch", diff_method="backprop")
        def circuit(inputs, weights):
            qml.AngleEmbedding(inputs, wires=range(n_qubits))
            if self.entangler == 'basic':
                qml.BasicEntanglerLayers(weights, wires=range(n_qubits))
            elif self.entangler == 'strong':
                qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
            return [qml.expval(obs) for obs in self.obs_list]

        weight_shapes = {"weights": (self.n_layers, self.n_qubits)}
        self.q_layer = TorchLayer(circuit, weight_shapes)

    def _generate_obs_list(self):
        pauli_list = []
        positions = list(range(self.n_qubits))
        for pauli_char in ["Z", "Y", "X"]:
            for combo in combinations(positions, self.k):
                # build pauli string
                pauli_str = ['I'] * self.n_qubits
                for idx in combo:
                    pauli_str[idx] = pauli_char

                obs = None
                # reverse to maintain indexing convention if needed
                for idx, p in enumerate(reversed(pauli_str)):
                    if p == 'I':
                        continue
                    elif p == 'X':
                        current = qml.PauliX(idx)
                    elif p == 'Y':
                        current = qml.PauliY(idx)
                    elif p == 'Z':
                        current = qml.PauliZ(idx)

                    if obs is None:
                        obs = current
                    else:
                        obs = obs @ current

                if obs is None:
                    obs = qml.Identity(0)

                pauli_list.append(obs)
                if len(pauli_list) == self.n_vertex:
                    return pauli_list
        return pauli_list

    def forward(self, x):
        x = self.q_layer(x)
        if self.nonlinear:
            x = self.nonlinear(self.alpha * x)
        # TorchLayer expects float inputs (torch.Tensor)
        return x



# Corrigir a classe CorrelatorQuantumSequentialNetwork
class CorrelatorQuantumSequentialNetwork(nn.Module):
    def __init__(self, n_qubits=4, n_layers=2, k=2, n_vertex=9, MK=1):
        super().__init__()
        self.blocks = nn.ModuleList([
            CorrelatorQuantumNeuralNetwork(n_qubits=n_qubits, n_layers=n_layers, k=k, n_vertex=n_vertex)
            for _ in range(MK)
        ])

    def forward(self, x):
        outs = []
        for block in self.blocks:
            outs.append(block(x))
        # outs: list of tensors shape [batch, n_vertex]
        # Stack and average along blocks
        stacked = tc.stack(outs, dim=0)  # shape [MK, batch, n_vertex]
        mean_over_blocks = tc.mean(stacked, dim=0)  # [batch, n_vertex]
        # Optionally reduce n_vertex dims to single output by averaging
        return mean_over_blocks.mean(dim=1, keepdim=True)  # [batch, 1]