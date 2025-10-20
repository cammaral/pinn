import torch as tc
import torch.nn as nn
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from sklearn.model_selection import train_test_split
from utils.get_data import convert_to_tensor
import pennylane as qml
from pennylane.qnn import TorchLayer
from utils.math import MSE

# ======================================================
# 📦 QuantumNeuralNetwork (Reused from qpinn.py)
# ======================================================
class QuantumNeuralNetwork(nn.Module):
    def __init__(self, n_qubits=4, n_layers=2):
        super().__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers

        # Device and Quantum Circuit setup
        dev = qml.device("default.qubit", wires=n_qubits)

        @qml.qnode(dev, interface="torch")
        def circuit(inputs, weights):
            # Input encoding (e.g., AngleEmbedding)
            qml.AngleEmbedding(inputs, wires=range(n_qubits))
            # Variational circuit (e.g., BasicEntanglerLayers)
            qml.BasicEntanglerLayers(weights, wires=range(n_qubits))
            # Measurements: expectation value for each qubit (n_qubits outputs)
            return [qml.expval(qml.PauliZ(wires=i)) for i in range(n_qubits)]

        # Weight shapes for the TorchLayer
        weight_shapes = {"weights": (n_layers, n_qubits)}
        # Pennylane QNN TorchLayer
        self.q_layer = TorchLayer(circuit, weight_shapes)

    def forward(self, x):
        # x.shape = [batch_size, n_qubits]
        return self.q_layer(x)

# ======================================================
# 📦 Hybrid Classical-Quantum-Classical (CQC) Model
# ======================================================
class Hybrid_CQC_PINN(nn.Module):
    """
    Classical-Quantum-Classical (CQC) Hybrid Model for PINN.
    Structure: Classical Layer -> Quantum Layer -> Classical Output Layer.
    """
    def __init__(self, input_dim=2, hidden_classical_dim=4, n_qubits=4, n_quantum_layers=2, output_dim=1):
        super().__init__()

        # --- Classical Input Block (C1) ---
        # Maps the 2 inputs (S, t) to the number of qubits
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_classical_dim),
            nn.Tanh(), # Or any other activation (e.g., Tanh, ReLU)
            nn.Linear(hidden_classical_dim, n_qubits),
            nn.Tanh() # Must output n_qubits for the QNN input
            # NOTE: The QNN uses AngleEmbedding, which is suited for inputs in [0, pi] or [0, 2pi].
            # You might need to add a scaling/normalization layer here, or rely on the QNN
            # to handle the input mapping from the classical layer's output range.
        )
        self.n_qubits = n_qubits

        # --- Quantum Block (Q) ---
        # Takes n_qubits inputs and produces n_qubits outputs (expval for each Z)
        self.quantum_block = QuantumNeuralNetwork(n_qubits=n_qubits, n_layers=n_quantum_layers)

        # --- Classical Output Block (C2) ---
        # Maps the n_qubits outputs from the QNN to the single output (V)
        self.output_layer = nn.Linear(n_qubits, output_dim)

    def forward(self, x):
        # 1. Classical Input Layer (C1)
        x = self.input_layer(x)
        x = np.pi*(1+x)  # Scale to [0, 2pi] for AngleEmbedding
        # 2. Quantum Layer (Q)
        # Output shape: [batch_size, n_qubits]
        x = self.quantum_block(x)
        
        # 3. Classical Output Layer (C2)
        # Output shape: [batch_size, output_dim=1]
        x = self.output_layer(x)
        return x
    
    # ======================================================
# 🏋️ Função de Treinamento Adaptada
# ======================================================

def train_hnn(S_train, t_train, S_terminal, t_terminal, V_terminal, 
             S_b0, t_b0, V_b0, S_bmax, t_bmax, V_bmax, 
             sigma=0.02, r=0.05, n_qubits=4, M=2, MK=1, neurons=32, gamma=0.9, 
             lr=0.01, epocas=100, weights=[1,1,1,1], seed=42, S_max=160, T=1.0, K=40): # Adicionar 'neurons' ao input

    # ... [Normalização e Conversão para Tensores - Conteúdo do qpinn.py permanece o mesmo] ...

    # Normalização dos inputs
    S_train /= S_max
    S_terminal /= S_max
    S_b0 /= S_max
    S_bmax /= S_max
    t_train /= T
    t_terminal /= T
    t_b0 /= T
    t_bmax /= T
    V_terminal /= K
    V_bmax /= K

    # Conversão para tensores (requer a função 'convert_to_tensor' do seu utilitário)
    # Supondo que 'convert_to_tensor' foi importado corretamente

    S_train = convert_to_tensor(S_train, requires_grad=True)
    t_train = convert_to_tensor(t_train, requires_grad=True)
    S_terminal = convert_to_tensor(S_terminal, requires_grad=True)
    t_terminal = convert_to_tensor(t_terminal, requires_grad=True)
    V_terminal = convert_to_tensor(V_terminal)
    S_b0 = convert_to_tensor(S_b0, requires_grad=True)
    t_b0 = convert_to_tensor(t_b0, requires_grad=True)
    V_b0 = convert_to_tensor(V_b0)
    S_bmax = convert_to_tensor(S_bmax, requires_grad=True)
    t_bmax = convert_to_tensor(t_bmax, requires_grad=True)
    V_bmax = convert_to_tensor(V_bmax)
    
    # Inicialização (Modificação Central)
    tc.manual_seed(seed)
    
    # Substitui a inicialização do QPINN/QBPINN
    model = Hybrid_CQC_PINN(
        hidden_classical_dim=neurons, 
        n_qubits=n_qubits, 
        n_quantum_layers=M
    )

    # ... [O restante do código de otimização e loop de treinamento permanece o mesmo] ...

    optimizer = tc.optim.Adam(params=model.parameters(), lr=lr)
    step_lr = tc.optim.lr_scheduler.StepLR(optimizer, gamma=gamma, step_size=epocas//4)
    LOSS = {
        'Total': [],
        'pde_loss': [],
        'terminal_loss': [],
        'boundary_0_loss': [],
        'boundary_max_loss': []
    }

    # Treinamento
    for epoch in tqdm(range(epocas), desc="Treinando CQC-PINN"):
        # O modelo (model) agora é o Hybrid_CQC_PINN, mas o uso é o mesmo
        V_pred = model(tc.cat([S_train, t_train], dim=1))
        solution = V_pred.reshape(-1, 1)
        #print(V_pred)
        # Cálculo de derivadas de segunda ordem (Black-Scholes PDE)
        df_dt = tc.autograd.grad(solution, t_train, grad_outputs=tc.ones_like(solution), create_graph=True)[0]
        df_ds = tc.autograd.grad(solution, S_train, grad_outputs=tc.ones_like(solution), create_graph=True)[0]
        d2f_d2s = tc.autograd.grad(df_ds, S_train, grad_outputs=tc.ones_like(solution), create_graph=True)[0]

        # Cálculo dos erros
        pde_residual = df_dt + 0.5 * sigma**2 * S_train**2 * d2f_d2s + r * S_train * df_ds - r * solution
        pde_loss = tc.mean(pde_residual**2)

        V_terminal_pred = model(tc.cat([S_terminal, t_terminal], dim=1))
        terminal_loss = tc.mean((V_terminal_pred - V_terminal) ** 2)

        V_b0_pred = model(tc.cat([S_b0, t_b0], dim=1))
        boundary_0_loss = tc.mean((V_b0_pred - V_b0) ** 2)

        V_bmax_pred = model(tc.cat([S_bmax, t_bmax], dim=1))
        boundary_max_loss = tc.mean((V_bmax_pred - V_bmax) ** 2)

        # Total Loss
        loss = (weights[0] * pde_loss + weights[1] * terminal_loss +
                weights[2] * boundary_0_loss + weights[3] * boundary_max_loss)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        step_lr.step()

        # Guarda histórico
        LOSS['Total'].append(loss.item())
        LOSS['pde_loss'].append(pde_loss.item())
        LOSS['terminal_loss'].append(terminal_loss.item())
        LOSS['boundary_0_loss'].append(boundary_0_loss.item())
        LOSS['boundary_max_loss'].append(boundary_max_loss.item())
    return model, LOSS

def test_model(S_test, t_test, V_true, model, S_max=160, T=1.0, K=40):
    V_pred = model(tc.cat([convert_to_tensor(S_test/S_max), convert_to_tensor(t_test /T)], dim=1)).detach().numpy()
    return MSE(V_true, V_pred*K), V_pred*K