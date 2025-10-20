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

# ======================================================
# 📦 Modelo de Rede Neural Quântica
# ======================================================

class QuantumNeuralNetwork(nn.Module):
    def __init__(self, n_qubits=4, n_layers=2, output_dim=1):
        super().__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers

        dev = qml.device("default.qubit", wires=n_qubits)

        @qml.qnode(dev, interface="torch")
        def circuit(inputs, weights):
            qml.AngleEmbedding(inputs, wires=range(n_qubits))
            qml.BasicEntanglerLayers(weights, wires=range(n_qubits))
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
        return x.mean(dim=1, keepdim=True)  # output shape: [batch_size, 1]


# ======================================================
# 🏋️ Função de Treinamento
# ======================================================

def train_nn(S_train, t_train, S_terminal, t_terminal, V_terminal, 
             S_b0, t_b0, V_b0, S_bmax, t_bmax, V_bmax, 
             sigma=0.02, r=0.05, n_qubits=4, M=2, MK=1, gamma=0.9, 
             lr=0.01, epocas=100, weights=[1,1,1,1], seed=42, S_max=160, T=1.0, K=40):
    
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
    # Conversão para tensores
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
    
    # Inicialização
    tc.manual_seed(seed)
    model = QuantumSequentialNetwork(n_qubits=n_qubits, n_layers=M, MK=MK)
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
    for epoch in tqdm(range(epocas), desc="Treinando QPINN"):
        V_pred = model(tc.cat([S_train, t_train], dim=1))
        solution = V_pred.reshape(-1, 1)
        print(V_pred)
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

# ======================================================
# 🔍 Funções de Teste
# ======================================================

def split_train_test(S, t, seed=42):
    return train_test_split(S, t, test_size=0.2, random_state=seed)

def MSE(V, U):
    return np.mean((V.reshape(-1) - U.reshape(-1))**2)

def test_model(S_test, t_test, V_true, model, S_max=160, T=1.0, K=40):
    S_test /= S_max
    t_test /=T
    V_true /= K
    V_pred = model(tc.cat([convert_to_tensor(S_test), convert_to_tensor(t_test)], dim=1)).detach().numpy()
    return MSE(V_true, V_pred), V_pred*40

# ======================================================
# 💾 Funções de Salvamento
# ======================================================
def format_weights(weights):
    return 'w' + '-'.join(str(w) for w in weights)

def save_model_and_loss(model, LOSS, r, K, T, S_max, sigma, neurons, M, MK, 
                        N_domain, N_boundary, N_terminal, epocas, lr, 
                        seed, seed1, seed2, weights,
                        output_dir="results/model"):

    base_path = Path(__file__).resolve().parents[1]
    out_path = base_path / output_dir
    out_path.mkdir(parents=True, exist_ok=True)
    weights_str = format_weights(weights)
    if MK>1:
        name = (f"qbpinn_r{r}_K{K}_T{T}_Smax{S_max}_sig{sigma}_n{neurons}_M{M}_MK{MK}"
                f"_Nd{N_domain}_Nb{N_boundary}_Nt{N_terminal}_ep{epocas}_{weights_str}_lr{lr}"
                f"_seed{seed}_Seed{seed1}_SEed{seed2}")
    else:
        name = (f"qpinn_r{r}_K{K}_T{T}_Smax{S_max}_sig{sigma}_n{neurons}_M{M}"
                f"_Nd{N_domain}_Nb{N_boundary}_Nt{N_terminal}_ep{epocas}_{weights_str}_lr{lr}"
                f"_seed{seed}_Seed{seed1}_SEed{seed2}")

    tc.save(model.state_dict(), out_path / f"{name}.pt")
    pd.DataFrame(LOSS).to_csv(out_path / f"{name}_loss.csv", index=False)


def save_V_error(erro, V_test, V_pred, r, K, T, S_max, sigma, neurons, M, MK,
                 N_domain, N_boundary, N_terminal, epocas, lr, seed, seed1, seed2, 
                  weights, output_dir="results/v_predicted"):

    base_path = Path(__file__).resolve().parents[1]
    out_path = base_path / output_dir
    out_path.mkdir(parents=True, exist_ok=True)
    weights_str = format_weights(weights)
    if MK > 1:
        name = (f"qbpinn_V_with_MSE_r{r}_K{K}_T{T}_Smax{S_max}_sig{sigma}_n{neurons}_M{M}_MK{MK}"
            f"_Nd{N_domain}_Nb{N_boundary}_Nt{N_terminal}_ep{epocas}_{weights_str}_lr{lr}"
            f"_seed{seed}_Seed{seed1}_SEed{seed2}")
    else:
        name = (f"qpinn_V_with_MSE_r{r}_K{K}_T{T}_Smax{S_max}_sig{sigma}_n{neurons}_M{M}"
            f"_Nd{N_domain}_Nb{N_boundary}_Nt{N_terminal}_ep{epocas}_{weights_str}_lr{lr}"
            f"_seed{seed}_Seed{seed1}_SEed{seed2}")

    results_df = pd.DataFrame({
        'V_test': np.array(V_test).flatten(),
        'V_pred': np.array(V_pred).flatten(),
        'MSE': erro
    })

    results_df.to_csv(out_path / f"{name}.csv", index=False)

def model_already_exists(r, K, T, S_max, sigma, n, m, MK,
                        N_domain, N_boundary, N_terminal, epocas, lr,
                        seed, seed1, seed2, weights,
                        output_dir="results/model"):

    base_path = Path(__file__).resolve().parents[1]
    weights_str = '-'.join(str(w) for w in weights)
    if MK>1:
        name = (f"qbpinn_r{r}_K{K}_T{T}_Smax{S_max}_sig{sigma}_n{n}_M{m}_MK{MK}"
                f"_Nd{N_domain}_Nb{N_boundary}_Nt{N_terminal}_ep{epocas}_w{weights_str}_lr{lr}"
                f"_seed{seed}_Seed{seed1}_SEed{seed2}.pt")
    else:
        name = (f"qpinn_r{r}_K{K}_T{T}_Smax{S_max}_sig{sigma}_n{n}_M{m}"
                f"_Nd{N_domain}_Nb{N_boundary}_Nt{N_terminal}_ep{epocas}_w{weights_str}_lr{lr}"
                f"_seed{seed}_Seed{seed1}_SEed{seed2}.pt")
    model_path = base_path / output_dir / name
    return model_path.exists()

