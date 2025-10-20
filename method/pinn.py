import torch as tc
import torch.nn as nn
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
#from sklearn.model_selection import train_test_split
from utils.get_data import convert_to_tensor
from utils.math import MSE
# ======================================================
# 📦 Modelo de Rede Neural
# ======================================================

class PINN_MLP(nn.Module):
    def __init__(self, neurons, M, activation=nn.Sigmoid(), output=1):
        super().__init__()
        self.hidden_layers = nn.ModuleList([nn.Linear(2, neurons)])
        self.hidden_layers.extend([nn.Linear(neurons, neurons) for _ in range(M-1)])
        self.output_layer = nn.Linear(neurons, output)
        self.activation = activation
        self.activation_name = activation.__class__.__name__.lower()

    def forward(self, x):
        for layer in self.hidden_layers:
            x = self.activation(layer(x))
        return self.output_layer(x)

# ======================================================
# 🏋️ Função de Treinamento
# ======================================================

def train_nn(S_train, t_train, V_train, S_terminal, t_terminal, V_terminal, 
             S_b0, t_b0, V_b0, S_bmax, t_bmax, V_bmax, arquitetura, 
             sigma=0.02, r=0.05, neurons=1, M=1, gamma=0.9, 
             lr=0.01, epocas=100, activation=nn.Sigmoid, weights=[1,1,1,1], seed=42, S_max=160, T=1.0, K=40, use_domain=False):
    
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
    
    LOSS = {
            'Total': [],
            'pde_loss': [],
            'terminal_loss': [],
            'boundary_0_loss': [],
            'boundary_max_loss': [],
        }

    if use_domain:
        LOSS = {
        'Total': [],
        'pde_loss': [],
        'terminal_loss': [],
        'boundary_0_loss': [],
        'boundary_max_loss': [],
        'domain_loss': []
        }
        V_train /= K
        V_train = convert_to_tensor(V_train)


    # Inicialização
    tc.manual_seed(seed)

    if arquitetura=='MLP':
        model = PINN_MLP(neurons=neurons, M=M, activation=activation)
    else:
        print('-----> ARQUITETURA INVALIDA  <-----')

    optimizer = tc.optim.Adam(params=model.parameters(), lr=lr)
    step_lr = tc.optim.lr_scheduler.StepLR(optimizer, gamma=gamma, step_size=epocas//4)
    

    # Treinamento
    for epoch in tqdm(range(epocas), desc="Treinando PINN"):
        V_pred = model(tc.cat([S_train, t_train], dim=1))
        solution = V_pred.reshape(-1, 1)

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
        beta=0
        if use_domain:
            domain_loss = tc.mean((V_pred - V_train) ** 2)
            beta=weights[4]
            loss = (weights[0] * pde_loss + weights[1] * terminal_loss +
                weights[2] * boundary_0_loss + weights[3] * boundary_max_loss +beta*domain_loss)
        else:
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
        if use_domain:
            LOSS['domain_loss'].append(domain_loss.item())

    return model, LOSS

# ======================================================
# 🔍 Funções de Teste
# ======================================================


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

def save_model_and_loss(model, LOSS, r, K, T, S_max, sigma, n, m, 
                        N_domain, N_boundary, N_terminal, epocas, lr, 
                        seed, seed1, activation_name, weights, use_domain=False,
                        output_dir="results/model"):
    base_path = Path(__file__).resolve().parents[1]
    out_path = base_path / output_dir
    out_path.mkdir(parents=True, exist_ok=True)
    weights_str = format_weights(weights)
    if use_domain:
        name = (f"pinn_r{r}_K{K}_T{T}_Smax{S_max}_sig{sigma}_n{n}_M{m}"
        f"_Nd{N_domain}_Nb{N_boundary}_Nt{N_terminal}_ep{epocas}_activation{activation_name}_{weights_str}_lr{lr}"
        f"_seed{seed}_Seed{seed1}_domain_true")
    else:
        name = (f"pinn_r{r}_K{K}_T{T}_Smax{S_max}_sig{sigma}_n{n}_M{m}"
        f"_Nd{N_domain}_Nb{N_boundary}_Nt{N_terminal}_ep{epocas}_activation{activation_name}_{weights_str}_lr{lr}"
        f"_seed{seed}_Seed{seed1}_domain_false")

    tc.save(model.state_dict(), out_path / f"{name}.pt")
    pd.DataFrame(LOSS).to_csv(out_path / f"{name}_loss.csv", index=False)


def save_V_error(erro, V_test, V_pred, r, K, T, S_max, sigma, n, m, 
                 N_domain, N_boundary, N_terminal, epocas, lr, seed, seed1, seed2, 
                 activation_name, weights, use_domain=None, output_dir="results/v_predicted"):

    base_path = Path(__file__).resolve().parents[1]
    out_path = base_path / output_dir
    out_path.mkdir(parents=True, exist_ok=True)
    weights_str = format_weights(weights)
    if use_domain:
        name = (f"pinn_r{r}_K{K}_T{T}_Smax{S_max}_sig{sigma}_n{n}_M{m}"
        f"_Nd{N_domain}_Nb{N_boundary}_Nt{N_terminal}_ep{epocas}_activation{activation_name}_w{weights_str}_lr{lr}"
        f"_seed{seed}_Seed{seed1}_SEed{seed2}_domain_true")
    else:
        name = (f"pinn_r{r}_K{K}_T{T}_Smax{S_max}_sig{sigma}_n{n}_M{m}"
        f"_Nd{N_domain}_Nb{N_boundary}_Nt{N_terminal}_ep{epocas}_activation{activation_name}_w{weights_str}_lr{lr}"
        f"_seed{seed}_Seed{seed1}_SEed{seed2}_domain_false")

    results_df = pd.DataFrame({
        'V_test': np.array(V_test).flatten(),
        'V_pred': np.array(V_pred).flatten(),
        'MSE': erro
    })

    results_df.to_csv(out_path / f"{name}.csv", index=False)

def model_already_exists(r, K, T, S_max, sigma, n, m,
                         N_domain, N_boundary, N_terminal, epocas, lr,
                         seed, seed1, seed2, activation_name, weights, use_domain=False,
                         output_dir="results/model"):
    
    base_path = Path(__file__).resolve().parents[1]
    weights_str = '-'.join(str(w) for w in weights)
    if use_domain:
        name = (f"pinn_r{r}_K{K}_T{T}_Smax{S_max}_sig{sigma}_n{n}_M{m}"
        f"_Nd{N_domain}_Nb{N_boundary}_Nt{N_terminal}_ep{epocas}_activation{activation_name}_w{weights_str}_lr{lr}"
        f"_seed{seed}_Seed{seed1}_SEed{seed2}_domain_true")
    else:
        name = (f"pinn_r{r}_K{K}_T{T}_Smax{S_max}_sig{sigma}_n{n}_M{m}"
        f"_Nd{N_domain}_Nb{N_boundary}_Nt{N_terminal}_ep{epocas}_activation{activation_name}_w{weights_str}_lr{lr}"
        f"_seed{seed}_Seed{seed1}_SEed{seed2}_domain_false")

    model_path = base_path / output_dir / name
    return model_path.exists()
