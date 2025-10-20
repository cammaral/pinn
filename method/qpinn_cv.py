import tensorflow as tf
import numpy as np
import strawberryfields as sf
from strawberryfields import ops
from tqdm import tqdm
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
tf.get_logger().setLevel('ERROR')

# ======================================================
# 📦 Modelo de Rede Neural Quântica com Strawberry Fields
# ======================================================

class MeanEnforcingDense(tf.keras.layers.Layer):
    def __init__(self, input_dim, output_dim=1):
        super().__init__()
        self.dense = tf.keras.layers.Dense(
            output_dim,
            use_bias=False,
            kernel_initializer='ones',  # vai ser sobrescrito no forward
            trainable=True
        )
        self.input_dim = input_dim

    def build(self, input_shape):
        # Força o shape da Dense
        self.dense.build(input_shape)

    def call(self, x):
        # Força os pesos a serem 1/n a cada chamada (simulando média)
        mean_kernel = tf.ones_like(self.dense.kernel) / self.input_dim
        self.dense.kernel.assign(mean_kernel)
        return self.dense(x)



class GaussianQuantumNeuralNetwork(tf.keras.Model):
    def __init__(self, n_modes=2, n_layers=2):
        super().__init__()
        self.n_modes = n_modes
        self.n_layers = n_layers
        self.cutoff = 10
        self.n_params = 16
        self.weight_shapes = (n_layers, self.n_params)

        self.q_weights = tf.Variable(
            tf.random.normal(self.weight_shapes, stddev=0.1),
            trainable=True,
            name="weights"
        )

        self.qprog = sf.Program(n_modes)
        self.eng = sf.Engine("tf", backend_options={"cutoff_dim": self.cutoff})

        # símbolos
        self.symbols = [
            [self.qprog.params(f"w_{l}_{i}") for i in range(self.n_params)]
            for l in range(n_layers)
        ]
        self._build()

    def _build(self):
        with self.qprog.context as q:
            self.S = self.qprog.params("S")
            self.t = self.qprog.params("t")
            ops.Dgate(self.S, 0.0) | q[0]
            ops.Dgate(self.t, 0.0) | q[1]

            for l in range(self.n_layers):
                w = self.symbols[l]
                ops.BSgate(w[0], w[1]) | (q[0], q[1])
                ops.Rgate(w[2]) | q[0]
                ops.Rgate(w[3]) | q[1]
                ops.Sgate(w[4], w[5]) | q[0]
                ops.Sgate(w[6], w[7]) | q[1]
                ops.BSgate(w[8], w[9]) | (q[0], q[1])
                ops.Rgate(w[10]) | q[0]
                ops.Rgate(w[11]) | q[1]
                ops.Dgate(w[12], 0.0) | q[0]
                ops.Dgate(w[13], 0.0) | q[1]
                ops.Kgate(w[14]) | q[0]
                ops.Kgate(w[15]) | q[1]

    def call(self, x):
        outputs = []
        for i in range(x.shape[0]):
            s = x[i, 0]
            t = x[i, 1]

            flat_weights = tf.reshape(self.q_weights, [-1])
            flat_symbols = [p for layer in self.symbols for p in layer]
            mapping = {p.name: v for p, v in zip(flat_symbols, flat_weights)}
            mapping["S"] = s
            mapping["t"] = t
            state = self.eng.run(self.qprog, args=mapping).state
            x0, _ = state.quad_expectation(0, 0)
            x1, _ = state.quad_expectation(1, 0)
            outputs.append(tf.stack([x0, x1]))
            self.eng.reset()
        return tf.stack(outputs)

class GaussianQuantumSequentialNetwork(tf.keras.Model):
    def __init__(self, n_modes=2, n_layers=2, MK=1, output_dim=1):
        super().__init__()
        self.blocks = [GaussianQuantumNeuralNetwork(n_modes=n_modes, n_layers=n_layers) for _ in range(MK)]
        #self.final_layer = tf.keras.layers.Dense(output_dim, use_bias=False)
        self.final_layer = MeanEnforcingDense(input_dim=n_modes)

        #self.final_layer = MeanDense(input_dim=n_modes)
        self.act = tf.keras.activations.tanh

    def call(self, x):
        for block in self.blocks:
            #print(x)
            x = block(x)
            #print(x)
            #x = tf.reduce_mean(x, axis=1, keepdims=True)
            x = self.final_layer(x)
            
            print('mean',x)
            print(tf.reduce_mean(x, axis=1, keepdims=True))
        return x


# ======================================================
# 🏋️ Função de Treinamento
# ======================================================

def train_nn(S_train, t_train, S_terminal, t_terminal, V_terminal, 
             S_b0, t_b0, V_b0, S_bmax, t_bmax, V_bmax, 
             sigma=0.02, r=0.05, n_modes=2, M=2, MK=1, gamma=0.9, 
             lr=0.01, epocas=100, weights=[1,1,1,1], seed=42, S_max=160, T=1.0, K=40):
    
    S_train = tf.convert_to_tensor(S_train / S_max, dtype=tf.float32)
    t_train = tf.convert_to_tensor(t_train / T, dtype=tf.float32)
    S_terminal = tf.convert_to_tensor(S_terminal / S_max, dtype=tf.float32)
    t_terminal = tf.convert_to_tensor(t_terminal / T, dtype=tf.float32)
    V_terminal = tf.convert_to_tensor(V_terminal / K, dtype=tf.float32)
    S_b0 = tf.convert_to_tensor(S_b0 / S_max, dtype=tf.float32)
    t_b0 = tf.convert_to_tensor(t_b0 / T, dtype=tf.float32)
    V_b0 = tf.convert_to_tensor(V_b0, dtype=tf.float32)
    S_bmax = tf.convert_to_tensor(S_bmax / S_max, dtype=tf.float32)
    t_bmax = tf.convert_to_tensor(t_bmax / T, dtype=tf.float32)
    V_bmax = tf.convert_to_tensor(V_bmax / K, dtype=tf.float32)

    x_input = tf.concat([S_train, t_train], axis=1)
    x_terminal = tf.concat([S_terminal, t_terminal], axis=1)
    x_b0 = tf.concat([S_b0, t_b0], axis=1)
    x_bmax = tf.concat([S_bmax, t_bmax], axis=1)

    tf.random.set_seed(seed)
    model = GaussianQuantumSequentialNetwork(n_modes=n_modes, n_layers=M, MK=MK)
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    LOSS = {
        'Total': [], 'pde_loss': [], 'terminal_loss': [], 'boundary_0_loss': [], 'boundary_max_loss': []
    }

    for epoch in tqdm(range(epocas), desc="Treinando CV-QPINN"):
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(x_input)
            with tf.GradientTape() as tape1:
                tape1.watch(x_input)
                V_pred = model(x_input)
                print(V_pred)
            df_dt = tape1.gradient(V_pred, x_input)[:, 1:2]
            df_ds = tape.gradient(V_pred, x_input)[:, 0:1]
            df2_ds2 = tape.gradient(df_ds, x_input)[:, 0:1]



            pde_residual = df_dt + 0.5 * sigma**2 * S_train**2 * df2_ds2 + r * S_train * df_ds - r * V_pred
            pde_loss = tf.reduce_mean(tf.square(pde_residual))
            
            terminal_loss = tf.reduce_mean(tf.square(model(x_terminal) - V_terminal))
            boundary_0_loss = tf.reduce_mean(tf.square(model(x_b0) - V_b0))
            boundary_max_loss = tf.reduce_mean(tf.square(model(x_bmax) - V_bmax))

            total_loss = (weights[0]*pde_loss + weights[1]*terminal_loss +
                        weights[2]*boundary_0_loss + weights[3]*boundary_max_loss)

        gradients = tape.gradient(total_loss, model.trainable_variables)

        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        LOSS['Total'].append(total_loss.numpy())
        LOSS['pde_loss'].append(pde_loss.numpy())
        LOSS['terminal_loss'].append(terminal_loss.numpy())
        LOSS['boundary_0_loss'].append(boundary_0_loss.numpy())
        LOSS['boundary_max_loss'].append(boundary_max_loss.numpy())

    return model, LOSS
# ======================================================
# 🔍 Funções de Teste
# ======================================================

def split_train_test(S, t, seed=42):
    return train_test_split(S, t, test_size=0.2, random_state=seed)

def MSE(V, U):
    return np.mean((V.reshape(-1) - U.reshape(-1))**2)

def test_model(S_test, t_test, V_true, model, S_max=160, T=1.0, K=40):
    S_test = tf.convert_to_tensor(S_test / S_max, dtype=tf.float32)
    t_test = tf.convert_to_tensor(t_test / T, dtype=tf.float32)
    V_true = V_true / K
    x_test = tf.concat([S_test, t_test], axis=1)
    V_pred = model(x_test).numpy()
    return MSE(V_true, V_pred), V_pred * K

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
    if MK > 1:
        name = (f"cvqbpinn_r{r}_K{K}_T{T}_Smax{S_max}_sig{sigma}_n{neurons}_M{M}_MK{MK}"
                f"_Nd{N_domain}_Nb{N_boundary}_Nt{N_terminal}_ep{epocas}_{weights_str}_lr{lr}"
                f"_seed{seed}_Seed{seed1}_SEed{seed2}")
    else:
        name = (f"cvqpinn_r{r}_K{K}_T{T}_Smax{S_max}_sig{sigma}_n{neurons}_M{M}"
                f"_Nd{N_domain}_Nb{N_boundary}_Nt{N_terminal}_ep{epocas}_{weights_str}_lr{lr}"
                f"_seed{seed}_Seed{seed1}_SEed{seed2}")

    model.save_weights(out_path / f"{name}.weights.h5")
    pd.DataFrame(LOSS).to_csv(out_path / f"{name}_loss.csv", index=False)

def save_V_error(erro, V_test, V_pred, r, K, T, S_max, sigma, neurons, M, MK,
                 N_domain, N_boundary, N_terminal, epocas, lr, seed, seed1, seed2, 
                 weights, output_dir="results/v_predicted"):

    base_path = Path(__file__).resolve().parents[1]
    out_path = base_path / output_dir
    out_path.mkdir(parents=True, exist_ok=True)
    weights_str = format_weights(weights)
    if MK > 1:
        name = (f"cvqbpinn_V_with_MSE_r{r}_K{K}_T{T}_Smax{S_max}_sig{sigma}_n{neurons}_M{M}_MK{MK}"
                f"_Nd{N_domain}_Nb{N_boundary}_Nt{N_terminal}_ep{epocas}_{weights_str}_lr{lr}"
                f"_seed{seed}_Seed{seed1}_SEed{seed2}")
    else:
        name = (f"cvqpinn_V_with_MSE_r{r}_K{K}_T{T}_Smax{S_max}_sig{sigma}_n{neurons}_M{M}"
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
    if MK > 1:
        name = (f"cvqbpinn_r{r}_K{K}_T{T}_Smax{S_max}_sig{sigma}_n{n}_M{m}_MK{MK}"
                f"_Nd{N_domain}_Nb{N_boundary}_Nt{N_terminal}_ep{epocas}_w{weights_str}_lr{lr}"
                f"_seed{seed}_Seed{seed1}_SEed{seed2}.h5")
    else:
        name = (f"cvqpinn_r{r}_K{K}_T{T}_Smax{S_max}_sig{sigma}_n{n}_M{m}"
                f"_Nd{N_domain}_Nb{N_boundary}_Nt{N_terminal}_ep{epocas}_w{weights_str}_lr{lr}"
                f"_seed{seed}_Seed{seed1}_SEed{seed2}.h5")
    model_path = base_path / output_dir / name
    return model_path.exists()


"""
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

# ======================================================
# 📦 Modelo de Rede Neural Quântica Contínua (CV - Gaussian)
# ======================================================

class GaussianQuantumNeuralNetwork(nn.Module):
    def __init__(self, n_modes=2, n_layers=2):
        super().__init__()
        dev = qml.device("default.gaussian", wires=n_modes)

        @qml.qnode(dev, interface="torch")
        def circuit(inputs, weights):
            S, t = inputs[0], inputs[1]
            qml.Displacement(S, 0.0, wires=0)
            qml.Displacement(t, 0.0, wires=1)


            for l in range(n_layers):
                for i in range(n_modes):
                    qml.Squeezing(weights[l, i, 0], weights[l, i, 1], wires=i)
                qml.Beamsplitter(weights[l, 0, 2], weights[l, 1, 2], wires=[0, 1])

            return [qml.expval(qml.X(i)) for i in range(n_modes)]

        weight_shapes = {"weights": (n_layers, n_modes, 3)}
        self.q_layer = TorchLayer(circuit, weight_shapes)

    def forward(self, x):
        return self.q_layer(x)




class GaussianQuantumSequentialNetwork(nn.Module):
    def __init__(self, n_modes=2, n_layers=2, MK=1, output_dim=1):
        super().__init__()
        self.blocks = nn.ModuleList([
            GaussianQuantumNeuralNetwork(n_modes=n_modes, n_layers=n_layers)
            for _ in range(MK)
        ])
        self.final_layer = nn.Linear(n_modes, output_dim)

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
            #print(x)
        return self.final_layer(x)

# ======================================================
# 🏋️ Função de Treinamento
# ======================================================

def train_nn(S_train, t_train, S_terminal, t_terminal, V_terminal, 
             S_b0, t_b0, V_b0, S_bmax, t_bmax, V_bmax, 
             sigma=0.02, r=0.05, n_modes=2, M=2, MK=1, gamma=0.9, 
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
    model = GaussianQuantumSequentialNetwork(n_modes=n_modes, n_layers=M, MK=MK)
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
    for epoch in tqdm(range(epocas), desc="Treinando CV-QPINN"):
        V_pred = model(tc.cat([S_train, t_train], dim=1))
        solution = V_pred.reshape(-1, 1)
        #S_train.retain_grad()
        #t_train.retain_grad()

        #df_dt = tc.autograd.grad(solution, t_train, grad_outputs=tc.ones_like(solution), create_graph=True)[0]
        df_dt = tc.autograd.grad(solution, t_train, grad_outputs=tc.ones_like(solution), create_graph=True)[0]
        if df_dt is None:
            print("⚠️ t_train não foi usado no modelo!")

        df_ds = tc.autograd.grad(solution, S_train, grad_outputs=tc.ones_like(solution), create_graph=True)[0]
        #print("df/dS:", df_ds)
        d2f_d2s = tc.autograd.grad(df_ds, S_train, grad_outputs=tc.ones_like(solution), create_graph=True, allow_unused=True)[0]
        if d2f_d2s is None:
            #print("⚠️ Segunda derivada d²f/dS² não existe — talvez a função seja linear.")
            d2f_d2s = tc.zeros_like(S_train)

    
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
        print(loss)
        optimizer.zero_grad()
        loss.backward()
        #print("Gradiente de S_train:", S_train.grad)

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
        name = (f"cvqbpinn_r{r}_K{K}_T{T}_Smax{S_max}_sig{sigma}_n{neurons}_M{M}_MK{MK}"
                f"_Nd{N_domain}_Nb{N_boundary}_Nt{N_terminal}_ep{epocas}_{weights_str}_lr{lr}"
                f"_seed{seed}_Seed{seed1}_SEed{seed2}")
    else:
        name = (f"cvqpinn_r{r}_K{K}_T{T}_Smax{S_max}_sig{sigma}_n{neurons}_M{M}"
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
        name = (f"cvqbpinn_V_with_MSE_r{r}_K{K}_T{T}_Smax{S_max}_sig{sigma}_n{neurons}_M{M}_MK{MK}"
            f"_Nd{N_domain}_Nb{N_boundary}_Nt{N_terminal}_ep{epocas}_{weights_str}_lr{lr}"
            f"_seed{seed}_Seed{seed1}_SEed{seed2}")
    else:
        name = (f"cvqpinn_V_with_MSE_r{r}_K{K}_T{T}_Smax{S_max}_sig{sigma}_n{neurons}_M{M}"
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
        name = (f"cvqbpinn_r{r}_K{K}_T{T}_Smax{S_max}_sig{sigma}_n{n}_M{m}_MK{MK}"
                f"_Nd{N_domain}_Nb{N_boundary}_Nt{N_terminal}_ep{epocas}_w{weights_str}_lr{lr}"
                f"_seed{seed}_Seed{seed1}_SEed{seed2}.pt")
    else:
        name = (f"cvqpinn_r{r}_K{K}_T{T}_Smax{S_max}_sig{sigma}_n{n}_M{m}"
                f"_Nd{N_domain}_Nb{N_boundary}_Nt{N_terminal}_ep{epocas}_w{weights_str}_lr{lr}"
                f"_seed{seed}_Seed{seed1}_SEed{seed2}.pt")
    model_path = base_path / output_dir / name
    return model_path.exists()
"""