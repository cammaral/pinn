import os
import pandas as pd
import numpy as np
import torch as tc
import torch.nn as nn
from tqdm import tqdm
import time
import json
import itertools

# --- Importe suas classes ---
from equation.option_pricing import BlackScholes
from optimize.option_princing import BlackScholeOptimizer
from method.nn import MLP, ResNet
from method.hnn import HybridCQN
from method.qnn import QuantumNeuralNetwork

# =============================================================================
# FUNÇÕES AUXILIARES: geração de grade
# =============================================================================

def generate_runs(base_config, sweep_params):
    """
    Cria combinações de parâmetros a partir de 'base_config' e 'sweep_params',
    gerando um dicionário por run + um run_id legível.
    """
    generated_experiments = []
    param_keys = list(sweep_params.keys())
    param_values = list(sweep_params.values())

    for combination in itertools.product(*param_values):
        new_config = base_config.copy()
        param_combination_dict = {}
        run_id_parts = [base_config.get("run_id_prefix", base_config.get("model_type", "run"))]

        for key, value in zip(param_keys, combination):
            param_combination_dict[key] = value
            if isinstance(value, nn.Module):
                val_str = value.__class__.__name__
            else:
                val_str = str(value)
            key_str = (
                key.replace("learning_rate", "lr")
                   .replace("hidden", "n")
                   .replace("layers", "l")
                   .replace("blocks", "b")
            )
            run_id_parts.append(f"{key_str}_{val_str}")

        new_config.update(param_combination_dict)
        new_config["run_id"] = "_".join(run_id_parts)
        generated_experiments.append(new_config)

    return generated_experiments

def pretty_print(config_list, num_to_show=5):
    for cfg in config_list[:num_to_show]:
        printable_cfg = {}
        for k, v in cfg.items():
            printable_cfg[k] = v.__class__.__name__ if isinstance(v, nn.Module) else v
        print(json.dumps(printable_cfg, indent=2))
    if len(config_list) > num_to_show:
        print(f"... e mais {len(config_list) - num_to_show} outros.")

# =============================================================================
# DEFINIÇÃO DOS EXPERIMENTOS
# =============================================================================

experiment_grid = []

# --- GRUPO: Teste de seeds / estabilidade (exemplo) ---
base_seed_test = {
    "model_type": "MLP",          # "MLP", "ResNet" ou "QPINN"
    "run_id_prefix": "mlp",
    "lr": 2e-3,
    "epochs": 15000,
    "activation": nn.Tanh(),
    "force": False,               # mude para True se quiser recalcular mesmo com artefatos existentes
}
sweep_seed = {
    "hidden": [2, 3, 5],
    "blocks": [1, 2, 3],
    "seed": [1900, 1905, 1924, 1925, 1926],
    #"seed": [1958, 1962, 1970, 1994, 2002, 1900, 1905, 1924, 1925, 1926],
}
experiment_grid.extend(generate_runs(base_seed_test, sweep_seed))

print(f"Total de {len(experiment_grid)} experimentos gerados para a grade.")
pretty_print(experiment_grid, num_to_show=3)

# =============================================================================
# AMBIENTE / PASTAS
# =============================================================================

RESULTS_DIR = "experimentos_pinn"
MODELS_DIR = os.path.join(RESULTS_DIR, "modelos_salvos")
LOSS_DIR   = os.path.join(RESULTS_DIR, "historicos_loss")
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(LOSS_DIR, exist_ok=True)

SUMMARY_CLASSIC_PATH = os.path.join(RESULTS_DIR, "sumario_classico.csv")
SUMMARY_HYBRID_PATH  = os.path.join(RESULTS_DIR, "sumario_hibrido.csv")
SUMMARY_QUANTUM_PATH = os.path.join(RESULTS_DIR, "sumario_quantico.csv")  # opcional / placeholder

headers_written = {
    SUMMARY_CLASSIC_PATH: os.path.exists(SUMMARY_CLASSIC_PATH),
    SUMMARY_HYBRID_PATH:  os.path.exists(SUMMARY_HYBRID_PATH),
    SUMMARY_QUANTUM_PATH: os.path.exists(SUMMARY_QUANTUM_PATH),
}

# =============================================================================
# VERIFICADORES: pular runs já calculadas
# =============================================================================

def artifact_paths(run_id, models_dir, loss_dir):
    model_save_path = os.path.join(models_dir, f"modelo_{run_id}.pth")
    loss_history_path = os.path.join(loss_dir, f"loss_{run_id}.json")
    return model_save_path, loss_history_path

def summary_has_run(summary_path, run_id):
    """
    Retorna True se o CSV de sumário já contém a run_id.
    Leitura leve (apenas coluna run_id).
    """
    if not os.path.exists(summary_path):
        return False
    try:
        df_ids = pd.read_csv(summary_path, usecols=["run_id"])
        return str(run_id) in set(df_ids["run_id"].astype(str))
    except Exception:
        # Se não conseguir ler, não bloqueia: assume que não está no sumário
        return False

def already_done(config, summary_path, models_dir=MODELS_DIR, loss_dir=LOSS_DIR, require_summary=True):
    """
    Verifica se a run já foi calculada:
      - existem modelo .pth e loss .json?
      - (opcional) já consta no sumário?
    Se config tiver 'force'=True, sempre retorna False (não pula).
    """
    if config.get("force", False):
        return False

    run_id = config["run_id"]
    model_path, loss_path = artifact_paths(run_id, models_dir, loss_dir)

    have_model   = os.path.exists(model_path)
    have_loss    = os.path.exists(loss_path)
    have_summary = summary_has_run(summary_path, run_id) if require_summary else True

    return have_model and have_loss and have_summary

# =============================================================================
# DADOS (gerar uma vez)
# =============================================================================

print("Gerando dados de treino e teste...")
bse = BlackScholes(eps=1e-10)
data_treino = bse.generate_data(seed=2025)
data_teste  = bse.generate_data(seed=42)

# =============================================================================
# LOOP DE TREINAMENTO E AVALIAÇÃO
# =============================================================================

print(f"Iniciando {len(experiment_grid)} experimentos...")
print(f"Resultados clássicos serão salvos em: {SUMMARY_CLASSIC_PATH}")
print(f"Resultados híbridos serão salvos em: {SUMMARY_HYBRID_PATH}")

for config in tqdm(experiment_grid, desc="Total de Experimentos"):

    run_id = config["run_id"]
    print(f"\n--- Iniciando Run: {run_id} ---")

    # Definir sumário alvo de acordo com o tipo de modelo
    model_type = config["model_type"]
    if model_type in ["MLP", "ResNet"]:
        summary_path = SUMMARY_CLASSIC_PATH
    elif model_type == "QPINN":
        summary_path = SUMMARY_HYBRID_PATH
    else:
        print(f"AVISO: Tipo de modelo '{model_type}' não reconhecido. Pulando run.")
        continue

    # Checagem: se já foi feito, pula
    if already_done(config, summary_path, MODELS_DIR, LOSS_DIR, require_summary=True):
        print(f"[SKIP] Artefatos e sumário já existem para '{run_id}'. Pulando cálculo.")
        continue

    # Reprodutibilidade
    seed = config.get('seed', 42)
    tc.manual_seed(seed)
    np.random.seed(seed)

    # Criação do modelo
    try:
        if model_type == "MLP":
            model = MLP(hidden=config['hidden'], blocks=config['blocks'], activation=config['activation'])
        elif model_type == "ResNet":
            model = ResNet(hidden=config['hidden'], blocks=config['blocks'], activation=config['activation'])
        elif model_type == "QPINN":
            qnn = QuantumNeuralNetwork(n_qubits=config['n_qubits'], n_layers=config['n_layers'])
            model = HybridCQN(classical_pre=None, qnn_block=qnn, classical_post=None)
        else:
            print(f"AVISO: Tipo de modelo '{model_type}' não reconhecido. Pulando run.")
            continue
    except Exception as e:
        print(f"Erro ao criar modelo para '{run_id}': {e}. Pulando run.")
        continue

    # Otimizador / treino
    opt = BlackScholeOptimizer(
        data_treino,
        model,
        epochs=config['epochs'],
        lr=config['lr'],
        weights=config.get('weights', [1, 1, 1, 1])
    )

    start_time = time.time()
    loss_history = opt.train(return_loss=True, return_all=False)  # retorna dicionário com chave 'Total'
    end_time = time.time()
    training_time_sec = end_time - start_time

    # Teste
    mse_norm, mse_unorm, _ = opt.test(data_teste, return_unormalized=True)
    num_params = opt.num_params
    print(f"Run '{run_id}' concluída. Tempo: {training_time_sec:.2f}s | MSE (Desnorm): {mse_unorm:.6f} | Parâmetros: {num_params}")

    # Salvar artefatos
    model_save_path, loss_history_path = artifact_paths(run_id, MODELS_DIR, LOSS_DIR)
    tc.save(model.state_dict(), model_save_path)
    try:
        with open(loss_history_path, 'w') as f:
            json.dump(loss_history, f)
    except Exception as e:
        print(f"AVISO: Não foi possível salvar o histórico de loss: {e}")

    # Registrar no sumário (por iteração)
    log_experimento = config.copy()
    log_experimento["mse_teste_normalizado"]     = mse_norm
    log_experimento["mse_teste_desnormalizado"]  = mse_unorm
    log_experimento["num_params"]                = num_params
    log_experimento["training_time_sec"]         = training_time_sec

    if loss_history and 'Total' in loss_history and loss_history['Total']:
        log_experimento["final_total_loss"]   = float(loss_history['Total'][-1])
        log_experimento["mean_last_100_loss"] = float(np.mean(loss_history['Total'][-100:]))
    else:
        log_experimento["final_total_loss"]   = None
        log_experimento["mean_last_100_loss"] = None

    if 'activation' in log_experimento:
        log_experimento['activation'] = str(log_experimento['activation'])

    try:
        df_run = pd.DataFrame([log_experimento])
        write_header = not headers_written[summary_path]
        df_run.to_csv(summary_path, mode='a', header=write_header, index=False)
        headers_written[summary_path] = True
    except Exception as e:
        print(f"ERRO ao salvar o sumário para a run '{run_id}': {e}")

# =============================================================================
# FINAL
# =============================================================================

print("\n--- TODOS OS EXPERIMENTOS CONCLUÍDOS ---")
print("Os sumários foram atualizados em tempo real.")
