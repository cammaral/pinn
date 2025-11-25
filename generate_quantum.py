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
# (Assumindo que estão no PYTHONPATH ou na mesma pasta)
from equation.option_pricing import BlackScholes
from optimize.option_princing import BlackScholeOptimizer
from method.nn import MLP, ResNet
from method.hnn import HybridCQN 
from method.qnn import QuantumNeuralNetwork, CorrelatorQuantumNeuralNetwork
from utils.save import *

# =============================================================================
# FUNÇÕES AUXILIARES
# =============================================================================

def generate_runs(base_config, sweep_params):
    """
    Função auxiliar para gerar uma lista de experimentos (runs).
    
    Ela pega uma 'base_config' (comum a todas as runs) e um 
    'sweep_params' (dicionário com listas de valores para variar)
    e cria todas as combinações.
    """
    generated_experiments = []
    param_keys = sweep_params.keys()
    param_values = sweep_params.values()
    
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
            
            key_str = key.replace("learning_rate", "lr").replace("hidden", "n").replace("layers", "l").replace("hidden", "h").replace("blocks", "b")
            run_id_parts.append(f"{key_str}_{val_str}")

        new_config.update(param_combination_dict)
        new_config["run_id"] = "_".join(run_id_parts)
        generated_experiments.append(new_config)
        
    return generated_experiments

def pretty_print(config_list, num_to_show=5):
    """Função para imprimir a grade de experimentos de forma legível."""
    for cfg in config_list[:num_to_show]:
        printable_cfg = {}
        for k, v in cfg.items():
            if isinstance(v, nn.Module):
                printable_cfg[k] = v.__class__.__name__
            else:
                printable_cfg[k] = v
        print(json.dumps(printable_cfg, indent=2))
    if len(config_list) > num_to_show:
        print(f"... e mais {len(config_list) - num_to_show} outros.")

# =============================================================================
# 1. DEFINIÇÃO DOS EXPERIMENTOS (Forma Dinâmica)
# =============================================================================

# Lista final que será usada pelo script
experiment_grid = []

# --- GRUPO 4: Testando efeito da Seed (Estabilidade) ---

device = 'cpu'
base_seed_test = {
    "model_type": "CQNN",
    "run_id_prefix": "cqnn_basic",
    "lr": 2e-3,
    "epochs": 15000,
    "activation": None, #nn.Tanh(),
    'entangler': 'basic'

}

sweep_seed = {
    "n_qubits": [4],
    "k": [2, 3],
    "n_vertex": [2, 3, 4, 5],
    #"n_layers": [1, 2, 3, 5],
    "n_layers": [3],
    "seed": [1924, 1925, 1926, 1973, 2025, 2024, 2012, 1958, 1962, 1997]
}

"""

sweep_seed = {
    "n_qubits": [2, 4],
    "n_layers": [3],
    #"seed": [1973, 2025, 2024, 2012, 1958, 1962, 1997]
    "seed": [1924, 1925, 1926, 1973, 2025, 2024, 2012, 1958, 1962, 1997]
}

base_seed_test = {
    "model_type": "QNN",
    "run_id_prefix": "qnn_basic",
    "lr": 2e-3,
    "epochs": 15000,
    "activation": None, #nn.Tanh(),
    'entangler': 'basic'
}
"""
experiment_grid.extend(generate_runs(base_seed_test, sweep_seed))


# --- Verificação ---
print(f"Total de {len(experiment_grid)} experimentos gerados para a grade.")
pretty_print(experiment_grid, num_to_show=3)



# =============================================================================
# 2. CONFIGURAÇÃO DO AMBIENTE
# =============================================================================

# Criar pastas para salvar os resultados
RESULTS_DIR = "experimentos_pinn"
MODELS_DIR = os.path.join(RESULTS_DIR, "modelos_salvos")
LOSS_DIR = os.path.join(RESULTS_DIR, "historicos_loss") 
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(LOSS_DIR, exist_ok=True) 

# Caminhos dos arquivos de sumário
SUMMARY_CLASSIC_PATH = os.path.join(RESULTS_DIR, "sumario_classico.csv")
SUMMARY_HYBRID_PATH = os.path.join(RESULTS_DIR, "sumario_hibrido.csv")
SUMMARY_QUANTUM_PATH = os.path.join(RESULTS_DIR, "sumario_quantico.csv")
SUMMARY_CQUANTUM_PATH = os.path.join(RESULTS_DIR, "sumario_cquantico.csv")  # Placeholder

# Dicionário para rastrear se o cabeçalho já foi escrito
headers_written = {
    SUMMARY_CLASSIC_PATH: os.path.exists(SUMMARY_CLASSIC_PATH),
    SUMMARY_HYBRID_PATH: os.path.exists(SUMMARY_HYBRID_PATH),
    SUMMARY_QUANTUM_PATH: os.path.exists(SUMMARY_QUANTUM_PATH),
    SUMMARY_CQUANTUM_PATH: os.path.exists(SUMMARY_CQUANTUM_PATH)
}

# Gerar os dados de treino e teste UMA VEZ
print("Gerando dados de treino e teste...")
bse = BlackScholes(eps=1e-10)
data_treino = bse.generate_data(seed=2025)
data_teste = bse.generate_data(seed=42)

# =============================================================================
# VERIFICADORES: pular runs já calculadas
# =============================================================================


# =============================================================================
# 3. LOOP DE TREINAMENTO E AVALIAÇÃO (MODIFICADO)
# =============================================================================
print(f"Iniciando {len(experiment_grid)} experimentos...")
print(f"Resultados clássicos serão salvos em: {SUMMARY_CLASSIC_PATH}")
print(f"Resultados híbridos serão salvos em: {SUMMARY_HYBRID_PATH}")

for config in tqdm(experiment_grid, desc="Total de Experimentos"):
    
    run_id = config["run_id"]
    model_type = config["model_type"]
    summary_path = resolve_summary_path(model_type)
    print(f"\n--- Iniciando Run: {run_id} ---")
    if run_already_done(run_id, summary_path=summary_path, model_dir=MODELS_DIR, loss_dir=LOSS_DIR):
        print(f"Pulando run '{run_id}' — já encontrada em sumário/artefatos.")
        continue
    # --- A. Garantir Reprodutibilidade (Aceita loop de seed) ---
    seed = config.get('seed', 42)
    tc.manual_seed(seed)
    np.random.seed(seed)

    # --- B. Criar o Modelo (Fábrica de Modelos) ---
    model_type = config["model_type"]
    model = None
    #summary_path = None 

    try:
        if model_type == "MLP":
            # <<< CORREÇÃO AQUI >>> (Estava usando 'hidden' e 'blocks' por engano)
            model = MLP(hidden=config['hidden'], blocks=config['blocks'], device=device,
                        activation=config['activation'])
            summary_path = SUMMARY_CLASSIC_PATH
        
        elif model_type == "ResNet":
            model = ResNet(hidden=config['hidden'], blocks=config['blocks'], device=device,
                           activation=config['activation'])
            summary_path = SUMMARY_CLASSIC_PATH
        
        elif model_type == "QNN": 
            qnn = QuantumNeuralNetwork(n_qubits=config['n_qubits'], 
                                       n_layers=config['n_layers'], device=device,
                                       entangler=config.get('entangler'))
            model = HybridCQN(classical_pre=None, qnn_block=qnn,device=device, classical_post=None)
            summary_path = SUMMARY_QUANTUM_PATH
        elif model_type == "CQNN": 
            qnn = CorrelatorQuantumNeuralNetwork(n_qubits=config['n_qubits'], 
                                       n_layers=config['n_layers'],
                                       k=config['k'],
                                       n_vertex=config['n_vertex'],
                                       nonlinear=False, device=device,
                                       entangler=config.get('entangler'))
            model = HybridCQN(classical_pre=None, qnn_block=qnn, device=device,classical_post=None)
            summary_path = SUMMARY_CQUANTUM_PATH
        elif model_type == "CQNN_nonlinear": 
            
            qnn = CorrelatorQuantumNeuralNetwork(n_qubits=config['n_qubits'], 
                                       n_layers=config['n_layers'],
                                       k=config['k'],
                                       n_vertex=config['n_vertex'],
                                       nonlinear=True, device=device,
                                       entangler=config.get('entangler'))
            model = HybridCQN(classical_pre=None, qnn_block=qnn,device=device, classical_post=None)
            summary_path = SUMMARY_CQUANTUM_PATH
        else:
            print(f"AVISO: Tipo de modelo '{model_type}' não reconhecido. Pulando run.")
            continue
    
    except Exception as e:
        print(f"Erro ao criar modelo para '{run_id}': {e}. Pulando run.")
        continue

    # --- C. Treinar o Modelo ---
    opt = BlackScholeOptimizer(
        data_treino, 
        model, 
        epochs=config['epochs'], 
        lr=config['lr'],
        device = device,
        weights=config.get('weights', [1,1,1,1])
    )
    
    start_time = time.time()
    
    # (return_all=False está OK, pois você só quer a loss total)
    loss_history = opt.train(return_loss=True, return_all=False) 
    
    end_time = time.time()
    training_time_sec = end_time - start_time
    
    # --- D. Testar e Coletar Métricas ---
    
    # <<< CORREÇÃO AQUI >>> (opt.test retorna 3 valores)
    mse_norm, mse_unorm, _ = opt.test(data_teste, return_unormalized=True)
    num_params = opt.num_params
    
    # <<< CORREÇÃO AQUI >>> (Exibindo o mse_unorm)
    print(f"Run '{run_id}' concluída. Tempo: {training_time_sec:.2f}s | MSE (Desnorm): {mse_unorm:.6f} | Parâmetros: {num_params}")

    # --- E. Salvar Artefatos (Modelo e Histórico de Loss) ---
    model_save_path = os.path.join(MODELS_DIR, f"modelo_{run_id}.pth")
    tc.save(model.state_dict(), model_save_path)
    
    # (Isto já estava salvando na pasta separada 'LOSS_DIR', como pedido)
    loss_history_path = os.path.join(LOSS_DIR, f"loss_{run_id}.json")
    try:
        with open(loss_history_path, 'w') as f:
            json.dump(loss_history, f)
    except Exception as e:
        print(f"AVISO: Não foi possível salvar o histórico de loss: {e}")

    # --- F. Salvar Resultados no Sumário (POR ITERAÇÃO) ---
    log_experimento = config.copy()
    
    # Adiciona métricas de teste e performance
    # <<< CORREÇÃO AQUI >>> (Logando ambos MSEs)
    log_experimento["mse_teste_normalizado"] = mse_norm
    log_experimento["mse_teste_desnormalizado"] = mse_unorm
    log_experimento["num_params"] = num_params
    log_experimento["training_time_sec"] = training_time_sec
    
    # Adiciona caminhos para artefatos
    log_experimento["model_path"] = model_save_path
    log_experimento["loss_history_path"] = loss_history_path

    # <<< MUDANÇA SOLICITADA AQUI >>>
    # Adiciona métricas de loss final
    if loss_history and 'Total' in loss_history and loss_history['Total']:
        # 1. Adiciona o último valor da loss
        log_experimento["final_total_loss"] = loss_history['Total'][-1]
        
        # 2. Adiciona a média dos últimos 100 valores
        # (np.mean lida automaticamente se houver menos de 100 épocas)
        mean_last_100_loss = np.mean(loss_history['Total'][-100:])
        log_experimento["mean_last_100_loss"] = mean_last_100_loss
    else:
        log_experimento["final_total_loss"] = None
        log_experimento["mean_last_100_loss"] = None
    
    # Limpa objetos não-serializáveis
    if 'activation' in log_experimento:
        log_experimento['activation'] = str(log_experimento['activation'])
    
    # Salva esta run no arquivo CSV apropriado
    try:
        df_run = pd.DataFrame([log_experimento])
        write_header = not headers_written[summary_path]
        df_run.to_csv(summary_path, mode='a', header=write_header, index=False)
        headers_written[summary_path] = True 
        
    except Exception as e:
        print(f"ERRO ao salvar o sumário para a run '{run_id}': {e}")
        
# =============================================================================
# 4. CONCLUSÃO
# =============================================================================

print("\n--- TODOS OS EXPERIMENTOS CONCLUÍDOS ---")
print("Os sumários foram atualizados em tempo real.")
