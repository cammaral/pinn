from utils import get_data
from config import r, K, T, S_max
from config_run import *
from method.qpinn_cv import (
    train_nn, split_train_test, test_model,
    save_model_and_loss, save_V_error, model_already_exists
)

#==================================================
# Método ÚNICO de treino (Etapa 2 + Etapa 3 unificadas)
#==================================================

def executar_treino(N_domains, N_boundarys, N_terminals, n_modes, Ms, MK, list_weights):
    """
    Executa o treino para diferentes seeds, domínios, arquiteturas, inicializações e pesos.
    """
    for seed in seeds:
        for N_domain in N_domains:
            for N_boundary in N_boundarys:
                for N_terminal in N_terminals:
                    print(f'\n=== Seed: {seed} | N_domain: {N_domain} | N_boundary: {N_boundary} | N_terminal: {N_terminal} ===')
                    data_sim = get_data.load_data_sim(r, K, T, S_max, N_domain, N_boundary, N_terminal, seed)

                    S_domain, t_domain = data_sim['domain']
                    S_terminal, t_terminal, V_terminal = data_sim['terminal']
                    S_b0, t_b0, V_b0 = data_sim['boundary_0']
                    S_bmax, t_bmax, V_bmax = data_sim['boundary_max']

                    for seed1 in seeds:
                        S_train, S_test, t_train, t_test = split_train_test(S_domain, t_domain, seed1)
                        V_test = get_data.black_scholes_call_price(S_test, t_test, T, K, r, sigma)
                        for mk in (MK if isinstance(MK, list) else [MK]):
                            for n in (n_modes if isinstance(n_modes, list) else [n_modes]):
                                for m in (Ms if isinstance(Ms, list) else [Ms]):
                                    for weights in (list_weights if isinstance(list_weights[0], list) else [list_weights]):
                                        print(f'>> Qubits: {n} | M: {m} | MK: {mk} | Weights: {weights} | Seed1: {seed1}')
                                        for seed2 in seeds:
                                            # Verificação antes do treino
                                            if model_already_exists(r, K, T, S_max, sigma, n, m, mk,
                                                                    N_domain, N_boundary, N_terminal, epocas, lr,
                                                                    seed, seed1, seed2, weights):
                                                print(f"[✓] Modelo já existe: Seed2={seed2} → pulando.")
                                                continue
                                            
                                            model, LOSS = train_nn(
                                                S_train, t_train, S_terminal, t_terminal, V_terminal,
                                                S_b0, t_b0, V_b0, S_bmax, t_bmax, V_bmax,
                                                sigma=sigma, r=r, n_modes=n_modes, M=m,MK=mk,
                                                gamma=gamma, lr=lr, epocas=epocas,
                                                weights=weights, seed=seed2
                                            )

                                            save_model_and_loss(
                                                model, LOSS, r, K, T, S_max, sigma, n, m, mk, N_domain, N_boundary, N_terminal, epocas, lr,
                                                seed, seed1, seed2,  weights, output_dir="results/model"
                                            )

                                            erro, V_pred = test_model(S_test, t_test, V_test, model)

                                            save_V_error(
                                                erro, V_test, V_pred, r, K, T, S_max, sigma, n, m, mk,
                                                N_domain, N_boundary, N_terminal, epocas, lr,
                                                seed, seed1, seed2, weights, output_dir="results/v_predicted"
                                            )



#==================================================
# Etapa 1: Geração de dados
#==================================================

if generate_data:
    print('\n===================\nInício: GERAÇÃO DE DADOS\n===================')
    for seed in seeds:
        for N_domain in N_domains:
            for N_boundary in N_boundarys:
                for N_terminal in N_terminals:
                    print(f'>> Seed={seed}, N_domain={N_domain}, N_boundary={N_boundary}, N_terminal={N_terminal}')
                    data_sim = get_data.generate_data(r, K, T, S_max, N_domain, N_boundary, N_terminal, seed)
                    get_data.save_data_sim(data_sim, r, K, T, S_max, N_domain, N_boundary, N_terminal, seed)
    print('\n===================\nFim: GERAÇÃO DE DADOS\n===================')


#==================================================
# Chamada única para treino (Etapas 2 e 3)
#==================================================

print('\n===================\nInício: TREINAMENTO PINN (único método)\n===================')
if train_pinn:
    executar_treino(N_domains, N_boundarys, N_terminals, n_modes, Ms, MK, list_weights)
print('\n===================\nFim: TREINAMENTO PINN\n===================')
