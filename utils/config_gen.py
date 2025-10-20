from itertools import product
import numpy as np


def generate_weight_combinations(n=4, dl=3):
    """
    Gera todas as combinações de pesos com valores entre 0 e 1 divididos em `dl` partes,
    excluindo o vetor com todos os zeros.

    Parameters:
        n (int): Tamanho da lista (número de pesos por vetor).
        dl (int): Número de divisões entre 0 e 1 (inclusive).

    Returns:
        list of lists: Combinações possíveis, exceto vetor nulo.
    """
    values = np.linspace(0, 1, dl)
    all_combinations = [list(comb) for comb in product(values, repeat=n)]
    
    # Remove vetores com todos os valores iguais a zero
    filtered = [v for v in all_combinations if not all(x == 0 for x in v)]
    
    return filtered