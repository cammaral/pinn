import numpy as np

def MSE(V, U):
    return np.mean((V.reshape(-1) - U.reshape(-1))**2)