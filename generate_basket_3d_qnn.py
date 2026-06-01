from basket_configs import RESULTS_DIR, DEVICE, BASKET_3D_PARAMS, DATA_3D, QNN_BASE, QNN_SWEEP_3D
from basket_experiment_utils import run_experiment_grid

if __name__ == "__main__":
    run_experiment_grid(QNN_BASE, QNN_SWEEP_3D, BASKET_3D_PARAMS, DATA_3D, RESULTS_DIR, device=DEVICE)
