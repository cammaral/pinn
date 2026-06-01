from basket_configs import RESULTS_DIR, DEVICE, BASKET_2D_PARAMS, DATA_2D, HQNN_BASE, HQNN_SWEEP_2D
from basket_experiment_utils import run_experiment_grid

if __name__ == "__main__":
    run_experiment_grid(HQNN_BASE, HQNN_SWEEP_2D, BASKET_2D_PARAMS, DATA_2D, RESULTS_DIR, device=DEVICE)
