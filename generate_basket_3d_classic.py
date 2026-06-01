from basket_configs import RESULTS_DIR, DEVICE, BASKET_3D_PARAMS, DATA_3D, CLASSIC_BASE, CLASSIC_SWEEP
from basket_experiment_utils import run_experiment_grid

if __name__ == "__main__":
    run_experiment_grid(CLASSIC_BASE, CLASSIC_SWEEP, BASKET_3D_PARAMS, DATA_3D, RESULTS_DIR, device=DEVICE)
