import subprocess
import sys

scripts = [
    "run_basket_2d_all.py",
    "run_basket_3d_all.py",
]

for script in scripts:
    print("=" * 80)
    print(f"Running {script}")
    print("=" * 80)
    subprocess.check_call([sys.executable, script])
