import subprocess
import sys

scripts = [
    "generate_basket_2d_classic.py",
    "generate_basket_2d_qnn.py",
    "generate_basket_2d_hqnn.py",
]

for script in scripts:
    print("=" * 80)
    print(f"Running {script}")
    print("=" * 80)
    subprocess.check_call([sys.executable, script])
