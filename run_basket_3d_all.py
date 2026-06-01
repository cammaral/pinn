import subprocess
import sys

scripts = [
    "generate_basket_3d_classic.py",
    "generate_basket_3d_qnn.py",
    "generate_basket_3d_hqnn.py",
]

for script in scripts:
    print("=" * 80)
    print(f"Running {script}")
    print("=" * 80)
    subprocess.check_call([sys.executable, script])
