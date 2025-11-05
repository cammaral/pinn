import torch as tc


def pick_torch_device(prefer: str = "auto"):
    prefer = prefer.lower()
    if prefer == "cuda":
        return tc.device("cuda" if tc.cuda.is_available() else "cpu")
    if prefer == "cpu":
        return tc.device("cpu")
    # auto
    return tc.device("cuda" if tc.cuda.is_available() else "cpu")

def pick_pl_backend(device_pref: str = "auto"):
    d = (device_pref or "auto").lower()
    if d == "cuda" and tc.cuda.is_available():
        return "lightning.gpu"
    if d == "auto" and tc.cuda.is_available():
        return "lightning.gpu"
    return "default.qubit"