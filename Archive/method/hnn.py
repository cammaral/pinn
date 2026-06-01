import torch.nn as nn
import torch as tc
from .qnn import QuantumSequentialNetwork, QuantumNeuralNetwork
from .nn import MLP, ResNet
from utils.device import pick_torch_device


class HybridCQN(nn.Module):
    """
    x -> (classical_pre | identidade) -> Linear(F->n_qubits) -> QNN -> (CQO: Linear->out | CQC: Linear->post_in -> classical_post)
    """
    def __init__(self, classical_pre: nn.Module | None, qnn_block: nn.Module,
                 classical_post: nn.Module | None = None,
                 input_dim: int = 2, output_dim: int = 1, post_in_dim: int | None = None,
                 device: str = "auto", dtype: tc.dtype = tc.float32):
        super().__init__()
        self.pre = classical_pre          # pode ser None
        self.qnn = qnn_block
        self.post = classical_post
        self.n_qubits = qnn_block.n_qubits
        self.n_output = qnn_block.n_output
        self.input_dim = input_dim

        # ===== NOVO: device/dtype =====
        self._torch_device = pick_torch_device(device)
        self._dtype = dtype
        # ==============================

        # F: dimensão de features após classical_pre (ou input_dim se pre=None)
        F = self._infer_out_dim(self.pre, input_dim)

        # Adaptador até o QNN (sem ativação)
        self.to_qubits = nn.Linear(F, self.n_qubits, bias=True)

        if self.post is None:  # C->Q->Output
            self.q_out = nn.Linear(self.n_output, output_dim, bias=True)
        else:                  # C->Q->C
            Dpost = post_in_dim or self._infer_first_linear_in(self.post) or input_dim
            self.decode_to_post = nn.Linear(self.n_qubits, Dpost, bias=True)

        # ===== NOVO: mover módulo inteiro p/ device/dtype =====
        self.to(self._torch_device, dtype=self._dtype)

    @tc.no_grad()
    def _infer_out_dim(self, module: nn.Module | None, in_dim: int) -> int:
        if module is None:
            return in_dim
        # ===== NOVO: garantir device/dtype no tensor dummy =====
        y = module(tc.zeros(1, in_dim, device=self._torch_device, dtype=self._dtype))
        return y.view(1, -1).shape[-1]

    def _infer_first_linear_in(self, module: nn.Module):
        for m in module.modules():
            if isinstance(m, nn.Linear):
                return m.in_features
        return None

    def forward(self, x):
        # ===== NOVO: garantir device/dtype da entrada =====
        x = x.to(self._torch_device, dtype=self._dtype)

        h = x if self.pre is None else self.pre(x)   # (B, F)
        q_in = self.to_qubits(h)                     # (B, n_qubits)  -- sem ativação
        q_out = self.qnn(q_in)                       # (B, n_qubits)
        if self.post is None:                        # CQO
            return self.q_out(q_out)                 # linear, sem ativação
        r_in = self.decode_to_post(q_out)
        return self.post(r_in)                       # CQC
