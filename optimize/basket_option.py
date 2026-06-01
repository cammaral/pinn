from __future__ import annotations

from typing import Dict, Any, List, Tuple

import numpy as np
import torch as tc
from tqdm import tqdm

from utils.device import pick_torch_device


class BasketOptionOptimizerND:
    """PINN optimizer for N-dimensional correlated arithmetic basket Black-Scholes PDE."""

    def __init__(
        self,
        data: Dict[str, Any],
        model,
        optimizer: str = "Adam",
        lr: float = 1e-3,
        epochs: int = 1000,
        sigmas=None,
        rho=None,
        r: float = 0.05,
        S_max=None,
        T: float = 1.0,
        V_max: float = 120.0,
        weights=None,
        device: str = "auto",
        dtype: tc.dtype = tc.float32,
    ):
        self.data = data
        self.model = model
        self.lr = float(lr)
        self.epochs = int(epochs)
        self.r = float(r)
        self.T = float(T)
        self.V_max = float(V_max)
        self.device = pick_torch_device(device)
        self.dtype = dtype
        self.model = self.model.to(self.device).to(self.dtype)

        self.S, self.t, self.V = data["domain"]
        self.ST, self.tT, self.VT = data["terminal"]
        self.boundaries = data.get("boundaries", [])

        self.n_assets = int(np.asarray(self.S).shape[1])
        self.input_dim = self.n_assets + 1

        self.S_max = np.asarray(S_max if S_max is not None else np.ones(self.n_assets) * 160.0, dtype=float).reshape(self.n_assets)
        self.sigmas = np.asarray(sigmas if sigmas is not None else np.ones(self.n_assets) * 0.2, dtype=float).reshape(self.n_assets)
        self.rho = np.asarray(rho if rho is not None else np.eye(self.n_assets), dtype=float).reshape(self.n_assets, self.n_assets)
        self.weights = np.asarray(weights if weights is not None else np.ones(2 + len(self.boundaries)), dtype=float)

        self.cov = tc.tensor(
            np.outer(self.sigmas, self.sigmas) * self.rho,
            dtype=self.dtype,
            device=self.device,
        )

        self.optimizer = self._get_optimizer(optimizer)
        self.num_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        self.loss = None

    def _get_optimizer(self, optimizer: str):
        if optimizer == "Adam":
            return tc.optim.Adam(self.model.parameters(), lr=self.lr)
        raise ValueError("Optimizer not recognized. Use 'Adam'.")

    def _to_tensor(self, arr, requires_grad=True):
        """
        Convert numpy/list data to a Torch tensor preserving its original 2D shape.

        Important: the original project helper convert_to_tensor() reshapes everything
        to (-1, 1), which is correct for the 1D Black-Scholes code but wrong here.
        For basket options, S has shape (N_points, n_assets), while t and V have
        shape (N_points, 1). Flattening S would turn (1000, 2) into (2000, 1) and
        break torch.cat([S_norm, t_norm], dim=1).
        """
        t = tc.as_tensor(arr, dtype=self.dtype, device=self.device)
        return t.clone().detach().requires_grad_(bool(requires_grad))

    def _normalize_S(self, S):
        return np.asarray(S, dtype=float) / self.S_max.reshape(1, -1)

    def _normalize_t(self, t):
        return np.asarray(t, dtype=float) / self.T

    def _normalize_V(self, V):
        return np.asarray(V, dtype=float) / self.V_max

    def derivatives(self, Vpred, S_norm, t_norm):
        ones = tc.ones_like(Vpred, device=self.device, dtype=self.dtype)
        dV_dt = tc.autograd.grad(Vpred, t_norm, grad_outputs=ones, create_graph=True, retain_graph=True)[0]
        dV_dS = tc.autograd.grad(Vpred, S_norm, grad_outputs=ones, create_graph=True, retain_graph=True)[0]

        hessian_cols = []
        for i in range(self.n_assets):
            grad_i = dV_dS[:, i:i+1]
            h_i = tc.autograd.grad(
                grad_i,
                S_norm,
                grad_outputs=tc.ones_like(grad_i),
                create_graph=True,
                retain_graph=True,
            )[0]
            hessian_cols.append(h_i)
        # H[b,i,j] = d/dS_j (dV/dS_i)
        H = tc.stack(hessian_cols, dim=1)
        return dV_dt, dV_dS, H

    def pde_loss(self, Vpred, S_norm, t_norm):
        dV_dt, dV_dS, H = self.derivatives(Vpred, S_norm, t_norm)
        drift = self.r * tc.sum(S_norm * dV_dS, dim=1, keepdim=True)

        diffusion = tc.zeros_like(Vpred)
        for i in range(self.n_assets):
            for j in range(self.n_assets):
                diffusion = diffusion + 0.5 * self.cov[i, j] * S_norm[:, i:i+1] * S_norm[:, j:j+1] * H[:, i, j:j+1]

        # normalized variables: t_norm = t / T, V_norm = V / V_max
        # Physical PDE divided by V_max gives (1/T) dV_norm/dt_norm + ...
        residual = (dV_dt / self.T) + diffusion + drift - self.r * Vpred
        return tc.mean(residual ** 2)

    def train(self, normalize: bool = True, return_loss: bool = False, return_all: bool = False):
        if not normalize:
            raise NotImplementedError("This optimizer expects normalized training, matching the project convention.")

        LOSS = {"Total": [], "pde_loss": [], "terminal_loss": [], "boundary_loss": []}
        for _, _, _, name in self.boundaries:
            LOSS[f"boundary_{name}_loss"] = []

        S_norm = self._to_tensor(self._normalize_S(self.S), requires_grad=True)
        t_norm = self._to_tensor(self._normalize_t(self.t), requires_grad=True)
        V_norm = self._to_tensor(self._normalize_V(self.V), requires_grad=False)

        ST_norm = self._to_tensor(self._normalize_S(self.ST), requires_grad=False)
        tT_norm = self._to_tensor(self._normalize_t(self.tT), requires_grad=False)
        VT_norm = self._to_tensor(self._normalize_V(self.VT), requires_grad=False)

        boundary_tensors = []
        for S_b, t_b, V_b, name in self.boundaries:
            boundary_tensors.append((
                self._to_tensor(self._normalize_S(S_b), requires_grad=False),
                self._to_tensor(self._normalize_t(t_b), requires_grad=False),
                self._to_tensor(self._normalize_V(V_b), requires_grad=False),
                name,
            ))

        for _ in tqdm(range(self.epochs), desc="Training basket"):
            self.optimizer.zero_grad()

            x_domain = tc.cat([S_norm, t_norm], dim=1)
            Vpred = self.model(x_domain).reshape(-1, 1)
            pde = self.pde_loss(Vpred, S_norm, t_norm)

            VTpred = self.model(tc.cat([ST_norm, tT_norm], dim=1)).reshape(-1, 1)
            terminal_loss = tc.mean((VTpred - VT_norm) ** 2)

            boundary_losses = []
            for S_b_norm, t_b_norm, V_b_norm, name in boundary_tensors:
                Vb_pred = self.model(tc.cat([S_b_norm, t_b_norm], dim=1)).reshape(-1, 1)
                boundary_losses.append((tc.mean((Vb_pred - V_b_norm) ** 2), name))

            boundary_total = sum(x[0] for x in boundary_losses) / max(len(boundary_losses), 1)
            self.loss = pde + terminal_loss + boundary_total
            self.loss.backward()
            self.optimizer.step()

            if return_loss:
                LOSS["Total"].append(float(self.loss.detach().cpu().item()))
                LOSS["pde_loss"].append(float(pde.detach().cpu().item()))
                LOSS["terminal_loss"].append(float(terminal_loss.detach().cpu().item()))
                LOSS["boundary_loss"].append(float(boundary_total.detach().cpu().item()))
                for val, name in boundary_losses:
                    LOSS[f"boundary_{name}_loss"].append(float(val.detach().cpu().item()))

        return LOSS if return_loss else None

    def test(self, data, normalize: bool = True, return_unormalized: bool = False):
        if not normalize:
            raise NotImplementedError("This optimizer expects normalized test inputs.")

        S_test, t_test, V_test = data["domain"]
        S_norm = self._to_tensor(self._normalize_S(S_test), requires_grad=False)
        t_norm = self._to_tensor(self._normalize_t(t_test), requires_grad=False)
        V_norm = self._to_tensor(self._normalize_V(V_test), requires_grad=False)

        self.model.eval()
        with tc.no_grad():
            pred = self.model(tc.cat([S_norm, t_norm], dim=1)).reshape(-1, 1)

        mse_norm = tc.mean((pred - V_norm) ** 2).item()
        pred_un = pred * self.V_max
        V_un = V_norm * self.V_max
        mse_un = tc.mean((pred_un - V_un) ** 2).item()

        if return_unormalized:
            return mse_norm, mse_un, pred_un.detach().cpu().numpy()
        return mse_norm, pred_un.detach().cpu().numpy()
