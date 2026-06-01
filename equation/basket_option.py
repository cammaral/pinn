from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np
from numpy.polynomial.hermite import hermgauss


class ArithmeticBasketOption:
    """
    N-dimensional arithmetic basket call option under correlated Black-Scholes.

    Payoff:
        max(sum_i w_i S_i - K, 0)

    Benchmark:
        Gauss-Hermite quadrature for the conditional expectation.

    Important for the project:
        - The Gauss-Hermite benchmark is cached and reused.
        - The generated train/test data follow the same style as BlackScholes.generate_data().
        - Data dictionary keys are: domain, terminal, boundaries.
    """

    def __init__(
        self,
        n_assets: int,
        S_max=160.0,
        T: float = 1.0,
        K: float = 40.0,
        r: float = 0.05,
        sigmas=None,
        rho=None,
        weights=None,
        option_type: str = "call",
        V_max: float | None = None,
        gh_order: int = 11,
        cache_dir: str | Path = "experimentos_pinn_basket/benchmarks",
        eps: float = 1e-10,
    ):
        self.n_assets = int(n_assets)
        self.T = float(T)
        self.K = float(K)
        self.r = float(r)
        self.option_type = option_type
        self.gh_order = int(gh_order)
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.eps = float(eps)

        if np.isscalar(S_max):
            self.S_max = np.full(self.n_assets, float(S_max), dtype=float)
        else:
            self.S_max = np.asarray(S_max, dtype=float).reshape(self.n_assets)

        if sigmas is None:
            sigmas = np.full(self.n_assets, 0.2)
        self.sigmas = np.asarray(sigmas, dtype=float).reshape(self.n_assets)

        if rho is None:
            rho = np.eye(self.n_assets)
        self.rho = np.asarray(rho, dtype=float).reshape(self.n_assets, self.n_assets)

        if weights is None:
            weights = np.full(self.n_assets, 1.0 / self.n_assets)
        self.weights = np.asarray(weights, dtype=float).reshape(self.n_assets)
        self.weights = self.weights / self.weights.sum()

        # V_max follows your 1D project convention. For average basket, max payoff is max(avg(Smax)-K,0).
        if V_max is None:
            V_max = max(float(np.dot(self.weights, self.S_max) - self.K), 1.0)
        self.V_max = float(V_max)

        self._validate()
        self._gh_z, self._gh_w = self._build_gh_nodes()

    def _validate(self) -> None:
        if self.option_type != "call":
            raise NotImplementedError("Only arithmetic basket call is implemented in this package.")
        if self.rho.shape != (self.n_assets, self.n_assets):
            raise ValueError("rho must be N x N.")
        if not np.allclose(self.rho, self.rho.T, atol=1e-10):
            raise ValueError("rho must be symmetric.")
        eig = np.linalg.eigvalsh(self.rho)
        if np.min(eig) <= 0:
            raise ValueError(f"rho must be positive definite. Min eigenvalue={np.min(eig)}")
        if np.any(self.sigmas <= 0):
            raise ValueError("All sigmas must be positive.")

    def _params_for_hash(self) -> Dict:
        return {
            "n_assets": self.n_assets,
            "S_max": self.S_max.tolist(),
            "T": self.T,
            "K": self.K,
            "r": self.r,
            "sigmas": self.sigmas.tolist(),
            "rho": self.rho.tolist(),
            "weights": self.weights.tolist(),
            "option_type": self.option_type,
            "V_max": self.V_max,
            "gh_order": self.gh_order,
        }

    def config_hash(self, extra: Dict | None = None) -> str:
        payload = self._params_for_hash()
        if extra:
            payload.update(extra)
        raw = json.dumps(payload, sort_keys=True).encode("utf-8")
        return hashlib.md5(raw).hexdigest()[:12]

    def _build_gh_nodes(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns:
            z_corr: [M,N] correlated standard normal GH nodes
            w:      [M] GH weights scaled for E[f(Z)]
        """
        x_1d, w_1d = hermgauss(self.gh_order)
        mesh_x = np.meshgrid(*([x_1d] * self.n_assets), indexing="ij")
        mesh_w = np.meshgrid(*([w_1d] * self.n_assets), indexing="ij")

        x = np.stack([m.reshape(-1) for m in mesh_x], axis=1)  # [M,N]
        w = np.prod(np.stack([m.reshape(-1) for m in mesh_w], axis=1), axis=1)
        w = w / (np.pi ** (self.n_assets / 2.0))

        # Hermite integrates exp(-x^2), so standard normal node is sqrt(2)*x.
        z_ind = np.sqrt(2.0) * x
        L = np.linalg.cholesky(self.rho)
        z_corr = z_ind @ L.T
        return z_corr.astype(float), w.astype(float)

    def payoff(self, S: np.ndarray) -> np.ndarray:
        S = np.asarray(S, dtype=float).reshape(-1, self.n_assets)
        basket = S @ self.weights
        return np.maximum(basket - self.K, 0.0).reshape(-1, 1)

    def price_gh(self, S: np.ndarray, t: np.ndarray, batch_size: int = 256) -> np.ndarray:
        """
        Gauss-Hermite price for arbitrary points.

        S: [B,N]
        t: [B,1] where t is PDE time, tau=T-t.
        """
        S = np.asarray(S, dtype=float).reshape(-1, self.n_assets)
        t = np.asarray(t, dtype=float).reshape(-1, 1)
        out = np.zeros((S.shape[0], 1), dtype=float)

        Z = self._gh_z  # [M,N]
        W = self._gh_w  # [M]

        for start in range(0, S.shape[0], batch_size):
            end = min(start + batch_size, S.shape[0])
            Sb = S[start:end]
            tb = t[start:end]
            tau = np.maximum(self.T - tb, 0.0)  # [B,1]

            at_maturity = tau.reshape(-1) <= self.eps
            if np.all(at_maturity):
                out[start:end] = self.payoff(Sb)
                continue

            # [B,N]
            drift = (self.r - 0.5 * self.sigmas**2) * tau
            vol = self.sigmas * np.sqrt(np.maximum(tau, self.eps))

            # ST [B,M,N]
            exponent = drift[:, None, :] + vol[:, None, :] * Z[None, :, :]
            ST = Sb[:, None, :] * np.exp(exponent)
            basket_T = np.tensordot(ST, self.weights, axes=([2], [0]))  # [B,M]
            payoff = np.maximum(basket_T - self.K, 0.0)
            price = np.exp(-self.r * tau.reshape(-1)) * (payoff @ W)

            if np.any(at_maturity):
                price[at_maturity] = self.payoff(Sb[at_maturity]).reshape(-1)

            out[start:end, 0] = price

        return out

    def finite_difference_greeks(
        self,
        S: np.ndarray,
        t: np.ndarray,
        h_rel: float = 1e-3,
        h_t: float = 1e-4,
    ) -> Dict[str, np.ndarray]:
        """
        Benchmark Greeks via finite differences around the GH price.
        This is intended for evaluation/cache, not for every training iteration.
        """
        S = np.asarray(S, dtype=float).reshape(-1, self.n_assets)
        t = np.asarray(t, dtype=float).reshape(-1, 1)
        B = S.shape[0]

        V0 = self.price_gh(S, t).reshape(-1)
        delta = np.zeros((B, self.n_assets), dtype=float)
        gamma = np.zeros((B, self.n_assets, self.n_assets), dtype=float)

        hS = np.maximum(h_rel * self.S_max, 1e-4)

        for i in range(self.n_assets):
            Sp = S.copy(); Sm = S.copy()
            Sp[:, i] += hS[i]
            Sm[:, i] = np.maximum(Sm[:, i] - hS[i], self.eps)
            Vp = self.price_gh(Sp, t).reshape(-1)
            Vm = self.price_gh(Sm, t).reshape(-1)
            delta[:, i] = (Vp - Vm) / (Sp[:, i] - Sm[:, i])
            gamma[:, i, i] = (Vp - 2.0 * V0 + Vm) / (0.5 * (Sp[:, i] - Sm[:, i]))**2

        for i in range(self.n_assets):
            for j in range(i + 1, self.n_assets):
                h_i = hS[i]
                h_j = hS[j]
                Spp = S.copy(); Spm = S.copy(); Smp = S.copy(); Smm = S.copy()
                Spp[:, i] += h_i; Spp[:, j] += h_j
                Spm[:, i] += h_i; Spm[:, j] = np.maximum(Spm[:, j] - h_j, self.eps)
                Smp[:, i] = np.maximum(Smp[:, i] - h_i, self.eps); Smp[:, j] += h_j
                Smm[:, i] = np.maximum(Smm[:, i] - h_i, self.eps); Smm[:, j] = np.maximum(Smm[:, j] - h_j, self.eps)
                Vpp = self.price_gh(Spp, t).reshape(-1)
                Vpm = self.price_gh(Spm, t).reshape(-1)
                Vmp = self.price_gh(Smp, t).reshape(-1)
                Vmm = self.price_gh(Smm, t).reshape(-1)
                gij = (Vpp - Vpm - Vmp + Vmm) / (4.0 * h_i * h_j)
                gamma[:, i, j] = gij
                gamma[:, j, i] = gij

        tp = np.minimum(t + h_t, self.T)
        tm = np.maximum(t - h_t, 0.0)
        Vtp = self.price_gh(S, tp).reshape(-1)
        Vtm = self.price_gh(S, tm).reshape(-1)
        denom = (tp - tm).reshape(-1)
        theta = (Vtp - Vtm) / np.maximum(denom, self.eps)

        return {"V": V0.reshape(-1, 1), "delta": delta, "gamma": gamma, "theta": theta.reshape(-1, 1)}

    def generate_data(
        self,
        N_domain: int = 1000,
        N_terminal: int = 500,
        N_boundary: int = 400,
        seed: int = 1924,
        cache: bool = True,
        tag: str = "train",
    ) -> Dict:
        """
        Generates domain, terminal and boundary samples.
        GH values are cached once for a fixed configuration and seed.
        """
        extra = {
            "N_domain": int(N_domain),
            "N_terminal": int(N_terminal),
            "N_boundary": int(N_boundary),
            "seed": int(seed),
            "tag": tag,
        }
        cache_path = self.cache_dir / f"basket_data_N{self.n_assets}_{self.config_hash(extra)}.npz"

        if cache and cache_path.exists():
            z = np.load(cache_path, allow_pickle=True)
            boundaries = []
            n_bound = int(z["n_boundaries"])
            for k in range(n_bound):
                boundaries.append((z[f"B{k}_S"], z[f"B{k}_t"], z[f"B{k}_V"], str(z[f"B{k}_name"])))
            return {
                "domain": (z["S_domain"], z["t_domain"], z["V_domain"]),
                "terminal": (z["S_terminal"], z["t_terminal"], z["V_terminal"]),
                "boundaries": boundaries,
                "metadata": json.loads(str(z["metadata_json"])),
            }

        rng = np.random.default_rng(seed)

        S_domain = rng.uniform(0.0, self.S_max, size=(int(N_domain), self.n_assets))
        t_domain = rng.uniform(0.0, self.T, size=(int(N_domain), 1))
        V_domain = self.price_gh(S_domain, t_domain)

        S_terminal = rng.uniform(0.0, self.S_max, size=(int(N_terminal), self.n_assets))
        t_terminal = self.T * np.ones((int(N_terminal), 1))
        V_terminal = self.payoff(S_terminal)

        boundaries = []
        n_per_boundary = max(1, int(N_boundary) // (2 * self.n_assets))
        for i in range(self.n_assets):
            for side, value in [("low", 0.0), ("high", self.S_max[i])]:
                S_b = rng.uniform(0.0, self.S_max, size=(n_per_boundary, self.n_assets))
                S_b[:, i] = value
                t_b = rng.uniform(0.0, self.T, size=(n_per_boundary, 1))
                V_b = self.price_gh(S_b, t_b)
                boundaries.append((S_b, t_b, V_b, f"S{i+1}_{side}"))

        metadata = {
            "class": "ArithmeticBasketOption",
            "params": self._params_for_hash(),
            "data": extra,
            "cache_path": str(cache_path),
        }

        if cache:
            payload = {
                "S_domain": S_domain,
                "t_domain": t_domain,
                "V_domain": V_domain,
                "S_terminal": S_terminal,
                "t_terminal": t_terminal,
                "V_terminal": V_terminal,
                "n_boundaries": len(boundaries),
                "metadata_json": json.dumps(metadata),
            }
            for k, (S_b, t_b, V_b, name) in enumerate(boundaries):
                payload[f"B{k}_S"] = S_b
                payload[f"B{k}_t"] = t_b
                payload[f"B{k}_V"] = V_b
                payload[f"B{k}_name"] = name
            np.savez_compressed(cache_path, **payload)

        return {
            "domain": (S_domain, t_domain, V_domain),
            "terminal": (S_terminal, t_terminal, V_terminal),
            "boundaries": boundaries,
            "metadata": metadata,
        }
