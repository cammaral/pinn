import torch as tc
from tqdm import tqdm
from utils.get_data import convert_to_tensor
from utils.device import pick_torch_device


class BlackScholeOptimizer:
    def __init__(
        self,
        data,
        model,
        optimizer='Adam',
        lr=1e-2,
        epochs=200,
        sigma=0.02,
        r=0.05,
        weights=[1, 1, 1, 1],
        device: str = "auto",
        dtype=tc.float32,
    ):
        self.sigma = sigma
        self.model = model
        self.lr = float(lr)
        self.r = r
        self.epochs = epochs
        self.weights = weights

        self.device = pick_torch_device(device)
        self.dtype = dtype
        self.model = self.model.to(self.device).to(self.dtype)

        self.S, self.t, self.V = data['domain']
        self.ST, self.tT, self.VT = data['terminal']
        self.S0, self.t0, self.V0 = data['b0']
        self.Smax, self.tmax, self.Vmax = data['bmax']

        self.optimizer = self._get_optimizer(optimizer)
        self.num_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        self.loss = None

    def _get_optimizer(self, optimizer):
        if optimizer == 'Adam':
            return tc.optim.Adam(params=self.model.parameters(), lr=self.lr)
        raise ValueError("Optimizer not recognized. Use 'Adam' or 'LBFGS'.")

    def derivatives(self, Vest, _S, _t):
        ones = tc.ones_like(Vest, device=self.device, dtype=self.dtype)
        df_dt = tc.autograd.grad(Vest, _t, grad_outputs=ones, create_graph=True)[0]
        df_ds = tc.autograd.grad(Vest, _S, grad_outputs=ones, create_graph=True)[0]
        d2f_d2s = tc.autograd.grad(df_ds, _S, grad_outputs=ones, create_graph=True)[0]
        return df_dt, df_ds, d2f_d2s

    def loss_function(self, Vest, Vgiven, df, _S, return_all=False):
        _VT, _V0, _Vmax = Vgiven
        df_dt, df_ds, d2f_d2s = df
        Vpred, VTpred, V0pred, Vmaxpred = Vest

        pde_loss = tc.mean(
            (df_dt + 0.5 * self.sigma**2 * _S**2 * d2f_d2s + self.r * _S * df_ds - self.r * Vpred) ** 2
        )
        terminal_loss = tc.mean((VTpred - _VT) ** 2)
        boundary_0_loss = tc.mean((V0pred - _V0) ** 2)
        boundary_max_loss = tc.mean((Vmaxpred - _Vmax) ** 2)

        loss = (
            self.weights[0] * pde_loss
            + self.weights[1] * terminal_loss
            + self.weights[2] * boundary_0_loss
            + self.weights[3] * boundary_max_loss
        )

        if return_all:
            return loss, pde_loss, terminal_loss, boundary_0_loss, boundary_max_loss
        return loss

    def test(self, data, S_max=160, T=1.0, V_max=120, normalize=True, return_unormalized=False, data_new=False):
        if not normalize:
            S_max = 1
            T = 1
            V_max = 1

        if data_new:
            _S_test = convert_to_tensor(
                data[0] / S_max, requires_grad=False, device=self.device, dtype=self.dtype
            ).reshape(-1, 1)
            _t_test = convert_to_tensor(
                data[1] / T, requires_grad=False, device=self.device, dtype=self.dtype
            ).reshape(-1, 1)
            _V_test = convert_to_tensor(
                data[2] / V_max, requires_grad=False, device=self.device, dtype=self.dtype
            ).reshape(-1, 1)
        else:
            _S_test = convert_to_tensor(
                data['domain'][0] / S_max, requires_grad=False, device=self.device, dtype=self.dtype
            )
            _t_test = convert_to_tensor(
                data['domain'][1] / T, requires_grad=False, device=self.device, dtype=self.dtype
            )
            _V_test = convert_to_tensor(
                data['domain'][2] / V_max, requires_grad=False, device=self.device, dtype=self.dtype
            )

        self.model.eval()
        with tc.no_grad():
            _V_pred = self.model(tc.cat([_S_test, _t_test], dim=1))

        _mse = tc.mean((_V_pred.reshape(-1, 1) - _V_test) ** 2).item()
        if return_unormalized:
            _mse_un = tc.mean((V_max * _V_pred.reshape(-1, 1) - V_max * _V_test) ** 2).item()
            return _mse, _mse_un, (_V_pred.reshape(-1, 1) * V_max).detach().cpu().numpy()
        return _mse, (_V_pred.reshape(-1, 1) * V_max).detach().cpu().numpy()

    def train(self, S_max=160, T=1.0, V_max=120, normalize=True, return_loss=False, return_all=False):
        LOSS = {
            'Total': [],
            'pde_loss': [],
            'terminal_loss': [],
            'boundary_0_loss': [],
            'boundary_max_loss': [],
        }

        if not normalize:
            S_max = 1
            T = 1
            V_max = 1

        _S = convert_to_tensor(self.S / S_max, device=self.device, dtype=self.dtype)
        _t = convert_to_tensor(self.t / T, device=self.device, dtype=self.dtype)
        _ST = convert_to_tensor(self.ST / S_max, device=self.device, dtype=self.dtype)
        _tT = convert_to_tensor(self.tT / T, device=self.device, dtype=self.dtype)
        _VT = convert_to_tensor(self.VT / V_max, device=self.device, dtype=self.dtype, requires_grad=False)
        _S0 = convert_to_tensor(self.S0 / S_max, device=self.device, dtype=self.dtype)
        _t0 = convert_to_tensor(self.t0 / T, device=self.device, dtype=self.dtype)
        _V0 = convert_to_tensor(self.V0 / V_max, device=self.device, dtype=self.dtype, requires_grad=False)
        _Smax = convert_to_tensor(self.Smax / S_max, device=self.device, dtype=self.dtype)
        _tmax = convert_to_tensor(self.tmax / T, device=self.device, dtype=self.dtype)
        _Vmax = convert_to_tensor(self.Vmax / V_max, device=self.device, dtype=self.dtype, requires_grad=False)

        if not _S.requires_grad:
            _S.requires_grad_(True)
        if not _t.requires_grad:
            _t.requires_grad_(True)

        for _ in tqdm(range(self.epochs), desc="Trainning"):
            Vpred = self.model(tc.cat([_S, _t], dim=1))
            VTpred = self.model(tc.cat([_ST, _tT], dim=1))
            V0pred = self.model(tc.cat([_S0, _t0], dim=1))
            Vmaxpred = self.model(tc.cat([_Smax, _tmax], dim=1))

            solution = Vpred.reshape(-1, 1)
            df = self.derivatives(solution, _S, _t)

            Vest = (
                Vpred.reshape(-1, 1),
                VTpred.reshape(-1, 1),
                V0pred.reshape(-1, 1),
                Vmaxpred.reshape(-1, 1),
            )
            Vgiven = (_VT, _V0, _Vmax)

            self.optimizer.zero_grad()

            if return_loss and return_all:
                self.loss, pde_loss, terminal_loss, boundary_0_loss, boundary_max_loss = self.loss_function(
                    Vest, Vgiven, df, _S, return_all=True
                )
            else:
                self.loss = self.loss_function(Vest, Vgiven, df, _S, return_all=False)

            self.loss.backward()
            self.optimizer.step()

            if return_loss:
                LOSS['Total'].append(self.loss.item())
                if return_all:
                    LOSS['pde_loss'].append(pde_loss.item())
                    LOSS['terminal_loss'].append(terminal_loss.item())
                    LOSS['boundary_0_loss'].append(boundary_0_loss.item())
                    LOSS['boundary_max_loss'].append(boundary_max_loss.item())

        return LOSS if return_loss else None


import torch as tc
from tqdm import tqdm
from utils.get_data import convert_to_tensor
from utils.device import pick_torch_device


class BlackScholeOptimizer:
    def __init__(
        self,
        data,
        model,
        optimizer='Adam',
        lr=1e-2,
        epochs=200,
        sigma=0.02,
        r=0.05,
        weights=[1, 1, 1, 1],
        device: str = "auto",
        dtype=tc.float32,
    ):
        self.sigma = sigma
        self.model = model
        self.lr = float(lr)
        self.r = r
        self.epochs = epochs
        self.weights = weights

        self.device = pick_torch_device(device)
        self.dtype = dtype
        self.model = self.model.to(self.device).to(self.dtype)

        self.S, self.t, self.V = data['domain']
        self.ST, self.tT, self.VT = data['terminal']
        self.S0, self.t0, self.V0 = data['b0']
        self.Smax, self.tmax, self.Vmax = data['bmax']

        self.optimizer = self._get_optimizer(optimizer)
        self.num_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        self.loss = None

    def _get_optimizer(self, optimizer):
        if optimizer == 'Adam':
            return tc.optim.Adam(params=self.model.parameters(), lr=self.lr)
        raise ValueError("Optimizer not recognized. Use 'Adam' or 'LBFGS'.")

    def derivatives(self, Vest, _S, _t):
        ones = tc.ones_like(Vest, device=self.device, dtype=self.dtype)
        df_dt = tc.autograd.grad(Vest, _t, grad_outputs=ones, create_graph=True)[0]
        df_ds = tc.autograd.grad(Vest, _S, grad_outputs=ones, create_graph=True)[0]
        d2f_d2s = tc.autograd.grad(df_ds, _S, grad_outputs=ones, create_graph=True)[0]
        return df_dt, df_ds, d2f_d2s

    def loss_function(self, Vest, Vgiven, df, _S, return_all=False):
        _VT, _V0, _Vmax = Vgiven
        df_dt, df_ds, d2f_d2s = df
        Vpred, VTpred, V0pred, Vmaxpred = Vest

        pde_loss = tc.mean(
            (df_dt + 0.5 * self.sigma**2 * _S**2 * d2f_d2s + self.r * _S * df_ds - self.r * Vpred) ** 2
        )
        terminal_loss = tc.mean((VTpred - _VT) ** 2)
        boundary_0_loss = tc.mean((V0pred - _V0) ** 2)
        boundary_max_loss = tc.mean((Vmaxpred - _Vmax) ** 2)

        loss = (
            self.weights[0] * pde_loss
            + self.weights[1] * terminal_loss
            + self.weights[2] * boundary_0_loss
            + self.weights[3] * boundary_max_loss
        )

        if return_all:
            return loss, pde_loss, terminal_loss, boundary_0_loss, boundary_max_loss
        return loss

    def test(self, data, S_max=160, T=1.0, V_max=120, normalize=True, return_unormalized=False, data_new=False):
        if not normalize:
            S_max = 1
            T = 1
            V_max = 1

        if data_new:
            _S_test = convert_to_tensor(
                data[0] / S_max, requires_grad=False, device=self.device, dtype=self.dtype
            ).reshape(-1, 1)
            _t_test = convert_to_tensor(
                data[1] / T, requires_grad=False, device=self.device, dtype=self.dtype
            ).reshape(-1, 1)
            _V_test = convert_to_tensor(
                data[2] / V_max, requires_grad=False, device=self.device, dtype=self.dtype
            ).reshape(-1, 1)
        else:
            _S_test = convert_to_tensor(
                data['domain'][0] / S_max, requires_grad=False, device=self.device, dtype=self.dtype
            )
            _t_test = convert_to_tensor(
                data['domain'][1] / T, requires_grad=False, device=self.device, dtype=self.dtype
            )
            _V_test = convert_to_tensor(
                data['domain'][2] / V_max, requires_grad=False, device=self.device, dtype=self.dtype
            )

        self.model.eval()
        with tc.no_grad():
            _V_pred = self.model(tc.cat([_S_test, _t_test], dim=1))

        _mse = tc.mean((_V_pred.reshape(-1, 1) - _V_test) ** 2).item()
        if return_unormalized:
            _mse_un = tc.mean((V_max * _V_pred.reshape(-1, 1) - V_max * _V_test) ** 2).item()
            return _mse, _mse_un, (_V_pred.reshape(-1, 1) * V_max).detach().cpu().numpy()
        return _mse, (_V_pred.reshape(-1, 1) * V_max).detach().cpu().numpy()

    def train(self, S_max=160, T=1.0, V_max=120, normalize=True, return_loss=False, return_all=False):
        LOSS = {
            'Total': [],
            'pde_loss': [],
            'terminal_loss': [],
            'boundary_0_loss': [],
            'boundary_max_loss': [],
        }

        if not normalize:
            S_max = 1
            T = 1
            V_max = 1

        _S = convert_to_tensor(self.S / S_max, device=self.device, dtype=self.dtype)
        _t = convert_to_tensor(self.t / T, device=self.device, dtype=self.dtype)
        _ST = convert_to_tensor(self.ST / S_max, device=self.device, dtype=self.dtype)
        _tT = convert_to_tensor(self.tT / T, device=self.device, dtype=self.dtype)
        _VT = convert_to_tensor(self.VT / V_max, device=self.device, dtype=self.dtype, requires_grad=False)
        _S0 = convert_to_tensor(self.S0 / S_max, device=self.device, dtype=self.dtype)
        _t0 = convert_to_tensor(self.t0 / T, device=self.device, dtype=self.dtype)
        _V0 = convert_to_tensor(self.V0 / V_max, device=self.device, dtype=self.dtype, requires_grad=False)
        _Smax = convert_to_tensor(self.Smax / S_max, device=self.device, dtype=self.dtype)
        _tmax = convert_to_tensor(self.tmax / T, device=self.device, dtype=self.dtype)
        _Vmax = convert_to_tensor(self.Vmax / V_max, device=self.device, dtype=self.dtype, requires_grad=False)

        if not _S.requires_grad:
            _S.requires_grad_(True)
        if not _t.requires_grad:
            _t.requires_grad_(True)

        for _ in tqdm(range(self.epochs), desc="Trainning"):
            Vpred = self.model(tc.cat([_S, _t], dim=1))
            VTpred = self.model(tc.cat([_ST, _tT], dim=1))
            V0pred = self.model(tc.cat([_S0, _t0], dim=1))
            Vmaxpred = self.model(tc.cat([_Smax, _tmax], dim=1))

            solution = Vpred.reshape(-1, 1)
            df = self.derivatives(solution, _S, _t)

            Vest = (
                Vpred.reshape(-1, 1),
                VTpred.reshape(-1, 1),
                V0pred.reshape(-1, 1),
                Vmaxpred.reshape(-1, 1),
            )
            Vgiven = (_VT, _V0, _Vmax)

            self.optimizer.zero_grad()

            if return_loss and return_all:
                self.loss, pde_loss, terminal_loss, boundary_0_loss, boundary_max_loss = self.loss_function(
                    Vest, Vgiven, df, _S, return_all=True
                )
            else:
                self.loss = self.loss_function(Vest, Vgiven, df, _S, return_all=False)

            self.loss.backward()
            self.optimizer.step()

            if return_loss:
                LOSS['Total'].append(self.loss.item())
                if return_all:
                    LOSS['pde_loss'].append(pde_loss.item())
                    LOSS['terminal_loss'].append(terminal_loss.item())
                    LOSS['boundary_0_loss'].append(boundary_0_loss.item())
                    LOSS['boundary_max_loss'].append(boundary_max_loss.item())

        return LOSS if return_loss else None


class BlackScholeGreeksOptimizer:
    """
    Modelo deve retornar 4 saídas por amostra:
        [V, delta_pred, gamma_pred, theta_pred]

    Agora com suporte a treino por fases.
    Cada fase pode ter:
        - epochs
        - lr
        - weights       = [w_pde, w_terminal, w_b0, w_bmax]
        - greek_weights = [w_delta, w_gamma, w_theta]
    """

    def __init__(
        self,
        data,
        model,
        optimizer='Adam',
        lr=1e-2,
        epochs=200,
        sigma=0.02,
        r=0.05,
        weights=[1, 1, 1, 1],
        greek_weights=[1, 1, 1],
        phase_schedule=None,
        reset_optimizer_each_phase=False,
        grad_clip=None,
        device: str = "auto",
        dtype=tc.float32,
    ):
        self.sigma = sigma
        self.model = model
        self.lr = float(lr)
        self.r = r
        self.base_epochs = int(epochs)
        self.base_weights = list(weights)
        self.base_greek_weights = list(greek_weights)
        self.reset_optimizer_each_phase = bool(reset_optimizer_each_phase)
        self.grad_clip = grad_clip
        self.optimizer_name = optimizer

        self.device = pick_torch_device(device)
        self.dtype = dtype
        self.model = self.model.to(self.device).to(self.dtype)

        self.S, self.t, self.V = data['domain']
        self.ST, self.tT, self.VT = data['terminal']
        self.S0, self.t0, self.V0 = data['b0']
        self.Smax, self.tmax, self.Vmax = data['bmax']

        self.phase_schedule = self._normalize_phase_schedule(phase_schedule)
        self.epochs = sum(phase["epochs"] for phase in self.phase_schedule)

        self.optimizer = self._get_optimizer(self.optimizer_name, self.lr)
        self.num_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        self.loss = None

    def _get_optimizer(self, optimizer, lr):
        if optimizer == 'Adam':
            return tc.optim.Adam(params=self.model.parameters(), lr=float(lr))
        raise ValueError("Optimizer not recognized. Use 'Adam'.")

    def _normalize_phase_schedule(self, phase_schedule):
        if phase_schedule is None:
            return [
                {
                    "name": "single_phase",
                    "epochs": int(self.base_epochs),
                    "lr": float(self.lr),
                    "weights": list(self.base_weights),
                    "greek_weights": list(self.base_greek_weights),
                }
            ]

        if not isinstance(phase_schedule, (list, tuple)) or len(phase_schedule) == 0:
            raise ValueError("phase_schedule deve ser uma lista não vazia de dicionários.")

        normalized = []
        for i, phase in enumerate(phase_schedule):
            if not isinstance(phase, dict):
                raise ValueError(f"Cada fase deve ser dict. Recebi {type(phase)} na posição {i}.")

            phase_name = str(phase.get("name", f"phase_{i+1}"))
            phase_epochs = int(phase.get("epochs", 0))
            phase_lr = float(phase.get("lr", self.lr))
            phase_weights = list(phase.get("weights", self.base_weights))
            phase_greek_weights = list(phase.get("greek_weights", self.base_greek_weights))

            if phase_epochs <= 0:
                raise ValueError(f"A fase '{phase_name}' precisa ter epochs > 0.")
            if len(phase_weights) != 4:
                raise ValueError(f"A fase '{phase_name}' precisa de 4 pesos em 'weights'.")
            if len(phase_greek_weights) != 3:
                raise ValueError(f"A fase '{phase_name}' precisa de 3 pesos em 'greek_weights'.")

            normalized.append(
                {
                    "name": phase_name,
                    "epochs": phase_epochs,
                    "lr": phase_lr,
                    "weights": [float(x) for x in phase_weights],
                    "greek_weights": [float(x) for x in phase_greek_weights],
                }
            )

        return normalized

    def _set_lr(self, new_lr):
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = float(new_lr)

    def _maybe_reset_optimizer(self, lr):
        if self.reset_optimizer_each_phase:
            self.optimizer = self._get_optimizer(self.optimizer_name, lr)
        else:
            self._set_lr(lr)

    def _split_output(self, out):
        if out.ndim == 1:
            out = out.unsqueeze(0)

        if out.shape[1] < 4:
            raise ValueError(
                f"O modelo deve retornar pelo menos 4 saídas por amostra "
                f"[V, delta, gamma, theta]. Recebi shape {tuple(out.shape)}."
            )

        V_pred = out[:, 0:1]
        delta_pred = out[:, 1:2]
        gamma_pred = out[:, 2:3]
        theta_pred = out[:, 3:4]
        return V_pred, delta_pred, gamma_pred, theta_pred

    def derivatives(self, Vest, _S, _t):
        ones_v = tc.ones_like(Vest, device=self.device, dtype=self.dtype)

        df_dt = tc.autograd.grad(
            Vest,
            _t,
            grad_outputs=ones_v,
            create_graph=True,
            retain_graph=True,
        )[0]

        df_ds = tc.autograd.grad(
            Vest,
            _S,
            grad_outputs=ones_v,
            create_graph=True,
            retain_graph=True,
        )[0]

        d2f_d2s = tc.autograd.grad(
            df_ds,
            _S,
            grad_outputs=tc.ones_like(df_ds, device=self.device, dtype=self.dtype),
            create_graph=True,
            retain_graph=True,
        )[0]

        return df_dt, df_ds, d2f_d2s

    def loss_function(
        self,
        Vest,
        Vgiven,
        df,
        _S,
        weights=None,
        greek_weights=None,
        return_all=False,
    ):
        _VT, _V0, _Vmax = Vgiven
        df_dt, df_ds, d2f_d2s = df

        Vpred, VTpred, V0pred, Vmaxpred, delta_pred, gamma_pred, theta_pred = Vest

        weights = self.base_weights if weights is None else weights
        greek_weights = self.base_greek_weights if greek_weights is None else greek_weights

        pde_loss = tc.mean(
            (
                df_dt
                + 0.5 * self.sigma**2 * _S**2 * d2f_d2s
                + self.r * _S * df_ds
                - self.r * Vpred
            ) ** 2
        )

        terminal_loss = tc.mean((VTpred - _VT) ** 2)
        boundary_0_loss = tc.mean((V0pred - _V0) ** 2)
        boundary_max_loss = tc.mean((Vmaxpred - _Vmax) ** 2)

        delta_loss = tc.mean((delta_pred - df_ds.detach()) ** 2)
        gamma_loss = tc.mean((gamma_pred - d2f_d2s.detach()) ** 2)
        theta_loss = tc.mean((theta_pred - df_dt.detach()) ** 2)

        loss = (
            weights[0] * pde_loss
            + weights[1] * terminal_loss
            + weights[2] * boundary_0_loss
            + weights[3] * boundary_max_loss
            + greek_weights[0] * delta_loss
            + greek_weights[1] * gamma_loss
            + greek_weights[2] * theta_loss
        )

        if return_all:
            return (
                loss,
                pde_loss,
                terminal_loss,
                boundary_0_loss,
                boundary_max_loss,
                delta_loss,
                gamma_loss,
                theta_loss,
            )
        return loss

    def test(
        self,
        data,
        S_max=160,
        T=1.0,
        V_max=120,
        normalize=True,
        return_unormalized=False,
        data_new=False
    ):
        if not normalize:
            S_max = 1
            T = 1
            V_max = 1

        if data_new:
            _S_test = convert_to_tensor(
                data[0] / S_max, requires_grad=False,
                device=self.device, dtype=self.dtype
            ).reshape(-1, 1)
            _t_test = convert_to_tensor(
                data[1] / T, requires_grad=False,
                device=self.device, dtype=self.dtype
            ).reshape(-1, 1)
            _V_test = convert_to_tensor(
                data[2] / V_max, requires_grad=False,
                device=self.device, dtype=self.dtype
            ).reshape(-1, 1)
        else:
            _S_test = convert_to_tensor(
                data['domain'][0] / S_max, requires_grad=False,
                device=self.device, dtype=self.dtype
            )
            _t_test = convert_to_tensor(
                data['domain'][1] / T, requires_grad=False,
                device=self.device, dtype=self.dtype
            )
            _V_test = convert_to_tensor(
                data['domain'][2] / V_max, requires_grad=False,
                device=self.device, dtype=self.dtype
            )

        self.model.eval()
        with tc.no_grad():
            out = self.model(tc.cat([_S_test, _t_test], dim=1))
            V_pred, delta_pred, gamma_pred, theta_pred = self._split_output(out)

        mse = tc.mean((V_pred - _V_test) ** 2).item()

        preds = {
            "V": V_pred.detach().cpu().numpy(),
            "delta": delta_pred.detach().cpu().numpy(),
            "gamma": gamma_pred.detach().cpu().numpy(),
            "theta": theta_pred.detach().cpu().numpy(),
        }

        if return_unormalized:
            preds_un = {
                "V": (V_pred * V_max).detach().cpu().numpy(),
                "delta": (delta_pred * (V_max / S_max)).detach().cpu().numpy(),
                "gamma": (gamma_pred * (V_max / (S_max ** 2))).detach().cpu().numpy(),
                "theta": (theta_pred * (V_max / T)).detach().cpu().numpy(),
            }
            mse_un = tc.mean(((V_pred * V_max) - (_V_test * V_max)) ** 2).item()
            return mse, mse_un, preds_un

        return mse, preds

    def train(self, S_max=160, T=1.0, V_max=120, normalize=True, return_loss=False, return_all=False):
        LOSS = {
            'Total': [],
            'pde_loss': [],
            'terminal_loss': [],
            'boundary_0_loss': [],
            'boundary_max_loss': [],
            'delta_loss': [],
            'gamma_loss': [],
            'theta_loss': [],
            'phase_idx': [],
            'epoch_global': [],
            'lr_history': [],
            'w_pde': [],
            'w_terminal': [],
            'w_b0': [],
            'w_bmax': [],
            'w_delta': [],
            'w_gamma': [],
            'w_theta': [],
        }

        if not normalize:
            S_max = 1
            T = 1
            V_max = 1

        _S = convert_to_tensor(self.S / S_max, device=self.device, dtype=self.dtype)
        _t = convert_to_tensor(self.t / T, device=self.device, dtype=self.dtype)

        _ST = convert_to_tensor(self.ST / S_max, device=self.device, dtype=self.dtype)
        _tT = convert_to_tensor(self.tT / T, device=self.device, dtype=self.dtype)
        _VT = convert_to_tensor(self.VT / V_max, device=self.device, dtype=self.dtype, requires_grad=False)

        _S0 = convert_to_tensor(self.S0 / S_max, device=self.device, dtype=self.dtype)
        _t0 = convert_to_tensor(self.t0 / T, device=self.device, dtype=self.dtype)
        _V0 = convert_to_tensor(self.V0 / V_max, device=self.device, dtype=self.dtype, requires_grad=False)

        _Smax = convert_to_tensor(self.Smax / S_max, device=self.device, dtype=self.dtype)
        _tmax = convert_to_tensor(self.tmax / T, device=self.device, dtype=self.dtype)
        _Vmax = convert_to_tensor(self.Vmax / V_max, device=self.device, dtype=self.dtype, requires_grad=False)

        if not _S.requires_grad:
            _S.requires_grad_(True)
        if not _t.requires_grad:
            _t.requires_grad_(True)

        phase_metadata = []
        epoch_global = 0

        pbar = tqdm(total=self.epochs, desc="Trainning")

        for phase_idx, phase in enumerate(self.phase_schedule):
            phase_name = phase["name"]
            phase_epochs = int(phase["epochs"])
            phase_lr = float(phase["lr"])
            phase_weights = phase["weights"]
            phase_greek_weights = phase["greek_weights"]

            self._maybe_reset_optimizer(phase_lr)

            phase_metadata.append(
                {
                    "phase_idx": phase_idx,
                    "name": phase_name,
                    "epochs": phase_epochs,
                    "lr": phase_lr,
                    "weights": phase_weights,
                    "greek_weights": phase_greek_weights,
                }
            )

            for _ in range(phase_epochs):
                out_domain = self.model(tc.cat([_S, _t], dim=1))
                Vpred, delta_pred, gamma_pred, theta_pred = self._split_output(out_domain)

                out_terminal = self.model(tc.cat([_ST, _tT], dim=1))
                VTpred, _, _, _ = self._split_output(out_terminal)

                out_b0 = self.model(tc.cat([_S0, _t0], dim=1))
                V0pred, _, _, _ = self._split_output(out_b0)

                out_bmax = self.model(tc.cat([_Smax, _tmax], dim=1))
                Vmaxpred, _, _, _ = self._split_output(out_bmax)

                df = self.derivatives(Vpred, _S, _t)

                Vest = (
                    Vpred,
                    VTpred,
                    V0pred,
                    Vmaxpred,
                    delta_pred,
                    gamma_pred,
                    theta_pred,
                )
                Vgiven = (_VT, _V0, _Vmax)

                self.optimizer.zero_grad()

                if return_loss and return_all:
                    (
                        self.loss,
                        pde_loss,
                        terminal_loss,
                        boundary_0_loss,
                        boundary_max_loss,
                        delta_loss,
                        gamma_loss,
                        theta_loss,
                    ) = self.loss_function(
                        Vest,
                        Vgiven,
                        df,
                        _S,
                        weights=phase_weights,
                        greek_weights=phase_greek_weights,
                        return_all=True,
                    )
                else:
                    self.loss = self.loss_function(
                        Vest,
                        Vgiven,
                        df,
                        _S,
                        weights=phase_weights,
                        greek_weights=phase_greek_weights,
                        return_all=False,
                    )

                self.loss.backward()

                if self.grad_clip is not None:
                    tc.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=float(self.grad_clip))

                self.optimizer.step()

                epoch_global += 1
                pbar.update(1)
                pbar.set_postfix(
                    phase=phase_name,
                    loss=f"{self.loss.item():.3e}",
                    lr=f"{phase_lr:.1e}",
                )

                if return_loss:
                    LOSS['Total'].append(self.loss.item())
                    LOSS['phase_idx'].append(phase_idx)
                    LOSS['epoch_global'].append(epoch_global)
                    LOSS['lr_history'].append(phase_lr)

                    LOSS['w_pde'].append(phase_weights[0])
                    LOSS['w_terminal'].append(phase_weights[1])
                    LOSS['w_b0'].append(phase_weights[2])
                    LOSS['w_bmax'].append(phase_weights[3])

                    LOSS['w_delta'].append(phase_greek_weights[0])
                    LOSS['w_gamma'].append(phase_greek_weights[1])
                    LOSS['w_theta'].append(phase_greek_weights[2])

                    if return_all:
                        LOSS['pde_loss'].append(pde_loss.item())
                        LOSS['terminal_loss'].append(terminal_loss.item())
                        LOSS['boundary_0_loss'].append(boundary_0_loss.item())
                        LOSS['boundary_max_loss'].append(boundary_max_loss.item())
                        LOSS['delta_loss'].append(delta_loss.item())
                        LOSS['gamma_loss'].append(gamma_loss.item())
                        LOSS['theta_loss'].append(theta_loss.item())

        pbar.close()
        LOSS["phase_metadata"] = phase_metadata
        return LOSS if return_loss else None