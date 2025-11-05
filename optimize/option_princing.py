import torch as tc
from tqdm import tqdm
from utils.get_data import convert_to_tensor
from utils.device import pick_torch_device


"""
class BlackScholeOptimizer:
    def __init__(self, data, model, optimizer='Adam', lr=10-2, epochs = 200,sigma=0.02, r=0.05, weights=[1,1,1,1]):
        self.sigma = sigma
        self.model = model
        self.lr = float(lr)
        self.r = r
        self.epochs = epochs
        self.weights = weights

        # dados
        self.S,  self.t,  self.V  = data['domain']
        self.ST, self.tT, self.VT = data['terminal']
        self.S0, self.t0, self.V0 = data['b0']
        self.Smax, self.tmax, self.Vmax = data['bmax']

        # otimizador
        self.optimizer = self._get_optimizer(optimizer)
        self.num_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        self.loss = None

    def _get_optimizer(self, optimizer):
        if optimizer == 'Adam':
            return tc.optim.Adam(params=self.model.parameters(), lr=self.lr)
        else:
            raise ValueError("Optimizer not recognized. Use 'Adam' or 'LBFGS'.")
    
    def derivatives(self, Vest, _S, _t):
        df_dt = tc.autograd.grad(Vest, _t, grad_outputs=tc.ones_like(Vest), create_graph=True)[0]
        df_ds = tc.autograd.grad(Vest, _S, grad_outputs=tc.ones_like(Vest), create_graph=True)[0]
        d2f_d2s = tc.autograd.grad(df_ds, _S, grad_outputs=tc.ones_like(Vest), create_graph=True)[0]
        return df_dt, df_ds, d2f_d2s

    def loss_function(self, Vest, Vgiven,df, _S,  return_all=False):
        _VT, _V0, _Vmax = Vgiven
        df_dt, df_ds, d2f_d2s = df
        Vpred, VTpred, V0pred, Vmaxpred = Vest

        #print(Vpred.shape, df_dt.shape, d2f_d2s.shape, df_ds.shape, _S.shape, _VT.shape, _V0.shape, _Vmax.shape)
        pde_loss = tc.mean((df_dt + 0.5 * self.sigma**2 * _S**2 * d2f_d2s + self.r * _S * df_ds - self.r * Vpred)**2)
        terminal_loss = tc.mean((VTpred - _VT) ** 2)
        boundary_0_loss = tc.mean((V0pred - _V0) ** 2)
        boundary_max_loss = tc.mean((Vmaxpred - _Vmax) ** 2)
        loss = (self.weights[0] * pde_loss + self.weights[1] * terminal_loss +
            self.weights[2] * boundary_0_loss + self.weights[3] * boundary_max_loss)
        if return_all:
            return loss, pde_loss, terminal_loss, boundary_0_loss, boundary_max_loss
        else:
            return loss
        

    def test(self, data, S_max=160, T=1.0, V_max=140, normalize=True, return_unormalized=False):
        if not normalize:
            S_max = 1
            T = 1
            V_max = 1

        _S_test = convert_to_tensor(data['domain'][0] / S_max, requires_grad=False)
        _t_test = convert_to_tensor(data['domain'][1] / T, requires_grad=False)
        _V_test = convert_to_tensor(data['domain'][2] / V_max, requires_grad=False)

        with tc.no_grad():
            _V_pred = self.model(tc.cat([_S_test, _t_test], dim=1))

        _mse = tc.mean((_V_pred.reshape(-1, 1) - _V_test) ** 2).item()
        if return_unormalized:
            _mse_un = tc.mean((V_max*_V_pred.reshape(-1, 1) - V_max*_V_test) ** 2).item()
            return _mse, _mse_un,  _V_pred.numpy()*V_max
        else:
            return _mse, _V_pred.numpy()*V_max
    def train(self, S_max=160, T=1.0, V_max=140, normalize=True, return_loss=False, return_all=False):
        LOSS = {
            'Total': [],
            'pde_loss': [],
            'terminal_loss': [],
            'boundary_0_loss': [],
            'boundary_max_loss': []
        }

        if not normalize:
            S_max = 1
            T = 1
            V_max = 1

        _S    = convert_to_tensor(self.S / S_max)         # needs requires_grad=True
        _t    = convert_to_tensor(self.t / T)              # idem
        _ST   = convert_to_tensor(self.ST / S_max)
        _tT   = convert_to_tensor(self.tT / T)
        _VT   = convert_to_tensor(self.VT / V_max,  requires_grad=False)
        _S0   = convert_to_tensor(self.S0 / S_max)
        _t0   = convert_to_tensor(self.t0 / T)
        _V0   = convert_to_tensor(self.V0 / V_max,  requires_grad=False)
        _Smax = convert_to_tensor(self.Smax / S_max)
        _tmax = convert_to_tensor(self.tmax / T)
        _Vmax = convert_to_tensor(self.Vmax / V_max, requires_grad=False)

        # GARANTA que _S e _t tenham gradientes (caso convert_to_tensor não faça isso)
        if not _S.requires_grad: _S.requires_grad_(True)
        if not _t.requires_grad: _t.requires_grad_(True)

        for epoch in tqdm(range(self.epochs), desc="Trainning"):
            Vpred    = self.model(tc.cat([_S,    _t],    dim=1))
            VTpred   = self.model(tc.cat([_ST,   _tT],   dim=1))
            V0pred   = self.model(tc.cat([_S0,   _t0],   dim=1))
            Vmaxpred = self.model(tc.cat([_Smax, _tmax], dim=1))

            solution = Vpred.reshape(-1, 1)
            df = self.derivatives(solution, _S, _t)

            Vest   = (Vpred.reshape(-1, 1), VTpred.reshape(-1, 1),
                    V0pred.reshape(-1, 1), Vmaxpred.reshape(-1, 1))
            Vgiven = (_VT, _V0, _Vmax)

            self.optimizer.zero_grad()

            if return_loss and return_all:
                self.loss, pde_loss, terminal_loss, boundary_0_loss, boundary_max_loss = \
                    self.loss_function(Vest, Vgiven, df, _S, return_all=True)
            else:
                self.loss = self.loss_function(Vest, Vgiven, df, _S, return_all=False)

            self.loss.backward()
            self.optimizer.step()

            # logging das perdas (dentro do loop!)
            if return_loss:
                LOSS['Total'].append(self.loss.item())
                if return_all:
                    LOSS['pde_loss'].append(pde_loss.item())
                    LOSS['terminal_loss'].append(terminal_loss.item())
                    LOSS['boundary_0_loss'].append(boundary_0_loss.item())
                    LOSS['boundary_max_loss'].append(boundary_max_loss.item())
        return LOSS if return_loss else None
"""

class BlackScholeOptimizer:
    def __init__(self, data, model, optimizer='Adam', lr=1e-2, epochs = 200,
                 sigma=0.02, r=0.05, weights=[1,1,1,1],
                 device: str = "auto", dtype=tc.float32):
        self.sigma = sigma
        self.model = model
        self.lr = float(lr)
        self.r = r
        self.epochs = epochs
        self.weights = weights

        # ======= NOVO: device/dtype =======
        self.device = pick_torch_device(device)
        self.dtype  = dtype
        self.model  = self.model.to(self.device).to(self.dtype)
        # ==================================

        # dados
        self.S,  self.t,  self.V  = data['domain']
        self.ST, self.tT, self.VT = data['terminal']
        self.S0, self.t0, self.V0 = data['b0']
        self.Smax, self.tmax, self.Vmax = data['bmax']

        # otimizador (depois de mover o modelo p/ device)
        self.optimizer = self._get_optimizer(optimizer)
        self.num_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        self.loss = None

    def _get_optimizer(self, optimizer):
        if optimizer == 'Adam':
            return tc.optim.Adam(params=self.model.parameters(), lr=self.lr)
        else:
            raise ValueError("Optimizer not recognized. Use 'Adam' or 'LBFGS'.")

    def derivatives(self, Vest, _S, _t):
        # ======= AJUSTE: grad_outputs no device/dtype =======
        ones = tc.ones_like(Vest, device=self.device, dtype=self.dtype)
        df_dt = tc.autograd.grad(Vest, _t, grad_outputs=ones, create_graph=True)[0]
        df_ds = tc.autograd.grad(Vest, _S, grad_outputs=ones, create_graph=True)[0]
        d2f_d2s = tc.autograd.grad(df_ds, _S, grad_outputs=ones, create_graph=True)[0]
        # ====================================================
        return df_dt, df_ds, d2f_d2s

    def loss_function(self, Vest, Vgiven,df, _S,  return_all=False):
        _VT, _V0, _Vmax = Vgiven
        df_dt, df_ds, d2f_d2s = df
        Vpred, VTpred, V0pred, Vmaxpred = Vest

        pde_loss = tc.mean((df_dt + 0.5 * self.sigma**2 * _S**2 * d2f_d2s + self.r * _S * df_ds - self.r * Vpred)**2)
        terminal_loss = tc.mean((VTpred - _VT) ** 2)
        boundary_0_loss = tc.mean((V0pred - _V0) ** 2)
        boundary_max_loss = tc.mean((Vmaxpred - _Vmax) ** 2)
        loss = (self.weights[0] * pde_loss + self.weights[1] * terminal_loss +
            self.weights[2] * boundary_0_loss + self.weights[3] * boundary_max_loss)
        if return_all:
            return loss, pde_loss, terminal_loss, boundary_0_loss, boundary_max_loss
        else:
            return loss

    def test(self, data, S_max=160, T=1.0, V_max=140, normalize=True, return_unormalized=False):
        if not normalize:
            S_max = 1
            T = 1
            V_max = 1

        # ======= AJUSTE: tensors no device/dtype =======
        _S_test = convert_to_tensor(data['domain'][0] / S_max, requires_grad=False,
                                    device=self.device, dtype=self.dtype)
        _t_test = convert_to_tensor(data['domain'][1] / T,   requires_grad=False,
                                    device=self.device, dtype=self.dtype)
        _V_test = convert_to_tensor(data['domain'][2] / V_max, requires_grad=False,
                                    device=self.device, dtype=self.dtype)
        # ===============================================

        self.model.eval()
        with tc.no_grad():
            _V_pred = self.model(tc.cat([_S_test, _t_test], dim=1))

        _mse = tc.mean((_V_pred.reshape(-1, 1) - _V_test) ** 2).item()
        if return_unormalized:
            _mse_un = tc.mean((V_max*_V_pred.reshape(-1, 1) - V_max*_V_test) ** 2).item()
            return _mse, _mse_un,  (_V_pred.reshape(-1,1)*V_max).detach().cpu().numpy()
        else:
            return _mse, (_V_pred.reshape(-1,1)*V_max).detach().cpu().numpy()

    def train(self, S_max=160, T=1.0, V_max=140, normalize=True, return_loss=False, return_all=False):
        LOSS = {
            'Total': [],
            'pde_loss': [],
            'terminal_loss': [],
            'boundary_0_loss': [],
            'boundary_max_loss': []
        }

        if not normalize:
            S_max = 1
            T = 1
            V_max = 1

        # ======= AJUSTE: tensors no device/dtype =======
        _S    = convert_to_tensor(self.S / S_max,  device=self.device, dtype=self.dtype)         # needs requires_grad=True
        _t    = convert_to_tensor(self.t / T,      device=self.device, dtype=self.dtype)         # idem
        _ST   = convert_to_tensor(self.ST / S_max, device=self.device, dtype=self.dtype)
        _tT   = convert_to_tensor(self.tT / T,     device=self.device, dtype=self.dtype)
        _VT   = convert_to_tensor(self.VT / V_max, device=self.device, dtype=self.dtype,  requires_grad=False)
        _S0   = convert_to_tensor(self.S0 / S_max, device=self.device, dtype=self.dtype)
        _t0   = convert_to_tensor(self.t0 / T,     device=self.device, dtype=self.dtype)
        _V0   = convert_to_tensor(self.V0 / V_max, device=self.device, dtype=self.dtype,  requires_grad=False)
        _Smax = convert_to_tensor(self.Smax / S_max, device=self.device, dtype=self.dtype)
        _tmax = convert_to_tensor(self.tmax / T,     device=self.device, dtype=self.dtype)
        _Vmax = convert_to_tensor(self.Vmax / V_max, device=self.device, dtype=self.dtype, requires_grad=False)
        # ===============================================

        # GARANTA que _S e _t tenham gradientes (caso convert_to_tensor não faça isso)
        if not _S.requires_grad: _S.requires_grad_(True)
        if not _t.requires_grad: _t.requires_grad_(True)

        for epoch in tqdm(range(self.epochs), desc="Trainning"):
            Vpred    = self.model(tc.cat([_S,    _t],    dim=1))
            VTpred   = self.model(tc.cat([_ST,   _tT],   dim=1))
            V0pred   = self.model(tc.cat([_S0,   _t0],   dim=1))
            Vmaxpred = self.model(tc.cat([_Smax, _tmax], dim=1))

            solution = Vpred.reshape(-1, 1)
            df = self.derivatives(solution, _S, _t)

            Vest   = (Vpred.reshape(-1, 1), VTpred.reshape(-1, 1),
                    V0pred.reshape(-1, 1), Vmaxpred.reshape(-1, 1))
            Vgiven = (_VT, _V0, _Vmax)

            self.optimizer.zero_grad()

            if return_loss and return_all:
                self.loss, pde_loss, terminal_loss, boundary_0_loss, boundary_max_loss = \
                    self.loss_function(Vest, Vgiven, df, _S, return_all=True)
            else:
                self.loss = self.loss_function(Vest, Vgiven, df, _S, return_all=False)

            self.loss.backward()
            self.optimizer.step()

            # logging das perdas (dentro do loop!)
            if return_loss:
                LOSS['Total'].append(self.loss.item())
                if return_all:
                    LOSS['pde_loss'].append(pde_loss.item())
                    LOSS['terminal_loss'].append(terminal_loss.item())
                    LOSS['boundary_0_loss'].append(boundary_0_loss.item())
                    LOSS['boundary_max_loss'].append(boundary_max_loss.item())
        return LOSS if return_loss else None
