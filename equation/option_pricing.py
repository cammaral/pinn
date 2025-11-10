import numpy as np
from scipy.stats import norm

class BlackScholes:
    def __init__(self, S_max=160, T=1.0, K=40, r=0.05, sigma=0.2, eps=1e-10):
        self.S_max = S_max
        self.T = T
        self.K = K
        self.r = r
        self.sigma = sigma
        self.eps = eps  # tolerância numérica

    def V(self, S, t, option_type='call'):
        S = np.array(S)
        t = np.array(t)
        tau = np.array(self.T - t)

        # payoff direto em maturidade/pós-maturidade
        if np.all(tau <= self.eps):
            if option_type == 'call':
                return np.maximum(S - self.K, 0.0)
            elif option_type == 'put':
                return np.maximum(self.K - S, 0.0)
            else:
                raise ValueError("option_type must be 'call' or 'put'")

        # garante tau >= 0 e evita log(0)
        tau = np.maximum(tau, 0.0)
        S_safe = np.maximum(S, self.eps)

        sqrt_tau = np.sqrt(np.maximum(tau, self.eps))
        d1 = (np.log(S_safe / self.K) + (self.r + 0.5 * self.sigma**2) * tau) / (self.sigma * sqrt_tau)
        d2 = d1 - self.sigma * sqrt_tau

        discK = self.K * np.exp(-self.r * tau)

        if option_type == 'call':
            price = S_safe * norm.cdf(d1) - discK * norm.cdf(d2)
            # condição de fronteira em S=0 -> 0
            price = np.where(S <= self.eps, 0.0, price)
            # em tau≈0, reforça payoff
            price = np.where(tau <= self.eps, np.maximum(S - self.K, 0.0), price)
            return price

        elif option_type == 'put':
            price = discK * norm.cdf(-d2) - S_safe * norm.cdf(-d1)
            # condição de fronteira em S=0 -> K e^{-r tau}
            price = np.where(S <= self.eps, discK, price)
            # em tau≈0, reforça payoff
            price = np.where(tau <= self.eps, np.maximum(self.K - S, 0.0), price)
            return price

        else:
            raise ValueError("option_type must be 'call' or 'put'")

    def generate_data(self, N_domain=1000, N_boundary=1000, N_terminal=1000, seed=1924):
        np.random.seed(seed)

        # Domain samples
        S_domain = np.random.uniform(0, self.S_max, (int(N_domain), 1))
        t_domain = np.random.uniform(0, self.T, (int(N_domain), 1))
        V_domain = self.V(S_domain, t_domain)

        # Terminal condition (payoff)
        S_terminal = np.random.uniform(0, self.S_max, (N_terminal, 1))
        t_terminal = self.T * np.ones((N_terminal, 1))
        V_terminal = self.V(S_terminal, t_terminal)

        # Boundary conditions
        S_boundary_0 = np.zeros((N_boundary // 2, 1))
        t_boundary_0 = np.random.uniform(0, self.T, (N_boundary // 2, 1))
        V_boundary_0 = self.V(S_boundary_0, t_boundary_0)

        S_boundary_max = self.S_max * np.ones((N_boundary // 2, 1))
        t_boundary_max = np.random.uniform(0, self.T, (N_boundary // 2, 1))
        V_boundary_max = self.V(S_boundary_max, t_boundary_max)

        return {
            'domain': (S_domain, t_domain, V_domain),
            'terminal': (S_terminal, t_terminal, V_terminal),
            'bmax': (S_boundary_max, t_boundary_max, V_boundary_max),
            'b0': (S_boundary_0, t_boundary_0, V_boundary_0)
        }



class Heston:
    def __init__(
        self,
        S_max=160.0, T=1.0, K=40.0, r=0.05,
        kappa=2.0, theta=0.04, sigma_v=0.3, rho=-0.7,  # parâmetros do Heston (não usados no proxy)
        V_max=1.0,  # limite superior para v (variância)
        eps=1e-10
    ):
        self.S_max  = S_max
        self.T      = T
        self.K      = K
        self.r      = r
        self.kappa  = kappa
        self.theta  = theta
        self.sigma_v= sigma_v
        self.rho    = rho
        self.V_max  = V_max
        self.eps    = eps  # tolerância numérica

    def _bs_price_with_sigma(self, S, tau, sigma, option_type='call'):
        """Preço Black–Scholes com vol σ informada (usa limites estáveis para σ→0)."""
        S = np.array(S)
        tau = np.array(tau)
        sigma = np.array(sigma)

        tau = np.maximum(tau, 0.0)
        S_safe = np.maximum(S, self.eps)
        discK = self.K * np.exp(-self.r * tau)

        # trata σ≈0: vira valor intrínseco descontado
        near_zero_sigma = sigma <= self.eps
        non_zero_sigma  = ~near_zero_sigma

        price = np.zeros_like(S_safe, dtype=float)

        if np.any(non_zero_sigma):
            sqrt_tau = np.sqrt(np.maximum(tau[non_zero_sigma], self.eps))
            d1 = (np.log(S_safe[non_zero_sigma] / self.K) +
                  (self.r + 0.5 * sigma[non_zero_sigma]**2) * tau[non_zero_sigma]) / (sigma[non_zero_sigma] * sqrt_tau)
            d2 = d1 - sigma[non_zero_sigma] * sqrt_tau

            if option_type == 'call':
                price_nonzero = S_safe[non_zero_sigma] * norm.cdf(d1) - discK[non_zero_sigma] * norm.cdf(d2)
            elif option_type == 'put':
                price_nonzero = discK[non_zero_sigma] * norm.cdf(-d2) - S_safe[non_zero_sigma] * norm.cdf(-d1)
            else:
                raise ValueError("option_type must be 'call' or 'put'")
            price[non_zero_sigma] = price_nonzero

        if np.any(near_zero_sigma):
            if option_type == 'call':
                price[near_zero_sigma] = np.maximum(S_safe[near_zero_sigma] - discK[near_zero_sigma], 0.0)
            elif option_type == 'put':
                price[near_zero_sigma] = np.maximum(discK[near_zero_sigma] - S_safe[near_zero_sigma], 0.0)

        return price

    def V(self, S, t, v, option_type='call'):
        """
        Proxy para V(S,t,v): BS com σ_local = sqrt(v).
        Mantém as mesmas salvaguardas de fronteira/payoff da tua classe original.
        """
        S   = np.array(S)
        t   = np.array(t)
        v   = np.array(v)
        tau = np.array(self.T - t)

        # payoff em maturidade/pós-maturidade
        if np.all(tau <= self.eps):
            if option_type == 'call':
                return np.maximum(S - self.K, 0.0)
            elif option_type == 'put':
                return np.maximum(self.K - S, 0.0)
            else:
                raise ValueError("option_type must be 'call' or 'put'")

        tau = np.maximum(tau, 0.0)
        S_safe = np.maximum(S, self.eps)
        sigma_local = np.sqrt(np.maximum(v, 0.0))

        # preço proxy (BS com sigma_local)
        price = self._bs_price_with_sigma(S_safe, tau, sigma_local, option_type=option_type)

        # fronteiras clássicas em S
        if option_type == 'call':
            # S=0 -> 0
            price = np.where(S <= self.eps, 0.0, price)
            # tau≈0 -> payoff
            price = np.where(tau <= self.eps, np.maximum(S - self.K, 0.0), price)
        else:
            # put: S=0 -> K e^{-r tau}
            discK = self.K * np.exp(-self.r * tau)
            price = np.where(S <= self.eps, discK, price)
            price = np.where(tau <= self.eps, np.maximum(self.K - S, 0.0), price)

        return price

    def generate_data(self, N_domain=1000, N_boundary=1000, N_terminal=1000, seed=1924, option_type='call'):
        """
        Mesmo padrão de saída da tua BlackScholes, com v incluído nas tuplas.
        Keys: 'domain', 'terminal', 'b0', 'bmax', e + duas novas: 'bv0', 'bvmax'.
        """
        np.random.seed(seed)

        # ------------------------
        # Domain samples (S, t, v)
        # ------------------------
        S_domain = np.random.uniform(0, self.S_max, (int(N_domain), 1))
        t_domain = np.random.uniform(0, self.T,     (int(N_domain), 1))
        v_domain = np.random.uniform(0, self.V_max, (int(N_domain), 1))
        V_domain = self.V(S_domain, t_domain, v_domain, option_type=option_type)

        # ------------------------
        # Terminal condition (t = T, payoff)
        # ------------------------
        S_terminal = np.random.uniform(0, self.S_max, (N_terminal, 1))
        t_terminal = self.T * np.ones((N_terminal, 1))
        # v é irrelevante no payoff, mas mantemos por consistência de shape
        v_terminal = np.random.uniform(0, self.V_max, (N_terminal, 1))
        V_terminal = self.V(S_terminal, t_terminal, v_terminal, option_type=option_type)

        # ------------------------
        # Boundary in S=0
        # ------------------------
        S_boundary_0  = np.zeros((N_boundary // 4, 1))
        t_boundary_0  = np.random.uniform(0, self.T,     (N_boundary // 4, 1))
        v_boundary_0  = np.random.uniform(0, self.V_max, (N_boundary // 4, 1))
        V_boundary_0  = self.V(S_boundary_0, t_boundary_0, v_boundary_0, option_type=option_type)

        # ------------------------
        # Boundary in S=S_max
        # ------------------------
        S_boundary_max = self.S_max * np.ones((N_boundary // 4, 1))
        t_boundary_max = np.random.uniform(0, self.T,     (N_boundary // 4, 1))
        v_boundary_max = np.random.uniform(0, self.V_max, (N_boundary // 4, 1))
        V_boundary_max = self.V(S_boundary_max, t_boundary_max, v_boundary_max, option_type=option_type)

        # ------------------------
        # Boundary in v=0
        # ------------------------
        v0 = np.zeros((N_boundary // 4, 1))
        S_bv0 = np.random.uniform(0, self.S_max, (N_boundary // 4, 1))
        t_bv0 = np.random.uniform(0, self.T,     (N_boundary // 4, 1))
        V_bv0 = self.V(S_bv0, t_bv0, v0, option_type=option_type)

        # ------------------------
        # Boundary in v=V_max
        # ------------------------
        vM = self.V_max * np.ones((N_boundary // 4, 1))
        S_bvM = np.random.uniform(0, self.S_max, (N_boundary // 4, 1))
        t_bvM = np.random.uniform(0, self.T,     (N_boundary // 4, 1))
        V_bvM = self.V(S_bvM, t_bvM, vM, option_type=option_type)

        return {
            'domain':   (S_domain, t_domain, v_domain, V_domain),
            'terminal': (S_terminal, t_terminal, v_terminal, V_terminal),
            'b0':       (S_boundary_0,  t_boundary_0,  v_boundary_0,  V_boundary_0),
            'bmax':     (S_boundary_max, t_boundary_max, v_boundary_max, V_boundary_max),
            'bv0':      (S_bv0, t_bv0, v0,  V_bv0),
            'bvmax':    (S_bvM, t_bvM, vM, V_bvM)
        }


"""
Implementar
- Plot 3D da superfície de preços
- Plot 2D data

"""