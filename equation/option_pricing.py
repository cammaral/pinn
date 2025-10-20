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


"""
Implementar
- Plot 3D da superfície de preços
- Plot 2D data

"""