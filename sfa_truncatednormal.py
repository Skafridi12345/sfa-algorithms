"""
Truncated-Normal Stochastic Cost Frontier (Energy Benchmarking)
---------------------------------------------------------------

This implements a TRUNCATED-NORMAL *cost* stochastic frontier.

Model direction:
    y = Xβ + v + u

- v is symmetric noise
- u ≥ 0 represents inefficiency and increases observed consumption
- u is normally distributed with mean μ and variance σ_u²,
  truncated at zero

Compared to the half-normal model, this allows inefficiency
to have a non-zero mean, providing additional flexibility.
"""

import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm


class TruncatedNormalSFA:
    """
    Truncated-Normal COST Stochastic Frontier Model.

    This specification allows the inefficiency term to have
    a non-zero mean before truncation, which can better capture
    systematic inefficiency in some datasets.
    """

    # -------------------------------------------------------------
    def __init__(self, y, X):
        # Store response and design matrix as float arrays
        self.y = np.asarray(y, float)
        self.X = np.asarray(X, float)
        self.n, self.k = self.X.shape

    # -------------------------------------------------------------
    def _unpack(self, theta):
        """
        Unpack parameter vector.

        Variance parameters are exponentiated to enforce positivity.
        """
        beta = theta[:self.k]
        mu = theta[self.k]
        sigma_v = np.exp(theta[self.k + 1])
        sigma_u = np.exp(theta[self.k + 2])
        return beta, mu, sigma_v, sigma_u

    # -------------------------------------------------------------
    # Log-likelihood (COST frontier, truncated-normal)
    # -------------------------------------------------------------
    def _loglik(self, theta):
        """
        Negative log-likelihood for the truncated-normal COST frontier.

        Includes the truncation normalisation term to ensure the
        inefficiency distribution integrates to one over u ≥ 0.
        """
        beta, mu, sigma_v, sigma_u = self._unpack(theta)

        eps = self.y - self.X @ beta
        sigma = np.sqrt(sigma_v**2 + sigma_u**2)
        lam = sigma_u / sigma_v

        # Composed error term
        z = (eps + mu) / sigma

        # Truncation normalisation (P(u ≥ 0))
        trunc = norm.cdf(mu / sigma_u)
        trunc = np.maximum(trunc, 1e-12)

        # Cost-frontier argument for truncated-normal model
        t = -(mu / sigma_u + lam * (eps + mu) / sigma)

        ll = (
            -np.log(sigma)
            + norm.logpdf(z)
            + norm.logcdf(t)
            - np.log(trunc)
        )

        return -np.sum(ll)

    # -------------------------------------------------------------
    # Analytical gradient
    # -------------------------------------------------------------
    def _gradient(self, theta):
        """
        Analytical score function for the truncated-normal COST frontier.

        Explicit gradients significantly improve numerical stability
        relative to finite-difference approximations.
        """
        beta, mu, sigma_v, sigma_u = self._unpack(theta)

        eps = self.y - self.X @ beta
        sigma = np.sqrt(sigma_v**2 + sigma_u**2)
        lam = sigma_u / sigma_v

        z = (eps + mu) / sigma
        phi_z = norm.pdf(z)

        trunc = norm.cdf(mu / sigma_u)
        trunc = np.maximum(trunc, 1e-12)
        trunc_pdf = norm.pdf(mu / sigma_u)

        # Cost-frontier argument
        t = -(mu / sigma_u + lam * (eps + mu) / sigma)
        phi_t = norm.pdf(t)
        Phi_t = np.maximum(norm.cdf(t), 1e-12)

        # ---------------------------------------------------------
        # dLL / dβ
        # ---------------------------------------------------------
        # β enters through both z and t via the residual
        term_beta = (
            phi_z * (1.0 / sigma)
            + phi_t / (Phi_t * sigma_v) * sigma_u / sigma
        )

        g_beta = -(term_beta[:, None] * self.X).sum(axis=0)

        # ---------------------------------------------------------
        # dLL / dμ
        # ---------------------------------------------------------
        dlog_phi_z_dmu = phi_z * (1.0 / sigma)
        dlog_Phi_t_dmu = (phi_t / Phi_t) * (-1 / sigma_u - lam / sigma)
        dlog_trunc_dmu = trunc_pdf / (trunc * sigma_u)

        g_mu = -np.sum(
            dlog_phi_z_dmu
            + dlog_Phi_t_dmu
            - dlog_trunc_dmu
        )

        # ---------------------------------------------------------
        # dLL / d log σ_v
        # ---------------------------------------------------------
        d_sigma_dsv = sigma_v / sigma
        d_lam_dsv = -sigma_u / (sigma_v**2)

        dt_dsv = (
            -(lam * (eps + mu) / sigma**2) * d_sigma_dsv
            - ((eps + mu) / sigma) * d_lam_dsv
        )

        g_sv = -np.sum(
            -(sigma_v / sigma)
            + phi_z * z * d_sigma_dsv
            + phi_t / Phi_t * dt_dsv
        ) * sigma_v

        # ---------------------------------------------------------
        # dLL / d log σ_u
        # ---------------------------------------------------------
        d_sigma_dsu = sigma_u / sigma
        d_lam_dsu = 1.0 / sigma_v

        dt_dsu = (
            -(1 / sigma_u)
            - lam * (eps + mu) * d_sigma_dsu / sigma**2
            - (eps + mu) / sigma * d_lam_dsu
        )

        dlog_trunc_dsu = -(mu / (sigma_u**2)) * trunc_pdf / trunc

        g_su = -np.sum(
            -(sigma_u / sigma)
            + phi_z * z * d_sigma_dsu
            + phi_t / Phi_t * dt_dsu
            + dlog_trunc_dsu
        ) * sigma_u

        return np.concatenate([g_beta, [g_mu, g_sv, g_su]])

    # -------------------------------------------------------------
    def fit(self):
        """
        Estimate model parameters via maximum likelihood.

        OLS is used only to initialise β and variance scales.
        """
        beta_ols = np.linalg.lstsq(self.X, self.y, rcond=None)[0]
        resid = self.y - self.X @ beta_ols
        sigma_ols = float(np.std(resid) + 1e-6)

        theta0 = np.concatenate([
            beta_ols,
            [0.0],                      # initial μ
            [np.log(sigma_ols * 0.5)],
            [np.log(sigma_ols * 0.7)]
        ])

        res = minimize(
            self._loglik,
            theta0,
            jac=self._gradient,
            method="L-BFGS-B",
            options={"maxiter": 600, "disp": False},
        )

        # Fallback to numerical gradient if needed
        if not res.success:
            res = minimize(
                self._loglik,
                theta0,
                method="L-BFGS-B",
                options={"maxiter": 600, "disp": False},
            )

        self.res = res
        self.theta = res.x
        self._postprocess()
        return self

    # -------------------------------------------------------------
    def _postprocess(self):
        """
        Compute conditional inefficiency estimates (JLMS)
        and Technical Efficiency scores.
        """
        beta, mu, sigma_v, sigma_u = self._unpack(self.theta)

        self.beta = beta
        self.mu = mu
        self.sigma_v = sigma_v
        self.sigma_u = sigma_u

        eps = self.y - self.X @ beta
        sigma2 = sigma_v**2 + sigma_u**2

        mu_star = (sigma_u**2 * (eps + mu)) / sigma2
        sigma_star = np.sqrt((sigma_v**2 * sigma_u**2) / sigma2)

        z_star = mu_star / sigma_star
        ratio = norm.pdf(z_star) / np.maximum(norm.cdf(z_star), 1e-12)

        u_hat = mu_star + sigma_star * ratio
        self.u_hat = np.maximum(u_hat, 0.0)

        # COST frontier efficiency
        self.TE = np.exp(-self.u_hat)
        self.frontier = self.X @ beta

    # -------------------------------------------------------------
    def summary(self):
        """
        Lightweight summary of fitted parameters and diagnostics.
        """
        return {
            "beta": self.beta,
            "mu": self.mu,
            "sigma_v": self.sigma_v,
            "sigma_u": self.sigma_u,
            "TE_mean": float(np.mean(self.TE)),
            "TE_median": float(np.median(self.TE)),
            "log_likelihood": -self.res.fun,
            "converged": self.res.success,
        }
