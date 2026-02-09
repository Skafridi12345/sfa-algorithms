"""
Half-Normal Stochastic Cost Frontier for Energy Benchmarking
-------------------------------------------------------------

This implements a HALF-NORMAL *cost* stochastic frontier model.

Model direction is important:
    y = Xβ + v + u

- v represents symmetric noise
- u ≥ 0 represents inefficiency and increases observed consumption

This is a COST frontier (not a production frontier).
Technical Efficiency is therefore computed as exp(-û).
"""

import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm


class HalfNormalSFA:
    """
    Half-Normal COST Stochastic Frontier Model.

    This class estimates a stochastic cost frontier where
    inefficiency is assumed to follow a half-normal distribution.
    """

    # --------------------------------------------------------------
    def __init__(self, y, X):
        # Store response and design matrix as float arrays
        self.y = np.asarray(y, float)
        self.X = np.asarray(X, float)
        self.n, self.k = self.X.shape

    # --------------------------------------------------------------
    def _unpack(self, theta):
        """
        Unpack parameter vector.

        Variance parameters are exponentiated to enforce positivity.
        """
        beta = theta[:self.k]
        sigma_v = np.exp(theta[self.k])
        sigma_u = np.exp(theta[self.k + 1])
        return beta, sigma_v, sigma_u

    # --------------------------------------------------------------
    # Log-likelihood (COST frontier)
    # --------------------------------------------------------------
    def _loglik(self, theta):
        """
        Negative log-likelihood for the half-normal COST frontier.

        The formulation follows standard SFA results adapted
        explicitly for a cost frontier (u enters with a positive sign).
        """
        beta, sigma_v, sigma_u = self._unpack(theta)

        eps = self.y - self.X @ beta
        sigma = np.sqrt(sigma_v**2 + sigma_u**2)
        lam = sigma_u / sigma_v

        z = eps / sigma
        t = (lam * eps) / sigma # t = -(lam * eps) / sigma for production frontier


        ll = (
            -np.log(sigma)
            + norm.logpdf(z)
            + np.log(2.0)
            + norm.logcdf(t)
        )

        return -np.sum(ll)

    # --------------------------------------------------------------
    # Analytical gradient
    # --------------------------------------------------------------
    def _gradient(self, theta):
        """
        Analytical score function for the half-normal COST frontier.

        Explicit gradients improve stability and convergence speed
        compared to numerical differentiation.
        """
        beta, sigma_v, sigma_u = self._unpack(theta)

        eps = self.y - self.X @ beta
        sigma = np.sqrt(sigma_v**2 + sigma_u**2)
        lam = sigma_u / sigma_v

        z = eps / sigma
        t = -(lam * eps) / sigma

        phi_z = norm.pdf(z)
        phi_t = norm.pdf(t)
        Phi_t = np.maximum(norm.cdf(t), 1e-12)

        # ----------------------------------------------------------
        # dLL / dβ
        # ----------------------------------------------------------
        term_beta = (z / sigma) + (lam * phi_t / (sigma * Phi_t))
        g_beta = -(term_beta[:, None] * self.X).sum(axis=0)

        # ----------------------------------------------------------
        # dLL / d log σ_v
        # ----------------------------------------------------------
        d_sigma_dsv = sigma_v / sigma
        d_lam_dsv = -sigma_u / (sigma_v**2)

        g_sv = (
            -(sigma_v / sigma)
            + phi_z * z * d_sigma_dsv
            + phi_t / Phi_t * (
                -(eps * d_lam_dsv) / sigma +
                lam * eps * d_sigma_dsv / sigma**2
            )
        )

        g_sv = -np.sum(g_sv) * sigma_v

        # ----------------------------------------------------------
        # dLL / d log σ_u
        # ----------------------------------------------------------
        d_sigma_dsu = sigma_u / sigma
        d_lam_dsu = 1.0 / sigma_v

        dt_dsu = (
            -(eps * d_lam_dsu) / sigma +
            lam * eps * d_sigma_dsu / sigma**2
        )

        g_su = (
            -(sigma_u / sigma)
            + phi_z * z * d_sigma_dsu
            + phi_t / Phi_t * dt_dsu
        )

        g_su = -np.sum(g_su) * sigma_u

        return np.concatenate([g_beta, [g_sv, g_su]])

    # --------------------------------------------------------------
    def fit(self):
        """
        Estimate model parameters via maximum likelihood.

        OLS is used only to initialise parameters.
        """
        beta_ols = np.linalg.lstsq(self.X, self.y, rcond=None)[0]
        resid = self.y - self.X @ beta_ols
        sigma_ols = float(np.std(resid) + 1e-6)

        theta0 = np.concatenate([
            beta_ols,
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

    # --------------------------------------------------------------
    # Post-processing (JLMS inefficiency)
    # --------------------------------------------------------------
    def _postprocess(self):
        """
        Compute conditional inefficiency estimates (JLMS)
        and Technical Efficiency scores.
        """
        beta, sigma_v, sigma_u = self._unpack(self.theta)

        self.beta = beta
        self.sigma_v = sigma_v
        self.sigma_u = sigma_u

        eps = self.y - self.X @ beta
        sigma2 = sigma_v**2 + sigma_u**2

        mu_star = (sigma_u**2 * eps) / sigma2
        sigma_star = (sigma_v * sigma_u) / np.sqrt(sigma2)

        z = mu_star / sigma_star
        ratio = norm.pdf(z) / np.maximum(norm.cdf(z), 1e-12)

        u_hat = mu_star + sigma_star * ratio
        self.u_hat = np.maximum(u_hat, 0.0)

        # COST frontier efficiency
        self.TE = np.exp(-self.u_hat)
        self.frontier = self.X @ beta

    # --------------------------------------------------------------
    def summary(self):
        """
        Lightweight summary of fitted parameters and diagnostics.
        """
        gamma = self.sigma_u**2 / (self.sigma_u**2 + self.sigma_v**2)
        lam = self.sigma_u / self.sigma_v

        return {
            "beta": self.beta,
            "sigma_v": self.sigma_v,
            "sigma_u": self.sigma_u,
            "lambda": lam,
            "gamma": gamma,
            "TE_mean": float(np.mean(self.TE)),
            "TE_median": float(np.median(self.TE)),
            "log_likelihood": -self.res.fun,
            "converged": self.res.success,
        }
