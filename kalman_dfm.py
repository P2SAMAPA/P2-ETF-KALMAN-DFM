"""
Kalman Smoother Dynamic Factor Model (State‑Space).
Estimates time‑varying factor loadings via EM on a rolling window.
"""
import numpy as np
import pandas as pd
from scipy.linalg import block_diag

class KalmanDFM:
    def __init__(self, n_assets, k_factors=3, em_iter=50):
        self.n = n_assets          # number of observed series (ETFs + maybe macro)
        self.k = k_factors
        self.em_iter = em_iter
        self._initialized = False

    def _init_params(self, Y):
        """Initialize parameters via PCA on first window."""
        T, n = Y.shape
        # Factor loadings (Lambda) from PCA on all data
        from sklearn.decomposition import PCA
        pca = PCA(n_components=self.k)
        factors = pca.fit_transform(Y)   # T x k
        Lambda = pca.components_.T       # n x k
        # State transition matrix A (k x k) – identity as start
        A = np.eye(self.k)
        # Covariance of state innovation (Q)
        Q = np.eye(self.k) * 0.01
        # Covariance of observation noise (R) – diagonal
        R = np.diag(np.var(Y - factors @ Lambda.T, axis=0))
        # Initial state mean and covariance
        mu0 = np.mean(factors, axis=0)
        Sigma0 = np.cov(factors.T)
        return Lambda, A, Q, R, mu0, Sigma0

    def _kalman_filter(self, Y, Lambda, A, Q, R, mu0, Sigma0):
        T, n = Y.shape
        k = self.k
        mu_pred = np.zeros((T, k))
        Sigma_pred = np.zeros((T, k, k))
        mu_upd = np.zeros((T, k))
        Sigma_upd = np.zeros((T, k, k))
        ll = 0.0

        mu_t = mu0
        Sigma_t = Sigma0
        for t in range(T):
            # Prediction step
            mu_pred_t = A @ mu_t
            Sigma_pred_t = A @ Sigma_t @ A.T + Q
            # Update step (observation)
            H = Lambda   # n x k
            v = Y[t] - H @ mu_pred_t
            S = H @ Sigma_pred_t @ H.T + R
            K = Sigma_pred_t @ H.T @ np.linalg.inv(S)
            mu_upd_t = mu_pred_t + K @ v
            Sigma_upd_t = (np.eye(k) - K @ H) @ Sigma_pred_t
            # Likelihood contribution
            ll += -0.5 * (n * np.log(2*np.pi) + np.linalg.slogdet(S)[1] + v @ np.linalg.solve(S, v))
            # Store
            mu_pred[t] = mu_pred_t
            Sigma_pred[t] = Sigma_pred_t
            mu_upd[t] = mu_upd_t
            Sigma_upd[t] = Sigma_upd_t
            # Next
            mu_t = mu_upd_t
            Sigma_t = Sigma_upd_t
        return mu_pred, Sigma_pred, mu_upd, Sigma_upd, ll

    def _kalman_smoother(self, Y, Lambda, A, Q, R, mu0, Sigma0):
        T, n = Y.shape
        k = self.k
        mu_pred, Sigma_pred, mu_upd, Sigma_upd, _ = self._kalman_filter(Y, Lambda, A, Q, R, mu0, Sigma0)
        # Backward pass
        mu_smooth = np.zeros((T, k))
        Sigma_smooth = np.zeros((T, k, k))
        mu_smooth[-1] = mu_upd[-1]
        Sigma_smooth[-1] = Sigma_upd[-1]
        for t in range(T-2, -1, -1):
            J = Sigma_upd[t] @ A.T @ np.linalg.inv(Sigma_pred[t+1])
            mu_smooth[t] = mu_upd[t] + J @ (mu_smooth[t+1] - mu_pred[t+1])
            Sigma_smooth[t] = Sigma_upd[t] + J @ (Sigma_smooth[t+1] - Sigma_pred[t+1]) @ J.T
        return mu_smooth, Sigma_smooth

    def _em_step(self, Y, Lambda, A, Q, R, mu0, Sigma0):
        """One EM iteration: E‑step (smoother) and M‑step (update parameters)."""
        T, n = Y.shape
        mu_smooth, Sigma_smooth = self._kalman_smoother(Y, Lambda, A, Q, R, mu0, Sigma0)
        # Expectation of cross products
        E_f = mu_smooth   # T x k
        E_ff = np.array([Sigma_smooth[t] + np.outer(mu_smooth[t], mu_smooth[t]) for t in range(T)])  # Txk x k
        # M‑step: update Lambda, A, Q, R
        # Lambda update: sum_t Y[t] * E_f[t]' * inv(sum_t E_ff[t])
        sum_Yf = np.zeros((n, k))
        sum_ff = np.zeros((k, k))
        for t in range(T):
            sum_Yf += np.outer(Y[t], E_f[t])
            sum_ff += E_ff[t]
        Lambda_new = sum_Yf @ np.linalg.inv(sum_ff)
        # A update: (sum_t E_f[t] * E_f[t-1]') * inv(sum_t E_ff[t-1])
        sum_f_fprev = np.zeros((k, k))
        sum_fprev_fprev = np.zeros((k, k))
        for t in range(1, T):
            sum_f_fprev += np.outer(E_f[t], E_f[t-1])
            sum_fprev_fprev += E_ff[t-1]
        A_new = sum_f_fprev @ np.linalg.inv(sum_fprev_fprev)
        # Q update: (sum_t E_ff[t] - A @ sum_fprev_f - A @ sum_fprev_f.T + A @ sum_fprev_fprev @ A.T) / (T-1)
        # simpler: average of (f_t - A f_{t-1})^2
        resid_state = np.zeros((k, k))
        for t in range(1, T):
            resid_state += E_ff[t] - A_new @ np.outer(E_f[t-1], E_f[t]) - np.outer(E_f[t], E_f[t-1]) @ A_new.T + A_new @ E_ff[t-1] @ A_new.T
        Q_new = resid_state / (T-1)
        # R update (observation noise)
        resid_obs = np.zeros((n, n))
        for t in range(T):
            resid_obs += np.outer(Y[t], Y[t]) - Lambda_new @ np.outer(E_f[t], Y[t]) - np.outer(Y[t], E_f[t]) @ Lambda_new.T + Lambda_new @ E_ff[t] @ Lambda_new.T
        R_new = resid_obs / T
        # mu0, Sigma0 from smoothed state at t=0
        mu0_new = mu_smooth[0]
        Sigma0_new = Sigma_smooth[0]
        return Lambda_new, A_new, Q_new, R_new, mu0_new, Sigma0_new

    def fit(self, Y):
        """Fit the dynamic factor model using EM on the whole matrix Y (T x n)."""
        if not self._initialized:
            self.Lambda, self.A, self.Q, self.R, self.mu0, self.Sigma0 = self._init_params(Y)
            self._initialized = True
        for it in range(self.em_iter):
            Lambda_new, A_new, Q_new, R_new, mu0_new, Sigma0_new = self._em_step(
                Y, self.Lambda, self.A, self.Q, self.R, self.mu0, self.Sigma0
            )
            # Small tolerance check (optional)
            delta = np.max(np.abs(Lambda_new - self.Lambda))
            self.Lambda, self.A, self.Q, self.R, self.mu0, self.Sigma0 = Lambda_new, A_new, Q_new, R_new, mu0_new, Sigma0_new
            if delta < 1e-4:
                break
        return self

    def forecast_returns(self, Y_last, horizon=1):
        """
        Given the last observation Y_last (1 x n), forecast next `horizon` steps.
        Returns: expected returns (mean) for each asset (n,)
        """
        # Use Kalman filter on the last observation to update state
        # We have parameters; we can run one step ahead prediction.
        # For simplicity, use the last smoothed factor mean and transition.
        # Here we use the state transition model directly.
        # We need the last filtered factor from the historical data.
        # We'll re‑run the smoother on the whole Y to get last smoothed factor.
        _, _, mu_upd, _, _ = self._kalman_filter(Y_last.reshape(1, -1), 
                                                 self.Lambda, self.A, self.Q, self.R, self.mu0, self.Sigma0)
        last_factor = mu_upd[-1]   # k
        # Forecast future factor: f_{t+1} = A * f_t
        f_next = self.A @ last_factor
        # Forecast returns: expected return = Lambda * f_next
        exp_return = self.Lambda @ f_next
        return exp_return

    def get_factor_loadings(self):
        """Return time‑varying loadings (constant for this EM version)."""
        return self.Lambda
