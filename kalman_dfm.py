"""
Kalman Smoother Dynamic Factor Model with EM estimation.
"""
import numpy as np
from sklearn.decomposition import PCA

class KalmanDFM:
    def __init__(self, n_assets, k_factors=3, em_iter=50):
        self.n = n_assets
        self.k = k_factors
        self.em_iter = em_iter

    def _init_params(self, Y):
        T, n = Y.shape
        pca = PCA(n_components=self.k)
        factors = pca.fit_transform(Y)
        Lambda = pca.components_.T
        A = np.eye(self.k)
        Q = np.eye(self.k) * 0.01
        R = np.diag(np.var(Y - factors @ Lambda.T, axis=0))
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
            mu_pred_t = A @ mu_t
            Sigma_pred_t = A @ Sigma_t @ A.T + Q
            H = Lambda
            v = Y[t] - H @ mu_pred_t
            S = H @ Sigma_pred_t @ H.T + R
            S_inv = np.linalg.inv(S)
            K = Sigma_pred_t @ H.T @ S_inv
            mu_upd_t = mu_pred_t + K @ v
            Sigma_upd_t = (np.eye(k) - K @ H) @ Sigma_pred_t
            ll += -0.5 * (n * np.log(2*np.pi) + np.linalg.slogdet(S)[1] + v @ S_inv @ v)
            mu_pred[t] = mu_pred_t
            Sigma_pred[t] = Sigma_pred_t
            mu_upd[t] = mu_upd_t
            Sigma_upd[t] = Sigma_upd_t
            mu_t = mu_upd_t
            Sigma_t = Sigma_upd_t
        return mu_pred, Sigma_pred, mu_upd, Sigma_upd, ll

    def _kalman_smoother(self, Y, Lambda, A, Q, R, mu0, Sigma0):
        T, n = Y.shape
        k = self.k
        mu_pred, Sigma_pred, mu_upd, Sigma_upd, _ = self._kalman_filter(Y, Lambda, A, Q, R, mu0, Sigma0)
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
        T, n = Y.shape
        mu_smooth, Sigma_smooth = self._kalman_smoother(Y, Lambda, A, Q, R, mu0, Sigma0)
        E_f = mu_smooth
        E_ff = np.array([Sigma_smooth[t] + np.outer(mu_smooth[t], mu_smooth[t]) for t in range(T)])
        # Update Lambda
        sum_Yf = np.zeros((n, self.k))
        sum_ff = np.zeros((self.k, self.k))
        for t in range(T):
            sum_Yf += np.outer(Y[t], E_f[t])
            sum_ff += E_ff[t]
        Lambda_new = sum_Yf @ np.linalg.inv(sum_ff)
        # Update A
        sum_f_fprev = np.zeros((self.k, self.k))
        sum_fprev_fprev = np.zeros((self.k, self.k))
        for t in range(1, T):
            sum_f_fprev += np.outer(E_f[t], E_f[t-1])
            sum_fprev_fprev += E_ff[t-1]
        A_new = sum_f_fprev @ np.linalg.inv(sum_fprev_fprev)
        # Update Q
        Q_new = np.zeros((self.k, self.k))
        for t in range(1, T):
            err = E_f[t] - A_new @ E_f[t-1]
            Q_new += np.outer(err, err) + Sigma_smooth[t] - A_new @ Sigma_smooth[t-1] - Sigma_smooth[t] @ A_new.T + A_new @ Sigma_smooth[t-1] @ A_new.T
        Q_new /= (T-1)
        # Update R
        R_new = np.zeros((n, n))
        for t in range(T):
            err = Y[t] - Lambda_new @ E_f[t]
            R_new += np.outer(err, err) + Lambda_new @ Sigma_smooth[t] @ Lambda_new.T
        R_new /= T
        # Ensure positive semidefiniteness
        Q_new = (Q_new + Q_new.T) / 2
        R_new = (R_new + R_new.T) / 2
        # Initial state
        mu0_new = mu_smooth[0]
        Sigma0_new = Sigma_smooth[0]
        return Lambda_new, A_new, Q_new, R_new, mu0_new, Sigma0_new

    def fit(self, Y):
        Lambda, A, Q, R, mu0, Sigma0 = self._init_params(Y)
        for it in range(self.em_iter):
            Lambda_new, A_new, Q_new, R_new, mu0_new, Sigma0_new = self._em_step(Y, Lambda, A, Q, R, mu0, Sigma0)
            delta = np.max(np.abs(Lambda_new - Lambda))
            Lambda, A, Q, R, mu0, Sigma0 = Lambda_new, A_new, Q_new, R_new, mu0_new, Sigma0_new
            if delta < 1e-4:
                break
        self.Lambda = Lambda
        self.A = A
        self.Q = Q
        self.R = R
        self.mu0 = mu0
        self.Sigma0 = Sigma0
        return self

    def forecast_returns(self, Y_last, horizon=1):
        # Run one-step prediction using last observation and current parameters
        # We'll use Kalman filter on the last observation to get updated state
        _, _, mu_upd, _, _ = self._kalman_filter(Y_last.reshape(1, -1), self.Lambda, self.A, self.Q, self.R, self.mu0, self.Sigma0)
        last_factor = mu_upd[-1]
        f_next = self.A @ last_factor
        exp_return = self.Lambda @ f_next
        return exp_return

    def get_factor_loadings(self):
        return self.Lambda
