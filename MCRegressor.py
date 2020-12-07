import numpy as np
from sklearn.base import BaseEstimator, MultiOutputMixin, RegressorMixin
from sklearn.metrics import r2_score
from sklearn.utils import check_array, check_X_y
from sklearn.utils.extmath import safe_sparse_dot
from sklearn.utils.validation import check_is_fitted

from optimization import (
    W_init,
    check_alpha,
    check_kappa,
    check_t,
    get_lambda,
    get_sigma,
    mc_gd_descent,
)


class MCRegressor(BaseEstimator, RegressorMixin, MultiOutputMixin):
    def __init__(
        self,
        alpha=0.01,
        t=0.9,
        kappa=21,
        random_key=None,
        a_hat=0.00,
        M=10,
        M_prime=0,
        lr=1e-2,
        N_steps=2000,
        init_method="zeros",
    ):
        self.alpha = alpha
        self.t = t
        self.kappa = kappa
        self.M = M
        self.M_prime = M_prime
        self.a_hat = a_hat
        self.lr = lr
        self.N_steps = N_steps
        self.init_method = init_method
        self.random_key = random_key

    def fit(self, X, phi_y):
        # check that X and y have correct shape
        X, phi_y = check_X_y(X, phi_y, multi_output=True)

        # params = dict with 'alpha', 't', 'kappa'
        self.n_features_ = X.shape[1]
        self.m_ = X.shape[0]
        self.dim_H_ = phi_y.shape[1]

        # check hyperparameters
        check_kappa(self.kappa, X)  # check that kappa is valid upper bound for X
        check_alpha(self.alpha)
        check_t(self.t)

        # pre-compute
        self.sigma_ = get_sigma(self.m_, self.alpha, self.t, self.kappa)
        self.lambda_ = get_lambda(self.t, self.sigma_, self.m_, self.alpha)
        print(f"Lambda: {self.lambda_}")
        print(f"Sigma: {self.sigma_}")

        # should be jnp from here --> add checks
        predictor_shape = (self.dim_H_, self.n_features_)
        W_0 = W_init(predictor_shape, self.init_method)

        # training with jax SGD
        print("MC descent")
        W, self.random_key = mc_gd_descent(
            X,
            phi_y,
            W_0,
            self.sigma_,
            self.a_hat,
            self.lambda_,
            self.lr,
            self.N_steps,
            self.M,
            self.M_prime,
            self.random_key,
        )
        W = np.array(W)
        self.coef_ = W
        return self

    def predict(self, X):
        check_is_fitted(self, ["coef_"])
        X = check_array(X)
        return safe_sparse_dot(X, self.coef_.T, dense_output=True)

    def score(self, X, phi_y):
        X, phi_y = check_X_y(X, phi_y, multi_output=True)
        phi_y_pred = self.predict(X)
        return r2_score(phi_y_pred, phi_y)
