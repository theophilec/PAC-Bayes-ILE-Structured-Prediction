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
    get_beta,
    get_lambda,
    get_sigma,
    relaxed_gd,
)


class RelaxedRegressor(BaseEstimator, RegressorMixin, MultiOutputMixin):
    def __init__(
        self,
        alpha=0.01,
        t=0.9,
        kappa=21,
        lr=1e-6,
        verbose_optimisation=True,
        tol=0,
        nmax=500,
        init_method="zeros",
    ):
        self.alpha = alpha
        self.t = t
        self.kappa = kappa
        self.lr = lr
        self.verbose_optimisation = verbose_optimisation
        self.tol = tol
        self.nmax = nmax
        self.init_method = init_method

    def fit(self, X, y):
        # check that X and y have correct shape
        self.X_, self.y_ = check_X_y(X, y, multi_output=True)

        self.n_features_ = X.shape[1]
        self.m_ = X.shape[0]
        self.dim_H_ = y.shape[1]

        # check hyperparameters
        check_kappa(self.kappa, X)  # check that kappa is valid upper bound for X
        check_alpha(self.alpha)
        check_t(self.t)

        # pre-compute
        self.sigma_ = get_sigma(self.m_, self.alpha, self.t, self.kappa)
        self.lambda_ = get_lambda(self.t, self.sigma_, self.m_, self.alpha)
        self.beta_ = get_beta(self.sigma_, self.dim_H_, X)

        # should be jnp here
        predictor_shape = (self.dim_H_, self.n_features_)
        W_0 = W_init(predictor_shape, self.init_method)

        # training with jax SGD
        W, status = relaxed_gd(
            X,
            y,
            W_0,
            self.lr,
            self.beta_,
            self.lambda_,
            self.tol,
            self.nmax,
            self.verbose_optimisation,
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
