import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, MultiOutputMixin
from sklearn.utils import check_array, check_X_y
from sklearn.utils.multiclass import unique_labels
from sklearn.utils.validation import check_is_fitted

from utils import (
    batch_phi,
    batch_psi,
    exhaustive,
    gaussian_sample,
    hamming_loss,
    score_bruteforce,
)


class StochasticILEMLClassifier(BaseEstimator, ClassifierMixin, MultiOutputMixin):
    def __init__(self, model, sigma=None):
        self.model = model  # e.g. Ridge(alpha=alpha, fit_intercept=False)
        self.sigma = sigma

    def fit(self, X, y):
        # Check that X and y have correct shape
        self.X_, self.y_ = check_X_y(X, y, multi_output=True)
        # Store the classes seen during fit
        self.classes_ = unique_labels(y)
        self.n_features_ = X.shape[1]
        self.m_ = X.shape[0]
        self.ell_ = y.shape[1]  # ell-binary multilabel clf

        phi_y = batch_phi(y)

        self.dim_H_ = phi_y.shape[1]

        self.grid = exhaustive(self.ell_)
        self.psi_ex = batch_psi(self.grid)
        self.model = self.model.fit(X, phi_y)  # , random_key=random_key)

        return self

    def decision_function(self, X, sample=True, n_samples=100):
        # surrogate predict
        # stochastic/deterministic decision function with n_samples
        check_is_fitted(self.model, ["coef_"])
        # Input validation
        X = check_array(X)
        if sample:
            # Return a sample from the Gaussian distribution with mean
            # self.model.coef_ and std sigma
            mus = self.model.coef_.T
            mus = mus[:, :, np.newaxis]
            mus = np.repeat(mus, n_samples, 2)  # TODO: remove repeat
            if self.sigma == None:
                self.sigma = self.model.sigma_
            weights = gaussian_sample(mus, self.sigma)
            # returns a tensor with the output of all examples and samples
            return np.tensordot(X, weights, axes=1)
        else:
            return self.model.predict(X)

    def predict(self, X, sample=True, n_samples=100):
        # Check is fit had been called
        check_is_fitted(self.model, ["coef_"])
        # Input validation
        X = check_array(X)

        if sample:
            # classification stochastic predict
            predictions = self.decision_function(X, sample=True, n_samples=n_samples)
            scores = score_bruteforce(predictions, self.psi_ex)[0]
            return np.moveaxis(self.grid[scores], 1, 2)
        else:
            # classification deterministic predict
            predictions = self.decision_function(X, sample=False)
            scores = score_bruteforce(predictions, self.psi_ex)[0]
            return self.grid[scores]

    def score(self, X, y, sample=True, n_samples=100):
        """Compute Hamming loss prediction and labels.
        Accepts 2 or 3-D y_pred.
        """
        y_pred = self.predict(X, sample=sample, n_samples=n_samples)
        return hamming_loss(y, y_pred)

    def surrogate_score(self, X, y, sample=True, n_samples=100):
        """Compute MSE of regression.
        Accepts 2 or 3-D g_x.
        """
        g_x = self.decision_function(X, sample=sample, n_samples=n_samples)
        phi_y = batch_phi(y)

        if g_x.ndim == 3:
            # in case we have a third dimension (because of multiple samples)
            phi_y = phi_y[:, :, np.newaxis]

        return np.mean((g_x - phi_y) ** 2)

    def bound_score(self, X, y, sample=True, n_samples=100, lambda_=1.0):
        g_x = self.decision_function(X, sample=sample, n_samples=n_samples)
        phi_y = batch_phi(y)
        if g_x.ndim == 3:
            # in case we have a third dimension (because of multiple samples)
            phi_y = phi_y[:, :, np.newaxis]

        return (
            np.linalg.norm(phi_y - g_x, axis=1).mean()
            + lambda_ * np.linalg.norm(self.model.coef_) ** 2
        )
