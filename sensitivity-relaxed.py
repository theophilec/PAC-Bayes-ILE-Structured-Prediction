import itertools
import pickle
import random

import ipdb
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import make_scorer
from sklearn.preprocessing import StandardScaler
from skmultilearn.dataset import load_dataset

from bounds import Gigere_bound, our_bound
from optimization import get_lambda, get_sigma
from RelaxedRegressor import RelaxedRegressor
from StochasticILE import StochasticILEMLClassifier
from utils import hamming_loss, mse_loss, run_crossvalidation, train_and_report
from VanillaILE import VanillaILE

dataset = "emotions"  # 'scene'

# load dataset
X_train, y_train, _, _ = load_dataset(dataset, "undivided")
# X_test, y_test, _, _ = load_dataset(dataset, "test")

# need to transform to dense arrays
y_train = y_train.toarray()
# y_test = y_test.toarray()
X_train = X_train.toarray()
# X_test = X_test.toarray()

# normalise data
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
# X_test = scaler.transform(X_test)

# add an addicional column of 1s to the data for the bias
X_train = np.c_[X_train, np.ones(X_train.shape[0])]
# X_test = np.c_[X_test, np.ones(X_test.shape[0])]

kappa = 22
alpha_tab = np.array(
    [1e-4, 1e-3, 1e-2, 1e-1, 0.2, 0.3, 0.4, 0.5]
)  # , 1 - 1e-4, 1 - 1e-3, 1 - 1e-2, 1 - 1e-1, 1 - 0.2, 1- 0.3, 1- 0.4,])
# t_tab = np.array([1e-4, 1e-3, 1e-2, 1e-1, 0.2, 0.3, 0.4, 0.5, 1 - 1e-4, 1 - 1e-3, 1 - 1e-2, 1 - 1e-1, 1 - 0.2, 1- 0.3, 1- 0.4,])
t_tab = np.array(
    [0.5, 1 - 1e-4, 1 - 1e-3, 1 - 1e-2, 1 - 1e-1, 1 - 0.2, 1 - 0.3, 1 - 0.4,]
)
lr_tab = [1e-8, 1e-4, 1e-3, 1e-2]
lr_tab.reverse()
t_tab.sort()
alpha_tab.sort()
# lambda_tab = np.logspace(-6, 0, 5)

results_ = []

m_ = X_train.shape[0]


for lr, t, alpha in itertools.product(lr_tab, t_tab, alpha_tab):
    regressor = RelaxedRegressor(
        kappa=kappa, lr=lr, alpha=alpha, t=t, nmax=1000, init_method="zeros"
    )
    mlclassifier = StochasticILEMLClassifier(model=regressor)
    mlclassifier = mlclassifier.fit(X_train, y_train)
    sigma_ = get_sigma(m_, alpha, t, kappa)
    lambda_ = get_lambda(t, sigma_, m_, alpha)
    results = {}
    results["lr"] = lr
    results["t"] = t
    results["alpha"] = alpha
    results["sigma"] = mlclassifier.model.sigma_
    results["lambda"] = mlclassifier.model.lambda_
    results["loss_Q"] = mlclassifier.bound_score(
        X_train, y_train, lambda_=lambda_, sample=True, n_samples=1000
    )
    results["Hamming_train_Q"] = mlclassifier.score(X_train, y_train, sample=True)
    results["MSE_train_Q"] = mlclassifier.surrogate_score(
        X_train, y_train, sample=True, n_samples=1000
    )
    results["Hamming_train_mean"] = mlclassifier.score(X_train, y_train, sample=False)
    results["MSE_train_mean"] = mlclassifier.surrogate_score(
        X_train, y_train, sample=False
    )

    print(results)
    results_.append(results)

with open(dataset + "-heatmap-relaxed.pkl", "wb") as f:
    pickle.dump(results_, f)
ipdb.set_trace()
