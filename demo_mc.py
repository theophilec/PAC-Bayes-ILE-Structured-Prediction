import jax
import numpy as np
from sklearn.preprocessing import StandardScaler
from skmultilearn.dataset import load_dataset

from MCRegressor import MCRegressor
from StochasticILE import StochasticILEMLClassifier

dataset = "emotions"

# load dataset
X_train, y_train, _, _ = load_dataset(dataset, "undivided")

# need to transform to dense arrays
y_train = y_train.toarray()
X_train = X_train.toarray()

# normalise data
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)

# add an additional column of 1s to the data for the bias
X_train = np.c_[X_train, np.ones(X_train.shape[0])]


# Bound parameters (these are checked before fitting)
alpha = 0.25
t = 0.5
kappa = 22

# training parameters
lr = 1e-3
a_hat = 0

# generate random key for jax
seed = 0
random_key = jax.random.PRNGKey(seed)
regressor = MCRegressor(
    kappa=kappa,
    lr=lr,
    alpha=alpha,
    t=t,
    M=20,
    N_steps=50,
    init_method="ones",
    random_key=random_key,
    a_hat=a_hat,
)
mlclassifier = StochasticILEMLClassifier(model=regressor)
mlclassifier = mlclassifier.fit(X_train, y_train)
print(f"Sampled Hamming loss: {mlclassifier.score(X_train, y_train, sample=True)}")
print(f"Mean Hamming loss: {mlclassifier.score(X_train, y_train, sample=False)}")
print(f"Sampled Bound loss: {mlclassifier.bound_score(X_train, y_train, sample=True, lambda_=mlclassifier.model.lambda_)}")
print(f"Mean Bound loss: {mlclassifier.bound_score(X_train, y_train, sample=False, lambda_=mlclassifier.model.lambda_)}")
