import numpy as np
from sklearn.preprocessing import StandardScaler
from skmultilearn.dataset import load_dataset

from RelaxedRegressor import RelaxedRegressor
from StochasticILE import StochasticILEMLClassifier


dataset = "emotions"

# load dataset
X_train, y_train, _, _ = load_dataset(dataset, "undivided")

# need to transform to dense arrays
y_train = y_train.toarray()
X_train = X_train.toarray()

scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)

# add an additional column of 1s to the data for the bias
X_train = np.c_[X_train, np.ones(X_train.shape[0])]


# Bound parameters (these are checked before fitting)
kappa = 22
alpha = 0.1
t = 0.9

# training parameters
lr = 1e-3
nmax = 2000
init_method = "ones"  # or "zeros"

regressor = RelaxedRegressor(
    kappa=kappa, lr=lr, alpha=alpha, t=t, nmax=nmax, init_method=init_method
)
mlclassifier = StochasticILEMLClassifier(model=regressor)
mlclassifier = mlclassifier.fit(X_train, y_train)

print(f"Sampled Hamming loss: {mlclassifier.score(X_train, y_train, sample=True)}")
print(f"Mean Hamming loss: {mlclassifier.score(X_train, y_train, sample=False)}")
print(f"Sampled Bound loss: {mlclassifier.bound_score(X_train, y_train, sample=True, lambda_=mlclassifier.model.lambda_)}")
print(f"Mean Bound loss: {mlclassifier.bound_score(X_train, y_train, sample=False, lambda_=mlclassifier.model.lambda_)}")
