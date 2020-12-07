import itertools

import numpy as np
from sklearn.model_selection import GridSearchCV


def hamming_loss(y, y_hat):
    assert len(y.shape) == 2
    if len(y_hat.shape) == 3:
        y = y[:, :, np.newaxis]
    return 1.0 - np.equal(y, y_hat).astype(int).mean()


def mse_loss(y, y_hat):
    phi_y = batch_phi(y)
    if len(y_hat.shape) == 3:
        phi_y = phi_y[:, :, np.newaxis]
    return np.mean((y_hat - phi_y) ** 2)


def exhaustive(n_labels):
    """Generate array of 2**n_labels binary vectors."""
    return np.array(list(itertools.product([0, 1], repeat=n_labels)))


# Hamming phi and psi
def interweave(y):
    arr = np.empty((y.shape[0], y.shape[1] + y.shape[1]), dtype=y.dtype)
    arr[:, 0::2] = np.array(np.logical_not(y), dtype="int")
    arr[:, 1::2] = y
    return arr


def batch_phi(y):
    """Batch encode labels."""
    return np.hstack(
        [np.ones(y.shape[0])[:, np.newaxis], -interweave(y) / np.sqrt(y.shape[1])]
    )


def batch_psi(y):
    return np.hstack(
        [np.ones(y.shape[0])[:, np.newaxis], interweave(y) / np.sqrt(y.shape[1])]
    )


def score_bruteforce(predicted_phi, psi_ex):
    # need tensordot to cover the cases when we are sampling
    out = np.tensordot(predicted_phi, psi_ex.T, axes=((1), (0)))
    if predicted_phi.ndim == 3:
        return np.argmin(out, 2), np.min(out, 2)
    else:
        return np.argmin(out, 1), np.min(out, 1)


def gaussian_sample(mu, sigma):
    epsilon = np.random.normal(0, 1, mu.shape)
    return mu + sigma * epsilon
