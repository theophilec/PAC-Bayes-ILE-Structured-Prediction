import chex  # testing for jax
import jax
import jax.numpy as jnp
from jax import jit
from jax.experimental import optimizers
from jax.scipy.stats.multivariate_normal import logpdf

# Validating hyperparameters


def check_kappa(kappa, X, tol=1e-5):
    norms = jnp.linalg.norm(X, axis=1)
    print(norms.max())
    assert (norms < kappa + tol).all()


def check_t(t):
    assert 0 < t < 1


def check_alpha(alpha):
    assert 0 < alpha < 1


# Computing subsequent parameters


def get_sigma(m, alpha, t, kappa):
    return t * m ** (1 - 2 * alpha) / kappa ** 2


def get_lambda(t, sigma, m, alpha):
    return 1 / (2 * t * sigma ** 2 * m ** alpha)


def get_beta(sigma, d_H, X):
    X_norms_2 = jnp.linalg.norm(X, axis=1) ** 2
    assert X.shape[0] == X_norms_2.shape[0]
    return sigma ** 2 * d_H * X_norms_2


def j_MSE_loss(y_pred, y_true):
    """
    Compute MSE loss between y_pred and y_true with JAX.

    Args:
        y_pred (ndarray): Array of rank 2 or 3
        y_true (ndarray): Array of rank 2
    Returns:
        float: MSE loss between y_pred and y_true where
        y_true is repeated if y_pred is of rank 3.

    """
    chex.assert_rank(y_true, 2)
    diff = y_pred - y_true
    chex.assert_equal_shape([diff, y_pred])
    return jnp.mean(diff ** 2)


@jit
def linear(W, X):
    """
    Compute linear transformation of X with coeff W.

    Args:
        W (ndarray): Array of rank 2 or 3. Shape should be (d_y, d_x, N_p).
        X (ndarray): Array of rank 2. Shape should be (N, d_x).
    Returns:
        ndarray: y = WX + b with rank promotion as necessary.

    y is (N, d_y) or (N, d_y, N_p)
    """
    assert W.shape[1] == X.shape[1]
    f = jnp.tensordot(X, W, axes=[1, 1])
    assert W.shape[0] == f.shape[1]
    return f


def partial_relaxed_loss(g_x, phi_y, beta):
    """
    Compute the "relaxed loss" from Cantelobre et al., 2020 (eq 34),
    without regularization term.

    Args:
        g_x (ndarray): regressed value (rank 2)
        phi_y (ndarray): phi(y) labels (rank 2)
        beta (ndarray): rank 1, size N
    Returns:
        float: see equation (34)
    """
    res = phi_y - g_x
    return jnp.sqrt(beta[:, jnp.newaxis] + (res ** 2)).mean()


def relaxed_predict_loss(W, X, phi_y, beta, lambda_):
    """Compute \hat J_c (see eq 34).

    Uses jax.numpy arrays and functions, for autodiff.
    """
    g_x = linear(W, X)
    g_norm_2 = (W ** 2).sum()
    return partial_relaxed_loss(g_x, phi_y, beta) + lambda_ * g_norm_2


# Gradient descent


def W_init(shape, input_method="zeros"):
    if input_method == "ones":
        return jnp.ones(shape)
    elif input_method == "zeros":
        return jnp.zeros(shape)
    else:
        raise NotImplementedError


def relaxed_gd(X, phi_y, W_0, lr, beta_, lambda_, tol, N_max, verbose=True):
    """Run gradient descent on relaxed_loss."""
    print(f"Lambda {lambda_}")
    dim_H_ = phi_y.shape[1]
    n_features_ = X.shape[1]
    predictor_shape = (
        dim_H_,
        n_features_,
    )
    assert W_0.shape == predictor_shape

    opt_init, opt_update, get_params = optimizers.sgd(lr)
    opt_state = opt_init(W_0)

    def step(opt_state, X, phi_y, beta, lambda_, i):
        value, grads = jax.value_and_grad(relaxed_predict_loss)(
            get_params(opt_state), X, phi_y, beta, lambda_
        )
        if (i < 10 or i % 100 == 0) and verbose:
            print(f"Step {i} value: \t\t{value.item()}")
        opt_state = opt_update(i, grads, opt_state)
        return value, opt_state

    old_value = jnp.inf
    diff = jnp.inf
    step_index = 0

    while step_index < N_max:
        value, opt_state = step(opt_state, X, phi_y, beta_, lambda_, step_index)

        diff = abs(old_value - value)
        old_value = value
        step_index += 1

    status = {
        "max_steps": step_index >= N_max,
        "tol_max": tol >= diff.item(),
        "diff": diff.item(),
        "step_index": step_index,
        "final_value": value.item(),
    }
    if verbose:
        print(status)

    return get_params(opt_state), status


## MC Gradient descent


def loss(V, X, phi_y):
    phi_y_hat = linear(V, X)
    phi_y_ = phi_y[:, :, jnp.newaxis]
    phi_y_ = phi_y_.repeat(V.shape[-1], 2)
    res = phi_y_hat - phi_y_
    return jnp.sqrt((res ** 2).sum(axis=1)).mean()


@jit
def losses(V, X, phi_y):
    phi_y_hat = linear(V, X)
    phi_y_ = phi_y[:, :, jnp.newaxis]
    phi_y_ = phi_y_.repeat(V.shape[-1], 2)
    res = phi_y_hat - phi_y_
    return jnp.sqrt((res ** 2).sum(axis=1)).mean(axis=0)


@jit
def cv(V, X, phi_y):
    """Compute control variate at V.

    Reference: equation 42."""
    M = V.shape[-1]
    phi_y_hat = linear(V, X)
    phi_y_ = phi_y[:, :, jnp.newaxis]
    phi_y_ = phi_y_.repeat(M, 2)
    res = phi_y_ - phi_y_hat
    return (res ** 2).sum(axis=1).mean(axis=0)


def cv_exp(W, sigma, X, phi_y):
    """Compute CV expectation over N(mu, sigma).

    where W = mu.reshape(dim_H, dim_F).
    """
    N = phi_y.shape[1] * X.shape[1]

    phi_y_hat = linear(W, X)
    res = phi_y - phi_y_hat
    mean_sqs = (res ** 2).sum(axis=1)
    assert mean_sqs.shape[0] == X.shape[0]

    variances = sigma ** 2 * N * (X ** 2).sum(axis=1)
    assert variances.shape[0] == X.shape[0]

    return mean_sqs.mean() + variances.mean()


grad_cv_exp = jax.grad(cv_exp)


def log_normal_batch(W, sigma, Vs):
    mu = W.ravel()
    Vs_ = Vs.reshape((-1, Vs.shape[-1])).T
    # Vs_ is of shape M, N
    assert mu.shape[0] == Vs_.shape[1]
    d = mu.shape[0]
    cov = sigma ** 2 * jnp.eye(d)
    return logpdf(Vs_, mu, cov)


jac_log_normal = jax.jacfwd(log_normal_batch)


def log_normal(W, sigma, V):
    mu = W.ravel()
    V_ = V.ravel()
    d = mu.shape[0]
    cov = sigma ** 2 * jnp.eye(d)
    return logpdf(V_, mu, cov)


grad_log_normal = jax.grad(log_normal)


def eta_M(W, sigma, Vs, X, phi_y, a_hat):
    M = Vs.shape[-1]
    L_Vs = losses(Vs, X, phi_y)
    assert L_Vs.shape[0] == M
    B_Vs = cv(Vs, X, phi_y)
    assert B_Vs.shape[0] == M
    jacobian_Vs = jac_log_normal(W, sigma, Vs)
    chex.assert_rank([L_Vs, B_Vs, a_hat], [1, 1, 0])
    diff = L_Vs - a_hat * B_Vs
    eta = jnp.tensordot(jacobian_Vs, diff, axes=[0, 0]) / M
    assert eta.shape == Vs[:, :, 0].shape
    return eta, L_Vs.mean()


def cv_full_grad(W, sigma, Vs, X, phi_y, a_hat, lambda_):
    # a_hat = 0 no covariate
    eta_M_, batch_loss = eta_M(W, sigma, Vs, X, phi_y, a_hat)
    # TODO
    return (
        eta_M_ + a_hat * grad_cv_exp(W, sigma, X, phi_y) + 2 * lambda_ * W,
        batch_loss,
    )


def sample(M, random_key, W, sigma):
    d_H, d_F = W.shape
    sample_key, random_key = jax.random.split(random_key)
    samples = jax.random.normal(sample_key, (d_H, d_F, M))
    return sigma * samples + W[:, :, jnp.newaxis], random_key


def mc_gd_step(X, phi_y, W, sigma, a_hat, M, lambda_, lr, random_key, verbose=True):
    Vs, random_key = sample(M, random_key, W, sigma)
    grad, batch_loss = cv_full_grad(W, sigma, Vs, X, phi_y, a_hat, lambda_)
    return W - lr * grad, random_key, batch_loss


def mc_gd_descent(
    X, phi_y, W_0, sigma, a_hat, lambda_, lr, N_steps, M, M_prime, random_key
):
    W = W_0
    for i in range(N_steps):
        # compute a_hat with M_prime
        W, random_key, batch_loss = mc_gd_step(
            X, phi_y, W, sigma, a_hat, M, lambda_, lr, random_key
        )
        if i < 10 or i % 10 == 0:
            print(f"Step {i} value: \t\t{batch_loss.item()}")
        if jnp.isnan(W).any():
            break
    return W, random_key
