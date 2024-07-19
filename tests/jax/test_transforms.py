import jax
import jax.numpy as jnp
import numpy as np
import pytest

from simplex_transforms.jax.transforms import (
    ALR,
    ILR,
    ExpandedSoftmax,
    NormalizedExponential,
    NormalizedGamma,
    StickbreakingAngular,
    StickbreakingLogistic,
    StickbreakingNormal,
    StickbreakingPowerLogistic,
    StickbreakingPowerNormal,
)
from simplex_transforms.jax.utils import vmap_over_leading_axes

jax.config.update("jax_enable_x64", True)

basic_transforms = [
    ALR,
    ILR,
    StickbreakingAngular,
    StickbreakingLogistic,
    StickbreakingNormal,
    StickbreakingPowerLogistic,
    StickbreakingPowerNormal,
]

expanded_transforms = [
    ExpandedSoftmax,
    NormalizedExponential,
    NormalizedGamma,
]


def _allclose(x, y, **kwargs):
    if isinstance(x, tuple) and isinstance(y, tuple):
        return jnp.all(jnp.array([_allclose(xi, yi, **kwargs) for xi, yi in zip(x, y)]))
    elif isinstance(x, tuple) or isinstance(y, tuple):
        raise ValueError("x and y must both be tuples or neither")
    else:
        return jnp.allclose(x, y, **kwargs)


def _make_helmert_matrix(N: int):
    H1 = jnp.full(N, 1 / jnp.sqrt(N))
    H22 = jnp.tril(jnp.ones((N - 1, N - 1)))
    ns = jnp.arange(1, N)
    H22 = H22.at[jnp.diag_indices_from(H22)].set(-ns) / jnp.sqrt(ns * (ns + 1)).reshape(
        -1, 1
    )
    H21 = H22[:, :1].at[0].multiply(-1)
    H = jnp.block([[H1], [H21, H22]])
    return H


def _make_ilr_semiorthogonal_matrix(N: int):
    H = _make_helmert_matrix(N)
    V = H.T[:, 1:]
    return V


def logdetjac(f):
    jac = jax.jacobian(f)

    def logdetjac_f(y):
        J = jac(y)
        if isinstance(J, tuple):
            J_r, J_x = J
            # stack J_r as a vector over J_x[:-1, :]
            J_square = jnp.concatenate(
                [jnp.expand_dims(J_r, axis=-2), J_x[..., :-1, :]], axis=-2
            )
        else:
            J_square = J[..., :-1, :]
        return jnp.linalg.slogdet(J_square)[1]

    return logdetjac_f


def get_random_params(transform_type, N: int, key) -> tuple:
    if transform_type is NormalizedGamma:
        alpha = jax.random.exponential(key, shape=(N,))
        return (alpha,)
    else:
        return ()


@pytest.mark.parametrize("seed", np.random.default_rng(0).integers(0, 1000, 5))
@pytest.mark.parametrize("N", [3, 5, 10])
@pytest.mark.parametrize("batch_dims", [(), (3,), (3, 4)])
@pytest.mark.parametrize("transform", basic_transforms + expanded_transforms)
def test_transform(transform, N, batch_dims, seed):
    key, subkey = jax.random.split(jax.random.key(seed))
    trans = transform(*get_random_params(transform, N, subkey))
    M = N if transform in expanded_transforms else N - 1
    y = jax.random.normal(key=key, shape=batch_dims + (M,))

    # check consistency of various methods
    x = trans.constrain(y)
    y2 = trans.unconstrain(x)
    x2, logJ = trans.constrain_with_logdetjac(y)
    assert _allclose(x, vmap_over_leading_axes(trans.constrain, y))
    assert _allclose(x, x2)
    assert jnp.allclose(y2, y)
    is_expanded = isinstance(x, tuple)
    logJ_expected = vmap_over_leading_axes(logdetjac(trans.constrain), y)
    assert jnp.allclose(logJ, logJ_expected)

    # verify basic properties are satisfied
    if is_expanded:
        r, x = x
        assert r.shape == batch_dims
        assert trans.default_prior(x).log_prob(r).shape == batch_dims
    assert x.shape[-1] == N
    assert jnp.all(x >= 0)
    assert jnp.allclose(jnp.sum(x, axis=-1), 1)


@pytest.mark.parametrize("N", [3, 5, 7])
def test_normalized_transforms_consistent(N, seed=42):
    alpha = jnp.ones(N)
    trans_exp = NormalizedExponential()
    trans_gamma = NormalizedGamma(alpha)

    y = jax.random.normal(key=jax.random.key(seed), shape=(N,))
    r_x = trans_exp.constrain(y)
    r, x = r_x
    assert _allclose(r_x, trans_gamma.constrain(y))
    assert jnp.allclose(trans_exp.unconstrain(r_x), trans_gamma.unconstrain(r_x))
    assert _allclose(
        trans_exp.constrain_with_logdetjac(y),
        trans_gamma.constrain_with_logdetjac(y),
    )
    assert jnp.allclose(
        trans_exp.default_prior(x).log_prob(r),
        trans_gamma.default_prior(x).log_prob(r),
    )


@pytest.mark.parametrize("N", [3, 5, 10])
def test_ilr_semiorthogonal_matrix_properties(N, seed=87):
    from simplex_transforms.jax.transforms import ilr

    V = _make_ilr_semiorthogonal_matrix(N)
    assert V.shape == (N, N - 1)
    assert jnp.allclose(V.T @ V, jnp.eye(N - 1))
    assert jnp.allclose(V.T @ jnp.ones(N), 0)
    y = jax.random.normal(key=jax.random.key(seed), shape=(N - 1,))
    z = V @ y
    assert jnp.allclose(ilr._get_V_mul_y(y), z)
    assert jnp.allclose(ilr._get_V_trans_mul_z(z), y)
