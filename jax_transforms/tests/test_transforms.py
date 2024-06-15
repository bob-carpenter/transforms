import jax
import jax.numpy as jnp
import numpy as np
import pytest

from .. import (
    ALR,
    ILR,
    ExpandedSoftmax,
    NormalizedExponential,
    StickbreakingAngular,
    StickbreakingLogistic,
    StickbreakingNormal,
    StickbreakingPowerLogistic,
    StickbreakingPowerNormal,
)
from ..util import vmap_over_leading_axes

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
]


def _allclose(x, y):
    if isinstance(x, tuple) and isinstance(y, tuple):
        return jnp.all(jnp.array([jnp.allclose(xi, yi) for xi, yi in zip(x, y)]))
    elif isinstance(x, tuple) or isinstance(y, tuple):
        raise ValueError("x and y must both be tuples or neither")
    else:
        return jnp.allclose(x, y)


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


@pytest.mark.parametrize("transform", basic_transforms + expanded_transforms)
@pytest.mark.parametrize("N", [3, 5, 10])
@pytest.mark.parametrize("batch_dims", [(), (3,), (3, 4)])
@pytest.mark.parametrize("seed", np.random.default_rng(0).integers(0, 1000, 10))
def test_transform(transform, N, batch_dims, seed):
    trans = transform()
    M = N if transform in expanded_transforms else N - 1
    y = jax.random.normal(key=jax.random.key(seed), shape=batch_dims + (M,))

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
        assert trans.default_prior(x).shape == batch_dims
        r, x = x
        assert r.shape == batch_dims
    assert x.shape[-1] == N
    assert jnp.all(x >= 0)
    assert jnp.allclose(jnp.sum(x, axis=-1), 1)


@pytest.mark.parametrize("N", [3, 5, 10])
def test_ilr_semiorthogonal_matrix_properties(N):
    import jax_transforms.ilr

    V = jax_transforms.ilr._make_semiorthogonal_matrix(N)
    assert V.shape == (N, N - 1)
    assert jnp.allclose(V.T @ V, jnp.eye(N - 1))
    assert jnp.allclose(V.T @ jnp.ones(N), 0)
