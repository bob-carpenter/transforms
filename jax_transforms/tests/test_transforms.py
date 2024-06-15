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


@pytest.mark.parametrize("transform", basic_transforms)
@pytest.mark.parametrize("N", [3, 5, 10])
@pytest.mark.parametrize("batch_dims", [(), (3,), (3, 4)])
@pytest.mark.parametrize("seed", np.random.default_rng(0).integers(0, 1000, 10))
def test_basic_transform(transform, N, batch_dims, seed):
    trans = transform()
    y = jax.random.normal(key=jax.random.key(seed), shape=batch_dims + (N - 1,))

    x = trans.constrain(y)
    assert x.shape[-1] == N
    assert jnp.all(x >= 0)
    assert jnp.allclose(jnp.sum(x, axis=-1), 1)
    assert jnp.allclose(x, vmap_over_leading_axes(trans.constrain, y))

    y2 = trans.unconstrain(x)
    assert jnp.allclose(y2, y)

    x2, logJ = trans.constrain_with_logdetjac(y)
    assert jnp.allclose(x2, x)

    logJ_expected = vmap_over_leading_axes(
        lambda y: jnp.linalg.slogdet(jax.jacobian(trans.constrain)(y)[:-1, :])[1],
        y,
    )
    assert jnp.allclose(logJ, logJ_expected)


@pytest.mark.parametrize("transform", expanded_transforms)
@pytest.mark.parametrize("N", [3, 5, 10])
@pytest.mark.parametrize("batch_dims", [(), (3,), (3, 4)])
@pytest.mark.parametrize("seed", np.random.default_rng(0).integers(0, 1000, 10))
def test_expanded_transform(transform, N, batch_dims, seed):
    trans = transform()
    y = jax.random.normal(key=jax.random.key(seed), shape=batch_dims + (N,))

    r_x = trans.constrain(y)
    x = r_x[..., 1:]
    assert x.shape[-1] == N
    assert jnp.all(x >= 0)
    assert jnp.allclose(jnp.sum(x, axis=-1), 1)
    assert jnp.allclose(r_x, vmap_over_leading_axes(trans.constrain, y))

    y2 = trans.unconstrain(r_x)
    assert jnp.allclose(y2, y)

    r_x2, logJ = trans.constrain_with_logdetjac(y)
    assert jnp.allclose(r_x2, r_x)

    logJ_expected = vmap_over_leading_axes(
        lambda y: jnp.linalg.slogdet(jax.jacobian(trans.constrain)(y)[:-1, :])[1],
        y,
    )
    assert jnp.allclose(logJ, logJ_expected)

    assert trans.default_prior(r_x).shape == batch_dims


@pytest.mark.parametrize("N", [3, 5, 10])
def test_ilr_semiorthogonal_matrix_properties(N):
    import jax_transforms.ilr

    V = jax_transforms.ilr._make_semiorthogonal_matrix(N)
    assert V.shape == (N, N - 1)
    assert jnp.allclose(V.T @ V, jnp.eye(N - 1))
    assert jnp.allclose(V.T @ jnp.ones(N), 0)
