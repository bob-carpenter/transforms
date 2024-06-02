import jax
import jax.numpy as jnp
import numpy as np
import pytest

from .. import (
    ALR,
    ILR,
    ExpandedSoftmax,
    NormalizedExponential,
    StickBreakingAngular,
    StickBreakingLogistic,
    StickBreakingNormal,
    StickBreakingPowerLogistic,
    StickBreakingPowerNormal,
)

jax.config.update("jax_enable_x64", True)

basic_transforms = [
    ALR,
    ILR,
    StickBreakingAngular,
    StickBreakingLogistic,
    StickBreakingNormal,
    StickBreakingPowerLogistic,
    StickBreakingPowerNormal,
]

expanded_transforms = [
    ExpandedSoftmax,
    NormalizedExponential,
]


@pytest.mark.parametrize("transform", basic_transforms)
@pytest.mark.parametrize("N", [3, 5, 10])
@pytest.mark.parametrize("seed", np.random.default_rng(0).integers(0, 1000, 10))
def test_basic_transform(transform, N, seed):
    trans = transform(N)
    y = jax.random.normal(key=jax.random.key(seed), shape=(N - 1,))

    x = trans.constrain(y)
    assert len(x) == N
    assert jnp.all(x >= 0)
    assert jnp.isclose(jnp.sum(x), 1)

    y2 = trans.unconstrain(x)
    assert jnp.allclose(y2, y)

    x2, logJ = trans.constrain_with_logdetjac(y)
    assert jnp.allclose(x2, x)

    J = jax.jacobian(trans.constrain)(y)
    logJ_expected = jnp.linalg.slogdet(J[:-1, :])[1]
    assert jnp.isclose(logJ, logJ_expected)


@pytest.mark.parametrize("transform", expanded_transforms)
@pytest.mark.parametrize("N", [3, 5, 10])
@pytest.mark.parametrize("seed", np.random.default_rng(0).integers(0, 1000, 10))
def test_expanded_transform(transform, N, seed):
    trans = transform(N)
    y = jax.random.normal(key=jax.random.key(seed), shape=(N,))

    r_x = trans.constrain(y)
    x = r_x[1:]
    assert len(x) == N
    assert jnp.all(x >= 0)
    assert jnp.isclose(jnp.sum(x), 1)

    y2 = trans.unconstrain(r_x)
    assert jnp.allclose(y2, y)

    r_x2, logJ = trans.constrain_with_logdetjac(y)
    assert jnp.allclose(r_x2, r_x)

    J = jax.jacobian(trans.constrain)(y)
    logJ_expected = jnp.linalg.slogdet(J[:-1, :])[1]
    assert jnp.isclose(logJ, logJ_expected)

    assert trans.default_prior(r_x).shape == ()
