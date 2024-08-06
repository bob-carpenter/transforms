from functools import partial
from typing import NamedTuple, Optional, Tuple

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Int


# utilities for adding, removing trailing dims for broadcasting x with y
def _expand_as(x, y):
    while x.ndim < y.ndim:
        x = jnp.expand_dims(x, axis=-1)
    return x


def _squeeze_as(x, y):
    while x.ndim > y.ndim:
        x = jnp.squeeze(x, axis=-1)
    return x


class WelfordState(NamedTuple):
    """State of the Welford algorithm for computing running mean and variance."""

    mean: Float[Array, "*batch dims"]
    sum_sq_delta: Float[Array, "*batch dims"]  # sum of squared differences from mean
    n: Int[Array, "*batch"]


def _update_welford_state(
    state: WelfordState, x: Float[Array, "*batch dims"]
) -> WelfordState:
    mean, sum_sq_delta, n_old = state
    n = _expand_as(n_old, mean) + 1
    delta = x - mean
    delta_n = delta / n
    mean += delta_n
    sum_sq_delta += (n - 1) * delta_n * delta
    return WelfordState(mean, sum_sq_delta, _squeeze_as(n, n_old))


def _combine_welford_states(s1: WelfordState, s2: WelfordState) -> WelfordState:
    mean1, sum_sq_delta1, n1 = s1
    mean2, sum_sq_delta2, n2 = s2
    n1, n2 = _expand_as(n1, mean1), _expand_as(n2, mean2)
    n = n1 + n2
    delta = mean2 - mean1
    delta_n2_n = n2 * delta / n
    mean = mean1 + delta_n2_n
    sum_sq_delta = sum_sq_delta1 + sum_sq_delta2 + n1 * delta * delta_n2_n
    return WelfordState(mean, sum_sq_delta, _squeeze_as(n, s1.n))


def _get_mean_and_se(state: WelfordState):
    mean, sum_sq_delta, n = state
    n = _expand_as(n, sum_sq_delta)
    se_mean = jnp.sqrt((sum_sq_delta / (n - 1)) / n)
    return mean, se_mean


def _create_initial_welford_state(dist, key, func, batch_shape) -> WelfordState:
    # sample from the distribution and call the function to get the shape of the output
    x = dist.sample(seed=key)
    y = func(x)
    shape = batch_shape + y.shape
    # initialize the state with zero draws
    return WelfordState(
        mean=jnp.zeros(shape),
        sum_sq_delta=jnp.zeros(shape),
        n=jnp.zeros(batch_shape, dtype=jnp.int32),
    )


@partial(jax.jit, static_argnums=(2, 3))
def compute_expectations_and_ses(
    key,
    dist,
    num_draws: int,
    batch_size: Optional[int] = None,
    funcs: Optional[Tuple[jax.tree_util.Partial]] = None,
) -> Tuple[jax.Array, jax.Array]:
    """Estimate the expectation of one or more functions and their standard errors over a distribution.

    Parameters
    ----------
    key : jax.random.PRNGKey
        The random key to use for generating samples.
    dist : Any
        The distribution whose mean with shape `(*dims)` will be estimated. Should adhere to the
        sampling API of `tensorflow_probability.substrates.jax.distributions.Distribution`.
    num_draws : int
        The number of samples to draw from `dist`.
    batch_size : int, optional
        The number of samples to draw in each batch. This must be a divisor of
        `num_draws` and should be chosen such that an array of size `(num_draws, *dims)` can fit into
        memory. If `None`, `batch_size` is set to `num_draws`.
    funcs : tuple of jax.tree_util.Partial, optional
        Functions to be applied to draws from `dist` before taking the expectation. Each function
        must broadcast over any batch dimensions of its inputs. If none are provided, then only the
        mean and its standard error are computed.

    Returns
    -------
    expectations : tuple of jax.Array
        The `i`th entry is the expected value of `func[i]` over the distribution.
    ses : tuple of jax.Array
        The estimated Monte Carlo standard error of each entry in `expectations`.
    """
    if batch_size is None:
        batch_size = num_draws
    num_batches = num_draws // batch_size
    assert num_draws % batch_size == 0, "num_draws must be divisible by batch_size"
    batch_shape = (batch_size,)

    if funcs is None:
        funcs = (jax.tree_util.Partial(lambda x: x),)

    def update_batch_state(states, key):
        x = dist.sample(seed=key, sample_shape=(batch_size,))
        states = tuple(
            (
                _update_welford_state(state, func(x))
                for func, state in zip(funcs, states)
            )
        )
        return states, None

    def combine_states(states1, states2):
        combined_states = tuple(
            (
                _combine_welford_states(state1, state2)
                for state1, state2 in zip(states1, states2)
            )
        )
        return combined_states, None

    # compute running expectation and sum of squared differences for multiple batches in parallel
    initial_states = tuple(
        (_create_initial_welford_state(dist, key, func, batch_shape) for func in funcs)
    )
    keys = jax.random.split(key, num_batches)
    states, _ = jax.lax.scan(update_batch_state, initial_states, keys)

    # combine each batch to a single running expectation and sum of squared differences
    initial_states = tuple(
        (_create_initial_welford_state(dist, key, func, ()) for func in funcs)
    )
    states, _ = jax.lax.scan(combine_states, initial_states, xs=states)

    means, ses = tuple(zip(*((_get_mean_and_se(state) for state in states))))
    return means, ses
