from typing import NamedTuple

import jax.numpy as jnp
from jaxtyping import Array, Float
from tensorflow_probability.substrates.jax import distributions as tfd

from ..transforms.alr import ALR


class MultiLogitNormal(NamedTuple):
    """Multivariate logit-normal distribution on the simplex.

    Parameters
    ----------
    mu : Array
        The location parameter
    L_Sigma : Array
        The lower Cholesky factor of the scale matrix parameter

    Notes
    -----
    This distribution adheres to part of the API of TensorFlow Probability's
    distribution objects:
    - `event_shape` property
    - `log_prob` method
    - `sample` method
    """

    mu: Float[Array, "*batch dim-1"]
    L_Sigma: Float[Array, "*batch dim-1 dim-1"]

    @property
    def event_shape(self):
        return self.distribution.event_shape

    @property
    def distribution(self):
        return tfd.MultivariateNormalTriL(self.mu, self.L_Sigma)

    def sample(self, *args, **kwargs) -> Float[Array, "*batch dim"]:
        y = self.distribution.sample(*args, **kwargs)
        return ALR().constrain(y)

    def log_prob(self, x: Float[Array, "*batch dim"]) -> Float[Array, "*batch"]:
        log_x = jnp.log(x)
        y = log_x[..., :-1] - log_x[..., -1:]
        return self.distribution.log_prob(y) - log_x.sum(axis=-1)
