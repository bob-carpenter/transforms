from dataclasses import dataclass

import jax
import jax.numpy as jnp
from tensorflow_probability.substrates.jax import distributions

from .expanded_softmax import ExpandedSoftmax


@dataclass
class NormalizedExponential:
    N: int

    def unconstrain(self, r_x):
        z = ExpandedSoftmax(self.N).unconstrain(r_x)
        return jax.scipy.stats.norm.ppf(
            distributions.Exponential(rate=1).cdf(jnp.exp(z))
        )

    def constrain(self, y):
        z = jnp.log(
            distributions.Exponential(rate=1).quantile(jax.scipy.stats.norm.cdf(y))
        )
        r_x = ExpandedSoftmax(self.N).constrain(z)
        return r_x

    def constrain_with_logdetjac(self, y):
        r_x = self.constrain(y)
        logJ = (
            jnp.sum(jax.scipy.stats.norm.logpdf(y), axis=-1)
            - jax.scipy.special.gammaln(self.N)
            - self.default_prior(r_x)
        )
        return r_x, logJ

    def default_prior(self, r_x):
        r = r_x[..., 0]
        return distributions.ExpGamma(concentration=self.N, rate=1).log_prob(r)
