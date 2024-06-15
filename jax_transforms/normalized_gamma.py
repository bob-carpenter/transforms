from dataclasses import dataclass

import jax
import jax.numpy as jnp
from tensorflow_probability.substrates.jax import bijectors, distributions

from .expanded_softmax import ExpandedSoftmax


@dataclass
class NormalizedGamma:
    alpha: jax.Array

    @property
    def distribution(self) -> distributions.Distribution:
        return distributions.ExpGamma(concentration=self.alpha, rate=1)

    def _dirichlet_log_prob(self, x):
        return distributions.Dirichlet(self.alpha).log_prob(x)

    def unconstrain(self, r_x):
        z = ExpandedSoftmax().unconstrain(r_x)
        return jax.scipy.stats.norm.ppf(self.distribution.cdf(z))

    def constrain(self, y):
        z = self.distribution.quantile(jax.scipy.stats.norm.cdf(y))
        r_x = ExpandedSoftmax().constrain(z)
        return r_x

    def constrain_with_logdetjac(self, y):
        r_x = self.constrain(y)
        r, x = r_x
        lp_norm = jnp.sum(jax.scipy.stats.norm.logpdf(y), axis=-1)
        lp_r = self.default_prior(x).log_prob(r)
        lp_dirichlet = self._dirichlet_log_prob(x)
        logJ = lp_norm - lp_dirichlet - lp_r
        return r_x, logJ

    def default_prior(self, x) -> distributions.Distribution:
        alpha_sum = jnp.sum(self.alpha, axis=-1)
        return distributions.ExpGamma(concentration=alpha_sum, rate=1)


@dataclass
class NormalizedExponential(NormalizedGamma):
    def __init__(self):
        super().__init__(alpha=jnp.ones(()))

    # the below are just more efficient implementations of the superclass methods
    @property
    def distribution(self) -> distributions.Distribution:
        return distributions.TransformedDistribution(
            distributions.Exponential(rate=1),
            bijectors.Log(),
        )

    def _dirichlet_log_prob(self, x):
        N = x.shape[-1]
        return jax.scipy.special.gammaln(N)

    def default_prior(self, x):
        N = x.shape[-1]
        return distributions.ExpGamma(concentration=N, rate=1)
