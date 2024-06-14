from dataclasses import dataclass

import jax
import jax.numpy as jnp


@dataclass
class ExpandedSoftmax:
    N: int

    def unconstrain(self, r_x):
        r = r_x[..., :1]
        x = r_x[..., 1:]
        return jnp.log(x) + r

    def constrain(self, y):
        r = jnp.expand_dims(jax.scipy.special.logsumexp(y, axis=-1), axis=-1)
        x = jnp.exp(y - r)
        r_x = jnp.concatenate([r, x], axis=-1)
        return r_x

    def constrain_with_logdetjac(self, y):
        r = jnp.expand_dims(jax.scipy.special.logsumexp(y, axis=-1), axis=-1)
        logx = y - r
        x = jnp.exp(logx)
        r_x = jnp.concatenate([r, x], axis=-1)
        logJ = jnp.sum(logx, axis=-1)
        return r_x, logJ

    def default_prior(self, r_x):
        r = r_x[..., 0]
        return jax.scipy.stats.norm.logpdf(r, loc=jnp.log(self.N), scale=1)
