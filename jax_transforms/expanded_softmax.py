import jax
import jax.numpy as jnp
from tensorflow_probability.substrates.jax import distributions


class ExpandedSoftmax:
    def unconstrain(self, r_x):
        r, x = r_x
        return jnp.log(x) + jnp.expand_dims(r, axis=-1)

    def constrain(self, y):
        r = jax.scipy.special.logsumexp(y, axis=-1)
        x = jnp.exp(y - jnp.expand_dims(r, axis=-1))
        return r, x

    def constrain_with_logdetjac(self, y):
        r = jax.scipy.special.logsumexp(y, axis=-1)
        logx = y - jnp.expand_dims(r, axis=-1)
        x = jnp.exp(logx)
        logJ = jnp.sum(logx, axis=-1)
        return (r, x), logJ

    def default_prior(self, x) -> distributions.Distribution:
        N = x.shape[-1]
        return distributions.Normal(loc=jnp.log(N), scale=1)
