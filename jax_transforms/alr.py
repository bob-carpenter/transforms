from dataclasses import dataclass

import jax
import jax.numpy as jnp


@dataclass
class ALR:
    N: int

    def unconstrain(self, x):
        return jnp.log(x[: self.N - 1]) - jnp.log(x[self.N - 1])

    def constrain(self, y):
        return jax.nn.softmax(jnp.append(y, 0))

    def constrain_with_logdetjac(self, y):
        z = jnp.append(y, 0)
        r = jax.scipy.special.logsumexp(z)
        logx = z - r
        x = jnp.exp(logx)
        logJ = jnp.sum(logx)
        return x, logJ
