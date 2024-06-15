import jax
import jax.numpy as jnp


class ALR:
    def unconstrain(self, x):
        return jnp.log(x[..., :-1]) - jnp.log(x[..., -1:])

    def constrain(self, y):
        z = jnp.concatenate([y, jnp.zeros(y.shape[:-1] + (1,))], axis=-1)
        return jax.nn.softmax(z, axis=-1)

    def constrain_with_logdetjac(self, y):
        z = jnp.concatenate([y, jnp.zeros(y.shape[:-1] + (1,))], axis=-1)
        logx = jax.nn.log_softmax(z, axis=-1)
        x = jnp.exp(logx)
        logJ = jnp.sum(logx, axis=-1)
        return x, logJ
