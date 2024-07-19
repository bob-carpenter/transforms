import jax
import jax.numpy as jnp


def _get_V_mul_y(y):
    N = y.shape[-1] + 1
    ns = jnp.arange(1, N)
    w = y / jnp.sqrt(ns * (ns + 1))
    w_rev_sum = jnp.flip(jnp.cumsum(jnp.flip(w, axis=-1), axis=-1), axis=-1)
    zeros = jnp.zeros(w_rev_sum.shape[:-1] + (1,))
    z = jnp.concatenate([w_rev_sum, zeros], axis=-1) - jnp.concatenate(
        [zeros, ns * w], axis=-1
    )
    return z


def _get_V_trans_mul_z(z):
    N = z.shape[-1]
    ns = jnp.arange(1, N)
    y = (jnp.cumsum(z[..., :-1], axis=-1) - ns * z[..., 1:]) / jnp.sqrt(ns * (ns + 1))
    return y


class ILR:
    def unconstrain(self, x):
        return _get_V_trans_mul_z(jnp.log(x))

    def constrain(self, y):
        z = _get_V_mul_y(y)
        return jax.nn.softmax(z, axis=-1)

    def constrain_with_logdetjac(self, y):
        N = y.shape[-1] + 1
        z = _get_V_mul_y(y)
        logx = jax.nn.log_softmax(z, axis=-1)
        x = jnp.exp(logx)
        logJ = jnp.sum(logx, axis=-1) + jnp.log(N) / 2
        return x, logJ
