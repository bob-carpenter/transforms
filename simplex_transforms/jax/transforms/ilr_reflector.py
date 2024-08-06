import jax
import jax.numpy as jnp


def _get_V_mul_y(y):
    N = y.shape[-1] + 1
    sqrtN = jnp.sqrt(N)
    z_N = jnp.sum(y, axis=-1, keepdims=True) / sqrtN
    z = jnp.concatenate([y - z_N / (sqrtN - 1), z_N], axis=-1)
    return z


def _get_V_trans_mul_z(z):
    N = z.shape[-1]
    sqrtN = jnp.sqrt(N)
    z_minus = z[..., :-1]
    z_N = z[..., -1:]
    y = z_minus + (z_N - jnp.sum(z_minus, axis=-1, keepdims=True) / (sqrtN - 1)) / sqrtN
    return y


class ILRReflector:
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
