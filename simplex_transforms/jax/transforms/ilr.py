import jax
import jax.numpy as jnp


def _make_helmert_matrix(N: int):
    H1 = jnp.full(N, 1 / jnp.sqrt(N))
    H22 = jnp.tril(jnp.ones((N - 1, N - 1)))
    ns = jnp.arange(1, N)
    H22 = H22.at[jnp.diag_indices_from(H22)].set(-ns) / jnp.sqrt(ns * (ns + 1)).reshape(
        -1, 1
    )
    H21 = H22[:, :1].at[0].multiply(-1)
    H = jnp.block([[H1], [H21, H22]])
    return H


def _make_semiorthogonal_matrix(N: int):
    H = _make_helmert_matrix(N)
    V = H.T[:, 1:]
    return V


class ILR:
    def unconstrain(self, x):
        N = x.shape[-1]
        V = _make_semiorthogonal_matrix(N)
        return jnp.dot(jnp.log(x), V)

    def constrain(self, y):
        N = y.shape[-1] + 1
        V = _make_semiorthogonal_matrix(N)
        return jax.nn.softmax(jnp.dot(y, V.T), axis=-1)

    def constrain_with_logdetjac(self, y):
        N = y.shape[-1] + 1
        V = _make_semiorthogonal_matrix(N)
        z = jnp.dot(y, V.T)
        logx = jax.nn.log_softmax(z, axis=-1)
        x = jnp.exp(logx)
        logJ = jnp.sum(logx, axis=-1) + jnp.log(N) / 2
        return x, logJ
