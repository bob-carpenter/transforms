from dataclasses import dataclass, field

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


@dataclass
class ILR:
    N: int
    V: jax.Array = field(init=False)

    def __post_init__(self):
        self.V = _make_semiorthogonal_matrix(self.N)

    def unconstrain(self, x):
        return self.V.T @ jnp.log(x)

    def constrain(self, y):
        return jax.nn.softmax(self.V @ y)

    def constrain_with_logdetjac(self, y):
        z = self.V @ y
        logx = jax.nn.log_softmax(z)
        x = jnp.exp(logx)
        logJ = jnp.sum(logx) + jnp.log(self.N) / 2
        return x, logJ
