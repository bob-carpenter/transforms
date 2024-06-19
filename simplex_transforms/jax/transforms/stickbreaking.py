from dataclasses import dataclass

import jax
import jax.numpy as jnp
from tensorflow_probability.substrates.jax import distributions

from ..utils import vmap_over_leading_axes


class StickbreakingBase:
    def _unconstrain_single(self, x):
        def _running_remainder(remainder, xi):
            return remainder - xi, remainder

        z = x[:-1] / jax.lax.scan(_running_remainder, 1, x[:-1])[1]
        return z

    def unconstrain(self, x):
        return vmap_over_leading_axes(self._unconstrain_single, x)

    def _constrain_single(self, z):
        def _break_stick(remainder, zi):
            xi = remainder * zi
            remainder -= xi
            return remainder, xi

        x_N, x_minus = jax.lax.scan(_break_stick, 1, z)
        x = jnp.append(x_minus, x_N)
        return x

    def constrain(self, z):
        return vmap_over_leading_axes(self._constrain_single, z)

    def constrain_with_logdetjac(self, z):
        N = z.shape[-1] + 1
        x = self.constrain(z)
        logJ = jnp.inner(jnp.arange(N - 2, 0, -1), jnp.log1p(-z[..., :-1]))
        return x, logJ


class StickbreakingCDF:
    def get_distribution(self, N: int) -> distributions.Distribution:
        raise NotImplementedError

    def unconstrain(self, x):
        N = x.shape[-1]
        z = StickbreakingBase().unconstrain(x)
        y = self.get_distribution(N).quantile(z)
        return y

    def constrain(self, y):
        N = y.shape[-1] + 1
        z = self.get_distribution(N).cdf(y)
        x = StickbreakingBase().constrain(z)
        return x

    def constrain_with_logdetjac(self, y):
        N = y.shape[-1] + 1
        distribution = self.get_distribution(N)
        z = distribution.cdf(y)
        x, logJ = StickbreakingBase().constrain_with_logdetjac(z)
        logJ += jnp.sum(distribution.log_prob(y), axis=-1)
        return x, logJ


class StickbreakingLogistic(StickbreakingCDF):
    def get_distribution(self, N: int) -> distributions.Logistic:
        return distributions.Logistic(loc=jnp.log(jnp.arange(N - 1, 0, -1)), scale=1)


class StickbreakingNormal(StickbreakingCDF):
    def get_distribution(self, N: int) -> distributions.Normal:
        return distributions.Normal(loc=jnp.log(jnp.arange(N - 1, 0, -1)) / 2, scale=1)


@dataclass
class StickbreakingPowerCDF:
    distribution: distributions.Distribution

    def unconstrain(self, x):
        N = x.shape[-1]
        z = StickbreakingBase().unconstrain(x)
        w = jnp.exp(jnp.arange(N - 1, 0, -1) * jnp.log1p(-z))
        y = self.distribution.quantile(w)
        return y

    def constrain(self, y):
        N = y.shape[-1] + 1
        w = self.distribution.cdf(y)
        z = 1 - w ** (1 / jnp.arange(N - 1, 0, -1))
        x = StickbreakingBase().constrain(z)
        return x

    def constrain_with_logdetjac(self, y):
        x = self.constrain(y)
        N = x.shape[-1]
        logJ = jnp.sum(
            self.distribution.log_prob(y), axis=-1
        ) - jax.scipy.special.gammaln(N)
        return x, logJ


class StickbreakingPowerLogistic(StickbreakingPowerCDF):
    def __init__(self):
        super().__init__(distributions.Logistic(loc=0, scale=1))


class StickbreakingPowerNormal(StickbreakingPowerCDF):
    def __init__(self):
        super().__init__(distributions.Normal(loc=0, scale=1))


class StickbreakingAngular:
    def unconstrain(self, x):
        z = StickbreakingBase().unconstrain(x)
        phi = jnp.arccos(jnp.sqrt(z))
        y = jax.scipy.special.logit(phi * 2 / jnp.pi)
        return y

    def constrain(self, y):
        phi = jnp.pi / 2 * jax.nn.sigmoid(y)
        z = jnp.cos(phi) ** 2
        x = StickbreakingBase().constrain(z)
        return x

    def constrain_with_logdetjac(self, y):
        phi = jnp.pi / 2 * jax.nn.sigmoid(y)
        cos_phi = jnp.cos(phi)
        z = cos_phi**2
        x, logJ = StickbreakingBase().constrain_with_logdetjac(z)
        N = x.shape[-1]
        logJ += jnp.sum(
            jnp.log(cos_phi)
            + jnp.log(jnp.sin(phi))
            + jnp.log(phi)
            + jnp.log1p(-phi * 2 / jnp.pi),
            axis=-1,
        ) + (N - 1) * jnp.log(2)
        return x, logJ
