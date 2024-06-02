from dataclasses import dataclass

import jax
import jax.numpy as jnp
from tensorflow_probability.substrates.jax import distributions


@dataclass
class StickbreakingBase:
    N: int

    def unconstrain(self, x):
        def _running_remainder(remainder, xi):
            return remainder - xi, remainder

        z = x[:-1] / jax.lax.scan(_running_remainder, 1, x[:-1])[1]
        return z

    def constrain(self, z):
        def _break_stick(remainder, zi):
            xi = remainder * zi
            remainder -= xi
            return remainder, xi

        x_N, x_minus = jax.lax.scan(_break_stick, 1, z)
        x = jnp.append(x_minus, x_N)
        return x

    def constrain_with_logdetjac(self, z):
        x = self.constrain(z)
        logJ = jnp.inner(jnp.arange(self.N - 2, 0, -1), jnp.log1p(-z[:-1]))
        return x, logJ


@dataclass
class StickbreakingCDF:
    N: int
    distribution: distributions.Distribution

    def unconstrain(self, x):
        z = StickbreakingBase(self.N).unconstrain(x)
        y = self.distribution.quantile(z)
        return y

    def constrain(self, y):
        z = self.distribution.cdf(y)
        x = StickbreakingBase(self.N).constrain(z)
        return x

    def constrain_with_logdetjac(self, y):
        z = self.distribution.cdf(y)
        x, logJ = StickbreakingBase(self.N).constrain_with_logdetjac(z)
        logJ += jnp.sum(self.distribution.log_prob(y))
        return x, logJ


@dataclass
class StickbreakingLogistic(StickbreakingCDF):
    def __init__(self, N: int):
        dist = distributions.Logistic(loc=jnp.log(jnp.arange(N - 1, 0, -1)), scale=1)
        super().__init__(N, dist)


@dataclass
class StickbreakingNormal(StickbreakingCDF):
    def __init__(self, N: int):
        dist = distributions.Normal(loc=jnp.log(jnp.arange(N - 1, 0, -1)) / 2, scale=1)
        super().__init__(N, dist)


@dataclass
class StickbreakingPowerCDF:
    N: int
    distribution: distributions.Distribution

    def unconstrain(self, x):
        z = StickbreakingBase(self.N).unconstrain(x)
        w = jnp.exp(jnp.arange(self.N - 1, 0, -1) * jnp.log1p(-z))
        y = self.distribution.quantile(w)
        return y

    def constrain(self, y):
        w = self.distribution.cdf(y)
        z = 1 - w ** (1 / jnp.arange(self.N - 1, 0, -1))
        x = StickbreakingBase(self.N).constrain(z)
        return x

    def constrain_with_logdetjac(self, y):
        x = self.constrain(y)
        logJ = jnp.sum(self.distribution.log_prob(y)) - jax.scipy.special.gammaln(
            self.N
        )
        return x, logJ


@dataclass
class StickbreakingPowerLogistic(StickbreakingPowerCDF):
    def __init__(self, N: int):
        dist = distributions.Logistic(loc=0, scale=1)
        super().__init__(N, dist)


@dataclass
class StickbreakingPowerNormal(StickbreakingPowerCDF):
    def __init__(self, N: int):
        dist = distributions.Normal(loc=0, scale=1)
        super().__init__(N, dist)


@dataclass
class StickbreakingAngular:
    N: int

    def unconstrain(self, x):
        z = StickbreakingBase(self.N).unconstrain(x)
        phi = jnp.arccos(jnp.sqrt(z))
        y = jax.scipy.special.logit(phi * 2 / jnp.pi)
        return y

    def constrain(self, y):
        phi = jnp.pi / 2 * jax.nn.sigmoid(y)
        z = jnp.cos(phi) ** 2
        x = StickbreakingBase(self.N).constrain(z)
        return x

    def constrain_with_logdetjac(self, y):
        phi = jnp.pi / 2 * jax.nn.sigmoid(y)
        cos_phi = jnp.cos(phi)
        z = cos_phi**2
        x, logJ = StickbreakingBase(self.N).constrain_with_logdetjac(z)
        logJ += jnp.sum(
            jnp.log(cos_phi)
            + jnp.log(jnp.sin(phi))
            + jnp.log(phi)
            + jnp.log1p(-phi * 2 / jnp.pi)
        ) + (self.N - 1) * jnp.log(2)
        return x, logJ
