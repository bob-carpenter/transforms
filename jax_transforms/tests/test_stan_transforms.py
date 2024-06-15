import os
from typing import NamedTuple

import arviz as az
import cmdstanpy
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from tensorflow_probability.substrates.jax import distributions as tfd

import jax_transforms

jax.config.update("jax_enable_x64", True)

basic_transforms = [
    "ALR",
    "ILR",
    "StickbreakingAngular",
    "StickbreakingLogistic",
    "StickbreakingNormal",
    "StickbreakingPowerLogistic",
    "StickbreakingPowerNormal",
]

expanded_transforms = [
    "ExpandedSoftmax",
    "NormalizedExponential",
]

project_dir = os.path.abspath(os.path.join(__file__, "..", "..", ".."))
targets_dir = os.path.join(project_dir, "targets")
transforms_dir = os.path.join(project_dir, "transforms")
stan_models = {}


class MultiLogitNormal(NamedTuple):
    mu: jax.Array
    L_Sigma: jax.Array

    @property
    def event_shape(self):
        return (self.mu.shape[0] + 1,)

    def log_prob(self, x):
        transform = jax_transforms.ALR()
        y = transform.unconstrain(x)
        logJ = transform.constrain_with_logdetjac(y)[1]
        lp_mvnorm = tfd.MultivariateNormalTriL(
            loc=self.mu, scale_tril=self.L_Sigma
        ).log_prob(y)
        return lp_mvnorm - logJ


def make_dirichlet_data(N: int, seed: int = 638):
    rng = np.random.default_rng(seed)
    alpha = rng.uniform(size=N)
    return {"N": N, "alpha": np.around(alpha, 4).tolist()}


def make_multi_logit_normal_data(N: int, seed: int = 638):
    rng = np.random.default_rng(seed)
    mu = rng.normal(size=N - 1)
    L_Sigma = np.tril(rng.normal(size=(N - 1, N - 1)))
    L_Sigma[np.diag_indices(N - 1)] = rng.uniform(size=N - 1)
    return {
        "N": N,
        "mu": np.around(mu, 4).tolist(),
        "L_Sigma": np.around(L_Sigma, 4).tolist(),
    }


def make_model_data(target: str, *args, **kwargs):
    if target == "dirichlet":
        return make_dirichlet_data(*args, **kwargs)
    elif target == "multi-logit-normal":
        return make_multi_logit_normal_data(*args, **kwargs)
    else:
        raise ValueError(f"Unknown target {target}")


def make_jax_distribution(target: str, params: dict):
    if target == "dirichlet":
        return tfd.Dirichlet(jnp.array(params["alpha"]))
    elif target == "multi-logit-normal":
        return MultiLogitNormal(
            mu=jnp.array(params["mu"]), L_Sigma=jnp.array(params["L_Sigma"])
        )
    else:
        raise ValueError(f"Unknown target {target}")


def make_stan_model(
    model_file: str, target_name: str, transform_name: str, log_scale: bool
) -> cmdstanpy.CmdStanModel:
    target_dir = os.path.join(targets_dir, target_name)
    transform_dir = os.path.join(transforms_dir, transform_name)
    space = "log_simplex" if log_scale else "simplex"
    model_code = f"""
    functions {{
    #include {target_name}_functions.stan
    #include {transform_name}_functions.stan
    }}
    #include {target_name}_data.stan
    #include {transform_name}_parameters_{space}.stan
    #include {target_name}_model_{space}.stan
    """
    with open(model_file, "w") as f:
        f.write(model_code)
    stanc_options = {"include-paths": ",".join([target_dir, transform_dir])}
    model = cmdstanpy.CmdStanModel(stan_file=model_file, stanc_options=stanc_options)
    return model


@pytest.mark.parametrize("N", [3, 5])
@pytest.mark.parametrize("log_scale", [False, True])
@pytest.mark.parametrize("target_name", ["dirichlet", "multi-logit-normal"])
@pytest.mark.parametrize("transform_name", basic_transforms + expanded_transforms)
def test_stan_and_jax_transforms_consistent(
    tmpdir, transform_name, target_name, N, log_scale, seed=638
):
    try:
        trans = getattr(jax_transforms, transform_name)()
    except AttributeError:
        pytest.skip(f"No JAX implementation of {transform_name}. Skipping.")
    if target_name != "dirichlet" and transform_name not in ["ALR", "ILR"]:
        pytest.skip(f"No need to test {transform_name} with {target_name}. Skipping.")

    constrain_with_logdetjac_vec = jax.vmap(
        jax.vmap(trans.constrain_with_logdetjac, 0), 0
    )

    data = make_model_data(target_name, N, seed=seed)
    dist = make_jax_distribution(target_name, data)
    log_prob = dist.log_prob

    # get compiled model or compile and add to cache
    model_key = (target_name, transform_name, log_scale)
    if model_key not in stan_models:
        model = make_stan_model(
            os.path.join(
                tmpdir,
                f"{target_name}_{transform_name}_{'log_simplex' if log_scale else 'simplex'}.stan",
            ),
            target_name,
            transform_name,
            log_scale,
        )
        stan_models[model_key] = model
    else:
        model = stan_models[(target_name, transform_name, log_scale)]

    result = model.sample(data=data, iter_sampling=100, sig_figs=9)
    idata = az.convert_to_inference_data(result)

    x_expected, lp_expected = constrain_with_logdetjac_vec(idata.posterior.y.data)
    if transform_name in expanded_transforms:
        r_expected, x_expected = x_expected
        lp_expected += trans.default_prior(x_expected).log_prob(r_expected)
    lp_expected += log_prob(x_expected)
    assert jnp.allclose(x_expected, idata.posterior.x.data, rtol=1e-4)
    assert jnp.allclose(lp_expected, idata.sample_stats.lp.data, rtol=1e-4)
