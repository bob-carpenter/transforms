import json
import os

import arviz as az
import bridgestan
import cmdstanpy
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import simplex_transforms.jax.targets as jax_targets
import simplex_transforms.jax.transforms as jax_transforms
import simplex_transforms.stan

jax.config.update("jax_enable_x64", True)

basic_transforms = [
    "ALR",
    "ILR",
    "StanStickbreaking",
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
bridgestan_models = {}

stan_version = cmdstanpy.cmdstan_version()
if stan_version is None:
    raise ValueError("Could not determine cmdstan version. It must be installed.")
stan_version = ".".join([str(i) for i in stan_version])
bridgestan_make_args = [
    "STAN_THREADS=true",
    "BRIDGESTAN_AD_HESSIAN=true",
    f"STANC3_VERSION={stan_version}",
]


def make_dirichlet_data(N: int, seed: int = 638):
    rng = np.random.default_rng(seed)
    alpha = rng.uniform(size=N)
    return {"N": N, "alpha": np.around(10 * alpha, 4).tolist()}


def make_multi_logit_normal_data(N: int, seed: int = 638):
    rng = np.random.default_rng(seed)
    mu = 0.01 * rng.normal(size=N - 1)
    L_Sigma = np.tril(rng.normal(size=(N - 1, N - 1)))
    diaginds = np.diag_indices(N - 1)
    L_Sigma[diaginds] = np.abs(L_Sigma[diaginds])
    sigma = 100 * np.random.uniform(size=N - 1)
    L_Sigma = np.diag(sigma / np.linalg.norm(L_Sigma, axis=1)) @ L_Sigma
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
        return jax_targets.Dirichlet(jnp.array(params["alpha"]))
    elif target == "multi-logit-normal":
        return jax_targets.MultiLogitNormal(
            jnp.array(params["mu"]), jnp.array(params["L_Sigma"])
        )
    else:
        raise ValueError(f"Unknown target {target}")


@pytest.mark.parametrize("N", [3, 5])
@pytest.mark.parametrize("log_scale", [False, True])
@pytest.mark.parametrize("target_name", ["dirichlet", "multi-logit-normal"])
@pytest.mark.parametrize("transform_name", basic_transforms + expanded_transforms)
def test_stan_and_jax_transforms_consistent(
    tmpdir, transform_name, target_name, N, log_scale, seed=638, stan_seed=348
):
    if transform_name == "StanStickbreaking":
        jax_transform_name = "StickbreakingLogistic"
    else:
        jax_transform_name = transform_name
    try:
        trans = getattr(jax_transforms, jax_transform_name)()
    except AttributeError:
        pytest.skip(f"No JAX implementation of {transform_name}. Skipping.")
    if target_name != "dirichlet" and transform_name not in ["ALR", "ILR"]:
        pytest.skip(f"No need to test {transform_name} with {target_name}. Skipping.")

    data = make_model_data(target_name, N, seed=seed)
    dist = make_jax_distribution(target_name, data)
    log_prob = dist.log_prob

    # get compiled model or compile and add to cache
    model_key = (target_name, transform_name, log_scale)
    if model_key not in stan_models:
        stan_code, include_paths = simplex_transforms.stan.make_stan_code(
            target_name, transform_name, log_scale
        )
        # save Stan code to file
        stan_file = os.path.join(
            tmpdir,
            f"{target_name}_{transform_name}_{'log_simplex' if log_scale else 'simplex'}.stan",
        )
        with open(stan_file, "w") as f:
            f.write(stan_code)

        # compile cmdstanpy model
        stanc_options = {"include-paths": ",".join(include_paths)}
        model = cmdstanpy.CmdStanModel(stan_file=stan_file, stanc_options=stanc_options)
        stan_models[model_key] = model

        # check that we can compile the bridgestan model
        stanc_args = ["--include-paths=" + ",".join(include_paths)]
        bridgestan_models[model_key] = bridgestan.compile_model(
            stan_file, stanc_args=stanc_args, make_args=bridgestan_make_args
        )
    else:
        model = stan_models[model_key]

    result = model.sample(data=data, iter_sampling=100, sig_figs=18, seed=stan_seed)
    idata = az.convert_to_inference_data(result)

    if transform_name == "StanStickbreaking":
        y = trans.unconstrain(idata.posterior.x.data)
    else:
        y = idata.posterior.y.data
    x_expected, lp_expected = trans.constrain_with_logdetjac(y)
    if transform_name in expanded_transforms:
        r_expected, x_expected = x_expected
        lp_expected += trans.default_prior(x_expected).log_prob(r_expected)
    lp_expected += log_prob(x_expected)
    assert jnp.allclose(x_expected, idata.posterior.x.data, rtol=1e-4)
    assert jnp.allclose(lp_expected, idata.sample_stats.lp.data, rtol=1e-4)


@pytest.mark.parametrize("N", [3, 5])
@pytest.mark.parametrize("log_scale", [False, True])
@pytest.mark.parametrize("transform_name", ["ALR", "ExpandedSoftmax"])
def test_none_target(tmpdir, transform_name, N, log_scale, seed=638):
    target_name = "none"
    trans = getattr(jax_transforms, transform_name)()

    data = {"N": N}
    data_str = json.dumps(data)

    # get compiled model or compile and add to cache
    model_key = (target_name, transform_name, log_scale)
    if model_key not in bridgestan_models:
        stan_code, include_paths = simplex_transforms.stan.make_stan_code(
            target_name,
            transform_name,
            log_scale,
        )
        # save Stan code to file
        stan_file = os.path.join(
            tmpdir,
            f"{target_name}_{transform_name}_{'log_simplex' if log_scale else 'simplex'}.stan",
        )
        with open(stan_file, "w") as f:
            f.write(stan_code)

        # check that we can compile the bridgestan model
        stanc_args = ["--include-paths=" + ",".join(include_paths)]
        model_file = bridgestan.compile_model(
            stan_file, stanc_args=stanc_args, make_args=bridgestan_make_args
        )
        bridgestan_models[model_key] = model_file
    else:
        model_file = bridgestan_models[model_key]
    model = bridgestan.StanModel(model_file, data=data_str)

    M = N - (transform_name in basic_transforms)

    y = np.random.default_rng(seed).normal(size=(100, M))
    x_expected, lp_expected = trans.constrain_with_logdetjac(y)
    if transform_name in expanded_transforms:
        r_expected, x_expected = x_expected
        lp_expected += trans.default_prior(x_expected).log_prob(r_expected)

    lp = np.apply_along_axis(
        lambda y: model.log_density(y, propto=False, jacobian=True), -1, y
    )
    assert np.allclose(lp, lp_expected, rtol=1e-4)
