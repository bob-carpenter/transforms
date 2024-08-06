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
cmdstanpy_models = {}
bridgestan_models = {}
bridgestan_make_args = ["STAN_THREADS=true", "BRIDGESTAN_AD_HESSIAN=true"]


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


def write_stan_file(tmpdir, target_name, transform_name, log_scale):
    stan_code, include_paths = simplex_transforms.stan.make_stan_code(
        target_name,
        transform_name,
        log_scale,
    )
    stan_file = os.path.join(
        tmpdir,
        f"{target_name}_{transform_name}_{'log_simplex' if log_scale else 'simplex'}.stan",
    )
    with open(stan_file, "w") as f:
        f.write(stan_code)
    return stan_file, include_paths


def get_bridgestan_model_file(tmpdir, target_name, transform_name, log_scale):
    global bridgestan_models
    model_key = (target_name, transform_name, log_scale)
    if model_key not in bridgestan_models:
        stan_file, include_paths = write_stan_file(
            tmpdir, target_name, transform_name, log_scale
        )
        stanc_args = ["--include-paths=" + ",".join(include_paths)]
        model_file = bridgestan.compile_model(
            stan_file, stanc_args=stanc_args, make_args=bridgestan_make_args
        )
        bridgestan_models[model_key] = model_file
    else:
        model_file = bridgestan_models[model_key]
    return model_file


def get_cmdstanpy_model(tmpdir, target_name, transform_name, log_scale):
    global cmdstanpy_models
    model_key = (target_name, transform_name, log_scale)
    if model_key not in cmdstanpy_models:
        stan_file, include_paths = write_stan_file(
            tmpdir, target_name, transform_name, log_scale
        )
        stanc_options = {"include-paths": ",".join(include_paths)}
        model = cmdstanpy.CmdStanModel(stan_file=stan_file, stanc_options=stanc_options)
        cmdstanpy_models[model_key] = model
    else:
        model = cmdstanpy_models[model_key]
    return model


def test_get_target_names():
    target_names = simplex_transforms.stan.get_target_names()
    assert target_names == sorted(["dirichlet", "multi-logit-normal", "none"])


def test_get_transform_names():
    transform_names = simplex_transforms.stan.get_transform_names()
    assert transform_names == sorted(basic_transforms + expanded_transforms)


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

    # check that we can compile the bridgestan model
    get_bridgestan_model_file(tmpdir, target_name, transform_name, log_scale)

    # get compiled model or compile and add to cache
    model = get_cmdstanpy_model(tmpdir, target_name, transform_name, log_scale)

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
    assert jnp.allclose(lp_expected, idata.sample_stats.lp.data, rtol=5e-3)


@pytest.mark.parametrize("N", [3, 5])
@pytest.mark.parametrize("log_scale", [False, True])
@pytest.mark.parametrize("transform_name", ["ALR", "ExpandedSoftmax"])
def test_none_target(tmpdir, transform_name, N, log_scale, seed=638):
    trans = getattr(jax_transforms, transform_name)()

    data = {"N": N}
    data_str = json.dumps(data)

    model_file = get_bridgestan_model_file(tmpdir, "none", transform_name, log_scale)
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


@pytest.mark.parametrize("N", [5, 10, 100, 1_000])
@pytest.mark.parametrize("log_scale", [False, True])
def test_stan_stickbreaking_consistency(tmpdir, N, log_scale, seed=234):
    data = {"N": N}
    data_str = json.dumps(data)

    model_ref = bridgestan.StanModel(
        get_bridgestan_model_file(tmpdir, "none", "StanStickbreaking", log_scale),
        data=data_str,
    )
    model = bridgestan.StanModel(
        get_bridgestan_model_file(tmpdir, "none", "StickbreakingLogistic", log_scale),
        data=data_str,
    )

    y = np.random.default_rng(seed).uniform(-2, 2, size=(100, N - 1))

    # get log-density and x or log_x from StickbreakingLogistic
    param_prefix = "log_x." if log_scale else "x."
    lp = np.apply_along_axis(
        lambda y: model.log_density(y, propto=False, jacobian=True), -1, y
    )
    param_names = model.param_names(include_tp=True)
    param_inds = np.array(
        [i for i, name in enumerate(param_names) if name.startswith(param_prefix)]
    )
    assert len(param_inds) == N
    params = np.apply_along_axis(
        lambda y: model.param_constrain(y, include_tp=True)[param_inds], -1, y
    )

    # get log-density and x or log_x from StanStickbreaking
    lp_ref = np.apply_along_axis(
        lambda y: model_ref.log_density(y, propto=False, jacobian=True), -1, y
    )
    param_names = model_ref.param_names(include_tp=True)
    param_inds = np.array(
        [i for i, name in enumerate(param_names) if name.startswith(param_prefix)]
    )
    assert len(param_inds) == N
    params_ref = np.apply_along_axis(
        lambda y: model_ref.param_constrain(y, include_tp=True)[param_inds], -1, y
    )

    # check that the log-densities and parameters are consistent
    assert np.allclose(lp, lp_ref, rtol=1e-6)
    assert np.allclose(params, params_ref, rtol=1e-6)
