import os
import tempfile

import arviz as az
import cmdstanpy
import jax
import jax.numpy as jnp
import pytest

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
stan_models_dir = os.path.join(project_dir, "transforms/simplex")
stan_models_logscale_dir = os.path.join(project_dir, "transforms/log_simplex")


@pytest.fixture(scope="module", params=basic_transforms + expanded_transforms)
def transform_and_model(request):
    transform_name = request.param
    model_file = os.path.join(stan_models_dir, f"{transform_name}.stan")
    model_code = open(model_file, "r").read()
    model_code = model_code.replace("target_density_lp(x, alpha)", "0")
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_model_fn = os.path.join(tmpdir, "model.stan")
        with open(tmp_model_fn, "w") as f:
            f.write(model_code)
        model = cmdstanpy.CmdStanModel(stan_file=tmp_model_fn)
        yield transform_name, model


@pytest.fixture(scope="module", params=basic_transforms + expanded_transforms)
def transform_and_model_logscale(request):
    transform_name = request.param
    model_file = os.path.join(stan_models_logscale_dir, f"{transform_name}.stan")
    model_code = open(model_file, "r").read()
    model_code = model_code.replace(
        "target_density_lp(log_x, alpha)", "sum(log_x[1:N - 1])"
    )
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_model_fn = os.path.join(tmpdir, "model.stan")
        with open(tmp_model_fn, "w") as f:
            f.write(model_code)
        model = cmdstanpy.CmdStanModel(stan_file=tmp_model_fn)
        yield transform_name, model


@pytest.mark.parametrize("N", [3, 5, 10])
def test_stan_and_jax_transforms_consistent(transform_and_model, N):
    transform_name, model = transform_and_model
    try:
        trans = getattr(jax_transforms, transform_name)()
    except AttributeError:
        pytest.skip(f"No JAX implementation of {transform_name}. Skipping.")
    constrain_with_logdetjac_vec = jax.vmap(
        jax.vmap(trans.constrain_with_logdetjac, 0), 0
    )
    data = {"N": N, "alpha": [1.0] * N}

    result = model.sample(data=data, iter_sampling=100)
    idata = az.convert_to_inference_data(result)

    x_expected, lp_expected = constrain_with_logdetjac_vec(idata.posterior.y.data)
    if transform_name in expanded_transforms:
        r_expected, x_expected = x_expected
        lp_expected += trans.default_prior(x_expected).log_prob(r_expected)

    assert jnp.allclose(x_expected, idata.posterior.x.data, atol=1e-5)
    assert jnp.allclose(lp_expected, idata.sample_stats.lp.data, atol=1e-5)


@pytest.mark.parametrize("N", [3, 5, 10])
def test_stan_and_jax_transforms_consistent_logscale(transform_and_model_logscale, N):
    transform_name, model = transform_and_model_logscale
    try:
        trans = getattr(jax_transforms, transform_name)()
    except AttributeError:
        pytest.skip(f"No JAX implementation of {transform_name}. Skipping.")
    constrain_with_logdetjac_vec = jax.vmap(
        jax.vmap(trans.constrain_with_logdetjac, 0), 0
    )
    data = {"N": N, "alpha": [1.0] * N}

    result = model.sample(data=data, iter_sampling=100)
    idata = az.convert_to_inference_data(result)

    x_expected, lp_expected = constrain_with_logdetjac_vec(idata.posterior.y.data)
    if transform_name in expanded_transforms:
        r_expected, x_expected = x_expected
        lp_expected += trans.default_prior(x_expected).log_prob(r_expected)
    log_x_expected = jnp.log(x_expected)
    assert jnp.allclose(log_x_expected, idata.posterior.log_x.data, atol=1e-5)
    assert jnp.allclose(x_expected, idata.posterior.x.data, atol=1e-5)
    assert jnp.allclose(lp_expected, idata.sample_stats.lp.data, atol=1e-5)
