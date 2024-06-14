import os

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
targets_dir = os.path.join(project_dir, "targets")
transforms_dir = os.path.join(project_dir, "transforms")
stan_models = {}


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


@pytest.mark.parametrize("N", [3, 5, 10])
@pytest.mark.parametrize("log_scale", [False, True])
@pytest.mark.parametrize("target_name", ["dirichlet"])
@pytest.mark.parametrize("transform_name", basic_transforms + expanded_transforms)
def test_stan_and_jax_transforms_consistent(
    tmpdir, transform_name, target_name, N, log_scale
):
    try:
        trans = getattr(jax_transforms, transform_name)(N)
    except AttributeError:
        pytest.skip(f"No JAX implementation of {transform_name}. Skipping.")
    constrain_with_logdetjac_vec = jax.vmap(
        jax.vmap(trans.constrain_with_logdetjac, 0), 0
    )
    data = {"N": N, "alpha": [1.0] * N}

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

    result = model.sample(data=data, iter_sampling=100)
    idata = az.convert_to_inference_data(result)

    x_expected, lp_expected = constrain_with_logdetjac_vec(idata.posterior.y.data)
    lp_expected += jax.scipy.special.gammaln(N)  # Dirichlet(1, ..., 1)
    if transform_name in expanded_transforms:
        lp_expected += jax.vmap(jax.vmap(trans.default_prior, 0), 0)(x_expected)
        x_expected = x_expected[:, :, 1:]
    assert jnp.allclose(x_expected, idata.posterior.x.data, atol=1e-5)
    assert jnp.allclose(lp_expected, idata.sample_stats.lp.data, atol=1e-5)
