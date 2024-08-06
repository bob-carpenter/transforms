import json

import arviz as az
import bridgestan
import numpy as np
import xarray as xr


def compute_error_and_ess(draw, mean_ref):
    mean = draw.mean(dim="draw")
    error = mean - mean_ref
    abs_rel_error = np.abs(error / mean_ref)
    se_sq = (error**2).mean(dim="chain")  # var with known mean
    ess_per_chain = draw.var(dim="draw", ddof=1) / se_sq
    ess_bulk = az.ess(draw)[draw.name]
    rhat = az.rhat(draw)
    assert isinstance(rhat, xr.Dataset)
    rhat = rhat[draw.name]
    return xr.Dataset(
        dict(
            mean=mean,
            mean_se=np.sqrt(se_sq),
            error=error,
            abs_rel_error=abs_rel_error,
            ess_per_chain=ess_per_chain,
            ess_bulk=ess_bulk,
            rhat=rhat,
        )
    )


def compute_sample_summaries(sample_stats: xr.Dataset) -> xr.Dataset:
    chain = sample_stats.chain
    n_draws = xr.DataArray(
        np.full_like(chain, len(sample_stats.draw)),
        dims="chain",
        coords={"chain": chain},
    )
    n_divergent = sample_stats.diverging.sum(dim="draw")
    step_size = xr.apply_ufunc(
        np.unique, sample_stats.step_size, input_core_dims=[["draw"]]
    )
    n_steps = sample_stats.n_steps.sum(dim="draw")
    n_steps_warmup = sample_stats.n_steps_warmup
    bfmi = xr.DataArray(az.bfmi(sample_stats), coords=dict(chain=sample_stats.chain))
    return xr.Dataset(
        dict(
            n_draws=n_draws,
            n_divergent=n_divergent,
            step_size=step_size,
            n_steps=n_steps,
            n_steps_warmup=n_steps_warmup,
            bfmi=bfmi,
        )
    )


def compute_estimate_summaries(
    idata: az.InferenceData, estimates_ref: dict, transform_model: bridgestan.StanModel
) -> az.InferenceData:
    if len(idata.groups()) == 0:
        return az.InferenceData()

    posterior = az.convert_to_dataset(idata, group="posterior")
    sample_stats = az.convert_to_dataset(idata, group="sample_stats")
    x = posterior.x
    x_mean_stats = compute_error_and_ess(x, estimates_ref["x_mean"])
    x2_mean_stats = compute_error_and_ess(x**2, estimates_ref["x2_mean"])

    if hasattr(posterior, "y"):
        y = posterior.y
    else:  # StanStickBreaking
        y = xr.apply_ufunc(
            transform_model.param_unconstrain,
            x,
            input_core_dims=[["x_dim_0"]],
            output_core_dims=[["y_dim_0"]],
            vectorize=True,
        )
    logJ = xr.apply_ufunc(
        lambda y: transform_model.log_density(y, propto=False, jacobian=True),
        y,
        input_core_dims=[["y_dim_0"]],
        vectorize=True,
    )
    entropy = -(sample_stats.lp - logJ)
    entropy_stats = compute_error_and_ess(entropy, estimates_ref["entropy"])

    sampling_stats = compute_sample_summaries(sample_stats)

    return az.InferenceData(
        x_mean_stats=x_mean_stats,
        x2_mean_stats=x2_mean_stats,
        entropy_stats=entropy_stats,
        sampling_stats=sampling_stats,
    )


smk = snakemake  # noqa: F821
sample_file, estimates_ref_file, transform_model_lib, data_file = smk.input
netcdf_file = smk.output[0]

idata = az.from_netcdf(sample_file)
with open(estimates_ref_file, "r") as f:
    estimates_ref = json.load(f)["values"]
with open(data_file, "r") as f:
    transform_data = json.dumps({"N": json.load(f)["N"]})
transform_model = bridgestan.StanModel(transform_model_lib, data=transform_data)
summary = compute_estimate_summaries(idata, estimates_ref, transform_model)
summary.to_netcdf(netcdf_file)
