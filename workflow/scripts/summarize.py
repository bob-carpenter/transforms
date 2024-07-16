import json

import arviz as az
import bridgestan
import numpy as np
import xarray as xr


def compute_error_and_ess(draw, mean_ref, num_steps):
    mean = draw.mean(dim="draw")
    error = mean - mean_ref
    abs_rel_error = np.abs(error / mean_ref)
    se_sq = (error**2).mean(dim="chain")  # var with known mean
    ess_per_chain = draw.var(dim="draw", ddof=1) / se_sq
    rel_ess_per_chain = ess_per_chain / num_steps
    ess_bulk = az.ess(draw)[draw.name]
    rel_ess_bulk = ess_bulk / num_steps.sum()
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
            rel_ess_bulk=rel_ess_bulk,
            rel_ess_per_chain=rel_ess_per_chain,
            rhat=rhat,
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
    n_steps_total = sample_stats.n_steps.sum(dim="draw") + sample_stats.n_steps_warmup
    x_mean_stats = compute_error_and_ess(x, estimates_ref["x_mean"], n_steps_total)
    x2_mean_stats = compute_error_and_ess(x**2, estimates_ref["x2_mean"], n_steps_total)

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
    entropy_stats = compute_error_and_ess(
        entropy, estimates_ref["entropy"], n_steps_total
    )

    return az.InferenceData(
        x_mean_stats=x_mean_stats,
        x2_mean_stats=x2_mean_stats,
        entropy_stats=entropy_stats,
        sample_stats=sample_stats,
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
