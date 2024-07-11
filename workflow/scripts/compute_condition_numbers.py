import arviz as az
import bridgestan
import numpy as np
import xarray as xr


def hessian_cond(y: np.ndarray) -> float:
    H = np.asanyarray(model.log_density_hessian(y)[2])
    try:
        return np.linalg.cond(H)
    except np.linalg.LinAlgError as e:
        if "SVD did not converge" in str(e):
            return np.nan
        else:
            raise


thin = snakemake.params["thin"]  # noqa: F821
model_file, data_file, sample_file = snakemake.input  # noqa: F821
output_file = snakemake.output[0]  # noqa: F821
idata = az.from_netcdf(sample_file)
if len(idata.groups()) > 0:
    model = bridgestan.StanModel(model_file, data=data_file)
    posterior = az.convert_to_dataset(idata, group="posterior")
    if "y" in posterior:
        y = posterior.y
    else:  # StanStickBreaking
        x = posterior.x
        y = xr.apply_ufunc(
            model.param_unconstrain,
            x,
            input_core_dims=[["x_dim_0"]],
            output_core_dims=[["y_dim_0"]],
            vectorize=True,
        )

    y_thinned = y.sel(draw=range(0, len(y.draw), thin))

    cond = xr.apply_ufunc(
        hessian_cond, y_thinned, input_core_dims=[["y_dim_0"]], vectorize=True
    )
    cond.name = "condition_number"
else:
    cond = xr.DataArray(np.nan, name="condition_number")
cond.to_netcdf(output_file)
