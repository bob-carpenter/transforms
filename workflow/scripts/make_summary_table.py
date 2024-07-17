import arviz as az
import pandas as pd
import xarray as xr


def summarize_estimate(ds: xr.Dataset) -> pd.DataFrame:
    param_dims = [dim for dim in ds.dims if dim not in ["chain", "draw"]]
    # per-chain stats (1 chain per row)
    max_abs_rel_error = ds["abs_rel_error"].max(dim=param_dims)
    min_ess_per_chain = ds["ess_per_chain"].min(dim=param_dims)
    min_rel_ess_per_chain = ds["rel_ess_per_chain"].min(dim=param_dims)
    ds_per_chain = xr.Dataset(
        dict(
            max_abs_rel_error=max_abs_rel_error,
            min_ess_per_chain=min_ess_per_chain,
            min_rel_ess_per_chain=min_rel_ess_per_chain,
        )
    )
    df_per_chain = ds_per_chain.to_dataframe().reset_index()
    df_per_chain["chain"] += 1
    # whole estimate stats (1 row)
    min_ess_bulk = ds["ess_bulk"].min(dim=param_dims)
    min_rel_ess_bulk = ds["rel_ess_bulk"].min(dim=param_dims)
    max_rhat = ds["rhat"].max(dim=param_dims)
    df_whole_estimate = pd.DataFrame(
        dict(
            chain=[pd.NA],
            min_ess_bulk=[min_ess_bulk.item()],
            min_rel_ess_bulk=[min_rel_ess_bulk.item()],
            max_rhat=[max_rhat.item()],
        )
    )
    return pd.concat(
        [
            ds_per_chain.to_dataframe().reset_index(),
            df_whole_estimate,
        ],
        axis=0,
    )


def make_summary_table(idata: az.InferenceData) -> pd.DataFrame:
    if len(idata.groups()) == 0:
        return pd.DataFrame()
    estimates = ["x_mean", "x2_mean", "entropy"]
    dfs = []
    for estimate in estimates:
        ds = getattr(idata, f"{estimate}_stats")
        df = summarize_estimate(ds)
        df.insert(0, "estimate", estimate)
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)


smk = snakemake  # noqa: F821
netcdf_file = smk.input[0]
csv_file = smk.output[0]
idata = az.from_netcdf(netcdf_file)
summary_df = make_summary_table(idata)
if len(summary_df) > 0:
    summary_df.insert(0, "target", smk.wildcards.target)
    summary_df.insert(1, "target_config", smk.wildcards.target_config)
    summary_df.insert(2, "transform", smk.wildcards.transform)
    summary_df.insert(3, "log_scale", smk.params.log_scale)
summary_df.to_csv(csv_file, index=False)
