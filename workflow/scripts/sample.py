import glob
import os
import re
import shutil
import traceback
import warnings

import arviz as az
import cmdstanpy


def sample(exe_file: str, data_file: str, csv_dir: str, **sample_kwargs) -> list[str]:
    model = cmdstanpy.CmdStanModel(exe_file=exe_file)
    try:
        model.sample(
            data=data_file, save_warmup=True, output_dir=csv_dir, **sample_kwargs
        )
    except Exception:
        warnings.warn("Failed to sample model. Traceback:\n" + traceback.format_exc())
        return []

    # save CSVs to tempdir for renaming e.g. "{target}_{transform}-_{chain_id}.csv" to "{transform}_{chain_id}.csv", and then move to csv_dir
    csv_files = {}
    for csv_file in glob.glob(os.path.join(csv_dir, "*.csv")):
        match = re.match(r"^.*_(\d+).csv", os.path.basename(csv_file))
        if match:
            chain_id = match.groups()[0]
        else:
            raise ValueError(f"Could not parse chain ID from {csv_file}")
        csv_files[int(chain_id)] = csv_file
    return [csv_files[k] for k in sorted(csv_files.keys())]


def create_inference_data(csv_files: list[str]) -> az.InferenceData:
    if len(csv_files) == 0:
        return az.InferenceData()
    idata = az.from_cmdstan(csv_files, save_warmup=True)
    if "sample_stats" not in idata:
        raise ValueError("InferenceData does not contain sample_stats group")
    if "warmup_sample_stats" in idata:
        # save number of warm-up steps to sample_stats
        idata["sample_stats"]["n_steps_warmup"] = idata["warmup_sample_stats"][
            "n_steps"
        ].sum("draw")
    # remove warm-up draws and sample_stats from data
    groups = {
        group: idata[group]
        for group in idata.groups()
        if not group.startswith("warmup")
    }
    idata = az.InferenceData(**groups)
    return idata


smk = snakemake  # noqa: F821
exe_file, data_file = smk.input
idata_file = smk.output[0]
csv_dir = smk.params["csv_dir"]
sample_kwargs = smk.params.config

shutil.rmtree(csv_dir, ignore_errors=True)
csv_files = sample(exe_file, data_file, csv_dir, **sample_kwargs)
idata = create_inference_data(csv_files)
idata.to_netcdf(idata_file)
