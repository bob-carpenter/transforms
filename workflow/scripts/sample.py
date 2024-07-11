import glob
import os
import re
import shutil
import tempfile

import arviz as az
import cmdstanpy


def sample(exe_file: str, data_file: str, csv_dir: str, **sample_kwargs) -> list[str]:
    model = cmdstanpy.CmdStanModel(exe_file=exe_file)
    fit = model.sample(data=data_file, save_warmup=True, time_fmt="", **sample_kwargs)
    # save CSVs to tempdir for renaming e.g. "{target}_{transform}-_{chain_id}.csv" to "{transform}_{chain_id}.csv", and then move to csv_dir
    csv_files = {}
    with tempfile.TemporaryDirectory() as tmpdir:
        fit.save_csvfiles(dir=tmpdir)
        csv_files_tmp = glob.glob(os.path.join(tmpdir, "*.csv"))
        for csv_file in csv_files_tmp:
            match = re.match(
                r"^[A-Za-z\-]+_(.+)-_(\d+).csv", os.path.basename(csv_file)
            )
            if match:
                transform_name, chain_id = match.groups()
                new_csv_basename = f"{transform_name}_{chain_id}.csv"
            else:
                new_csv_basename = os.path.basename(csv_file)
            new_csv_file = os.path.join(csv_dir, new_csv_basename)
            shutil.move(csv_file, new_csv_file)
            csv_files[int(chain_id)] = new_csv_file
    return [csv_files[k] for k in sorted(csv_files.keys())]


def create_inference_data(csv_files: list[str]) -> az.InferenceData:
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

csv_files = sample(exe_file, data_file, csv_dir, **sample_kwargs)
idata = create_inference_data(csv_files)
idata.to_netcdf(idata_file)
