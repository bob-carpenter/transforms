import glob
import os
import re
import tempfile

import cmdstanpy


def sample(exe_file, data_file, csv_dir, **sample_kwargs):
    model = cmdstanpy.CmdStanModel(exe_file=exe_file)
    fit = model.sample(data=data_file, save_warmup=True, time_fmt="", **sample_kwargs)
    # save CSVs to tempdir for renaming e.g. "{target}_{transform}-_{chain_id}.csv" to "{transform}_{chain_id}.csv", and then move to csv_dir
    csv_files = []
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
            os.rename(csv_file, new_csv_file)
            csv_files.append(new_csv_file)
    return csv_files


smk = snakemake  # noqa: F821
exe_file, data_file = smk.input
csv_files = smk.output
csv_dir = smk.params["csv_dir"]
sample_kwargs = smk.params.config

csv_files_actual = sample(exe_file, data_file, csv_dir, **sample_kwargs)
assert sorted(csv_files) == sorted(csv_files_actual)
