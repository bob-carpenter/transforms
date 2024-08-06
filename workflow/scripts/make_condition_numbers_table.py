import pandas as pd
import xarray as xr
import re

input_files = snakemake.input  # noqa: F821
csv_file = snakemake.output[0]  # noqa: F821

def split_path(path):
    # Use regex to match the required pattern and extract components
    pattern = r".*/([^/]+)/([^/]+)/([^_/]+)_([^/]+)\.nc"
    match = re.match(pattern, path)
    
    if match:
        # Extracted components from the regex groups
        target = match.group(1)
        target_config = match.group(2)
        transform = match.group(3)
        space = match.group(4)
        
        return target, target_config, transform, space
    else:
        raise ValueError("The path format is incorrect.")

columns = None
with open(csv_file, "w") as f:
    for i, input_file in enumerate(input_files):
        cond = xr.load_dataarray(input_file)
        if cond.size == 1:
            continue
        df = cond.to_dataframe().reset_index()
        target, target_config, transform, space = split_path(input_file)
        log_scale = space == "log_simplex"
        df['chain'] += 1
        df.insert(0, "target", target)
        df.insert(1, "target_config", target_config)
        df.insert(2, "transform", transform)
        df.insert(3, "log_scale", log_scale)
        if i == 0:
            csv_kwargs = {}
        else:
            csv_kwargs = dict(columns=columns, header=False, mode='a')
        df.to_csv(f, index=False, **csv_kwargs)
