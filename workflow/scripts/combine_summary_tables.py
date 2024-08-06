import pandas as pd

input_files = snakemake.input  # noqa: F821
output_file = snakemake.output[0]  # noqa: F821

integer_cols = ["chain", "n_draws", "n_divergent", "n_steps", "n_steps_warmup"]
dfs = []
for input_file in input_files:
    try:
        df = pd.read_csv(input_file, dtype={c: pd.Int64Dtype() for c in integer_cols})
    except pd.errors.EmptyDataError:
        continue
    dfs.append(df)

df_combined = pd.concat(dfs, ignore_index=True)
df_combined.to_csv(output_file, index=False)
