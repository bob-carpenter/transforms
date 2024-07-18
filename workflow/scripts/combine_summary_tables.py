import pandas as pd

input_files = snakemake.input  # noqa: F821
output_file = snakemake.output[0]  # noqa: F821

dfs = []
for input_file in input_files:
    try:
        df = pd.read_csv(input_file, dtype={"chain": pd.Int64Dtype()})
    except pd.errors.EmptyDataError:
        continue
    dfs.append(df)

df_combined = pd.concat(dfs, ignore_index=True)
df_combined.to_csv(output_file, index=False)
