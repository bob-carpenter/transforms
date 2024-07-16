import shutil

input_files = snakemake.input  # noqa: F821
output_file = snakemake.output[0]  # noqa: F821

with open(output_file, "w") as output:
    for i, input_file in enumerate(input_files):
        with open(input_file, "r") as input:
            if i > 0:
                next(input)
            shutil.copyfileobj(input, output)
