from simplex_transforms.stan import make_stan_code

smk = snakemake  # noqa: F821
transform = smk.params["transform"]
target = smk.params["target"]
log_scale = smk.params["space"] == "log_simplex"
out_file = smk.output[0]
stan_code, _ = make_stan_code(target, transform, log_scale=True, use_include=False)
with open(out_file, "w") as f:
    f.write(stan_code)
