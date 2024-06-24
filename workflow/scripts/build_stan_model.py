import cmdstanpy

from simplex_transforms.stan import make_stan_code


def save_stan_file(out_file, target, transform, log_scale=False):
    stan_code, _ = make_stan_code(
        target, transform, log_scale=log_scale, use_include=False
    )
    with open(out_file, "w") as f:
        f.write(stan_code)


def compile_cmdstan_model(stan_file):
    model = cmdstanpy.CmdStanModel(stan_file=stan_file)
    return model


smk = snakemake  # noqa: F821
transform = smk.params["transform"]
target = smk.params["target"]
log_scale = smk.params["space"] == "log_simplex"
stan_file = smk.output[0]
save_stan_file(stan_file, target, transform, log_scale=log_scale)
compile_cmdstan_model(stan_file)
