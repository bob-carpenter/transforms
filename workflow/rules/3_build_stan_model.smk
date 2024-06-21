# Build Stan models for the given target, transform, and space.
# Compile once so they don't need to be recompiled every time.


rule compile_bridgestan_model:
    input:
        "results/models/{target}_{transform}_{space}.stan",
    output:
        "results/models/{target}_{transform}_{space}_model.so",
    run:
        import bridgestan

        # enable multi-threading and use AD for Hessian computation
        bridgestan.compile_model(
            input[0], make_args=["STAN_THREADS=true", "BRIDGESTAN_AD_HESSIAN=true"]
        )


rule compile_cmdstan_model:
    input:
        "results/models/{target}_{transform}_{space}.stan",
    output:
        "results/models/{target}_{transform}_{space}",
    run:
        import cmdstanpy

        cmdstanpy.CmdStanModel(stan_file=input[0])


rule build_stan_model:
    params:
        target=lambda wildcards: wildcards.target,
        transform=lambda wildcards: wildcards.transform,
        space=lambda wildcards: wildcards.space,
    output:
        "results/models/{target}_{transform}_{space}.stan",
    script:
        "../scripts/build_stan_model.py"
