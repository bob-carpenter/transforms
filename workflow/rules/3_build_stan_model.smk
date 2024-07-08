# Build Stan models for the given target, transform, and space.
# Compile once so they don't need to be recompiled every time.
rule build_stan_model:
    params:
        target=lambda wildcards: wildcards.target,
        transform=lambda wildcards: wildcards.transform,
        space=lambda wildcards: wildcards.space,
    output:
        "results/models/{target}_{transform}_{space}.stan",
        "results/models/{target}_{transform}_{space}",  # compiled cmdstan model
        "results/models/{target}_{transform}_{space}_model.so",  # compiled bridgestan model
    resources:
        mem_mb=4_000,
    conda:
        config["conda-environment"]
    script:
        "../scripts/build_stan_model.py"
