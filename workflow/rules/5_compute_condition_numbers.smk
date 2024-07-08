# Compute Hessian condition numbers
rule compute_condition_numbers:
    input:
        "results/models/{target}_{transform}_{space}_model.so",
        "results/target_data/{target}/{target_config}.json",
        "results/samples/{target}/{target_config}/{transform}_{space}.nc",
    params:
        target=lambda wildcards: wildcards.target,
        transform=lambda wildcards: wildcards.transform,
        thin=config["condition_numbers"]["thin"],
    output:
        "results/condition_numbers/{target}/{target_config}/{transform}_{space}.nc",
    resources:
        mem_mb=2_000,
        runtime=300,
    conda:
        config["conda-environment"]
    script:
        "../scripts/compute_condition_numbers.py"
