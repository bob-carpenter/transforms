# Compute expectations wrt to parameterized target
rule compute_target_expectations:
    input:
        "results/target_data/{target_config}.json",
    params:
        target=lambda wildcards: target_configs[wildcards.target_config][0],
        config=config["target_expectations"],
    output:
        "results/target_expectations/{target_config}.json",
    resources:
        mem_mb=2_000,
    conda:
        config["conda-environment"]
    script:
        "../scripts/compute_target_expectations.py"
