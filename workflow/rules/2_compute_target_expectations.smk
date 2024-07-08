def mem_mb_for_target_expectations(wildcards) -> int:
    bytes_per_double = 64
    scale_factor = 10
    batch_size = 1_000_000
    N = target_configs[wildcards.target_config][1]["N"]
    num_bytes = bytes_per_double * batch_size * N * scale_factor
    return max(2_000, num_bytes // 1_000_000)


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
        mem_mb=mem_mb_for_target_expectations,
    conda:
        config["conda-environment"]
    script:
        "../scripts/compute_target_expectations.py"
