# Draw samples from models
rule sample:
    input:
        "results/models/{target}_{transform}_{space}",
        "results/target_data/{target}/{target_config}.json",
    params:
        config=config["sample"],
        csv_dir="results/samples/{target}/{target_config}/{transform}_{space}",
    output:
        "results/samples/{target}/{target_config}/{transform}_{space}.nc",
    resources:
        cpus_per_task=8,
        mem_mb=mem_mb_for_samples,
        runtime=300,
    conda:
        config["conda-environment"]
    script:
        "../scripts/sample.py"
