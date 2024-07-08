# Draw samples from models
rule sample:
    input:
        "results/models/{target}_{transform}_{space}",
        "results/target_data/{target}/{target_config}.json",
    params:
        config=config["sample"],
        csv_dir="results/samples/{target}/{target_config}",
    output:
        [
            *sample_csv_files,
            "results/samples/{target}/{target_config}/{transform}_{space}.nc",
        ],
    conda:
        config["conda-environment"]
    script:
        "../scripts/sample.py"
