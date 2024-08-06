# Generate parameters for a given target configuration
rule generate_target_data:
    params:
        target_and_params=lambda wildcards: target_configs[wildcards.target_config],
        decimals=config["target_data"]["decimals"],
    output:
        "results/target_data/{target_config}.json",
    script:
        "../scripts/generate_target_data.py"
