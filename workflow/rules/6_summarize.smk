# Compute summary error and ESS, format as summary tables, and combine
rule summarize:
    input:
        "results/samples/{target}/{target_config}/{transform}_{space}.nc",
        "results/target_expectations/{target}/{target_config}.json",
        "results/models/none_{transform}_{space}_model.so",
        "results/target_data/{target}/{target_config}.json",
    output:
        "results/summaries/{target}/{target_config}/{transform}_{space}.nc",
    resources:
        mem_mb=mem_mb_for_samples,
    conda:
        config["conda-environment"]
    script:
        "../scripts/summarize.py"


rule make_summary_table:
    input:
        "results/summaries/{target}/{target_config}/{transform}_{space}.nc",
    output:
        "results/summaries/{target}/{target_config}/{transform}_{space}.csv",
    params:
        log_scale=lambda wildcards: wildcards.space == "log_simplex",
    conda:
        config["conda-environment"]
    script:
        "../scripts/make_summary_table.py"


rule combine_summary_tables:
    input:
        summary_csv_files,
    output:
        "results/summaries/all.csv",
    script:
        "../scripts/combine_summary_tables.py"
