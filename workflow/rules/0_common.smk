import itertools
import os

import simplex_transforms
import simplex_transforms.stan

target_configs = {}


# Create parameter combinations for each target
def make_target_configs(config):
    global target_configs
    for target, params in config["target_parameters"].items():
        param_combinations = [
            dict(zip(params.keys(), values))
            for values in itertools.product(*params.values())
        ]
        for combination in param_combinations:
            target_config = os.path.join(
                target, "_".join([f"{k}{v}" for k, v in combination.items()])
            )
            target_configs[target_config] = (target, combination)


def mem_mb_for_samples(wildcards) -> int:
    bytes_per_double = 8
    num_chains = config["sample"]["chains"]
    num_iters = config["sample"]["iter_sampling"]
    target_config = f"{wildcards.target}/{wildcards.target_config}"
    N = target_configs[target_config][1]["N"]
    scale_factor = 10
    num_bytes = bytes_per_double * num_chains * num_iters * N * scale_factor
    return max(800, num_bytes // 1_000_000)


chain_ids = range(1, config["sample"]["chains"] + 1)

if config["transforms"] == "all":
    transforms = simplex_transforms.stan.get_transform_names()
else:
    transforms = config["transforms"]
targets = list(config["target_data"]["target_parameters"].keys())

make_target_configs(config["target_data"])

sample_csv_files = expand(
    "results/samples/{target}/{target_config}/{transform}_{space}_{chain_id}.csv",
    chain_id=chain_ids,
    allow_missing=True,
)
