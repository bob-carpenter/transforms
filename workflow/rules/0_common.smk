import itertools

import simplex_transforms
import simplex_transforms.stan


# Create parameter combinations for each target
def make_target_configs(config):
    target_configs = {}
    for target, params in config["target_parameters"].items():
        param_combinations = [
            dict(zip(params.keys(), values))
            for values in itertools.product(*params.values())
        ]
        for combination in param_combinations:
            target_config = (
                target + "/" + "_".join([f"{k}{v}" for k, v in combination.items()])
            )
            target_configs[target_config] = (target, combination)
    return target_configs


chain_ids = range(1, config["sample"]["chains"] + 1)

if config["transforms"] == "all":
    transforms = simplex_transforms.stan.get_transform_names()
else:
    transforms = config["transforms"]
targets = list(config["target_data"]["target_parameters"].keys())

target_configs = make_target_configs(config["target_data"])
