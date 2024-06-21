import json
from typing import Sequence

import jax
import jax.numpy as jnp
import numpy as np
import scipy.stats

from simplex_transforms.jax.expectation import compute_expectations_and_ses
from simplex_transforms.jax.targets import MultiLogitNormal

jax.config.update("jax_enable_x64", True)


def compute_dirichlet_expectations(N: int, alpha: Sequence):
    alpha_array = np.asarray(alpha)
    x_mean = scipy.stats.dirichlet.mean(alpha_array)
    return dict(
        values=dict(
            x_mean=x_mean,
            x2_mean=scipy.stats.dirichlet.var(alpha_array) + x_mean**2,
            entropy=scipy.stats.dirichlet.entropy(alpha_array),
        )
    )


def estimate_multi_logit_normal_expectations(
    N: int,
    mu: np.ndarray,
    L_Sigma: np.ndarray,
    num_draws=10**6,
    batch_size=10**4,
    seed=0,
):
    dist = MultiLogitNormal(jnp.array(mu), jnp.array(L_Sigma))
    funcs = tuple(
        jax.tree_util.Partial(f)
        for f in (lambda x: x, jnp.square, lambda x: -dist.log_prob(x))
    )

    key = jax.random.PRNGKey(seed)
    expectations, ses = compute_expectations_and_ses(
        key, dist, num_draws, batch_size=batch_size, funcs=funcs
    )
    results = dict(
        values=dict(
            x_mean=expectations[0], x2_mean=expectations[1], entropy=expectations[2]
        ),
        standard_errors=dict(
            x_mean=ses[0],
            x2_mean=ses[1],
            entropy=ses[2],
        ),
        num_draws=dict(x_mean=num_draws, x2_mean=num_draws, entropy=num_draws),
    )
    return results


def compute_target_expectations(target: str, params: dict, **kwargs):
    if target == "dirichlet":
        return compute_dirichlet_expectations(**params)
    elif target == "multi-logit-normal":
        return estimate_multi_logit_normal_expectations(**params, **kwargs)
    else:
        raise ValueError(f"Unknown target {target}")


def write_results(out_file, data):
    data_fmt = {}
    for k, d in data.items():
        d_fmt = {}
        for k2, v2 in d.items():
            try:
                d_fmt[k2] = v2.tolist()
            except AttributeError:
                d_fmt[k2] = v2
        data_fmt[k] = d_fmt
    with open(out_file, "w") as f:
        json.dump(data_fmt, f)


smk = snakemake  # noqa: F821
model_data = smk.input[0]
out_file = smk.output[0]
target = smk.params["target"]
kwargs = smk.params["config"]

with open(model_data) as f:
    params = json.load(f)
expectations = compute_target_expectations(target, params, **kwargs)
write_results(out_file, expectations)
