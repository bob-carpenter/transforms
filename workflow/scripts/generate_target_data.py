import json
from typing import Union

import numpy as np


def as_float(x: str) -> float:
    try:
        return float(int(x))
    except ValueError:
        return float(x)


def generate_dirichlet_data(N: int, alpha: Union[float, str]) -> dict:
    if isinstance(alpha, str):
        alpha_range = list(map(as_float, alpha.split("-", 2)))
        alpha_array = np.linspace(alpha_range[0], alpha_range[1], N)
    else:
        alpha_array = np.full(N, alpha)
    return dict(N=N, alpha=alpha_array)


def generate_multi_logit_normal_data(
    N: int, rho: float, scale: Union[float, str]
) -> dict:
    mu = np.zeros(N - 1)
    inds = np.arange(N - 1)
    cor = rho ** np.abs(inds - inds[:, np.newaxis])
    L_cor = np.linalg.cholesky(cor)
    if isinstance(scale, str):
        scale_range = list(map(as_float, scale.split("-", 2)))
        scale_array = np.linspace(scale_range[0], scale_range[1], N - 1)
    else:
        scale_array = np.full(N - 1, scale)
    L_Sigma = scale_array[:, np.newaxis] * L_cor
    return dict(N=N, mu=mu, L_Sigma=L_Sigma)


def generate_model_data(target: str, params: dict) -> dict:
    if target == "dirichlet":
        return generate_dirichlet_data(**params)
    elif target == "multi-logit-normal":
        return generate_multi_logit_normal_data(**params)
    else:
        raise ValueError(f"Unknown target {target}")


def save_model_data(data: dict, path: str, decimals: int = 4) -> None:
    formatted_data = {}
    for key, value in data.items():
        if isinstance(value, np.ndarray):
            formatted_data[key] = np.around(value, decimals).tolist()
        else:
            formatted_data[key] = value

    with open(path, "w") as f:
        json.dump(formatted_data, f)


smk = snakemake  # noqa: F821
data = generate_model_data(*smk.params["target_and_params"])
save_model_data(data, smk.output[0], decimals=smk.params["decimals"])
