import os
from typing import Tuple

_TARGETS_DIR = os.path.join(os.path.dirname(__file__), "targets")
_TRANSFORMS_DIR = os.path.join(os.path.dirname(__file__), "transforms")


def _include_or_read(file_path: str, use_include: bool) -> str:
    if use_include:
        return f"#include {os.path.basename(file_path)}"
    else:
        return open(file_path).read()


def make_stan_code(
    target_name: str,
    transform_name: str,
    log_scale: bool = False,
    use_include: bool = True,
) -> Tuple[str, list[str]]:
    """Make Stan code for a target-transform pair.

    Arguments
    ---------
    target_name : str
        Name of the target distribution.
    transform_name : str
        Name of the transform.
    log_scale : bool, optional
        Whether to sample the log-simplex instead of the simplex, by default False.
    use_include : bool, optional
        Whether to use `#include` statements in the Stan code, by default True.
        If False, the files are read and included directly.

    Returns
    -------
    model_code : str
        The Stan code
    include_paths : list[str]
        List of directories where the Stan code includes, if any, are located.
    """
    target_dir = os.path.join(_TARGETS_DIR, target_name)
    transform_dir = os.path.join(_TRANSFORMS_DIR, transform_name)
    space = "log_simplex" if log_scale else "simplex"

    model_code = "\n".join(
        [
            "functions {",
            _include_or_read(
                os.path.join(target_dir, f"{target_name}_functions.stan"), use_include
            ),
            _include_or_read(
                os.path.join(transform_dir, f"{transform_name}_functions.stan"),
                use_include,
            ),
            "}",
            _include_or_read(
                os.path.join(target_dir, f"{target_name}_data.stan"), use_include
            ),
            _include_or_read(
                os.path.join(
                    transform_dir, f"{transform_name}_parameters_{space}.stan"
                ),
                use_include,
            ),
            _include_or_read(
                os.path.join(target_dir, f"{target_name}_model_{space}.stan"),
                use_include,
            ),
        ]
    )
    model_code = model_code.replace("\n\n", "\n")
    header_includes = [target_dir, transform_dir] if use_include else []

    return model_code, header_includes
