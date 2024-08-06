# transforms

Implementations of transforms between an unconstrained real vector and a point on the simplex for evaluation in the [Stan](https://mc-stan.org) probabilistic programming language.

## Installation

For basic installation, run

```bash
pip install .
```

## Usage

### Stan implementations

To get a Stan model using a transform `<transform>` to sample a target `<target>`, call

```python
from simplex_transforms.stan import make_stan_code

make_stan_code(<target>, <transform>)
```

### JAX implementations

To get a JAX implementation of the same transform, first make sure the optional JAX dependencies have been installed:

```bash
pip install .[jax]
```

Then call
```python
from simplex_transforms.jax.transforms import <transform>
```

## Analyses

To reproduce all analyses, first install the package in a conda environment named `simplex_transforms`
```bash
pip install .[workflow]
```

Then, from this directory, run
```bash
snakemake --cores all --use-conda
```
Results will then be stored in `results`.
