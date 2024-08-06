import jax
import jax.numpy as jnp
import pytest
from tensorflow_probability.substrates.jax import distributions as tfd

from simplex_transforms.jax.expectation import compute_expectations_and_ses

jax.config.update("jax_enable_x64", True)


@pytest.mark.parametrize(
    "dist", [tfd.Normal(3, 5), tfd.Dirichlet(jnp.arange(1.0, 6.0))]
)
@pytest.mark.parametrize("num_draws", [1_000, 10_000_000])
def test_compute_expectations_and_ses_univariate(
    dist, num_draws, key=jax.random.PRNGKey(348), alpha=0.05
):
    batch_size = num_draws // 10

    # define the functions to compute expectations for
    funcs = (jax.tree_util.Partial(lambda x: x), jax.tree_util.Partial(dist.log_prob))

    # compute expectations and standard errors
    expectations, ses = compute_expectations_and_ses(
        key, dist, num_draws, batch_size, funcs
    )

    # verify expectations using Central Limit Theorem
    true_expectations = (dist.mean(), -dist.entropy())

    # adjust alpha for multiple comparisons
    num_tests = sum([exp.size for exp in true_expectations])
    alpha_test = 1 - (1 - alpha) ** (1 / num_tests)
    bound = jax.scipy.stats.norm.ppf(1 - alpha_test / 2)

    # compare the estimated expectations to the true expectations
    for exp, se, exp_true in zip(expectations, ses, true_expectations):
        z = (exp - exp_true) / se
        assert jnp.all(jnp.abs(z) < bound)
