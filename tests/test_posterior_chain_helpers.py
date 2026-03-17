from __future__ import annotations

from types import SimpleNamespace

import jax
import numpy as np
import numpyro
import numpyro.distributions as dist
import pytest
from numpyro.infer import MCMC, NUTS

from surrogatenn_dsge import chunk_stats, epsilon_means_from_chain, theta_draws


def _chain_model() -> None:
    numpyro.sample("alpha", dist.Normal(0.0, 1.0))
    numpyro.sample("beta", dist.Normal(0.0, 1.0))


def _run_chain() -> MCMC:
    kernel = NUTS(_chain_model)
    mcmc = MCMC(kernel, num_warmup=4, num_samples=5, num_chains=1, progress_bar=False)
    mcmc.run(jax.random.PRNGKey(7))
    return mcmc


def test_theta_draws_matches_numpyro_samples_and_mapping_layout() -> None:
    mcmc = _run_chain()
    samples = mcmc.get_samples()

    expected = np.column_stack(
        [
            np.asarray(samples["beta"], dtype=np.float64),
            np.asarray(samples["alpha"], dtype=np.float64),
        ]
    )
    np.testing.assert_allclose(
        theta_draws(mcmc, ("beta", "alpha")),
        expected,
        rtol=0.0,
        atol=0.0,
    )
    np.testing.assert_allclose(
        theta_draws(samples, ("beta", "alpha")),
        expected,
        rtol=0.0,
        atol=0.0,
    )

    with pytest.raises(ValueError, match="not found in chain samples"):
        theta_draws(samples, ("missing",))


def test_epsilon_means_from_chain_parses_greek_sites_and_warns_on_length_mismatch() -> None:
    payload = {
        "ε[1,1]": np.asarray([1.0, 3.0], dtype=np.float64),
        "ϵ[2,1]": np.asarray([2.0, 4.0], dtype=np.float64),
        "ε[1,2]": np.asarray([5.0, 7.0], dtype=np.float64),
        "alpha": np.asarray([0.0, 0.0], dtype=np.float64),
    }

    eps_mean = epsilon_means_from_chain(payload)
    np.testing.assert_allclose(
        eps_mean,
        np.asarray([[2.0, 6.0], [3.0, np.nan]], dtype=np.float64),
        rtol=0.0,
        atol=0.0,
        equal_nan=True,
    )

    with pytest.warns(UserWarning, match="max_t does not match sample_idx length"):
        truncated = epsilon_means_from_chain(payload, sample_idx=[5])
    np.testing.assert_allclose(
        truncated,
        np.asarray([[2.0], [3.0]], dtype=np.float64),
        rtol=0.0,
        atol=0.0,
    )

    assert epsilon_means_from_chain({"alpha": np.asarray([1.0, 2.0])}) is None


def test_chunk_stats_supports_numpyro_mcmc_and_checkpoint_like_payloads() -> None:
    mcmc = _run_chain()

    accept, divergences, step_size = chunk_stats(mcmc)
    assert accept is not None and np.isfinite(accept)
    assert divergences is not None and divergences >= 0
    assert step_size is not None and np.isfinite(step_size)

    wrapped = {"chain": mcmc}
    wrapped_stats = chunk_stats(wrapped)
    np.testing.assert_allclose(wrapped_stats[0], accept, rtol=0.0, atol=0.0)
    assert wrapped_stats[1] == divergences
    np.testing.assert_allclose(wrapped_stats[2], step_size, rtol=0.0, atol=0.0)

    synthetic = {
        "extra_fields": {
            "accept_prob": np.asarray([0.6, 0.8], dtype=np.float64),
            "diverging": np.asarray([False, True]),
        },
        "last_state": SimpleNamespace(
            adapt_state=SimpleNamespace(step_size=np.asarray(0.125, dtype=np.float64))
        ),
    }
    synthetic_stats = chunk_stats(synthetic)
    np.testing.assert_allclose(synthetic_stats[0], 0.7, rtol=0.0, atol=1e-12)
    assert synthetic_stats[1] == 1
    np.testing.assert_allclose(synthetic_stats[2], 0.125, rtol=0.0, atol=0.0)

