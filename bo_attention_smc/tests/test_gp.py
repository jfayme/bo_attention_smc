"""The GP with the per-molecule term.

The term must be one exactly where two reactions share a molecule; the GP
that carries it must be the reference's posterior at the GP's own settings
(its marginal likelihood and its predictions), so that the MAP term and the
sampled term are the same model; and its fit must record each weight.

    python -m unittest bo_attention_smc.tests.test_gp

They read the stored bh_1 MACE cell and run on one torch thread, as the
campaigns do.
"""

import unittest

import numpy as np
import torch

from bo_attention_smc.core.campaign import make_pool
from bo_attention_smc.core.cells import load_cell, molecules, reagents
from bo_attention_smc.core.gp import (
    Prediction,
    SameMolecule,
    build_gp,
    fit,
    fitted,
)
from bo_attention_smc.tests.reference import Posterior

CELL = "cells/bh_1__mace__attention.npz"
ROWS = [3, 50, 101, 222, 333, 404, 505, 606, 707, 780]


def setUpModule() -> None:  # noqa: N802 -- unittest's name
    """Run on one torch thread, as the campaigns do."""
    torch.set_num_threads(1)


class TestTerm(unittest.TestCase):
    """The same-molecule kernel, and the GP that carries it."""

    def setUp(self) -> None:
        """Read the cell, as the term's GP sees it."""
        self.cell = load_cell(CELL)
        self.pool = make_pool(self.cell, "geom", "bh_1")

    def test_same_molecule(self) -> None:
        """One exactly where two reactions share a molecule of a reagent."""
        pool = self.pool
        kernel = SameMolecule(active_dims=torch.tensor([pool.n_features]))
        # as Normalize maps the inputs
        low, high = pool.bounds
        x = (pool.x[ROWS] - low) / (high - low)
        k = kernel(x, x).to_dense()
        ids = torch.as_tensor(molecules("bh_1")[ROWS, 0])
        expected = (ids[:, None] == ids[None, :]).to(k.dtype)
        self.assertTrue(torch.equal(k, expected))
        diagonal = kernel(x, x, diag=True)
        self.assertTrue(
            torch.equal(diagonal, torch.ones(len(ROWS), dtype=k.dtype))
        )

    def test_term_is_the_reference(self) -> None:
        """At a fitted GP's settings, the reference gives the same model."""
        pool = self.pool
        gp = build_gp(
            pool.x[ROWS],
            pool.y[ROWS],
            pool.prior,
            pool.bounds,
            pool.n_features,
        )
        fit(gp)
        kernels = gp.covar_module.kernels
        # a GP without the term carrying the fitted Matern, noise and mean
        bounds = pool.bounds[:, : pool.n_features]
        null = build_gp(
            pool.features[ROWS],
            pool.y[ROWS],
            pool.prior,
            bounds,
            pool.n_features,
        )
        raw = kernels[0].raw_lengthscale.data
        null.covar_module.raw_lengthscale.data = raw
        null.likelihood.noise = gp.likelihood.noise.detach()
        null.mean_module.constant.data = gp.mean_module.constant.data
        posterior = Posterior(null, pool.features, ROWS, None, pool.molecules)
        weights = torch.stack(
            [k.raw_outputscale.detach() for k in kernels[1:]]
        )
        theta = torch.cat([posterior.start[: -len(weights)], weights])

        gp.train()
        marginal = gp.likelihood(gp(*gp.train_inputs)).log_prob(
            gp.train_targets
        )
        gp.eval()
        torch.testing.assert_close(
            posterior.log_likelihood(theta), marginal.detach()
        )
        candidates = np.arange(0, 790, 7)
        by_gp = Prediction.of_gp(gp, pool.x[candidates])
        by_posterior = posterior.predict(theta.unsqueeze(0), candidates)
        torch.testing.assert_close(
            by_gp.mean, by_posterior.mean, rtol=1e-6, atol=1e-6
        )
        torch.testing.assert_close(
            by_gp.variance, by_posterior.variance, rtol=1e-6, atol=1e-6
        )

    def test_fit_records_weights(self) -> None:
        """The term's weights start at their prior's mode and are recorded."""
        pool = self.pool
        gp = build_gp(
            pool.x[ROWS],
            pool.y[ROWS],
            pool.prior,
            pool.bounds,
            pool.n_features,
        )
        starts = [k.outputscale.item() for k in gp.covar_module.kernels[1:]]
        np.testing.assert_allclose(starts, 0.25)
        self.assertIn(fit(gp), ("SUCCESS", "STOPPED"))
        found = fitted(gp, pool.prior, pool.reagents)
        expected = ["ell_ratio", "noise"]
        expected += [f"w[{name}]" for name in reagents("bh_1")]
        self.assertEqual(list(found), expected)
        self.assertTrue(all(np.isfinite(v) for v in found.values()))


if __name__ == "__main__":
    unittest.main()
