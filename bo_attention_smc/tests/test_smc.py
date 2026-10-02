"""The fast posterior is the reference's, particle for particle; SMC works.

posterior.py computes the log posterior, its gradient and the predictions
for a whole population at once; tests/reference.py computes them one
particle at a time, plainly. They must agree on MACE with conformers, MACE
with numeric settings (Shields), T5 tokens, the per-molecule term, the
`default` rule's LogNormal prior and the head on every channel, on the CPU
and on the GPU where there is one. The sampler must reach a known
Gaussian target, and move a real campaign state.

    python -m unittest bo_attention_smc.tests.test_smc

"""

import dataclasses
import math
import unittest

import numpy as np
import torch

from bo_attention_smc.core import smc
from bo_attention_smc.core.campaign import make_pool
from bo_attention_smc.core.cells import load_cell
from bo_attention_smc.core.gp import build_gp
from bo_attention_smc.core.posterior import (
    F64,
    Bank,
    Data,
    log_posterior,
    log_prior,
    predict,
)
from bo_attention_smc.tests.reference import Posterior, SpanHead

CELLS = {
    "conformers": ("bh_1", "cells/bh_1__mace__attention.npz"),
    "settings": ("shields", "cells/shields__mace__attention.npz"),
    "tokens": ("bh_1", "cells/bh_1__t5_augm__attention.npz"),
}
ROWS = [3, 50, 101, 222, 333, 404, 505, 606, 707, 780, 12, 34]
N_MAX = 20
DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


def target(
    name: str,
    term: bool = False,
    device: str = "cpu",
    rule: str = "geom",
    head: str = "pca8",
) -> tuple[Posterior, Bank, Data]:
    """Build the reference posterior on ROWS, and the fast bank and data."""
    dataset, path = CELLS[name]
    cell = load_cell(path)
    pool = make_pool(cell, rule, dataset if term else None)
    bounds = pool.bounds[:, : pool.n_features]
    gp = build_gp(
        pool.features[ROWS], pool.y[ROWS], pool.prior, bounds, pool.n_features
    )
    molecules = pool.molecules if term else None
    torch.manual_seed(0)
    posterior = Posterior(
        gp,
        pool.features,
        ROWS,
        SpanHead(cell, bounds, pca8=head == "pca8"),
        molecules,
    )
    bank = Bank.of(cell, bounds, head, molecules, device=device)
    data = Data.of(gp, ROWS, pool.prior, N_MAX, device)
    return posterior, bank, data


class TestPosterior(unittest.TestCase):
    """The batched log posterior, gradient and predictions."""

    def check(self, posterior: Posterior, bank: Bank, data: Data) -> None:
        """Compare every quantity with the reference, particle by particle."""
        self.assertEqual(bank.dimension, len(posterior.start))
        generator = torch.Generator().manual_seed(1)
        noise = torch.randn(3, len(posterior.start), generator=generator)
        batch = (
            posterior.start + torch.tensor([0.0, 0.3, 1.0])[:, None] * noise
        )
        device = bank.device
        mine = log_posterior(batch.to(device), bank, data).cpu()
        leaf = batch.clone().to(device).requires_grad_(True)
        log_posterior(leaf, bank, data).sum().backward()
        mean, variance, noise_variance = predict(batch.to(device), bank, data)
        everything = np.arange(bank.n_pool)
        for k, theta in enumerate(batch):
            theirs = posterior.log_posterior(theta)
            self.assertAlmostEqual(
                float(mine[k]), float(theirs), delta=1e-8 * abs(float(theirs))
            )
            their_leaf = theta.clone().requires_grad_(True)
            posterior.log_posterior(their_leaf).backward()
            np.testing.assert_allclose(
                leaf.grad[k].cpu().numpy(),
                their_leaf.grad.numpy(),
                rtol=1e-6,
                atol=1e-8,
            )
            prediction = posterior.predict(theta.unsqueeze(0), everything)
            np.testing.assert_allclose(
                mean[k].cpu().numpy(),
                prediction.mean[0].numpy(),
                rtol=1e-8,
                atol=1e-8,
            )
            np.testing.assert_allclose(
                variance[k].cpu().numpy(),
                prediction.variance[0].numpy(),
                rtol=1e-6,
                atol=1e-8,
            )
            self.assertAlmostEqual(
                float(noise_variance[k]), float(prediction.noise[0]), places=10
            )

    def test_conformer_bank(self) -> None:
        """MACE on bh_1: atoms with a conformer axis."""
        for device in DEVICES:
            with self.subTest(device=device):
                self.check(*target("conformers", device=device))

    def test_numeric_settings(self) -> None:
        """MACE on Shields: temperature and concentration columns too."""
        self.check(*target("settings"))

    def test_token_bank(self) -> None:
        """T5 on bh_1: tokens without a conformer axis."""
        self.check(*target("tokens"))

    def test_per_molecule_term(self) -> None:
        """The term, on every device."""
        for device in DEVICES:
            with self.subTest(device=device):
                self.check(*target("conformers", term=True, device=device))

    def test_default_rule(self) -> None:
        """BoTorch's LogNormal length-scale prior, and its reference draws."""
        for device in DEVICES:
            with self.subTest(device=device):
                self.check(
                    *target(
                        "conformers", term=True, device=device, rule="default"
                    )
                )
        _, bank, data = target("conformers", rule="default")
        assert data.prior.lognormal is not None
        loc, scale = data.prior.lognormal
        self.assertAlmostEqual(loc, math.sqrt(2) + 0.5 * math.log(43))
        generator = torch.Generator().manual_seed(4)
        drawn = smc.draw_reference(bank, data, 4000, generator)
        raw = drawn[:, : bank.n_lengthscales]
        log_ell = torch.log(torch.nn.functional.softplus(raw))
        self.assertAlmostEqual(float(log_ell.mean()), loc, delta=0.02)
        self.assertAlmostEqual(float(log_ell.std()), scale, delta=0.02)

    def test_full_head(self) -> None:
        """The head on every channel, no projection."""
        for name in CELLS:
            for device in DEVICES:
                with self.subTest(cell=name, device=device):
                    self.check(*target(name, device=device, head="all"))

    def test_held_head(self) -> None:
        """A narrower head prior changes the log prior as N(0, s^2) says."""
        posterior, bank, data = target("conformers")
        held = dataclasses.replace(bank, head_scale=0.5)
        generator = torch.Generator().manual_seed(2)
        batch = posterior.start + torch.randn(
            4, len(posterior.start), generator=generator, dtype=F64
        )
        start = bank.n_lengthscales + 2
        head = batch[:, start : start + bank.n_head]
        expected = (
            -0.5 * (head / 0.5) ** 2 - math.log(0.5) + 0.5 * head**2
        ).sum(-1)
        change = log_prior(batch, held, data) - log_prior(batch, bank, data)
        torch.testing.assert_close(change, expected)
        drawn = smc.draw_reference(held, data, 4000, generator)
        spread = drawn[:, start : start + bank.n_head].std()
        self.assertAlmostEqual(float(spread), 0.5, delta=0.02)


class TestSampler(unittest.TestCase):
    """The tempering loop reaches its target."""

    def test_gaussian(self) -> None:
        """From N(0, I) to a correlated Gaussian: moments and evidence."""
        d = 5
        generator = torch.Generator().manual_seed(0)
        mean = torch.arange(1.0, d + 1.0, dtype=F64)
        a = torch.randn(d, d, generator=generator, dtype=F64)
        covariance = a @ a.T / d + 0.2 * torch.eye(d, dtype=F64)
        precision = torch.linalg.inv(covariance)

        def log_base(theta: torch.Tensor) -> torch.Tensor:
            return -0.5 * (theta**2).sum(-1)

        def log_target(theta: torch.Tensor) -> torch.Tensor:
            r = theta - mean
            return -0.5 * torch.einsum("pi,ij,pj->p", r, precision, r)

        settings = smc.Settings(particles=4000, mcmc_steps=10)
        particles = torch.randn(4000, d, generator=generator, dtype=F64)
        particles, health = smc.temper(
            particles, log_base, log_target, settings, 0.3, generator
        )
        self.assertTrue(health.reached)
        np.testing.assert_allclose(particles.mean(0), mean, atol=0.1)
        np.testing.assert_allclose(
            torch.cov(particles.T), covariance, atol=0.15
        )
        # the estimate's sd is about 0.07 at 4000 particles (5 seeds)
        expected = 0.5 * float(torch.linalg.slogdet(covariance)[1])
        self.assertAlmostEqual(health.log_evidence, expected, delta=0.2)

    def test_campaign_state(self) -> None:
        """First and next posterior of a real state, on every device."""
        cell = load_cell(CELLS["conformers"][1])
        pool = make_pool(cell, "geom")
        settings = smc.Settings(particles=64, mcmc_steps=2)
        for device in DEVICES:
            with self.subTest(device=device):
                bank = Bank.of(cell, pool.bounds, "pca8", device=device)
                datas = []
                for rows in (ROWS, [*ROWS, 5]):
                    gp = build_gp(
                        pool.x[rows],
                        pool.y[rows],
                        pool.prior,
                        pool.bounds,
                        pool.n_features,
                    )
                    datas.append(Data.of(gp, rows, pool.prior, N_MAX, device))
                generator = torch.Generator(device).manual_seed(0)
                particles, health = smc.first(
                    bank, datas[0], settings, generator
                )
                self.assertEqual(particles.shape, (64, bank.dimension))
                self.assertTrue(torch.isfinite(particles).all())
                self.assertTrue(health.reached)
                moved, health = smc.advance(
                    particles,
                    bank,
                    datas[0],
                    datas[1],
                    settings,
                    health.step_size,
                    generator,
                )
                self.assertTrue(health.reached)
                self.assertTrue(torch.isfinite(moved).all())
                self.assertFalse(math.isnan(health.acceptance))


if __name__ == "__main__":
    unittest.main()
