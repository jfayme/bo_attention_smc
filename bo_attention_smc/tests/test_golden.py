"""bo_attention_smc must give the numbers recorded in golden.json.

golden.json is the golden file of the study's bo_fixes package, copied
unchanged: what the code of the study's commit e9a613f computed for short
campaigns and fixed posteriors. Two parts of it are checked here:

- the `null` campaigns (seed 3, geom, 5 random then 4 BO picks) with their
  per-fit diagnostics, which the null's records now name `noise`,
  `ell_ratio` (the old `ell_mean` over ell_0) and `log_score` (the old
  `log_score_null`);
- the posterior of the sampled arms at fixed vectors: its log density,
  gradient, predictions and LogEI, computed by tests/reference.py, the
  one-particle version that test_smc.py checks posterior.py against.

The choices must match exactly and the numbers to 1e-9, relatively. The
golden NUTS campaigns are not checked: NUTS is gone.

The pools:

- `learned`: bh_1 with fake atoms. Every molecule gets random atoms seeded
  from a hash of its SMILES, in 1 to 3 conformers weighted by the Boltzmann
  populations of a fake energy ladder: the attention path, conformers
  included, without a model;
- `fixed`: 80 random points of the unit cube under a smooth bump of yield;
- `real`, `real_t5`, `real_morgan`: the study's Shields cells under MACE
  (a conformer bank), the chemistry T5 (a token bank) and Morgan, in
  cells/. A skipped cell is a hole in the check.

The tests run on one thread, as the numbers were recorded, from the
repository root:

    python -m unittest bo_attention_smc.tests.test_golden

"""

import json
import math
import unittest
import zlib
from pathlib import Path
from typing import Any

import numpy as np
import torch

from bo_attention_smc.core.campaign import make_pool, map_step, run
from bo_attention_smc.core.cells import (
    DATASETS,
    Cell,
    Described,
    build_attention,
    load_cell,
    read_reactions,
)
from bo_attention_smc.core.conformer import ConformerEnsemble
from bo_attention_smc.core.gp import build_gp, fit, make_prior, mean_distance
from bo_attention_smc.tests.reference import Posterior, SpanHead

GOLDEN = Path(__file__).resolve().parent / "golden.json"
CELLS = Path("cells")
RTOL = 1e-9

# The sampled arms of each pool whose posterior the golden file holds.
SAMPLED = {
    "learned": ("hmc_null", "hmc_span", "hmc_span_pca8"),
    "fixed": (),
    "real": ("hmc_span_pca8",),
    "real_t5": ("hmc_span_pca8",),
    "real_morgan": ("hmc_null",),
}

# The real cells, by their key in the golden file.
REAL_CELLS = {
    "real": "shields__mace__attention.npz",
    "real_t5": "shields__t5_augm__attention.npz",
    "real_morgan": "shields__morgan.npz",
}


def learned_cell() -> Cell:
    """Build bh_1 with fake atoms, as the study's tests did."""
    reactions = read_reactions("bh_1")
    described = {}
    for name in DATASETS["bh_1"].compounds:
        smiles = reactions[name].unique().tolist()
        descriptors, ensembles = [], []
        for s in smiles:
            # crc32 rather than hash(): the same draws in every process
            rng = np.random.default_rng(zlib.crc32(s.encode()))
            k, n = int(rng.integers(1, 4)), int(rng.integers(2, 7))
            descriptors.append(rng.normal(size=(k, n, 12)))
            ladder = np.random.default_rng(zlib.crc32(s.encode()))
            ensembles.append(
                ConformerEnsemble(
                    numbers=np.full(n, 6, dtype=np.int64),
                    coords=np.zeros((k, n, 3), dtype=np.float32),
                    energies=np.sort(ladder.normal(size=k)) * 0.01,
                    charge=0,
                    multiplicity=1,
                )
            )
        described[name] = Described(
            smiles, "fake", descriptors=descriptors, ensembles=ensembles
        )
    return build_attention(
        reactions, "bh_1", described, "decorr0.7", "boltzmann"
    )


def fixed_cell() -> Cell:
    """Build 80 random points of the unit cube under a bump of yield."""
    rng = np.random.default_rng(0)
    features = rng.random((80, 3))
    objective = np.exp(-np.sum((features - 0.7) ** 2, axis=1) / 0.08)
    objective += 0.01 * rng.normal(size=80)
    return Cell(objective, features, blocks=())


def null_campaign(cell: Cell) -> dict:
    """Run the golden null campaign; name its fits as the golden file."""
    pool = make_pool(cell, "geom")
    found = run(pool.y, map_step(pool), seed=3, n_bo=4)
    fits = found["fits"]
    return {
        "sampled_indices": found["sampled_indices"],
        "fits": {
            "noise_null": fits["noise"],
            "ell_mean": [r * pool.prior.ell_0 for r in fits["ell_ratio"]],
            "log_score_null": fits["log_score"],
        },
    }


def posterior_numbers(cell: Cell, arm: str) -> dict:
    """Evaluate the reference posterior at the golden vectors.

    The null is fitted on 20 rows drawn with seed 11, and the head drawn
    from the torch stream seeded 5; the vectors are the start and two
    perturbations of it drawn with seed 12.
    """
    pool = make_pool(cell, "geom")
    head = None
    if arm != "hmc_null":
        head = SpanHead(cell, pool.bounds, pca8=arm == "hmc_span_pca8")
    rows = np.random.default_rng(11).choice(len(pool.y), 20, replace=False)
    rows = torch.as_tensor(rows)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(5)
        gp = build_gp(
            pool.x[rows],
            pool.y[rows],
            pool.prior,
            pool.bounds,
            pool.n_features,
        )
        fit(gp)
        posterior = Posterior(gp, pool.x, rows, head)
    start = posterior.start
    noise = np.random.default_rng(12).normal(0.0, 0.1, (2, len(start)))
    noise = torch.as_tensor(noise)
    thetas = [start, start + noise[0], start + noise[1]]
    values, gradients = [], []
    for theta in thetas:
        theta = theta.clone().requires_grad_(True)
        value = posterior.log_posterior(theta)
        (gradient,) = torch.autograd.grad(value, theta)
        values.append(float(value.detach()))
        gradients.append([float(g) for g in gradient[:12]])
    candidates = np.setdiff1d(np.arange(len(pool.y)), rows.numpy())[:25]
    prediction = posterior.predict(torch.stack(thetas), candidates)
    best = float(pool.y[rows].max())
    return {
        "dimension": len(start),
        "log_posterior": values,
        "gradient_head": gradients,
        "mean": prediction.mean.reshape(-1).tolist(),
        "variance": prediction.variance.reshape(-1).tolist(),
        "log_ei": prediction.log_ei(best).tolist(),
    }


class TestGolden(unittest.TestCase):
    """The golden numbers, recomputed."""

    golden: dict
    threads: int

    @classmethod
    def setUpClass(cls: type["TestGolden"]) -> None:
        """Read the golden file, and run on one thread, as it was written."""
        cls.golden = json.loads(GOLDEN.read_text(encoding="utf-8"))
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls: type["TestGolden"]) -> None:
        """Give the thread count back."""
        torch.set_num_threads(cls.threads)

    def close(
        self,
        ours: Any,  # noqa: ANN401 -- JSON values of any kind
        theirs: Any,  # noqa: ANN401
        where: str,
    ) -> None:
        """Compare nested numbers to RTOL, and everything else exactly."""
        if isinstance(theirs, dict):
            self.assertEqual(set(ours), set(theirs), where)
            for key in theirs:
                self.close(ours[key], theirs[key], f"{where}.{key}")
        elif isinstance(theirs, list):
            self.assertEqual(len(ours), len(theirs), where)
            for i, (a, b) in enumerate(zip(ours, theirs, strict=True)):
                self.close(a, b, f"{where}[{i}]")
        elif isinstance(theirs, float) and math.isnan(theirs):
            self.assertTrue(math.isnan(ours), where)
        elif isinstance(theirs, float):
            tolerance = RTOL * max(1.0, abs(theirs))
            self.assertAlmostEqual(ours, theirs, delta=tolerance, msg=where)
        else:
            self.assertEqual(ours, theirs, where)

    def check(self, key: str, cell: Cell) -> None:
        """Compare one pool's null campaign and posteriors with golden."""
        with self.subTest(pool=key, arm="null"):
            recorded = self.golden[key]["null"]["campaign"]
            self.close(null_campaign(cell), recorded, f"{key}.null")
        for arm in SAMPLED[key]:
            with self.subTest(pool=key, arm=arm):
                recorded = self.golden[key][arm]["posterior"]
                found = posterior_numbers(cell, arm)
                self.close(found, recorded, f"{key}.{arm}.posterior")

    def real(self, key: str) -> None:
        """Compare a real cell; skipped when cells/ does not hold it."""
        path = CELLS / REAL_CELLS[key]
        if not path.is_file():
            self.skipTest(f"{path} is not available")
        self.check(key, load_cell(path))

    def test_learned(self) -> None:
        """The attention fixture: mean pooling and both span heads."""
        self.check("learned", learned_cell())

    def test_fixed(self) -> None:
        """A fixed-feature pool: the null."""
        self.check("fixed", fixed_cell())

    def test_real_mace(self) -> None:
        """Shields/MACE: a conformer bank, the null and the PCA-8 head."""
        self.real("real")

    def test_real_t5(self) -> None:
        """Shields/chemistry T5: a token bank, no conformer axis."""
        self.real("real_t5")

    def test_real_morgan(self) -> None:
        """Shields/Morgan: fingerprints, the null and its posterior."""
        self.real("real_morgan")


class TestPrior(unittest.TestCase):
    """chen and geom centre the prior where the study's sweep recorded."""

    def test_stored_cells(self) -> None:
        """Every cell of the sweep: its mean distance and both centres."""
        path = CELLS / "cells.json"
        if not path.is_file():
            self.skipTest(f"{path} is not available")
        recorded = json.loads(path.read_text(encoding="utf-8"))
        for name, entry in recorded.items():
            if entry["status"] != "ok":
                continue
            cell = load_cell(CELLS / f"{name.replace('/', '__')}.npz")
            with self.subTest(cell=name):
                self.assertEqual(mean_distance(cell.features), entry["D_bar"])
                # the sweep predates the rule `default`
                for rule in ("chen", "geom"):
                    ell_0 = make_prior(cell.features, rule).ell_0
                    self.assertEqual(ell_0, entry[f"ell_{rule}"])


if __name__ == "__main__":
    unittest.main()
