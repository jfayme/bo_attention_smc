"""Recorded campaigns, replayed: the same choices and the same scores.

The first BO steps of campaigns the study recorded with its old packages
(bo_per_molecule, bo_smc) are run again by this package, and must choose
the same reactions and score each choice to the last bit:

- MAP: mean pooling and the term under chen (bo_per_molecule_config),
  the rule `default` and Morgan (fill/map);
- SMC, on the GPU: the head with the term (bo_smc_chen), the term alone
  (bo_smc_term), the head alone (fill/smc) and the head on every channel
  (compress/smc).

An SMC campaign pads every step to its last, 55 reactions; a replay pads
to the same, since another padding changes the last bits, and through
resampling the choices. The records are in records/; a missing one is
skipped, and the SMC replays need CUDA.

    python -m unittest bo_attention_smc.tests.test_replay

"""

import json
import unittest
from pathlib import Path

import torch

from bo_attention_smc.core.campaign import (
    MODELS,
    N_BO,
    N_INIT,
    make_pool,
    make_step,
    run,
)
from bo_attention_smc.core.cells import load_cell
from bo_attention_smc.runs.run import cell_path

# folder, cell, model, rule, seed
MAP_CAMPAIGNS = (
    ("records/bo_per_molecule_config", "shields/mace", "null", "chen", 40),
    ("records/bo_per_molecule_config", "shields/mace", "identity", "chen", 40),
    ("records/bo_per_molecule_config", "bh_1/t5_augm", "identity", "chen", 40),
    ("records/fill/map", "shields/t5_augm", "null", "default", 7),
    ("records/fill/map", "bh_1/morgan", "identity", "geom", 3),
    ("records/fill/map", "bh_full/morgan", "null", "chen", 11),
)
SMC_CAMPAIGNS = (
    ("records/bo_smc_chen", "shields/mace", "smc_identity_head", "chen", 40),
    ("records/bo_smc_term", "bh_1/t5_augm", "smc_identity", "chen", 25),
    ("records/fill/smc", "bh_full/mace", "smc_span_pca8", "geom", 22),
    ("records/compress/smc", "shields/mace", "smc_span", "chen", 3),
)


def recorded(
    folder: str, cell: str, model: str, rule: str, seed: int
) -> dict | None:
    """Return a recorded campaign, or None when records/ lacks it."""
    path = Path(folder) / "campaigns.jsonl"
    if not path.is_file():
        return None
    wanted = (cell, model, rule, seed, "ok")
    for line in path.read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        if (r["cell"], r["arm"], r["rule"], r["seed"], r["status"]) == wanted:
            return r
    return None


class TestReplay(unittest.TestCase):
    """The first steps of recorded campaigns, choice for choice."""

    def replay(self, campaign: tuple, n_bo: int, device: str) -> None:
        """Replay one campaign's first BO steps, and compare."""
        folder, cell_name, model, rule, seed = campaign
        theirs = recorded(*campaign)
        if theirs is None:
            self.skipTest(f"{folder} does not hold {campaign}")
        dataset, featuriser = cell_name.split("/")
        cell = load_cell(cell_path(dataset, featuriser))
        pool = make_pool(cell, rule, dataset if MODELS[model].term else None)
        step = make_step(model, cell, pool, seed, N_INIT + N_BO, device)
        mine = run(pool.y, step, seed, n_bo)
        self.assertEqual(
            mine["sampled_indices"],
            theirs["sampled_indices"][: N_INIT + n_bo],
        )
        # the null's records named its scores `log_score_null`
        key = (
            "log_score" if "log_score" in theirs["fits"] else "log_score_null"
        )
        self.assertEqual(mine["fits"]["log_score"], theirs["fits"][key][:n_bo])

    def test_map(self) -> None:
        """MAP: six BO steps of each campaign, on one CPU thread."""
        threads = torch.get_num_threads()
        torch.set_num_threads(1)
        try:
            for campaign in MAP_CAMPAIGNS:
                with self.subTest(campaign=campaign):
                    self.replay(campaign, 6, "cpu")
        finally:
            torch.set_num_threads(threads)

    @unittest.skipUnless(torch.cuda.is_available(), "SMC ran on the GPU")
    def test_smc(self) -> None:
        """SMC: three BO steps of each campaign, on the GPU."""
        threads = torch.get_num_threads()
        torch.set_num_threads(1)
        try:
            for campaign in SMC_CAMPAIGNS:
                with self.subTest(campaign=campaign):
                    self.replay(campaign, 3, "cuda")
        finally:
            torch.set_num_threads(threads)


if __name__ == "__main__":
    unittest.main()
