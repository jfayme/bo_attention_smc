"""Lift and coverage are the numbers the study recorded.

Every campaign recorded in records/ is summarised again from its choices
and the dataset's yields, and must give its recorded metrics exactly: lift
needs the random baseline at the campaign's own length, 55 experiments in
all of them. The folders are those of the per-molecule term and of SMC:
the confirmations, the table's runs, the SMC term test and the head on
every channel. A missing folder is skipped.

    python -m unittest bo_attention_smc.tests.test_metrics

"""

import json
import unittest
from pathlib import Path

import numpy as np

from bo_attention_smc.core.cells import read_reactions
from bo_attention_smc.diagnostics_and_metrics.metrics import (
    outcome,
    random_auc,
)

RECORDS = Path("records")

# What the study recorded, and outcome recomputes.
KEYS = (
    "auc",
    "lift",
    "coverage_top5",
    "simple_regret",
    "best_found",
    "first_top5_hit",
)

# The run folders of records/.
FOLDERS = (
    "bo_per_molecule",
    "bo_per_molecule_confirm",
    "bo_per_molecule_config",
    "bo_smc",
    "bo_smc_confirm",
    "bo_smc_chen",
    "bo_smc_term",
    "fill/map",
    "fill/smc",
    "compress/smc",
)


class TestMetrics(unittest.TestCase):
    """Recorded campaigns, summarised again."""

    def check(self, folder: Path, pools: dict, baselines: dict) -> None:
        """Recompute every completed campaign of a run folder."""
        records = folder / "campaigns.jsonl"
        if not records.is_file():
            self.skipTest(f"{records} is not available")
        checked = 0
        for line in records.read_text(encoding="utf-8").splitlines():
            record = json.loads(line)
            if record["status"] != "ok":
                continue
            dataset = record["cell"].split("/")[0]
            pool = pools[dataset]
            sampled = pool[record["sampled_indices"]]
            found = outcome(sampled, pool, baselines[dataset])
            for key in KEYS:
                # exactly equal, NaN (never in the top 5 %) equal to NaN
                np.testing.assert_equal(
                    found[key],
                    record["metrics"][key],
                    f"{folder} {record['cell']} {key}",
                )
            checked += 1
        self.assertGreater(checked, 0)

    def test_records(self) -> None:
        """Every campaign of every folder of records/."""
        pools, baselines = {}, {}
        for dataset in ("bh_full", "bh_1", "shields"):
            objective = read_reactions(dataset)["objective"]
            pools[dataset] = objective.to_numpy(dtype=np.float64)
            baselines[dataset] = random_auc(pools[dataset], 55)
        for folder in FOLDERS:
            with self.subTest(folder=folder):
                self.check(RECORDS / folder, pools, baselines)


if __name__ == "__main__":
    unittest.main()
