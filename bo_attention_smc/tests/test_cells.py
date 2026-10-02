"""Cells rebuilt from the data are the cells the study stored; molecules.

Shields is rebuilt under Morgan, the chemistry T5 and MACE, with attention
for the two with atoms, and compared with the study's cells as prepare.py
converted them into cells/:

- Morgan and T5, array for array. T5 ran on the GPU for the study, so its
  check needs one;
- MACE in structure (the channels, the molecules of every reaction, the
  mask, the conformer weights) and in value to 1e-4 only: MACE on the GPU
  does not repeat itself, by about 5e-6 from one run to the next.

The MACE check reads the conformers the study cached in conformers/, and
both model checks load their models: about half a minute in all.

The molecules the term reads from a dataset (cells.molecules) must be
those an attention cell's atoms name, number for number, on every stored
attention cell.

    python -m unittest bo_attention_smc.tests.test_cells

"""

import unittest
from pathlib import Path

import numpy as np
import torch

from bo_attention_smc.core.cells import (
    Cell,
    build_attention,
    build_fixed,
    describe,
    load_cell,
    molecules,
    read_reactions,
)
from bo_attention_smc.runs.prepare import same

CELLS = Path("cells")


def rebuild(featuriser: str, aggregation: str) -> Cell:
    """Build a Shields cell as prepare.py does, at its defaults."""
    reactions = read_reactions("shields")
    described = describe(reactions, "shields", featuriser, aggregation)
    if aggregation == "attention":
        return build_attention(reactions, "shields", described, "decorr0.7")
    return build_fixed(reactions, "shields", described, "decorr0.7")


class TestRebuild(unittest.TestCase):
    """Shields, rebuilt from its CSV and compared with the stored cells."""

    def stored(self, name: str) -> Cell:
        """Read a stored cell; skip when cells/ does not hold it."""
        path = CELLS / f"{name}.npz"
        if not path.is_file():
            self.skipTest(f"{path} is not available")
        return load_cell(path)

    def test_morgan(self) -> None:
        """Fingerprints: array for array."""
        stored = self.stored("shields__morgan")
        self.assertTrue(same(rebuild("morgan", "mean"), stored))

    @unittest.skipUnless(torch.cuda.is_available(), "T5 ran on the GPU")
    def test_t5(self) -> None:
        """Chemistry T5 tokens: array for array."""
        stored = self.stored("shields__t5_augm__attention")
        self.assertTrue(same(rebuild("t5_augm", "attention"), stored))

    def test_mace(self) -> None:
        """MACE atoms: the same structure, the same values to 1e-4."""
        stored = self.stored("shields__mace__attention")
        cell = rebuild("mace", "attention")
        self.assertEqual(cell.blocks, stored.blocks)
        ours, theirs = cell.atoms, stored.atoms
        assert ours is not None and theirs is not None
        self.assertEqual(ours.pooling, theirs.pooling)
        for name in ("mask", "weights", "index", "settings"):
            self.assertTrue(
                torch.equal(getattr(ours, name), getattr(theirs, name)), name
            )
        for a, b in zip(ours.columns, theirs.columns, strict=True):
            self.assertTrue(torch.equal(a, b))
        np.testing.assert_allclose(cell.features, stored.features, atol=1e-4)
        np.testing.assert_allclose(ours.values, theirs.values, atol=1e-4)


class TestMolecules(unittest.TestCase):
    """The molecules of a dataset, as its attention cells number them."""

    def test_stored_cells(self) -> None:
        """Every stored attention cell's atoms name the same molecules."""
        paths = sorted(CELLS.glob("*__attention.npz"))
        if not paths:
            self.skipTest(f"{CELLS} holds no attention cell")
        for path in paths:
            with self.subTest(cell=path.stem):
                dataset = path.stem.split("__")[0]
                atoms = load_cell(path).atoms
                assert atoms is not None
                np.testing.assert_array_equal(
                    molecules(dataset), atoms.index.numpy()
                )


if __name__ == "__main__":
    unittest.main()
