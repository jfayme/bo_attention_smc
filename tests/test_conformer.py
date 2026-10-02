"""Tests for the conformer ensemble and the clustering symmetry guard.

None of these touch the MACE calculator. Building a ConformerGenerator is
cheap because the potential is loaded lazily, and every assertion here is
about RDKit geometry or about arithmetic on energies, so the suite runs
without a GPU.

"""

import unittest

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem

from conformer import EV_TO_KCAL, ConformerEnsemble, ConformerGenerator

RDLogger.DisableLog("rdApp.*")

# Four interchangeable CF3 groups give 31,104 symmetry mappings, well above
# any sensible threshold, on a molecule small enough to embed instantly.
SYMMETRIC = "FC(F)(F)C(C(F)(F)F)(C(F)(F)F)C(F)(F)F"

# Two mappings, so the guard must leave it alone.
PLAIN = "Cc1ccccc1"

# Every heavy atom sits in the ring, so stripping can remove nothing.
NO_TERMINALS = "C1CCCCC1"

# Two terminal methyls hanging off a ring-bound carbon.
BRANCHED = "CC(C)c1ccccc1"


def embed_heavy(smiles: str, n_confs: int = 3) -> Chem.Mol:
    """Embed a molecule and return it with its hydrogens removed.

    Parameters
    ----------
    smiles
        SMILES of the molecule to embed.
    n_confs
        Number of conformers to generate.

    Returns
    -------
    mol
        Heavy-atom molecule carrying the embedded conformers.

    """
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    params = AllChem.ETKDGv3()
    params.randomSeed = 0xF00D
    AllChem.EmbedMultipleConfs(mol, numConfs=n_confs, params=params)
    return Chem.RemoveHs(mol)


class TestConformerEnsemble(unittest.TestCase):
    """Derived quantities exposed by the relaxed conformer ensemble."""

    def setUp(self) -> None:
        """Build a five-conformer ensemble with a known energy ladder."""
        self.energies = np.array(
            [-100.0, -99.96, -99.93, -99.88, -99.80], dtype=np.float64
        )
        rng = np.random.default_rng(0xF00D)
        self.coords = rng.normal(size=(5, 3, 3)).astype(np.float32)
        self.numbers = np.array([6, 6, 8], dtype=np.int64)
        self.ensemble = ConformerEnsemble(
            numbers=self.numbers,
            coords=self.coords,
            energies=self.energies,
            charge=-1,
            multiplicity=1,
        )

    def test_len_counts_the_conformers(self) -> None:
        """The ensemble length is the number of relaxed conformers."""
        self.assertEqual(len(self.ensemble), 5)

    def test_lowest_returns_the_first_conformer(self) -> None:
        """Conformers are energy-sorted, so the minimum is the first."""
        np.testing.assert_array_equal(self.ensemble.lowest, self.coords[0])

    def test_as_atoms_packs_numbers_coordinates_and_charge(self) -> None:
        """The calculator triple carries the lowest-energy geometry."""
        numbers, coords, charge = self.ensemble.as_atoms
        np.testing.assert_array_equal(numbers, self.numbers)
        np.testing.assert_array_equal(coords, self.coords[0])
        self.assertEqual(charge, -1)

    def test_relative_energies_are_zero_at_the_minimum(self) -> None:
        """The lowest conformer is the reference for the ladder."""
        relative = self.ensemble.relative_energies_kcal
        self.assertAlmostEqual(float(relative[0]), 0.0)
        self.assertTrue((relative >= 0.0).all())

    def test_relative_energies_convert_ev_to_kcal(self) -> None:
        """Differences are scaled by the electronvolt conversion."""
        expected = (self.energies - self.energies.min()) * EV_TO_KCAL
        np.testing.assert_allclose(
            self.ensemble.relative_energies_kcal, expected
        )

    def test_boltzmann_weights_sum_to_one(self) -> None:
        """The populations are normalised."""
        weights = self.ensemble.boltzmann_weights(298.15)
        self.assertAlmostEqual(float(weights.sum()), 1.0, places=12)

    def test_boltzmann_weights_decrease_with_energy(self) -> None:
        """A higher conformer is never more populated than a lower one."""
        weights = self.ensemble.boltzmann_weights(298.15)
        self.assertTrue((np.diff(weights) <= 0.0).all())

    def test_boltzmann_weights_collapse_onto_the_minimum(self) -> None:
        """As the temperature falls, only the minimum stays populated."""
        weights = self.ensemble.boltzmann_weights(1.0)
        np.testing.assert_allclose(weights[0], 1.0, atol=1e-9)
        np.testing.assert_allclose(weights[1:], 0.0, atol=1e-9)

    def test_boltzmann_weights_ignore_a_constant_offset(self) -> None:
        """Shifting every energy leaves the populations unchanged."""
        shifted = ConformerEnsemble(
            numbers=self.numbers,
            coords=self.coords,
            energies=self.energies + 500.0,
            charge=-1,
            multiplicity=1,
        )
        np.testing.assert_allclose(
            shifted.boltzmann_weights(298.15),
            self.ensemble.boltzmann_weights(298.15),
        )

    def test_single_conformer_ensemble_is_well_defined(self) -> None:
        """A lone minimum takes the whole population."""
        single = ConformerEnsemble(
            numbers=self.numbers,
            coords=self.coords[:1],
            energies=self.energies[:1],
            charge=0,
            multiplicity=1,
        )
        self.assertEqual(len(single), 1)
        np.testing.assert_allclose(single.boltzmann_weights(298.15), [1.0])
        np.testing.assert_allclose(single.relative_energies_kcal, [0.0])


class TestTerminalStripping(unittest.TestCase):
    """Removal of the terminal heavy atoms used by the symmetry guard."""

    def setUp(self) -> None:
        """Build a generator; the MACE potential is never touched."""
        self.generator = ConformerGenerator()

    def test_only_degree_one_heavy_atoms_are_removed(self) -> None:
        """The two isopropyl methyls go and the ring is left alone."""
        heavy = embed_heavy(BRANCHED)
        core = self.generator._strip_terminal_atoms(heavy)
        expected = sum(1 for atom in heavy.GetAtoms() if atom.GetDegree() != 1)
        self.assertEqual(core.GetNumAtoms(), expected)
        self.assertLess(core.GetNumAtoms(), heavy.GetNumAtoms())

    def test_kept_atoms_keep_their_coordinates(self) -> None:
        """Stripping carries every conformer across unchanged."""
        heavy = embed_heavy(BRANCHED)
        keep = [
            atom.GetIdx() for atom in heavy.GetAtoms() if atom.GetDegree() != 1
        ]
        core = self.generator._strip_terminal_atoms(heavy)
        self.assertEqual(core.GetNumConformers(), heavy.GetNumConformers())
        for conf_id in range(core.GetNumConformers()):
            np.testing.assert_allclose(
                heavy.GetConformer(conf_id).GetPositions()[keep],
                core.GetConformer(conf_id).GetPositions(),
            )

    def test_a_molecule_without_terminal_atoms_is_unchanged(self) -> None:
        """Every cyclohexane carbon is in the ring, so none is dropped."""
        heavy = embed_heavy(NO_TERMINALS)
        core = self.generator._strip_terminal_atoms(heavy)
        self.assertEqual(core.GetNumAtoms(), heavy.GetNumAtoms())


class TestSymmetryGuard(unittest.TestCase):
    """The automorphism threshold that chooses the clustering reference."""

    def setUp(self) -> None:
        """Pin the threshold so the test does not depend on settings."""
        self.generator = ConformerGenerator()
        self.generator.max_automorphisms = 5000

    def test_low_symmetry_molecule_is_left_alone(self) -> None:
        """Toluene has two mappings, far below the threshold."""
        heavy = embed_heavy(PLAIN)
        reference = self.generator._rmsd_reference(heavy, PLAIN)
        self.assertEqual(reference.GetNumAtoms(), heavy.GetNumAtoms())

    def test_high_symmetry_molecule_is_stripped(self) -> None:
        """Four interchangeable CF3 groups push it past the threshold."""
        heavy = embed_heavy(SYMMETRIC)
        with self.assertLogs("conformer", level="WARNING") as captured:
            reference = self.generator._rmsd_reference(heavy, SYMMETRIC)
        self.assertLess(reference.GetNumAtoms(), heavy.GetNumAtoms())
        self.assertEqual(
            reference.GetNumAtoms(),
            self.generator._strip_terminal_atoms(heavy).GetNumAtoms(),
        )
        # Switching metric must never be silent.
        self.assertIn("symmetry mappings", captured.output[0])

    def test_raising_the_threshold_disables_the_guard(self) -> None:
        """Above every count in the molecule, the guard must not fire."""
        self.generator.max_automorphisms = 10**6
        heavy = embed_heavy(SYMMETRIC)
        reference = self.generator._rmsd_reference(heavy, SYMMETRIC)
        self.assertEqual(reference.GetNumAtoms(), heavy.GetNumAtoms())


if __name__ == "__main__":
    unittest.main()
