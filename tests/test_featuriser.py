"""Tests for the pooling, aggregation and coverage helpers.

None of these load a model. Every assertion is about arithmetic on a
hand-built descriptor matrix or on a ConformerEnsemble assembled directly,
so the suite runs without a GPU and without the MACE or AIMNet2 weights.

"""

import unittest

import numpy as np

from conformer import ConformerEnsemble
from featuriser import (
    FIXED_AGGREGATIONS,
    POOL_STAT_NAMES,
    POOLINGS,
    STAT_NAMES,
    MLIPFeaturizer,
)

# The atom-axis collapse and the conformer-axis collapse, and their widths.
aggregate_atoms = MLIPFeaturizer.fixed_aggregation
aggregated_width = MLIPFeaturizer.fixed_aggregation_width
pool_conformers = MLIPFeaturizer.conformers_pooling
pooled_width = MLIPFeaturizer.conformers_pooling_width


def make_ensemble(
    n_conformers: int = 3, n_atoms: int = 3
) -> ConformerEnsemble:
    """Build an ensemble with a known energy ladder.

    Parameters
    ----------
    n_conformers
        Number of conformers the ensemble holds.
    n_atoms
        Number of atoms each conformer has.

    Returns
    -------
    ensemble
        Ensemble whose energies rise linearly from the minimum.

    """
    rng = np.random.default_rng(0xF00D)
    return ConformerEnsemble(
        numbers=np.full(n_atoms, 6, dtype=np.int64),
        coords=rng.normal(size=(n_conformers, n_atoms, 3)).astype(np.float32),
        energies=np.linspace(-100.0, -99.9, n_conformers),
        charge=0,
        multiplicity=1,
    )


class TestAtomAggregation(unittest.TestCase):
    """Collapsing a per-atom descriptor matrix to one conformer vector."""

    def setUp(self) -> None:
        """Build a small matrix whose statistics are known by hand."""
        # rows chosen so mean, std and max are all distinct
        self.per_atom = np.array(
            [[0.0, 1.0], [2.0, 1.0], [4.0, 7.0]], dtype=np.float64
        )

    def test_mean_averages_every_atom(self) -> None:
        """The default mode is the plain column mean."""
        np.testing.assert_allclose(
            aggregate_atoms(self.per_atom, "mean"), [2.0, 3.0]
        )

    def test_mean_std_max_concatenates_three_blocks(self) -> None:
        """The blocks follow the order named in POOL_STAT_NAMES."""
        vector = aggregate_atoms(self.per_atom, "mean_std_max")
        np.testing.assert_allclose(vector[:2], self.per_atom.mean(axis=0))
        np.testing.assert_allclose(vector[2:4], self.per_atom.std(axis=0))
        np.testing.assert_allclose(vector[4:], self.per_atom.max(axis=0))

    def test_every_mode_is_invariant_to_atom_order(self) -> None:
        """Pooling is a symmetric function, so a shuffle changes nothing."""
        order = np.array([2, 0, 1])
        for mode in FIXED_AGGREGATIONS:
            with self.subTest(mode=mode):
                np.testing.assert_allclose(
                    aggregate_atoms(self.per_atom[order], mode),
                    aggregate_atoms(self.per_atom, mode),
                )

    def test_unknown_mode_is_rejected(self) -> None:
        """A mistyped mode fails rather than silently averaging."""
        with self.assertRaises(ValueError):
            aggregate_atoms(self.per_atom, "nonsense")

    def test_a_one_dimensional_matrix_is_rejected(self) -> None:
        """The atom axis must be present even for a single atom."""
        with self.assertRaises(ValueError):
            aggregate_atoms(self.per_atom[0], "mean")


class TestAtomAggregationWidth(unittest.TestCase):
    """The width `fixed_aggregation` returns, without computing it."""

    def test_width_matches_the_aggregated_vector_for_every_mode(self) -> None:
        """The promised width is the width actually produced."""
        per_atom = np.arange(12.0).reshape(4, 3)
        for mode in FIXED_AGGREGATIONS:
            with self.subTest(mode=mode):
                self.assertEqual(
                    aggregated_width(3, mode),
                    len(aggregate_atoms(per_atom, mode)),
                )

    def test_mean_std_max_triples_the_width(self) -> None:
        """One block per name in POOL_STAT_NAMES."""
        self.assertEqual(
            aggregated_width(64, "mean_std_max"), 64 * len(POOL_STAT_NAMES)
        )

    def test_unknown_mode_is_rejected(self) -> None:
        """The width of an unknown mode is an error, not a guess."""
        with self.assertRaises(ValueError):
            aggregated_width(64, "nonsense")


class TestConformerPooling(unittest.TestCase):
    """Collapsing a per-conformer descriptor matrix to one vector."""

    def setUp(self) -> None:
        """Build three conformers with distinct descriptor rows."""
        self.features = np.array(
            [[0.0, 1.0], [2.0, 3.0], [4.0, 11.0]], dtype=np.float64
        )
        self.ensemble = make_ensemble(n_conformers=3)

    def test_lowest_takes_the_first_row(self) -> None:
        """The ensemble is energy-sorted, so row zero is the minimum."""
        np.testing.assert_allclose(
            pool_conformers(self.features, self.ensemble, "lowest"), [0.0, 1.0]
        )

    def test_boltzmann_is_a_convex_combination(self) -> None:
        """The weighted vector lies inside the range of the rows."""
        vector = pool_conformers(
            self.features, self.ensemble, "boltzmann", 298.15
        )
        self.assertTrue((vector >= self.features.min(axis=0)).all())
        self.assertTrue((vector <= self.features.max(axis=0)).all())

    def test_boltzmann_collapses_onto_the_minimum_when_cold(self) -> None:
        """As the temperature falls the weighting becomes `lowest`."""
        np.testing.assert_allclose(
            pool_conformers(self.features, self.ensemble, "boltzmann", 1.0),
            self.features[0],
            atol=1e-9,
        )

    def test_ensemble_stats_concatenates_four_blocks(self) -> None:
        """The blocks follow the order named in STAT_NAMES."""
        vector = pool_conformers(
            self.features, self.ensemble, "ensemble_stats"
        )
        self.assertEqual(len(vector), 2 * len(STAT_NAMES))
        np.testing.assert_allclose(vector[:2], self.features.min(axis=0))
        np.testing.assert_allclose(vector[2:4], self.features.max(axis=0))
        np.testing.assert_allclose(vector[4:6], self.features.mean(axis=0))
        np.testing.assert_allclose(vector[6:], self.features.std(axis=0))

    def test_a_single_conformer_collapses_every_mode(self) -> None:
        """With one conformer the spread is zero and the row is the answer."""
        features = np.array([[1.0, 2.0]])
        ensemble = make_ensemble(n_conformers=1)
        np.testing.assert_allclose(
            pool_conformers(features, ensemble, "lowest"), [1.0, 2.0]
        )
        np.testing.assert_allclose(
            pool_conformers(features, ensemble, "boltzmann"), [1.0, 2.0]
        )
        np.testing.assert_allclose(
            pool_conformers(features, ensemble, "ensemble_stats"),
            [1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 0.0, 0.0],
        )

    def test_unknown_mode_is_rejected(self) -> None:
        """A mistyped mode fails rather than falling through to the stats."""
        with self.assertRaises(ValueError):
            pool_conformers(self.features, self.ensemble, "nonsense")

    def test_mismatched_conformer_count_is_rejected(self) -> None:
        """Descriptor rows and conformers have to describe one search."""
        with self.assertRaises(ValueError):
            pool_conformers(
                self.features, make_ensemble(n_conformers=2), "lowest"
            )


class TestConformerPoolingWidth(unittest.TestCase):
    """The width `conformers_pooling` returns, without computing it."""

    def test_width_matches_the_pooled_vector_for_every_mode(self) -> None:
        """The promised width is the width actually produced."""
        features = np.arange(12.0).reshape(3, 4)
        ensemble = make_ensemble(n_conformers=3)
        for mode in POOLINGS:
            with self.subTest(mode=mode):
                self.assertEqual(
                    pooled_width(4, mode),
                    len(pool_conformers(features, ensemble, mode)),
                )

    def test_unknown_mode_is_rejected(self) -> None:
        """The width of an unknown mode is an error, not a guess."""
        with self.assertRaises(ValueError):
            pooled_width(64, "nonsense")


if __name__ == "__main__":
    unittest.main()
