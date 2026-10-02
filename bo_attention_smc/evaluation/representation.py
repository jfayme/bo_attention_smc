r"""Whether the chemistry is in a molecule's description, before any GP.

For the dominant reagent (the Shields ligands, the bh_full aryl halides),
each molecule's mean yield over the pool is predicted from the other
molecules alone (leave one out), under several descriptions of a molecule:

- `pool_kept`: the mean over its atoms of the channels the study keeps
  (decorrelation 0.7: what the GP sees under mean pooling);
- `pool_full`: the mean over its atoms of all of MACE's channels;
- `key_kept`, `key_full`: the key atom alone (the ligand's phosphorus, the
  aryl halide's halogen), kept or all channels: what a perfect attention
  head would give;
- `shell_full`: the key atom and its bonded neighbours, all channels;
- `heavy_atoms`: the count of heavy atoms, a crude size proxy;
- `random`: Gaussian columns as wide as `pool_full` (20 draws): the floor.

Two predictors, on columns standardised over the molecules: k nearest
neighbours (k = 3, inverse distance) and a small GP (scikit-learn: RBF and
white noise). Each is scored by Spearman's rho between the leave-one-out
predictions and the true means; and with no fitting, the Pearson
correlation of the feature distance with the yield gap over pairs of
molecules.

It reads the study's MACE attention cells and the same cells kept on every
channel (built by prepare.py with --reduction none), and the conformers.

    python -m bo_attention_smc.evaluation.representation \
        --full cells/full

"""

import argparse
import warnings
from pathlib import Path

import numpy as np
from rdkit import Chem
from scipy.spatial.distance import pdist
from scipy.stats import pearsonr, spearmanr
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel

from bo_attention_smc.core.cells import load_cell, molecules, reagents
from bo_attention_smc.evaluation.chemistry import (
    KEY_ATOMS,
    atom_labels,
    conformers,
    named_molecules,
)
from bo_attention_smc.evaluation.predict import DOMINANT
from bo_attention_smc.runs.run import cell_path

# the datasets whose dominant reagent (predict.DOMINANT) is examined
DATASETS = ("shields", "bh_full")
# atoms closer than this (angstrom) to the key atom are its neighbours
BOND = 2.0


def standardise(x: np.ndarray) -> np.ndarray:
    """Standardise each column over the molecules; drop constant ones."""
    sd = x.std(axis=0)
    keep = sd > 1e-9
    return (x[:, keep] - x[:, keep].mean(axis=0)) / sd[keep]


def leave_one_out(x: np.ndarray, y: np.ndarray, how: str) -> np.ndarray:
    """Predict each molecule's yield from the others, by `knn` or `gp`."""
    predicted = np.empty_like(y)
    for i in range(len(y)):
        train = np.arange(len(y)) != i
        xt, xi = x[train], x[i : i + 1]
        if how == "knn":
            d = np.linalg.norm(xt - xi, axis=1) + 1e-9
            near = np.argsort(d)[:3]
            predicted[i] = np.sum(y[train][near] / d[near]) / np.sum(
                1 / d[near]
            )
        else:
            scale = np.median(pdist(xt)) or 1.0
            kernel = ConstantKernel(1.0) * RBF(scale) + WhiteKernel(0.1)
            gp = GaussianProcessRegressor(
                kernel,
                normalize_y=True,
                n_restarts_optimizer=2,
                random_state=0,
            )
            gp.fit(xt, y[train])
            predicted[i] = gp.predict(xi)[0]
    return predicted


def score(x: np.ndarray, y: np.ndarray) -> dict:
    """Score one description: both predictors and the distance test."""
    z = standardise(x)
    return {
        "knn": spearmanr(leave_one_out(z, y, "knn"), y)[0],
        "gp": spearmanr(leave_one_out(z, y, "gp"), y)[0],
        "dist~gap": pearsonr(pdist(z), pdist(y[:, None]))[0],
    }


def describe(dataset: str, full_cells: Path) -> tuple[dict, np.ndarray]:
    """Every description of the dominant reagent's molecules, and yields.

    Returns
    -------
    descriptions, y
        Each description, one row per molecule, and each molecule's mean
        yield over the pool.

    """
    reagent = DOMINANT[dataset]
    key = KEY_ATOMS[reagent]
    small = load_cell(cell_path(dataset, "mace"))
    full = load_cell(full_cells / f"{dataset}__mace__attention.npz")
    atoms, every = small.atoms, full.atoms
    if atoms is None or every is None:
        raise ValueError("both cells must be attention cells")
    c = reagents(dataset).index(reagent)
    kept = atoms.columns[c].numpy()
    numbers = molecules(dataset)[:, c]
    found: dict[str, list] = {
        name: []
        for name in (
            "pool_kept",
            "pool_full",
            "key_kept",
            "key_full",
            "shell_full",
            "heavy_atoms",
        )
    }
    y = []
    # both cells number the molecules as the dataset does
    for m, smiles in sorted(named_molecules(dataset)[c].items()):
        n = int(atoms.mask[m, 0].sum())
        kept_values = atoms.values[m, 0, :n].numpy()
        all_values = every.values[m, 0, :n].numpy()
        labels = atom_labels(smiles, "mace")
        coords = conformers(smiles).coords[0]
        k = [i for i, label in enumerate(labels) if label in key][0]
        distance = np.linalg.norm(coords - coords[k], axis=1)
        shell = [k, *np.flatnonzero((distance > 0) & (distance < BOND))]
        found["pool_kept"].append(kept_values[:, kept].mean(axis=0))
        found["pool_full"].append(all_values.mean(axis=0))
        found["key_kept"].append(kept_values[k, kept])
        found["key_full"].append(all_values[k])
        found["shell_full"].append(all_values[shell].mean(axis=0))
        heavy = Chem.MolFromSmiles(smiles).GetNumHeavyAtoms()
        found["heavy_atoms"].append([heavy])
        y.append(small.objective[numbers == m].mean())
    return {k: np.array(v, dtype=float) for k, v in found.items()}, np.array(y)


def main(argv: list[str] | None = None) -> None:
    """Print every description's scores, for both datasets."""
    parser = argparse.ArgumentParser(
        prog="python -m bo_attention_smc.evaluation.representation",
        description=__doc__.split("\n")[0],
    )
    parser.add_argument(
        "--full",
        type=Path,
        default=Path("cells/full"),
        help="the MACE attention cells on every channel",
    )
    args = parser.parse_args(argv)
    # the small GP's optimiser warns on near-flat likelihoods, as expected
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    for dataset in DATASETS:
        found, y = describe(dataset, args.full)
        print(
            f"== {dataset}: {len(y)} {DOMINANT[dataset]}s, leave-one-out "
            "Spearman (kNN, GP) and distance~yield-gap r"
        )
        for name, x in found.items():
            s = score(x, y)
            print(
                f"   {name:12s} dim {x.shape[1]:5d}  kNN {s['knn']:+.2f}  "
                f"GP {s['gp']:+.2f}  dist~gap {s['dist~gap']:+.2f}"
            )
        rng = np.random.default_rng(0)
        width = found["pool_full"].shape[1]
        floor = [score(rng.normal(size=(len(y), width)), y) for _ in range(20)]
        print(
            f"   {'random':12s} dim {width:5d}  kNN "
            f"{np.mean([r['knn'] for r in floor]):+.2f}  GP "
            f"{np.mean([r['gp'] for r in floor]):+.2f}  dist~gap "
            f"{np.mean([r['dist~gap'] for r in floor]):+.2f}  (mean of 20)"
        )
        size = found["heavy_atoms"][:, 0]
        print(
            "   heavy-atom count vs yield, Spearman "
            f"{spearmanr(size, y)[0]:+.2f}"
        )


if __name__ == "__main__":
    main()
