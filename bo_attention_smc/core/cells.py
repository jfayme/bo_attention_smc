"""From a dataset of reactions to a cell: what a campaign's GP sees.

A cell is one dataset described by one featuriser under one aggregation,
for example Shields under MACE with attention. It is built in four steps:

1. The reactions are read (`read_reactions`): one row per measured
   reaction, with the SMILES of its molecules, its numeric settings and its
   yield.
2. The unique molecules of each reaction component are described
   (`describe`) by featuriser.py. Morgan fingerprints and the T5 encoders
   read the SMILES; the MLIPs (MACE, AIMNet2) read each molecule's relaxed
   conformers, found by conformer.py and cached on disk
   (`conformer_ensembles`), since the conformer search is by far the
   slowest step.
3. The descriptors are reduced. Under a fixed aggregation (`mean`,
   `mean_std_max`, or Morgan's bits) each molecule is one vector, and each
   component keeps the columns a reduction chooses (`build_fixed`). Under
   `attention` the atoms are kept, on the channels the same reduction
   chooses, for the GP's head to pool (`build_attention`).
4. The cell is saved as an .npz file (`save_cell`, `load_cell`).

A cell does not name its molecules. The per-molecule kernel term needs to
know which reactions share a molecule, and reads it from the dataset
(`molecules`): a cell lists the reactions in the dataset's order.

The reductions, one per name: `decorr0.3`, `decorr0.5` and `decorr0.7`
visit the columns by decreasing variance and keep each one unless it
correlates with a kept one beyond the threshold (`decorrelate`); `pca64`
keeps principal component scores up to 98 % of the variance and at most
64; `none` keeps every column. Only the column-keeping ones can feed a
head, which reads channels.

Numeric settings (Shields' temperature and concentration) enter raw, one
column each: the GP's Normalize is their only scaling.

These are the operations of the study's dataset.py, representation.py,
reduce.py and attention.py, so a cell rebuilt here is the stored one, array
for array, for Morgan and T5. MACE on the GPU does not repeat itself to the
last bits (about 5e-6 between two runs), so neither do its cells.

"""

import hashlib
import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from bo_attention_smc.core.conformer import (
    ConformerEnsemble,
    ConformerGenerator,
)
from bo_attention_smc.core.featuriser import (
    FEATURIZER_SETTINGS,
    MLIPFeaturizer,
    MorganFeaturizer,
    T5Featurizer,
)

logger = logging.getLogger(__name__)

F64 = torch.float64

# The datasets and the conformer cache, in the repository's own folders.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = PROJECT_ROOT / "data"
CONFORMER_CACHE = PROJECT_ROOT / "conformers"

FEATURISERS = ("morgan", "t5_vanilla", "t5_augm", "mace", "aimnet2")
AGGREGATIONS = ("mean", "mean_std_max", "attention")
REDUCTIONS = ("decorr0.3", "decorr0.5", "decorr0.7", "pca64", "none")
POOLINGS = ("lowest", "boltzmann", "ensemble_stats")

# The elements each potential describes: MACE-MH-1 is trained from H to Ac,
# AIMNet2 (wB97M-D3) on H B C N O F Si P S Cl As Se Br I. A component with
# another element (the Shields bases' K and Cs) goes whole to the fallback,
# so that a column means the same thing for every molecule.
COVERAGE = {
    "mace": frozenset(range(1, 90)),
    "aimnet2": frozenset({1, 5, 6, 7, 8, 9, 14, 15, 16, 17, 33, 34, 35, 53}),
}
FALLBACK = {"aimnet2": "mace"}

# A column varying by less than this is dropped before any reduction.
CONSTANT_RANGE = 1e-8


@dataclass(frozen=True)
class Dataset:
    """Where a dataset lives and how its columns are read.

    Attributes
    ----------
    path
        The CSV file, relative to the data folder.
    target
        The column holding the yield, to be maximised.
    compounds
        The columns holding the SMILES of the reaction components.
    numeric
        The columns holding numeric settings.

    """

    path: str
    target: str
    compounds: tuple[str, ...]
    numeric: tuple[str, ...] = ()


DATASETS = {
    "bh_full": Dataset(
        "Buchwald_Hartwig.csv",
        "yield",
        ("Ligand", "Additive", "Base", "Aryl halide"),
    ),
    "bh_1": Dataset(
        "bh_reaction_1.csv",
        "objective",
        ("ligand", "additive", "base", "aryl halide"),
    ),
    "shields": Dataset(
        "shields_dataset.csv",
        "yield",
        ("Solvent_SMILES", "Base_SMILES", "Ligand_SMILES"),
        ("Temp_C", "Concentration"),
    ),
}


@dataclass(frozen=True, eq=False)
class Atoms:
    """The atoms of every molecule of a cell, for an attention head to pool.

    Attributes
    ----------
    values
        Per-atom descriptors of every molecule, standardised, on the
        channels the head reads: shape (M, K, N, C) with a conformer axis
        (MLIPs) or (M, N, C) without (T5).
    mask
        True where an atom is real rather than padding, shape (M, K, N) or
        (M, N).
    weights
        Weight of each conformer, shape (M, K), under `lowest` (the first
        conformer) and `boltzmann`; None otherwise.
    pooling
        How the conformers are collapsed after the atoms: one of POOLINGS,
        or None without a conformer axis.
    index
        The molecule of each component in every reaction, rows of
        `values`, shape (n, n_components).
    columns
        The columns of a pooled molecule each component keeps, one tensor
        per component.
    settings
        The numeric settings of every reaction, shape (n, S); S may be 0.

    """

    values: torch.Tensor
    mask: torch.Tensor
    weights: torch.Tensor | None
    pooling: str | None
    index: torch.Tensor
    columns: tuple[torch.Tensor, ...]
    settings: torch.Tensor


@dataclass(frozen=True, eq=False)
class Cell:
    """One dataset under one embedding, as the GP sees it.

    Attributes
    ----------
    objective
        Yield of every reaction of the pool, shape (n,).
    features
        The mean-pooled, reduced features, shape (n, D): each component's
        columns, then the numeric settings. For fingerprints and fixed
        aggregations they are the GP's inputs; for attention cells they set
        the prior and the input range.
    blocks
        Each component's columns in `features`, one slice per component.
    atoms
        The atoms behind the features, under attention; None for
        fingerprints and fixed aggregations.

    """

    objective: np.ndarray
    features: np.ndarray
    blocks: tuple[slice, ...]
    atoms: Atoms | None = None


@dataclass(frozen=True, eq=False)
class Described:
    """The unique molecules of one component, as a featuriser left them.

    Attributes
    ----------
    smiles
        The molecules, in the order they first appear in the reactions.
    used
        The featuriser that described them, after any fallback.
    features
        Under a fixed aggregation, one row per molecule, a failed molecule
        as a row of NaN; None under attention.
    descriptors
        Under attention, the atoms of each molecule: (K, N, C) from an
        MLIP, (L, C) tokens from T5, None for a failed molecule; empty
        under a fixed aggregation.
    ensembles
        The conformers behind each MLIP descriptor under attention, for
        the conformer weights; empty otherwise.

    """

    smiles: list[str]
    used: str
    features: np.ndarray | None = None
    descriptors: list = field(default_factory=list)
    ensembles: list = field(default_factory=list)


def read_reactions(name: str) -> pd.DataFrame:
    """Read the measured reactions of a dataset.

    Parameters
    ----------
    name
        A key of DATASETS.

    Returns
    -------
    reactions
        The compound and numeric columns and `objective`, one row per
        reaction. A row missing a value is dropped, and a repeated reaction
        keeps its first measurement, without averaging.

    """
    spec = DATASETS[name]
    source = pd.read_csv(DATA_ROOT / spec.path)
    key = list(spec.compounds + spec.numeric)
    target = pd.to_numeric(source[spec.target], errors="coerce")
    reactions = (
        source[key]
        .assign(objective=target)
        .dropna()
        .drop_duplicates(subset=key)
        .astype({column: float for column in spec.numeric})
    )
    return reactions.reset_index(drop=True)


def reagents(dataset: str) -> list[str]:
    """Name a dataset's reagents as records do: `aryl halide`, `ligand`."""
    names = DATASETS[dataset].compounds
    return [name.replace("_SMILES", "").lower() for name in names]


def molecules(dataset: str) -> np.ndarray:
    """Return each reaction's molecule of every reagent of a dataset.

    Parameters
    ----------
    dataset
        A key of DATASETS.

    Returns
    -------
    molecules
        Shape (n, C): for every reaction, each reagent's molecule as an
        integer. They are numbered as an attention cell numbers its bank
        (Atoms.index): the first reagent's molecules from 0 in the order
        they first appear, the next reagent's after them, and so on. Two
        reactions share a molecule exactly when they share its number.

    """
    reactions = read_reactions(dataset)
    columns, offset = [], 0
    for name in DATASETS[dataset].compounds:
        codes = pd.factorize(reactions[name])[0]
        columns.append(offset + codes)
        offset += codes.max() + 1
    return np.column_stack(columns)


def cached(smiles: str) -> Path:
    """Return where a molecule's conformers are cached, by its SMILES."""
    digest = hashlib.sha256(smiles.encode("utf-8")).hexdigest()[:32]
    return CONFORMER_CACHE / f"{digest}.npz"


def conformer_ensembles(
    smiles: list[str], generator: ConformerGenerator
) -> list[ConformerEnsemble | None]:
    """Read each molecule's relaxed conformers from the cache, or find them.

    Parameters
    ----------
    smiles
        The molecules.
    generator
        The conformer search run on a cache miss.

    Returns
    -------
    ensembles
        One per molecule, None where the search failed.

    Notes
    -----
    The cache holds one .npz per molecule, named by a hash of its SMILES.
    The name says nothing of the [conformer] settings: changing them means
    clearing the cache. A failed search is not cached, so it is tried again
    next time.

    """
    ensembles = []
    for s in smiles:
        path = cached(s)
        ensemble = ConformerEnsemble.load(path, s)
        if ensemble is None:
            try:
                ensemble = generator(s)
            except (ValueError, RuntimeError) as error:
                logger.warning("conformer search failed for %r: %s", s, error)
            else:
                ensemble.save(path, s)
        ensembles.append(ensemble)
    return ensembles


def describe(
    reactions: pd.DataFrame,
    dataset: str,
    featuriser: str,
    aggregation: str,
    pooling: str = FEATURIZER_SETTINGS["pooling"],
    device: str | None = None,
) -> dict[str, Described]:
    """Describe the unique molecules of every component of a dataset.

    Parameters
    ----------
    reactions
        The dataset's reactions, from `read_reactions`.
    dataset
        Its name, a key of DATASETS.
    featuriser
        One of FEATURISERS.
    aggregation
        One of AGGREGATIONS; Morgan has no atoms and ignores it.
    pooling
        How an MLIP's conformers are collapsed under a fixed aggregation.
    device
        Torch device of the models; None reads their settings tables.

    Returns
    -------
    described
        One entry per component, in the dataset's order.

    """
    learned = aggregation == "attention"
    described = {}
    if featuriser == "morgan":
        morgan = MorganFeaturizer()
        for name in DATASETS[dataset].compounds:
            smiles = reactions[name].unique().tolist()
            features = np.full((len(smiles), morgan.n_bits), np.nan)
            for row, s in enumerate(smiles):
                try:
                    features[row] = morgan([s])[0]
                except ValueError as error:
                    logger.warning("Morgan failed for %r: %s", s, error)
            described[name] = Described(smiles, "morgan", features=features)
        return described

    if featuriser.startswith("t5_"):
        encoder = T5Featurizer(t5=featuriser[3:], device=device)
        for name in DATASETS[dataset].compounds:
            smiles = reactions[name].unique().tolist()
            values = encoder(smiles, aggregation)
            if learned:
                entry = Described(smiles, featuriser, descriptors=list(values))
            else:
                features = np.asarray(values, dtype=np.float64)
                entry = Described(smiles, featuriser, features=features)
            described[name] = entry
        return described

    mlips = {}
    for mlip in (featuriser, FALLBACK.get(featuriser)):
        if mlip is not None:
            mlips[mlip] = MLIPFeaturizer(
                mlip=mlip,
                device=device,
                aggregation=aggregation,
                pooling=pooling,
            )
    generator = ConformerGenerator()
    for name in DATASETS[dataset].compounds:
        smiles = reactions[name].unique().tolist()
        ensembles = conformer_ensembles(smiles, generator)
        found = [row for row, e in enumerate(ensembles) if e is not None]
        molecules = [e for e in ensembles if e is not None]
        present = {int(z) for e in molecules for z in e.numbers}
        used = featuriser
        if not present <= COVERAGE[featuriser]:
            used = FALLBACK[featuriser]
            logger.warning("%s: %s goes to %s", name, featuriser, used)
        values = mlips[used](molecules)
        if learned:
            descriptors = [None] * len(smiles)
            for row, tensor in zip(found, values, strict=True):
                descriptors[row] = tensor
            described[name] = Described(
                smiles, used, descriptors=descriptors, ensembles=ensembles
            )
        else:
            features = np.full((len(smiles), np.shape(values)[1]), np.nan)
            features[found] = values
            described[name] = Described(smiles, used, features=features)
    return described


def decorrelate(features: np.ndarray, threshold: float) -> np.ndarray:
    """Keep the columns no kept column correlates with beyond a threshold.

    Parameters
    ----------
    features
        One row per molecule, shape (n, d).
    threshold
        The largest absolute correlation a kept column may have.

    Returns
    -------
    kept
        Indices of the kept columns, ascending.

    Notes
    -----
    The columns are visited by decreasing variance (ties in index order),
    so each correlated cluster keeps its most varied member. With two
    molecules every pair of columns is perfectly correlated, and exactly
    one column survives.

    """
    features = np.asarray(features, dtype=np.float64)
    if features.shape[1] < 2:
        return np.arange(features.shape[1])
    correlation = np.abs(np.corrcoef(features, rowvar=False))
    kept: list[int] = []
    for column in np.argsort(-features.var(axis=0), kind="stable"):
        if not kept or correlation[column, kept].max() <= threshold:
            kept.append(int(column))
    return np.sort(np.asarray(kept, dtype=np.int64))


def keep_columns(features: np.ndarray, reduction: str) -> np.ndarray:
    """Find the columns a column-keeping reduction keeps, ascending."""
    if reduction == "none":
        return np.arange(features.shape[1])
    return decorrelate(features, float(reduction.removeprefix("decorr")))


def reduce_columns(features: np.ndarray, reduction: str) -> np.ndarray:
    """Reduce one component's molecules, shape (n, d), by a reduction."""
    if reduction != "pca64":
        return features[:, keep_columns(features, reduction)]
    centred = features - features.mean(axis=0)
    _, singular, axes = np.linalg.svd(centred, full_matrices=False)
    # an SVD fixes each axis up to its sign, which depends on the BLAS: the
    # largest loading is made positive, so every machine agrees
    largest = axes[np.arange(len(axes)), np.abs(axes).argmax(axis=1)]
    axes = axes * np.sign(largest)[:, None]
    explained = np.cumsum(singular**2) / np.sum(singular**2)
    k = int(np.searchsorted(explained, 0.98) + 1)
    k = min(k, 64, features.shape[0] - 1, features.shape[1])
    return centred @ axes[:k].T


def positions(
    reactions: pd.DataFrame, name: str, smiles: list[str]
) -> np.ndarray:
    """Where each reaction's molecule of a component sits in `smiles`."""
    position = {s: row for row, s in enumerate(smiles)}
    return reactions[name].map(position).to_numpy(dtype=np.int64)


def numeric_columns(reactions: pd.DataFrame, dataset: str) -> np.ndarray:
    """Stack the raw numeric settings of every reaction, shape (n, S)."""
    columns = [
        reactions[name].to_numpy(dtype=np.float64)[:, None]
        for name in DATASETS[dataset].numeric
    ]
    return np.hstack(columns) if columns else np.zeros((len(reactions), 0))


def build_fixed(
    reactions: pd.DataFrame,
    dataset: str,
    described: dict[str, Described],
    reduction: str,
) -> Cell:
    """Build a cell of one vector per molecule.

    Parameters
    ----------
    reactions
        The dataset's reactions.
    dataset
        Its name.
    described
        Every component, under a fixed aggregation or Morgan.
    reduction
        One of REDUCTIONS.

    Returns
    -------
    cell
        Each component's reduced columns, then the numeric settings.

    Raises
    ------
    ValueError
        If a molecule failed, or none of a component's columns varies (a
        one-molecule component is no search dimension).

    Notes
    -----
    The study imputed up to 1 % of a component's molecules; with at most
    22 molecules per component, one failure is already more, so a failure
    stops the build, as it does under attention.

    """
    parts, blocks, start = [], [], 0
    for name in DATASETS[dataset].compounds:
        entry = described[name]
        raw = np.array(entry.features, dtype=np.float64)
        if not np.isfinite(raw).all():
            raise ValueError(f"{dataset}/{name}: a molecule failed")
        varying = raw[:, np.ptp(raw, axis=0) > CONSTANT_RANGE]
        if varying.shape[1] == 0:
            raise ValueError(f"{dataset}/{name}: no column varies")
        reduced = reduce_columns(varying, reduction)
        parts.append(reduced[positions(reactions, name, entry.smiles)])
        blocks.append(slice(start, start + reduced.shape[1]))
        start += reduced.shape[1]
    parts.append(numeric_columns(reactions, dataset))
    objective = reactions["objective"].to_numpy(dtype=np.float64)
    return Cell(objective, np.hstack(parts), tuple(blocks))


def collapse_conformers(
    per_conformer: torch.Tensor,
    mask: torch.Tensor,
    weights: torch.Tensor | None,
    pooling: str | None,
) -> torch.Tensor:
    """Collapse the conformers of each molecule, after its atoms.

    Parameters
    ----------
    per_conformer
        One vector per conformer, shape (M, K, C), or (M, C) without a
        conformer axis.
    mask
        True where an atom is real, shape (M, K, N).
    weights
        Weight of each conformer, shape (M, K), for `lowest`, `boltzmann`.
    pooling
        One of POOLINGS, or None without a conformer axis.

    Returns
    -------
    pooled
        Shape (M, C), or (M, 4 C) under `ensemble_stats`: the minimum,
        maximum, mean and standard deviation over the real conformers.

    Notes
    -----
    This is MLIPFeaturizer.conformers_pooling in torch, so that a head's
    gradient passes through it. The variance is floored before its square
    root: one conformer has none, and the root's gradient at zero is NaN.

    """
    if pooling is None:
        return per_conformer
    if pooling != "ensemble_stats":
        return torch.einsum("...k,...kc->...c", weights, per_conformer)
    real = mask.any(dim=-1).unsqueeze(-1)
    counts = real.sum(dim=-2).clamp(min=1)
    mean = (per_conformer * real).sum(dim=-2) / counts
    spread = ((per_conformer - mean.unsqueeze(-2)) * real) ** 2
    variance = spread.sum(dim=-2) / counts
    std = variance.clamp_min(torch.finfo(variance.dtype).tiny).sqrt()
    low = per_conformer.masked_fill(~real, float("inf")).amin(dim=-2)
    high = per_conformer.masked_fill(~real, float("-inf")).amax(dim=-2)
    return torch.cat([low, high, mean, std], dim=-1)


def build_attention(
    reactions: pd.DataFrame,
    dataset: str,
    described: dict[str, Described],
    reduction: str,
    pooling: str = FEATURIZER_SETTINGS["pooling"],
    temperature: float = FEATURIZER_SETTINGS["temperature_k"],
) -> Cell:
    """Build a cell whose molecules the GP's head pools from their atoms.

    Parameters
    ----------
    reactions
        The dataset's reactions.
    dataset
        Its name.
    described
        Every component, under attention, all by one featuriser.
    reduction
        A column-keeping one of REDUCTIONS.
    pooling
        How the conformers are collapsed after the atoms, one of POOLINGS.
    temperature
        Temperature (K) of the `boltzmann` pooling.

    Returns
    -------
    cell
        The bank of every component's molecules, each reaction's molecules
        in it, and the features of mean pooling.

    Raises
    ------
    ValueError
        If the components were described by different featurisers (one
        head cannot score two), a molecule failed, or none of a
        component's columns varies.

    Notes
    -----
    The channels are chosen as the fixed `mean` aggregation's columns are,
    on each component's mean-pooled molecules, so that at mean pooling the
    GP sees the fixed arm's columns. The head reads every channel some
    component keeps, standardised over every atom of the bank, and each
    component then keeps its own columns of the pooled molecules.

    """
    entries = [described[name] for name in DATASETS[dataset].compounds]
    used = sorted({entry.used for entry in entries})
    if len(used) != 1:
        raise ValueError(f"{dataset}: one head cannot score {used}")
    if any(d is None for entry in entries for d in entry.descriptors):
        raise ValueError(f"{dataset}: a molecule failed")
    conformers = bool(entries[0].ensembles)

    kept: list[np.ndarray] = []
    for name, entry in zip(DATASETS[dataset].compounds, entries, strict=True):
        # each molecule as the fixed `mean` aggregation collapses it
        vectors = []
        if conformers:
            for values, ensemble in zip(
                entry.descriptors, entry.ensembles, strict=True
            ):
                means = [
                    MLIPFeaturizer.fixed_aggregation(np.asarray(atoms), "mean")
                    for atoms in values
                ]
                vectors.append(
                    MLIPFeaturizer.conformers_pooling(
                        np.stack(means), ensemble, pooling, temperature
                    )
                )
        else:
            for values in entry.descriptors:
                vectors.append(
                    MLIPFeaturizer.fixed_aggregation(
                        np.asarray(values), "mean"
                    )
                )
        molecules = np.stack(vectors)
        varying = np.flatnonzero(np.ptp(molecules, axis=0) > CONSTANT_RANGE)
        if len(varying) == 0:
            raise ValueError(f"{dataset}/{name}: no column varies")
        kept.append(varying[keep_columns(molecules[:, varying], reduction)])

    # a kept column of a pooled molecule is a (statistic, channel) pair:
    # under ensemble_stats each channel gives 4 columns, otherwise one
    n_channels = np.shape(entries[0].descriptors[0])[-1]
    channels = np.unique(np.concatenate([k % n_channels for k in kept]))
    where = {int(channel): i for i, channel in enumerate(channels)}
    columns = []
    for component in kept:
        pooled_columns = [
            (c // n_channels) * len(channels) + where[int(c % n_channels)]
            for c in component
        ]
        columns.append(torch.tensor(pooled_columns, dtype=torch.long))

    # one bank over every component's molecules, in component order
    descriptors = [np.asarray(d) for e in entries for d in e.descriptors]
    ensembles = [x for e in entries for x in e.ensembles] if conformers else []
    if pooling == "lowest" and conformers:
        # only the lowest conformer reaches the kernel: the bank keeps it alone
        descriptors = [d[:1] for d in descriptors]
    flat = [d.reshape(-1, n_channels) for d in descriptors]
    atoms = np.concatenate([f.astype(np.float64)[:, channels] for f in flat])
    mean, scale = atoms.mean(axis=0), atoms.std(axis=0)
    scale = np.where(scale > 0.0, scale, 1.0)
    shape = np.max([d.shape[:-1] for d in descriptors], axis=0).tolist()
    values = torch.zeros((len(descriptors), *shape, len(channels)), dtype=F64)
    mask = torch.zeros((len(descriptors), *shape), dtype=torch.bool)
    for row, (d, f) in enumerate(zip(descriptors, flat, strict=True)):
        standard = (
            np.asarray(f, dtype=np.float64)[:, channels] - mean
        ) / scale
        region = (row, *(slice(0, size) for size in d.shape[:-1]))
        values[region] = torch.as_tensor(
            standard.reshape(*d.shape[:-1], -1), dtype=F64
        )
        mask[region] = True

    # without a conformer axis there is nothing to collapse
    collapse = pooling if conformers else None
    weights = None
    if conformers and pooling != "ensemble_stats":
        weights = torch.zeros(mask.shape[:2], dtype=F64)
        for row, ensemble in enumerate(ensembles):
            if pooling == "lowest":
                weights[row, 0] = 1.0  # the ensemble is energy-sorted
            else:
                weights[row, : len(ensemble)] = torch.as_tensor(
                    ensemble.boltzmann_weights(temperature), dtype=F64
                )

    # the features of mean pooling: each molecule's mean atom per conformer
    real = mask.unsqueeze(-1)
    counts = real.sum(dim=-2).clamp(min=1)
    pooled = (values * real).sum(dim=-2) / counts
    pooled = collapse_conformers(pooled, mask, weights, collapse).numpy()

    index, parts, blocks, start, offset = [], [], [], 0, 0
    for name, entry, own in zip(
        DATASETS[dataset].compounds, entries, columns, strict=True
    ):
        rows = offset + positions(reactions, name, entry.smiles)
        index.append(rows)
        parts.append(pooled[rows][:, own.numpy()])
        blocks.append(slice(start, start + len(own)))
        start += len(own)
        offset += len(entry.smiles)
    settings = numeric_columns(reactions, dataset)
    atoms = Atoms(
        values=values,
        mask=mask,
        weights=weights,
        pooling=collapse,
        index=torch.as_tensor(np.column_stack(index)),
        columns=tuple(columns),
        settings=torch.as_tensor(settings, dtype=F64),
    )
    return Cell(
        objective=reactions["objective"].to_numpy(dtype=np.float64),
        features=np.hstack([*parts, settings]),
        blocks=tuple(blocks),
        atoms=atoms,
    )


def pool_atoms(scores: torch.Tensor, atoms: Atoms) -> torch.Tensor:
    """Average each molecule's atoms by the softmax of their scores.

    Parameters
    ----------
    scores
        Score of every atom, shaped as the atoms' mask.
    atoms
        A cell's atoms.

    Returns
    -------
    pooled
        One vector per molecule, shape (M, C), its conformers collapsed.

    Notes
    -----
    Equal scores give each molecule's mean atom: mean pooling, the null. A
    head's scores weight the atoms instead; posterior.py pools them by the
    same operations, for every particle at once. These are the operations
    of the study's attention head (attention.AttentionPool), so the pooled
    vectors are its own to the last bit.

    """
    # a padded conformer has no atoms, and the softmax of -inf alone is NaN:
    # it is scored as if real, and the mask discards it
    empty = ~atoms.mask.any(dim=-1, keepdim=True)
    scores = scores.masked_fill(~(atoms.mask | empty), float("-inf"))
    attention = torch.softmax(scores, dim=-1) * atoms.mask
    pooled = torch.einsum("...n,...nc->...c", attention, atoms.values)
    return collapse_conformers(
        pooled, atoms.mask, atoms.weights, atoms.pooling
    )


def null_features(cell: Cell) -> torch.Tensor:
    """Mean-pool every reaction's molecules: the null GP's inputs.

    Parameters
    ----------
    cell
        The cell.

    Returns
    -------
    x
        Shape (n, D): each component's kept channels of its molecule's
        mean atom, then the numeric settings; for a cell without atoms, its
        features as they are.

    Notes
    -----
    These are the stored `features` again, but pooled by the head's
    operations (`pool_atoms`, equal scores) rather than by a plain mean:
    the two agree to about 1e-16, and the study's GP saw these. The stored
    features still set the prior and the input range, as they did.

    """
    atoms = cell.atoms
    if atoms is None:
        return torch.tensor(cell.features, dtype=F64)
    equal = torch.zeros(atoms.mask.shape, dtype=F64)
    pooled = pool_atoms(equal, atoms)
    parts = []
    for component, kept in enumerate(atoms.columns):
        parts.append(pooled[atoms.index[:, component]][:, kept])
    parts.append(atoms.settings)
    return torch.cat(parts, dim=-1)


def save_cell(cell: Cell, path: Path) -> None:
    """Write a cell to an .npz file, every array as it is."""
    arrays = {
        "objective": cell.objective,
        "features": cell.features,
        "blocks": np.array([[b.start, b.stop] for b in cell.blocks]),
    }
    atoms = cell.atoms
    if atoms is not None:
        arrays.update(
            values=atoms.values.numpy(),
            mask=atoms.mask.numpy(),
            pooling=np.array(atoms.pooling or ""),
            index=atoms.index.numpy(),
            columns=np.concatenate([c.numpy() for c in atoms.columns]),
            widths=np.array([len(c) for c in atoms.columns]),
            settings=atoms.settings.numpy(),
        )
        if atoms.weights is not None:
            arrays["weights"] = atoms.weights.numpy()
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)


def load_cell(path: str | Path) -> Cell:
    """Read a cell written by `save_cell`."""
    with np.load(path) as saved:
        blocks = tuple(slice(int(a), int(b)) for a, b in saved["blocks"])
        cell = Cell(saved["objective"], saved["features"], blocks)
        if "values" not in saved:
            return cell
        splits = np.cumsum(saved["widths"])[:-1]
        weights = None
        if "weights" in saved:
            weights = torch.from_numpy(saved["weights"])
        atoms = Atoms(
            values=torch.from_numpy(saved["values"]),
            mask=torch.from_numpy(saved["mask"]),
            weights=weights,
            pooling=str(saved["pooling"]) or None,
            index=torch.from_numpy(saved["index"]),
            columns=tuple(
                torch.from_numpy(c) for c in np.split(saved["columns"], splits)
            ),
            settings=torch.from_numpy(saved["settings"]),
        )
        return Cell(cell.objective, cell.features, blocks, atoms)
