"""The posterior that SMC samples: the GP's settings and an attention head.

A model with a head describes each molecule by a weighted average of its
atoms (MACE) or tokens (T5) instead of their plain mean. The head scores
every atom as s = w . tanh(V h + b), and the molecule is the softmax-weighted
average of its atoms, on every channel, its conformers then collapsed as
the cell says. The atom's inputs h are

- `pca8`: its 8 principal components over the bank's atoms, whitened, with
  4 hidden units: 40 weights whatever the descriptor width;
- `all`: every channel, with as many hidden units (at most 4) as keep the
  head to 10 weights per molecule.

The span constraint. A free head can move a molecule anywhere in the hull
of its atoms, which stretches distances several-fold and turns the head
into a length-scale corrector. So each pooled column is mapped linearly
onto the range the mean-pooled features cover, over the molecules its
reagent uses: the head can reorder molecules within a column but not
rescale it, and the length-scale prior keeps its meaning.

The parameters are one flat vector per particle,

    theta = [raw length-scales (D), u, mean, V, b, w, raw weights (C)],

with length-scale = softplus(raw), noise = floor + softplus(u) and term
weight = softplus(raw weight), as GPyTorch parameterises them. The log
posterior is the Matern-5/2 marginal likelihood of the standardised yields
(plus the per-molecule term, if any), the length-scale, noise and weight
priors of gp.py, N(0, head_scale^2) on every head weight, and the log
Jacobians of the softplus maps.

Every function takes a batch of theta, shape (P, dimension), and returns
one value per particle, so a whole population goes through the GPU at once;
torch's autograd gives the gradients. tests/test_smc.py checks the log
posterior, its gradient and the predictions against a plain one-particle
version (tests/reference.py), particle by particle. Three things differ
from that version, for speed, and none changes a value:

- The measured reactions are padded to a fixed number, the campaign's
  last, with a mask. A padded row is decoupled (unit variance, no noise, a
  zero residual), so it adds nothing to the likelihood.
- Kernels are computed from inner products instead of pairwise
  differences, which would need a (P, n, n, D) array: in float64 on a GPU
  that array is the cost of every gradient. The two agree to rounding.
- Predictions cover the whole pool at once.

SMC campaigns were recorded with these operations in this order; changing
either changes the last bits and, through resampling, the choices.

"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from botorch.models import SingleTaskGP
from torch.nn.functional import logsigmoid, softplus

from bo_attention_smc.core.cells import Cell
from bo_attention_smc.core.gp import (
    MIN_VARIANCE,
    NOISE_LOC,
    NOISE_SCALE,
    TERM_CONCENTRATION,
    TERM_RATE,
    Prior,
)

F64 = torch.float64
SQRT5 = math.sqrt(5.0)
TINY = torch.finfo(F64).tiny
LOG_2PI = math.log(2.0 * math.pi)

HEADS = ("pca8", "all")

# Hidden units of the head; the `all` head has fewer where the bank has too
# few molecules for them: at most 10 weights per molecule.
HIDDEN = 4
MAX_WEIGHTS_PER_MOLECULE = 10.0

# Principal components the `pca8` head scores atoms on.
N_COMPONENTS = 8


@dataclass(frozen=True, eq=False)
class Component:
    """How one reagent's molecules enter the kernel.

    Attributes
    ----------
    molecules
        The reagent's rows of the bank.
    kept
        The pooled channels the reagent keeps.
    low, high
        The mean-pooled features' range over those channels: the span map
        puts each pooled column onto it.
    where
        For every reaction of the pool, its molecule's position among the
        reagent's molecules.

    """

    molecules: torch.Tensor
    kept: torch.Tensor
    low: torch.Tensor
    high: torch.Tensor
    where: torch.Tensor


@dataclass(frozen=True, eq=False)
class Bank:
    """What a campaign's posteriors share: the atoms, the pool, the layout.

    Attributes
    ----------
    inputs
        What the head scores for every atom: its 8 whitened principal
        components (`pca8`) or its channels (`all`); (M, K, N, I), or
        (M, N, I) without conformers.
    values
        Every atom's channels, what the head pools.
    mask
        True where an atom is real.
    weights
        Conformer weights (M, K), or None without conformers.
    components
        One Component per reagent.
    settings
        Every reaction's numeric settings, (n, S).
    offset, coefficient
        Normalize's map of the features onto the unit box.
    molecules
        Every reaction's molecule of each reagent, (n, C), for the
        per-molecule term; None without it.
    n_hidden
        The head's hidden units.
    head_scale
        The prior sd of every head weight.

    """

    inputs: torch.Tensor
    values: torch.Tensor
    mask: torch.Tensor
    weights: torch.Tensor | None
    components: tuple[Component, ...]
    settings: torch.Tensor
    offset: torch.Tensor
    coefficient: torch.Tensor
    molecules: torch.Tensor | None
    n_hidden: int
    head_scale: float

    @classmethod
    def of(
        cls: type[Bank],
        cell: Cell,
        bounds: torch.Tensor,
        head: str,
        molecules: torch.Tensor | None = None,
        head_scale: float = 1.0,
        device: str | torch.device = "cpu",
    ) -> Bank:
        """Build the bank of a cell for one head.

        Parameters
        ----------
        cell
            A cell with atoms.
        bounds
            The range of the cell's features, shape (2, D).
        head
            One of HEADS.
        molecules
            Each reaction's molecule of every reagent, (n, C), to carry the
            per-molecule term; None without it.
        head_scale
            The prior sd of every head weight.
        device
            Where the tensors live.

        Raises
        ------
        ValueError
            If the cell has no atoms (a fingerprint, a fixed aggregation),
            or collapses its conformers by `ensemble_stats`, which is not
            ported.

        """
        atoms = cell.atoms
        if atoms is None:
            raise ValueError("a cell without atoms has no head to sample")
        if atoms.pooling not in (None, "lowest", "boltzmann"):
            raise ValueError(f"conformer pooling {atoms.pooling!r} not ported")
        inputs, n_hidden = atoms.values, HIDDEN
        if head == "pca8":
            real = atoms.values[atoms.mask]
            centre = real.mean(dim=0)
            _, singular, vt = torch.linalg.svd(
                real - centre, full_matrices=False
            )
            sd = singular[:N_COMPONENTS] / math.sqrt(len(real) - 1)
            # whitened: unit variance and no correlation over the bank's atoms
            inputs = (atoms.values - centre) @ (vt[:N_COMPONENTS].T / sd)
        else:
            # V and b hold n_hidden * (n_inputs + 1) weights, w n_hidden
            n_inputs, n_molecules = inputs.shape[-1], len(atoms.values)
            while (
                n_hidden * (n_inputs + 2) / n_molecules
                > MAX_WEIGHTS_PER_MOLECULE
            ):
                n_hidden -= 1
        components = []
        for c, block in enumerate(cell.blocks):
            own = torch.unique(atoms.index[:, c])
            # where each of the reagent's molecules sits among them
            position = torch.full((len(atoms.values),), -1)
            position[own] = torch.arange(len(own))
            components.append(
                Component(
                    own.to(device),
                    atoms.columns[c].to(device),
                    bounds[0, block].to(device, F64),
                    bounds[1, block].to(device, F64),
                    position[atoms.index[:, c]].to(device),
                )
            )
        weights = None
        if atoms.pooling is not None and atoms.weights is not None:
            weights = atoms.weights.to(device, F64)
        return cls(
            inputs.to(device, F64),
            atoms.values.to(device, F64),
            atoms.mask.to(device),
            weights,
            tuple(components),
            atoms.settings.to(device, F64),
            bounds[0].to(device, F64),
            (bounds[1] - bounds[0]).to(device, F64),
            None if molecules is None else molecules.to(device, F64),
            n_hidden,
            head_scale,
        )

    @property
    def n_lengthscales(self) -> int:
        """Length-scales: one per feature column."""
        return len(self.offset)

    @property
    def n_inputs(self) -> int:
        """What the head scores per atom."""
        return self.inputs.shape[-1]

    @property
    def n_head(self) -> int:
        """Weights of the head: V, b and w."""
        return self.n_hidden * (self.n_inputs + 2)

    @property
    def n_reagents(self) -> int:
        """Weights of the term: one per reagent, 0 without the term."""
        return 0 if self.molecules is None else self.molecules.shape[1]

    @property
    def dimension(self) -> int:
        """Length of theta."""
        return self.n_lengthscales + 2 + self.n_head + self.n_reagents

    @property
    def n_pool(self) -> int:
        """Reactions in the pool."""
        return self.settings.shape[0]

    @property
    def device(self) -> torch.device:
        """Where the tensors live."""
        return self.values.device


@dataclass(frozen=True, eq=False)
class Data:
    """One step's measured reactions, padded to a fixed number.

    Attributes
    ----------
    rows
        The measured reactions, then zeros, shape (n_max,).
    mask
        1.0 for a measured reaction, 0.0 for padding.
    y
        Their yields standardised by their own mean and sd, as the GP holds
        them; zero for padding.
    y_mean, y_std
        That standardisation.
    prior
        The length-scale prior.
    floor
        The noise floor, as GPyTorch holds it (in float32).
    n
        The measured reactions' number.

    """

    rows: torch.Tensor
    mask: torch.Tensor
    y: torch.Tensor
    y_mean: float
    y_std: float
    prior: Prior
    floor: float
    n: int

    @classmethod
    def of(
        cls: type[Data],
        gp: SingleTaskGP,
        rows: list[int],
        prior: Prior,
        n_max: int,
        device: str | torch.device = "cpu",
    ) -> Data:
        """Read one step's data from the GP built on the measured rows."""
        n = len(rows)
        if n > n_max:
            raise ValueError(f"{n} measured reactions, room for {n_max}")
        pad = n_max - n
        measured = torch.as_tensor(rows)
        y = gp.train_targets.detach()
        noise_covar = gp.likelihood.noise_covar
        return cls(
            torch.cat([measured, measured.new_zeros(pad)]).to(device),
            torch.cat([torch.ones(n), torch.zeros(pad)]).to(device, F64),
            torch.cat([y, y.new_zeros(pad)]).to(device, F64),
            float(gp.outcome_transform.means.detach().squeeze()),
            float(gp.outcome_transform.stdvs.detach().squeeze()),
            prior,
            float(noise_covar.raw_noise_constraint.lower_bound),
            n,
        )


@dataclass(frozen=True, eq=False)
class Parts:
    """theta, named, one row per particle."""

    raw_lengthscale: torch.Tensor
    lengthscale: torch.Tensor
    u: torch.Tensor
    noise: torch.Tensor
    constant: torch.Tensor
    V: torch.Tensor  # noqa: N815 -- the head's matrix, as the docs name it
    b: torch.Tensor
    w: torch.Tensor
    head: torch.Tensor
    raw_weights: torch.Tensor
    weights: torch.Tensor


def unpack(theta: torch.Tensor, bank: Bank, floor: float) -> Parts:
    """Name the parts of a batch of theta, shape (P, dimension)."""
    d, h, i = bank.n_lengthscales, bank.n_hidden, bank.n_inputs
    raw = theta[:, :d]
    u = theta[:, d]
    start = d + 2
    head = theta[:, start : start + bank.n_head]
    raw_weights = theta[:, start + bank.n_head :]
    return Parts(
        raw,
        softplus(raw),
        u,
        floor + softplus(u),
        theta[:, d + 1],
        head[:, : h * i].reshape(-1, h, i),
        head[:, h * i : h * i + h],
        head[:, h * i + h :],
        head,
        raw_weights,
        softplus(raw_weights),
    )


def pool_bank(p: Parts, bank: Bank) -> torch.Tensor:
    """Pool every molecule of the bank, per particle: (P, M, C)."""
    hidden = torch.einsum("...i,phi->p...h", bank.inputs, p.V)
    shape = (p.b.shape[0],) + (1,) * (bank.inputs.dim() - 1) + (-1,)
    hidden = torch.tanh(hidden + p.b.reshape(shape))
    scores = torch.einsum("p...h,ph->p...", hidden, p.w)
    # a padded conformer has no atoms: it is scored as if real, and the
    # mask discards it (cells.pool_atoms)
    empty = ~bank.mask.any(dim=-1, keepdim=True)
    scores = scores.masked_fill(~(bank.mask | empty), float("-inf"))
    attention = torch.softmax(scores, dim=-1) * bank.mask
    pooled = torch.einsum("p...n,...nc->p...c", attention, bank.values)
    if bank.weights is not None:
        pooled = torch.einsum("mk,pmkc->pmc", bank.weights, pooled)
    return pooled


def features(
    p: Parts, bank: Bank, rows: torch.Tensor, pooled: torch.Tensor
) -> torch.Tensor:
    """Return the kernel's inputs of some reactions: (P, n, D).

    Each reagent's kept channels of its pooled molecule, mapped onto the
    mean-pooled features' span, then the numeric settings.
    """
    parts = []
    for component in bank.components:
        own = pooled[:, component.molecules][:, :, component.kept]
        lowest = own.amin(dim=1, keepdim=True)
        span = (own.amax(dim=1, keepdim=True) - lowest).clamp_min(TINY)
        own = component.low + (own - lowest) * (
            (component.high - component.low) / span
        )
        parts.append(own[:, component.where[rows]])
    settings = bank.settings[rows]
    parts.append(settings.expand(len(pooled), *settings.shape))
    x = torch.cat(parts, dim=-1)
    return (x - bank.offset) / bank.coefficient


def matern52(z: torch.Tensor, ls: torch.Tensor) -> torch.Tensor:
    """Matern-5/2 correlation of a set with itself, per particle: (P, n, n).

    From inner products, without the (P, n, n, D) array of differences.
    The diagonal is set to distance zero exactly, where the inner products
    would leave rounding.
    """
    a = z / ls[:, None, :]
    squares = (a**2).sum(-1)
    squared = (
        squares.unsqueeze(-1)
        + squares.unsqueeze(-2)
        - 2.0 * a @ a.transpose(-1, -2)
    )
    diagonal = torch.eye(z.shape[1], dtype=torch.bool, device=z.device)
    squared = squared.masked_fill(diagonal, 0.0)
    r = torch.sqrt(squared.clamp_min(1e-36))
    return (1.0 + SQRT5 * r + 5.0 / 3.0 * r**2) * torch.exp(-SQRT5 * r)


def matern52_inner(
    z1: torch.Tensor, z2: torch.Tensor, ls: torch.Tensor
) -> torch.Tensor:
    """Matern-5/2 correlation between two sets, per particle."""
    a, b = z1 / ls[:, None, :], z2 / ls[:, None, :]
    squared = (
        (a**2).sum(-1).unsqueeze(-1)
        + (b**2).sum(-1).unsqueeze(-2)
        - 2.0 * a @ b.transpose(-1, -2)
    )
    r = torch.sqrt(squared.clamp_min(1e-36))
    return (1.0 + SQRT5 * r + 5.0 / 3.0 * r**2) * torch.exp(-SQRT5 * r)


def same_molecule(
    bank: Bank,
    rows_a: torch.Tensor,
    rows_b: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    """Return the per-molecule term between two sets: (P, na, nb)."""
    if bank.molecules is None:
        raise ValueError("the bank carries no molecules for the term")
    a, b = bank.molecules[rows_a], bank.molecules[rows_b]
    same = (a.T.unsqueeze(-1) == b.T.unsqueeze(-2)).to(F64)
    return torch.einsum("pc,cij->pij", weights, same)


def train_covariance(
    p: Parts, bank: Bank, data: Data, z: torch.Tensor
) -> torch.Tensor:
    """Return the measured reactions' covariance, padded, per particle."""
    covariance = matern52(z, p.lengthscale)
    if bank.n_reagents:
        covariance = covariance + same_molecule(
            bank, data.rows, data.rows, p.weights
        )
    m = data.mask
    return covariance * (m[:, None] * m[None, :]) + torch.diag_embed(
        (1.0 - m) + p.noise[:, None] * m
    )


def train_factor(
    p: Parts, bank: Bank, data: Data, z: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Cholesky factors of the measured reactions' covariance, padded.

    Returns
    -------
    chol, failed
        One factor per particle, and where it failed (not positive
        definite); a failed particle's factor is garbage.

    """
    covariance = train_covariance(p, bank, data, z)
    chol, info = torch.linalg.cholesky_ex(covariance)
    return chol, info != 0


def log_likelihood(
    theta: torch.Tensor, bank: Bank, data: Data
) -> torch.Tensor:
    """Log marginal likelihood per particle, (P,); -inf where it fails."""
    p = unpack(theta, bank, data.floor)
    z = features(p, bank, data.rows, pool_bank(p, bank))
    chol, failed = train_factor(p, bank, data, z)
    residual = (data.y - p.constant[:, None]) * data.mask
    alpha = torch.cholesky_solve(residual.unsqueeze(-1), chol).squeeze(-1)
    value = (
        -0.5 * (residual * alpha).sum(-1)
        - torch.log(torch.diagonal(chol, dim1=-2, dim2=-1)).sum(-1)
        - 0.5 * data.n * LOG_2PI
    )
    return torch.where(failed | ~torch.isfinite(value), -torch.inf, value)


def gamma_log_prob(
    x: torch.Tensor, concentration: float, rate: float
) -> torch.Tensor:
    """Log Gamma(x; concentration, rate)."""
    return (
        concentration * math.log(rate)
        + (concentration - 1.0) * torch.log(x)
        - rate * x
        - math.lgamma(concentration)
    )


def lengthscale_log_prob(x: torch.Tensor, prior: Prior) -> torch.Tensor:
    """Log prior of the length-scales: the Gamma, or the LogNormal."""
    if prior.lognormal is None:
        return gamma_log_prob(x, prior.concentration, prior.rate)
    loc, scale = prior.lognormal
    log_x = torch.log(x)
    return (
        -log_x
        - math.log(scale)
        - 0.5 * LOG_2PI
        - (log_x - loc) ** 2 / (2 * scale**2)
    )


def log_prior(theta: torch.Tensor, bank: Bank, data: Data) -> torch.Tensor:
    """Sum the priors and the softplus maps' log Jacobians, per particle."""
    p = unpack(theta, bank, data.floor)
    scale = lengthscale_log_prob(p.lengthscale, data.prior)
    log_noise = torch.log(p.noise)
    noise = (
        -log_noise
        - math.log(NOISE_SCALE)
        - 0.5 * LOG_2PI
        - (log_noise - NOISE_LOC) ** 2 / (2 * NOISE_SCALE**2)
    )
    head = (
        -0.5 * (p.head / bank.head_scale) ** 2
        - math.log(bank.head_scale)
        - 0.5 * LOG_2PI
    )
    jacobian = logsigmoid(p.raw_lengthscale).sum(-1) + logsigmoid(p.u)
    value = scale.sum(-1) + noise + head.sum(-1) + jacobian
    if bank.n_reagents:
        value = value + (
            gamma_log_prob(p.weights, TERM_CONCENTRATION, TERM_RATE).sum(-1)
            + logsigmoid(p.raw_weights).sum(-1)
        )
    return value


def log_posterior(theta: torch.Tensor, bank: Bank, data: Data) -> torch.Tensor:
    """Log posterior density per particle, up to a constant."""
    return log_likelihood(theta, bank, data) + log_prior(theta, bank, data)


@torch.no_grad()
def predict(
    theta: torch.Tensor, bank: Bank, data: Data, batch: int = 16
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Predict every reaction of the pool under every particle.

    Parameters
    ----------
    theta
        The particles, (P, dimension).
    bank, data
        The campaign's bank and this step's data.
    batch
        Particles predicted at once: the whole pool under many particles
        does not fit on the GPU.

    Returns
    -------
    mean, variance, noise
        Latent mean and variance (P, n_pool), noise variance (P,), in the
        yields' units.

    """
    means, variances, noises = [], [], []
    everything = torch.arange(bank.n_pool, device=bank.device)
    for chunk in theta.split(batch):
        p = unpack(chunk, bank, data.floor)
        pooled = pool_bank(p, bank)
        z = features(p, bank, data.rows, pooled)
        zx = features(p, bank, everything, pooled)
        chol, _ = train_factor(p, bank, data, z)
        cross = matern52_inner(zx, z, p.lengthscale)
        prior_variance = torch.ones_like(p.constant)
        if bank.n_reagents:
            cross = cross + same_molecule(
                bank, everything, data.rows, p.weights
            )
            prior_variance = prior_variance + p.weights.sum(-1)
        cross = cross * data.mask
        residual = (data.y - p.constant[:, None]) * data.mask
        alpha = torch.cholesky_solve(residual.unsqueeze(-1), chol)
        mean = p.constant[:, None] + (cross @ alpha).squeeze(-1)
        v = torch.linalg.solve_triangular(
            chol, cross.transpose(-1, -2), upper=False
        )
        variance = (prior_variance[:, None] - (v**2).sum(-2)).clamp_min(
            MIN_VARIANCE
        )
        means.append(data.y_mean + data.y_std * mean)
        variances.append(variance * data.y_std**2)
        noises.append(p.noise * data.y_std**2)
    return torch.cat(means), torch.cat(variances), torch.cat(noises)
