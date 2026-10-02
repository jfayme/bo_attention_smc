"""One BO campaign, under any of the study's models.

A campaign starts from 5 reactions drawn at random from the pool and then
chooses 50 more, one at a time. Before each choice the model is rebuilt on
every reaction measured so far, each reaction not measured yet is scored by
its log expected improvement (LogEI), and the best one is measured next:
its yield is read from the pool. Each choice is scored before it joins the
data, by the log density and the z-score of its yield under the model (the
prequential score), which says how well the model predicted it.

A model is a combination of parts, named by the labels the study's records
use (`MODELS`):

| label               | kernel          | head                   | inference |
|---------------------|-----------------|------------------------|-----------|
| `null`              | Matern          | none (mean pooling)    | MAP       |
| `identity`          | Matern + term   | none                   | MAP       |
| `smc_span_pca8`     | Matern          | pca8                   | SMC       |
| `smc_span`          | Matern          | all channels           | SMC       |
| `smc_identity_head` | Matern + term   | pca8                   | SMC       |
| `smc_identity`      | Matern + term   | pca8, held at mean     | SMC       |
|                     |                 | pooling (prior sd 1e-3)|           |

The features (MACE, T5, Morgan) are the cell's, and the rule centring the
length-scale prior is chosen per campaign. A model without a head is fitted
by MAP at every step (gp.py). One with a head is sampled by SMC (smc.py),
its particles carried from one step to the next, and LogEI is averaged over
the particles. The term needs each reaction's molecules, which the dataset
names (cells.molecules), so it works on every cell, Morgan included; a head
needs atoms, so not on Morgan.

Two random streams. The random start comes from numpy, seeded 1337 + seed,
so every model of a (cell, rule, seed) starts from the same reactions. The
torch stream is seeded with the same number inside the campaign, and an
SMC model draws from its own generator, seeded with the same number again.
The first index wins a tie.

"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import torch
from botorch.acquisition import LogExpectedImprovement
from botorch.models import SingleTaskGP
from torch.nn.functional import softplus

from bo_attention_smc.core import smc
from bo_attention_smc.core.cells import (
    Cell,
    molecules,
    null_features,
    reagents,
)
from bo_attention_smc.core.gp import (
    Prediction,
    Prior,
    build_gp,
    fit,
    fitted,
    make_prior,
    pool_bounds,
)
from bo_attention_smc.core.posterior import F64, Bank, Data, predict

# The design of every campaign: 5 random reactions then 50 chosen, and the
# seed offset.
N_INIT = 5
N_BO = 50
SEED_BASE = 1337

# A head prior this narrow holds the head at mean pooling (to about 1e-3):
# the per-molecule term sampled by SMC, as its pre-registered test ran it.
HELD = 1e-3


@dataclass(frozen=True)
class Model:
    """What a label stands for.

    Attributes
    ----------
    term
        Whether the kernel carries the per-molecule term.
    head
        None (mean pooling, fitted by MAP), or a head sampled by SMC: one
        of posterior.HEADS.
    head_scale
        The prior sd of every head weight.

    """

    term: bool = False
    head: str | None = None
    head_scale: float = 1.0


MODELS = {
    "null": Model(),
    "identity": Model(term=True),
    "smc_span_pca8": Model(head="pca8"),
    "smc_span": Model(head="all"),
    "smc_identity_head": Model(term=True, head="pca8"),
    "smc_identity": Model(term=True, head="pca8", head_scale=HELD),
}


@dataclass(frozen=True, eq=False)
class Pool:
    """A cell as a campaign's GP reads it.

    Attributes
    ----------
    x
        Every reaction's input: its mean-pooled features, then, with the
        term, each reagent's molecule.
    y
        Every reaction's yield.
    prior
        The length-scale prior, centred on the cell's stored features.
    bounds
        Normalize's bounds, shape (2, x's width): the stored features'
        range, then 0 and the largest number of each molecule column.
    n_features
        The columns the Matern reads, the first.
    reagents
        The reagents of the molecule columns; empty without the term.

    """

    x: torch.Tensor
    y: torch.Tensor
    prior: Prior
    bounds: torch.Tensor
    n_features: int
    reagents: list[str]

    @property
    def features(self) -> torch.Tensor:
        """The columns the Matern reads, for every reaction."""
        return self.x[:, : self.n_features]

    @property
    def molecules(self) -> torch.Tensor:
        """Each reagent's molecule, for every reaction, shape (n, C)."""
        return self.x[:, self.n_features :]


def make_pool(cell: Cell, rule: str, dataset: str | None = None) -> Pool:
    """Describe a cell to a campaign's GP.

    Parameters
    ----------
    cell
        The cell.
    rule
        One of gp.RULES.
    dataset
        The cell's dataset, to give the GP each reaction's molecules for
        the term; None for a model without the term.

    Returns
    -------
    pool
        The inputs are the null features (cells.null_features), as the
        study's GP saw them; the prior and the bounds come from the stored
        features, which differ from them by about 1e-16.

    """
    x = null_features(cell)
    y = torch.tensor(cell.objective, dtype=F64)
    bounds = torch.tensor(pool_bounds(cell.features), dtype=F64)
    prior = make_prior(cell.features, rule)
    if dataset is None:
        return Pool(x, y, prior, bounds, x.shape[1], [])
    numbers = torch.as_tensor(molecules(dataset), dtype=F64)
    top = numbers.amax(dim=0).clamp_min(1.0)
    span = torch.stack([torch.zeros_like(top), top])
    return Pool(
        torch.cat([x, numbers], dim=1),
        y,
        prior,
        torch.cat([bounds, span], dim=1),
        x.shape[1],
        reagents(dataset),
    )


@dataclass(eq=False)
class Fitted:
    """A GP fitted by MAP, as the campaign asks of a model."""

    gp: SingleTaskGP
    x: torch.Tensor

    def log_ei(self, rows: np.ndarray, best_f: float) -> torch.Tensor:
        """BoTorch's LogEI of some reactions."""
        with torch.no_grad():
            acquisition = LogExpectedImprovement(self.gp, best_f=best_f)
            return acquisition(self.x[rows].unsqueeze(1))

    def predict(self, rows: object) -> Prediction:
        """Predict some reactions."""
        return Prediction.of_gp(self.gp, self.x[rows])


@dataclass(eq=False)
class Sampled:
    """The particles' predictions of the whole pool, as the campaign asks."""

    mean: torch.Tensor
    variance: torch.Tensor
    noise: torch.Tensor

    def log_ei(self, rows: np.ndarray, best_f: float) -> torch.Tensor:
        """LogEI of some reactions, averaged over the particles."""
        return self.predict(rows).log_ei(best_f)

    def predict(self, rows: object) -> Prediction:
        """Predict some reactions: a mixture over the particles."""
        rows = torch.as_tensor(np.asarray(rows))
        return Prediction(
            self.mean[:, rows], self.variance[:, rows], self.noise
        )


# Builds the model on the measured reactions: the model, and what to record
# of it.
Step = Callable[[list[int]], tuple[Fitted | Sampled, dict]]


def map_step(pool: Pool) -> Step:
    """Return the step of a model without a head: a GP fitted by MAP."""

    def step(rows: list[int]) -> tuple[Fitted, dict]:
        gp = build_gp(
            pool.x[rows],
            pool.y[rows],
            pool.prior,
            pool.bounds,
            pool.n_features,
        )
        status = fit(gp)
        found = fitted(gp, pool.prior, pool.reagents)
        return Fitted(gp, pool.x), {"status": status, **found}

    return step


class SMCStep:
    """The step of a model with a head: SMC, the particles carried along.

    Parameters
    ----------
    cell
        The cell, with atoms.
    pool
        The cell as the GP reads it; with molecule columns, the kernel
        carries the per-molecule term.
    head
        One of posterior.HEADS.
    head_scale
        The prior sd of every head weight.
    seed
        The campaign's seed, which also seeds the sampler's generator.
    n_max
        Reactions the campaign ends with: the padding of every step.
    device
        Where the particles live: the GPU where there is one.

    """

    def __init__(
        self,
        cell: Cell,
        pool: Pool,
        head: str,
        head_scale: float,
        seed: int,
        n_max: int,
        device: str = "cpu",
    ) -> None:
        self.pool = pool
        self.n_max = n_max
        self.device = torch.device(device)
        self.settings = smc.Settings()
        numbers = pool.molecules if pool.reagents else None
        bounds = pool.bounds[:, : pool.n_features]
        self.bank = Bank.of(
            cell, bounds, head, numbers, head_scale, self.device
        )
        self.generator = torch.Generator(self.device)
        self.generator.manual_seed(SEED_BASE + seed)
        self.particles: torch.Tensor | None = None
        self.data: Data | None = None
        self.step_size = self.settings.step_size

    def __call__(self, rows: list[int]) -> tuple[Sampled, dict]:
        """Move the particles to this step's posterior, and predict."""
        tic = time.perf_counter()
        pool = self.pool
        # the GP without the term, built but not fitted: the standardised
        # yields and the noise floor, as the MAP models see them
        gp = build_gp(
            pool.features[rows],
            pool.y[rows],
            pool.prior,
            pool.bounds[:, : pool.n_features],
            pool.n_features,
        )
        data = Data.of(gp, rows, pool.prior, self.n_max, self.device)
        if self.particles is None or self.data is None:
            particles, health = smc.first(
                self.bank, data, self.settings, self.generator
            )
        else:
            particles, health = smc.advance(
                self.particles,
                self.bank,
                self.data,
                data,
                self.settings,
                self.step_size,
                self.generator,
            )
        self.particles, self.data = particles, data
        self.step_size = health.step_size
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        sampled = time.perf_counter()
        mean, variance, noise = predict(particles, self.bank, data)
        if self.device.type == "cuda":
            # the peak of the pool's prediction, given back to the other
            # campaigns sharing the GPU
            torch.cuda.empty_cache()
        record = {
            "status": "SMC" if health.reached else "UNREACHED",
            "smc_tempering": health.tempering_steps,
            "smc_acceptance": health.acceptance,
            "smc_step_size": health.step_size,
            "smc_log_evidence": health.log_evidence,
            "smc_seconds": sampled - tic,
            "predict_seconds": time.perf_counter() - sampled,
        }
        if pool.reagents:
            raw = particles[:, -self.bank.n_reagents :]
            medians = softplus(raw).median(dim=0).values.cpu().numpy()
            for name, value in zip(pool.reagents, medians, strict=True):
                record[f"w[{name}]"] = float(value)
        return Sampled(mean.cpu(), variance.cpu(), noise.cpu()), record


def make_step(
    label: str,
    cell: Cell,
    pool: Pool,
    seed: int,
    n_max: int,
    device: str = "cpu",
) -> Step:
    """Return a label's model-building step: MAP, or SMC with a head."""
    model = MODELS[label]
    if model.head is None:
        return map_step(pool)
    return SMCStep(
        cell, pool, model.head, model.head_scale, seed, n_max, device
    )


def run(y: torch.Tensor, step: Step, seed: int, n_bo: int) -> dict:
    """Run one campaign, the model at each choice built by `step`.

    Parameters
    ----------
    y
        Every reaction's yield.
    step
        Builds the model on the measured reactions.
    seed
        Campaign seed.
    n_bo
        Reactions chosen after the random start.

    Returns
    -------
    record
        `sampled_indices`, every reaction measured, in order; `fits`, one
        list per diagnostic with an entry per choice, among them `log_score`
        and `z`, the prequential score of each choice; `seconds`.

    """
    tic = time.perf_counter()
    n = len(y)
    rng = np.random.default_rng(SEED_BASE + seed)
    sampled = rng.choice(n, N_INIT, replace=False).tolist()
    fits = []
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(SEED_BASE + seed)
        for _ in range(n_bo):
            start = time.perf_counter()
            candidates = np.setdiff1d(np.arange(n), sampled)
            best = float(y[sampled].max())
            model, found = step(list(sampled))
            scores = model.log_ei(candidates, best)
            # the first index wins a tie
            chosen = int(candidates[np.argmax(scores.numpy())])
            measured = y[chosen : chosen + 1]
            prediction = model.predict([chosen])
            mean, variance = prediction.moments()
            found["log_score"] = float(prediction.log_density(measured))
            found["z"] = float((measured - mean) / variance.sqrt())
            found["seconds"] = time.perf_counter() - start
            fits.append(found)
            sampled.append(chosen)
    return {
        "sampled_indices": sampled,
        "fits": {name: [fit[name] for fit in fits] for name in fits[0]},
        "seconds": time.perf_counter() - tic,
    }


def run_campaign(
    cell: Cell,
    dataset: str,
    label: str,
    rule: str,
    seed: int,
    n_bo: int = N_BO,
    device: str = "cpu",
) -> dict:
    """Run one campaign of a label on a cell; `run` says what it returns."""
    model = MODELS[label]
    pool = make_pool(cell, rule, dataset if model.term else None)
    step = make_step(label, cell, pool, seed, N_INIT + n_bo, device)
    return run(pool.y, step, seed, n_bo)
