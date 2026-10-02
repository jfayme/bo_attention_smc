"""The Gaussian process (GP) of every campaign, its priors, and its fit.

The GP predicts a reaction's yield from its features. Its kernel, the
similarity of two reactions, is Matern-5/2 with one length-scale per column
(ARD) and no amplitude, since the yields are standardised by the measured
ones. The inputs are mapped onto the unit box by the range of the pool's
features (`pool_bounds`), and the noise is Gaussian.

Optionally the kernel carries the per-molecule term:

    k(a, b) = Matern(x_a, x_b) + sum_c w_c [a and b use the same molecule c]

one weight w_c per reagent. The Matern reads the features only; the term
lets the GP learn each molecule's own effect, whatever its features say.

The priors:

- every length-scale: a Gamma centred by a rule (`make_prior`), or under
  the rule `default` BoTorch's own LogNormal;
- the noise: LogNormal(-4, 1), floored at 1e-4, as BoTorch 0.18 sets it,
  written out here so that it cannot move with BoTorch;
- every weight w_c: Gamma(2, 4), whose mode, 0.25, gives each molecule an
  own effect of sd 0.5 standardised units on top of its features'.

MAP fitting (`fit`) is one L-BFGS-B run from the priors' centres. The
sampled models of posterior.py read their priors and standardised yields
from a GP built here, so that both kinds of model see the same ones.

"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass

import numpy as np
import torch
from botorch.acquisition.analytic import _log_ei_helper
from botorch.models import SingleTaskGP
from botorch.models.transforms import Normalize, Standardize
from botorch.optim.fit import fit_gpytorch_mll_scipy
from gpytorch.constraints import GreaterThan
from gpytorch.kernels import AdditiveKernel, Kernel, MaternKernel, ScaleKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.priors import GammaPrior, LogNormalPrior
from linear_operator.utils.errors import NanError, NotPSDError
from scipy.spatial.distance import pdist

F64 = torch.float64

RULES = ("chen", "geom", "default")

# The Gamma prior's coefficient of variation, one width for chen and geom,
# so that only the centre differs between them.
CV = 0.3

# The noise prior, LogNormal(loc, scale), and the floor of the noise.
NOISE_LOC = -4.0
NOISE_SCALE = 1.0
NOISE_FLOOR = 1e-4

# The Gamma(concentration, rate) prior of every weight of the term; each
# weight starts at its mode.
TERM_CONCENTRATION = 2.0
TERM_RATE = 4.0

# u* maximises u |k'(u)| for the Matern-5/2 kernel: the kernel varies most
# at distance u* times the length-scale, so geom centres it at the pool's
# mean distance over u*.
USTAR = float((1.0 + np.sqrt(3.0)) / np.sqrt(5.0))

# The mean distance is taken over at most this many reactions, drawn with
# seed 0 from a larger pool.
MAX_POINTS = 2000

# Posterior variances are floored before a square root, as BoTorch's
# LogExpectedImprovement floors them.
MIN_VARIANCE = 1e-12


def pool_bounds(features: np.ndarray) -> np.ndarray:
    """Range of every column, shape (2, D), mapped onto [0, 1] for the GP.

    A constant column gets an upper bound one above its value, so that it
    maps to zero rather than to 0 / 0.
    """
    lower, upper = features.min(axis=0), features.max(axis=0)
    return np.stack([lower, np.where(upper > lower, upper, lower + 1.0)])


@dataclass(frozen=True)
class Prior:
    """The prior on every length-scale: a Gamma, or BoTorch's LogNormal.

    Attributes
    ----------
    ell_0
        Where every length-scale starts: the Gamma's mean, or the
        LogNormal's mode, where BoTorch starts it.
    concentration
        The Gamma's shape parameter; NaN under the LogNormal.
    rate
        The Gamma's rate parameter; NaN under the LogNormal.
    lognormal
        None for the Gamma; else the (loc, scale) of the LogNormal that
        replaces it, under the rule `default`.

    """

    ell_0: float
    concentration: float
    rate: float
    lognormal: tuple[float, float] | None = None

    def module(self) -> GammaPrior | LogNormalPrior:
        """Build the GPyTorch prior, with float64 parameters."""
        if self.lognormal is not None:
            loc, scale = self.lognormal
            return LogNormalPrior(
                torch.tensor(loc, dtype=F64), torch.tensor(scale, dtype=F64)
            )
        return GammaPrior(
            torch.tensor(self.concentration, dtype=F64),
            torch.tensor(self.rate, dtype=F64),
        )


def mean_distance(features: np.ndarray) -> float:
    """Mean pairwise distance of a pool mapped onto the unit box.

    Parameters
    ----------
    features
        The pool's features, shape (n, D).

    Returns
    -------
    distance
        Over every pair of reactions, or of 2000 drawn with seed 0 from a
        larger pool.

    """
    lower, upper = pool_bounds(features)
    unit = (features - lower) / (upper - lower)
    if len(unit) > MAX_POINTS:
        rng = np.random.default_rng(0)
        unit = unit[rng.choice(len(unit), MAX_POINTS, replace=False)]
    return float(pdist(unit).mean())


def make_prior(features: np.ndarray, rule: str) -> Prior:
    """Centre the length-scale prior by a rule.

    Parameters
    ----------
    features
        The pool's features, shape (n, D).
    rule
        `chen`: 0.4 sqrt(D) + 4, the published rule of Chen, Fleck and
        Stuyver (JCTC 2026), blind to the pool. `geom`: the pool's mean
        pairwise distance in the unit box over u*. Both give Gamma(a,
        a / ell_0) with a = 1 / CV**2. `default`: no rule, the prior
        BoTorch 0.18 gives SingleTaskGP, LogNormal(sqrt(2) + log(D) / 2,
        sqrt(3)) (Hvarfner et al. 2024), each length-scale starting at its
        mode, as BoTorch starts it.

    Returns
    -------
    prior
        The Gamma prior, with mean ell_0; or, under `default`, the
        LogNormal, with mode ell_0.

    """
    if rule == "default":
        loc = float(np.sqrt(2.0) + 0.5 * np.log(features.shape[1]))
        scale = float(np.sqrt(3.0))
        mode = float(np.exp(loc - scale * scale))
        return Prior(mode, float("nan"), float("nan"), (loc, scale))
    if rule == "chen":
        ell_0 = float(0.4 * np.sqrt(features.shape[1]) + 4.0)
    else:
        ell_0 = mean_distance(features) / USTAR
    concentration = 1.0 / (CV * CV)
    return Prior(ell_0, concentration, concentration / ell_0)


class SameMolecule(Kernel):
    """One between reactions that use the same molecule, else zero.

    It reads one column, a reagent's molecule, and is positive
    semi-definite: a block of ones per molecule.
    """

    has_lengthscale = False

    def forward(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor,
        diag: bool = False,
        **params: object,
    ) -> torch.Tensor:
        """Compare the molecules of two sets of reactions."""
        a, b = x1[..., 0], x2[..., 0]
        if diag:
            return (a == b).to(x1.dtype)
        return (a.unsqueeze(-1) == b.unsqueeze(-2)).to(x1.dtype)


def build_gp(
    x: torch.Tensor,
    y: torch.Tensor,
    prior: Prior,
    bounds: torch.Tensor,
    n_features: int,
) -> SingleTaskGP:
    """Build the GP on the measured reactions, without fitting it.

    Parameters
    ----------
    x
        The measured reactions' inputs, shape (n, n_features + C): their
        features, then, with the term, each reagent's molecule.
    y
        Their yields, shape (n,).
    prior
        The length-scale prior.
    bounds
        Normalize's bounds, shape (2, n_features + C); a molecule column's
        are 0 and its largest number, and equal molecules stay equal.
    n_features
        The columns the Matern reads, the first. Each column after them
        adds one SameMolecule term, its weight a ScaleKernel's outputscale.

    Returns
    -------
    gp
        Every length-scale and weight at its prior's centre, the noise at
        its prior's mode, the yields standardised by their own mean and sd.

    """
    with_term = x.shape[1] > n_features
    # active_dims only with the term: the GP without it is built by the
    # operations every recorded null campaign used
    active = {"active_dims": torch.arange(n_features)} if with_term else {}
    matern = MaternKernel(
        nu=2.5,
        ard_num_dims=n_features,
        lengthscale_prior=prior.module(),
        **active,
    ).to(F64)
    matern.lengthscale = torch.full((1, n_features), prior.ell_0, dtype=F64)
    kernel = matern
    if with_term:
        terms = [matern]
        for column in range(n_features, x.shape[1]):
            weight = ScaleKernel(
                SameMolecule(active_dims=torch.tensor([column])),
                outputscale_prior=GammaPrior(
                    torch.tensor(TERM_CONCENTRATION, dtype=F64),
                    torch.tensor(TERM_RATE, dtype=F64),
                ),
            ).to(F64)
            weight.outputscale = torch.tensor(
                (TERM_CONCENTRATION - 1.0) / TERM_RATE, dtype=F64
            )
            terms.append(weight)
        kernel = AdditiveKernel(*terms)
    # Python floats, so that the noise starts at the prior's mode in
    # float32, as BoTorch's default likelihood does
    noise_prior = LogNormalPrior(loc=NOISE_LOC, scale=NOISE_SCALE)
    likelihood = GaussianLikelihood(
        noise_prior=noise_prior,
        noise_constraint=GreaterThan(
            NOISE_FLOOR, transform=None, initial_value=noise_prior.mode
        ),
    )
    return SingleTaskGP(
        x,
        y.unsqueeze(-1),
        likelihood=likelihood,
        covar_module=kernel,
        input_transform=Normalize(x.shape[1], bounds=bounds),
        outcome_transform=Standardize(1),
    )


def fit(gp: SingleTaskGP) -> str:
    """Fit a GP's settings by one L-BFGS-B run on its marginal likelihood.

    Returns
    -------
    status
        How the run ended: SUCCESS, STOPPED (at the iteration limit),
        FAILURE (scipy gave up; its last point is kept) or ERROR (the
        likelihood could not be evaluated; the GP is back where it
        started).

    Notes
    -----
    BoTorch's fit_gpytorch_mll would retry a failed fit from random draws
    of every prior, silently and from the campaign's torch stream. Here a
    fit is never retried (BO_CHANGELOG.md, 2026-09-25).

    """
    mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
    start = copy.deepcopy(gp.state_dict())
    mll.train()
    try:
        status = fit_gpytorch_mll_scipy(mll).status.name
    except (NotPSDError, NanError):
        gp.load_state_dict(start)
        status = "ERROR"
    mll.eval()
    return status


def fitted(gp: SingleTaskGP, prior: Prior, reagents: list[str]) -> dict:
    """Read what a fit found, for a campaign's record.

    Returns
    -------
    found
        `ell_ratio`, the geometric mean length-scale over the prior's
        centre; `noise`, standardised; and with the term `w[reagent]`, each
        reagent's weight.

    """
    kernel = gp.covar_module
    matern = kernel.kernels[0] if reagents else kernel
    lengthscale = matern.lengthscale.detach().reshape(-1)
    found = {
        "ell_ratio": float(lengthscale.log().mean().exp()) / prior.ell_0,
        "noise": float(gp.likelihood.noise.detach().squeeze()),
    }
    for c, name in enumerate(reagents):
        weight = kernel.kernels[1 + c].outputscale.detach()
        found[f"w[{name}]"] = float(weight)
    return found


@dataclass(frozen=True)
class Prediction:
    """Predictions of m reactions: an equal mixture of S Gaussians.

    Attributes
    ----------
    mean
        Latent mean under each component, in the yields' units, shape
        (S, m).
    variance
        Latent variance under each component, same units squared, shape
        (S, m).
    noise
        Noise variance of each component, same units squared, shape (S,).

    Notes
    -----
    A fitted GP is a mixture of one (`of_gp`), a sampled model one
    component per particle.

    """

    mean: torch.Tensor
    variance: torch.Tensor
    noise: torch.Tensor

    @classmethod
    def of_gp(
        cls: type[Prediction], gp: SingleTaskGP, x: torch.Tensor
    ) -> Prediction:
        """Predict some reactions with a fitted GP.

        Parameters
        ----------
        gp
            The fitted GP, with its Standardize outcome transform.
        x
            The reactions' inputs, shape (m, D).

        Returns
        -------
        prediction
            A mixture of one, in the yields' units.

        """
        with torch.no_grad():
            posterior = gp.posterior(x)
            stdv = gp.outcome_transform.stdvs.squeeze()
            noise = gp.likelihood.noise.detach().reshape(1) * stdv**2
        return cls(
            posterior.mean.squeeze(-1).unsqueeze(0),
            posterior.variance.squeeze(-1).unsqueeze(0),
            noise,
        )

    def log_density(self, y: torch.Tensor) -> torch.Tensor:
        """Log density of measured yields, noise included, shape (m,)."""
        total = self.variance + self.noise.unsqueeze(-1)
        log_p = -0.5 * (
            torch.log(2 * math.pi * total) + (y - self.mean) ** 2 / total
        )
        return torch.logsumexp(log_p, dim=0) - math.log(len(self.mean))

    def moments(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Mean and variance of a new measurement under the mixture.

        Returns
        -------
        mean, variance
            Shape (m,); the variance includes the noise and the spread of
            the components' means.

        """
        mean = self.mean.mean(dim=0)
        spread = ((self.mean - mean) ** 2).mean(dim=0)
        within = (self.variance + self.noise.unsqueeze(-1)).mean(dim=0)
        return mean, within + spread

    def log_ei(self, best_f: float) -> torch.Tensor:
        """Log expected improvement over `best_f`, averaged over components.

        Parameters
        ----------
        best_f
            Best yield measured so far.

        Returns
        -------
        log_ei
            Shape (m,): log of the mean of the components' EI. For one
            component it is BoTorch's LogExpectedImprovement, with the same
            helper and variance floor.

        """
        sigma = self.variance.clamp_min(MIN_VARIANCE).sqrt()
        per = _log_ei_helper((self.mean - best_f) / sigma) + sigma.log()
        return torch.logsumexp(per, dim=0) - math.log(len(per))
