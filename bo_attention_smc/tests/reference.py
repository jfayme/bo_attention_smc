"""A plain, one-particle-at-a-time version of the sampled posterior.

posterior.py computes the posterior for a whole population of particles at
once on the GPU, with a few tricks for speed (padding, kernels from inner
products). This is the same model written plainly, one particle at a time,
by pairwise differences: the study's own code (bo_hmc's Posterior and
SpanHead, bo_per_molecule's IdentityPosterior), with the NUTS sampler
removed. The tests check posterior.py against it particle by particle
(test_smc.py), and check it against the numbers the study recorded
(golden.json, test_golden.py) and against the GP with the term
(test_gp.py), so the chain runs from the recorded numbers to the fast code.

"""

from __future__ import annotations

import math

import torch
from botorch.models import SingleTaskGP
from torch import nn
from torch.nn.functional import linear, logsigmoid, softplus

from bo_attention_smc.core.cells import Cell, pool_atoms
from bo_attention_smc.core.gp import (
    MIN_VARIANCE,
    TERM_CONCENTRATION,
    TERM_RATE,
    Prediction,
)

F64 = torch.float64
SQRT5 = math.sqrt(5.0)

# The head's hidden units, the cap of the head on every channel, and the
# PCA-8 head's components, as posterior.py has them.
HIDDEN = 4
MAX_WEIGHTS_PER_MOLECULE = 10.0
N_COMPONENTS = 8


def span_map(
    pooled: torch.Tensor, low: torch.Tensor, high: torch.Tensor
) -> torch.Tensor:
    """Map every column of a reagent's pooled molecules onto [low, high]."""
    lowest = pooled.amin(dim=0)
    span = (pooled.amax(dim=0) - lowest).clamp_min(
        torch.finfo(pooled.dtype).tiny
    )
    return low + (pooled - lowest) * ((high - low) / span)


def matern52(
    z1: torch.Tensor, z2: torch.Tensor, lengthscale: torch.Tensor
) -> torch.Tensor:
    """Matern-5/2 correlation by pairwise differences, shape (m1, m2)."""
    scaled = (z1.unsqueeze(-2) - z2.unsqueeze(-3)) / lengthscale
    r = torch.sqrt((scaled**2).sum(-1).clamp_min(1e-36))
    return (1.0 + SQRT5 * r + 5.0 / 3.0 * r**2) * torch.exp(-SQRT5 * r)


class SpanHead:
    """An attention head that pools molecules within the null's span.

    Parameters
    ----------
    cell
        A cell with atoms.
    bounds
        The null features' range, shape (2, D).
    pca8
        If True, atoms are scored on their top 8 principal components;
        otherwise on every channel.

    """

    def __init__(self, cell: Cell, bounds: torch.Tensor, pca8: bool) -> None:
        if cell.atoms is None:
            raise ValueError("a cell without atoms has no head")
        self.atoms = cell.atoms
        n_inputs = self.atoms.values.shape[-1]
        self.projection = None
        n_hidden = HIDDEN
        if pca8:
            real = self.atoms.values[self.atoms.mask]
            centre = real.mean(dim=0)
            _, singular, vt = torch.linalg.svd(
                real - centre, full_matrices=False
            )
            sd = singular[:N_COMPONENTS] / math.sqrt(len(real) - 1)
            self.projection = (centre, vt[:N_COMPONENTS].T / sd)
            n_inputs = N_COMPONENTS
        else:
            n_molecules = len(self.atoms.values)
            while (
                n_hidden * (n_inputs + 2) / n_molecules
                > MAX_WEIGHTS_PER_MOLECULE
            ):
                n_hidden -= 1
        self.shapes = {
            "V": (n_hidden, n_inputs),
            "b": (n_hidden,),
            "w": (1, n_hidden),
        }
        self.molecules, self.positions, self.low, self.high = [], [], [], []
        for component, block in enumerate(cell.blocks):
            molecules = torch.unique(self.atoms.index[:, component])
            position = torch.full((len(self.atoms.values),), -1)
            position[molecules] = torch.arange(len(molecules))
            self.molecules.append(molecules)
            self.positions.append(position)
            self.low.append(bounds[0, block])
            self.high.append(bounds[1, block])

    def draw(self) -> torch.Tensor:
        """Draw starting weights from the torch stream, as the study did.

        w is zero, so the head starts at mean pooling exactly. V and b are
        drawn as torch initialises a linear layer, in float32, and w is
        drawn too and then zeroed, as the study's head was.
        """
        n_hidden, n_inputs = self.shapes["V"]
        scorer = nn.Linear(n_inputs, n_hidden)
        nn.Linear(n_hidden, 1, bias=False)
        start = [scorer.weight.flatten(), scorer.bias, torch.zeros(n_hidden)]
        return torch.cat(start).detach().double()

    def features(self, head: dict, rows: object) -> torch.Tensor:
        """Pool the bank with some weights, and assemble some reactions."""
        inputs = self.atoms.values
        if self.projection is not None:
            centre, matrix = self.projection
            inputs = (self.atoms.values - centre) @ matrix
        hidden = torch.tanh(linear(inputs, head["V"], head["b"]))
        scores = linear(hidden, head["w"]).squeeze(-1)
        pooled = pool_atoms(scores, self.atoms)
        rows = torch.as_tensor(rows)
        index = self.atoms.index[rows]
        parts = []
        for component, kept in enumerate(self.atoms.columns):
            own = pooled[self.molecules[component]][:, kept]
            own = span_map(own, self.low[component], self.high[component])
            parts.append(own[self.positions[component][index[:, component]]])
        parts.append(self.atoms.settings[rows])
        return torch.cat(parts, dim=-1)


class Posterior:
    """The posterior of the GP's settings, a head's weights and the term's.

    theta = [raw length-scales, u, mean, V, b, w, raw term weights].

    Parameters
    ----------
    gp
        A GP without the term, built on the measured reactions (gp.py):
        its priors, noise floor, Normalize and standardised yields are read
        from it, and theta starts at its settings.
    x
        The null features of every reaction, shape (n, D).
    rows
        The measured reactions.
    head
        The head; None for the null features.
    molecules
        Each reaction's molecule of every reagent, (n, C), for the term;
        None without it.

    """

    def __init__(
        self,
        gp: SingleTaskGP,
        x: torch.Tensor,
        rows: object,
        head: SpanHead | None = None,
        molecules: torch.Tensor | None = None,
    ) -> None:
        kernel = gp.covar_module
        self.lengthscale_prior = kernel.lengthscale_prior
        self.n_lengthscales = kernel.raw_lengthscale.numel()
        noise_covar = gp.likelihood.noise_covar
        self.noise_prior = noise_covar.noise_prior
        self.floor = float(noise_covar.raw_noise_constraint.lower_bound)
        self.head_prior = torch.distributions.Normal(
            torch.tensor(0.0, dtype=F64), torch.tensor(1.0, dtype=F64)
        )
        self.weight_prior = torch.distributions.Gamma(
            torch.tensor(TERM_CONCENTRATION, dtype=F64),
            torch.tensor(TERM_RATE, dtype=F64),
        )
        self.normalize = gp.input_transform
        self.y = gp.train_targets.detach()
        self.y_mean = gp.outcome_transform.means.detach().squeeze()
        self.y_std = gp.outcome_transform.stdvs.detach().squeeze()
        self.x = x
        self.rows = torch.as_tensor(rows)
        self.head = head
        self.molecules = molecules
        self.n_reagents = 0 if molecules is None else molecules.shape[1]
        noise = gp.likelihood.noise.detach().reshape(())
        excess = (noise - self.floor).clamp_min(1e-12)
        start = [
            kernel.raw_lengthscale.detach().reshape(-1),
            torch.log(torch.expm1(excess)).reshape(1),
            gp.mean_module.constant.detach().reshape(1),
        ]
        if head is not None:
            start.append(head.draw())
        if molecules is not None:
            weight = (TERM_CONCENTRATION - 1.0) / TERM_RATE
            weights = torch.full((self.n_reagents,), weight, dtype=F64)
            start.append(torch.log(torch.expm1(weights)))
        self.start = torch.cat(start).to(F64)

    def unpack(self, theta: torch.Tensor) -> dict:
        """Name the parts of a flat vector."""
        d, c = self.n_lengthscales, self.n_reagents
        end = len(theta) - c
        head, i = {}, d + 2
        if self.head is not None:
            for name, shape in self.head.shapes.items():
                size = math.prod(shape)
                head[name] = theta[i : i + size].view(shape)
                i += size
        return {
            "raw_lengthscale": theta[:d],
            "lengthscale": softplus(theta[:d]),
            "u": theta[d],
            "noise": self.floor + softplus(theta[d]),
            "constant": theta[d + 1],
            "head": head,
            "head_flat": theta[d + 2 : end],
            "raw_w": theta[end:],
            "w": softplus(theta[end:]),
        }

    def features(self, p: dict, rows: object) -> torch.Tensor:
        """Return the kernel's inputs of some reactions under one theta."""
        if self.head is None:
            return self.normalize.transform(self.x[rows])
        return self.normalize.transform(self.head.features(p["head"], rows))

    def covariance(
        self, p: dict, rows_a: object, rows_b: object
    ) -> torch.Tensor:
        """Return the latent covariance between two sets of reactions."""
        k = matern52(
            self.features(p, rows_a),
            self.features(p, rows_b),
            p["lengthscale"],
        )
        if self.molecules is not None:
            a = self.molecules[torch.as_tensor(rows_a)]
            b = self.molecules[torch.as_tensor(rows_b)]
            same = (a.T.unsqueeze(-1) == b.T.unsqueeze(-2)).to(F64)
            k = k + (p["w"][:, None, None] * same).sum(0)
        return k

    def log_likelihood(self, theta: torch.Tensor) -> torch.Tensor:
        """Log marginal likelihood of the standardised yields."""
        p = self.unpack(theta)
        n = len(self.rows)
        covariance = self.covariance(p, self.rows, self.rows)
        covariance = covariance + p["noise"] * torch.eye(n, dtype=F64)
        chol, info = torch.linalg.cholesky_ex(covariance)
        if int(info) != 0:
            return theta.sum() * 0.0 - math.inf
        residual = (self.y - p["constant"]).unsqueeze(-1)
        alpha = torch.cholesky_solve(residual, chol).squeeze(-1)
        return (
            -0.5 * residual.squeeze(-1) @ alpha
            - torch.log(torch.diagonal(chol)).sum()
            - 0.5 * n * math.log(2.0 * math.pi)
        )

    def log_posterior(self, theta: torch.Tensor) -> torch.Tensor:
        """Log posterior density, up to a constant."""
        p = self.unpack(theta)
        log_prior = (
            self.lengthscale_prior.log_prob(p["lengthscale"]).sum()
            + self.noise_prior.log_prob(p["noise"]).sum()
            + self.head_prior.log_prob(p["head_flat"]).sum()
            + self.weight_prior.log_prob(p["w"]).sum()
        )
        jacobian = (
            logsigmoid(p["raw_lengthscale"]).sum()
            + logsigmoid(p["u"])
            + logsigmoid(p["raw_w"]).sum()
        )
        return self.log_likelihood(theta) + log_prior + jacobian

    def predict(self, thetas: torch.Tensor, rows: object) -> Prediction:
        """Predict some reactions under every theta, in the yields' units."""
        means, variances, noises = [], [], []
        with torch.no_grad():
            for theta in thetas:
                p = self.unpack(theta)
                n = len(self.rows)
                chol = torch.linalg.cholesky(
                    self.covariance(p, self.rows, self.rows)
                    + p["noise"] * torch.eye(n, dtype=F64)
                )
                cross = self.covariance(p, rows, self.rows)
                residual = (self.y - p["constant"]).unsqueeze(-1)
                alpha = torch.cholesky_solve(residual, chol).squeeze(-1)
                mean = p["constant"] + cross @ alpha
                v = torch.linalg.solve_triangular(chol, cross.T, upper=False)
                prior_variance = 1.0 + p["w"].sum()
                variance = (prior_variance - (v**2).sum(0)).clamp_min(
                    MIN_VARIANCE
                )
                means.append(self.y_mean + self.y_std * mean)
                variances.append(variance * self.y_std**2)
                noises.append(p["noise"] * self.y_std**2)
        return Prediction(
            torch.stack(means), torch.stack(variances), torch.stack(noises)
        )
