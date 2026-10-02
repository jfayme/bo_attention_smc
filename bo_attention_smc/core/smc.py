"""Sequential Monte Carlo (SMC) over a campaign's posteriors, on the GPU.

A BO campaign asks for a new posterior after every measurement, and each
differs from the last by one reaction (and by the standardisation of the
yields, which moves with it). So a population of particles is carried from
one posterior to the next (the IBIS idea: Chopin 2002):

1. At the first step the particles are drawn from a reference, the priors
   with N(0, 1) on the GP's mean (which has none), and tempered to the
   first posterior.
2. At every later step they are tempered from the last posterior to the
   new one.

Tempering from a density `base` to a density `target` moves through
base^(1 - lambda) target^lambda, lambda from 0 to 1. Each tempering step:

- picks the next lambda so that the effective sample size of the
  reweighted particles is `target_ess` of their number (bisection);
- resamples them (systematic resampling);
- moves each by a few HMC transitions at the new lambda. The inverse mass
  matrix is the particles' variance, and the step size is adapted towards
  an acceptance of 0.65 and carried from one BO step to the next.

At least one tempering step runs even when lambda can jump to 1, so the
particles are refreshed at every BO step. The whole population moves at
once: one batched log density and one autograd gradient per leapfrog step.

Every draw comes from one torch generator, seeded by the campaign; the
Gamma draws of the reference are made by numpy, seeded from it, since
torch's Gamma sampler takes no generator.

"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import torch
from torch.nn.functional import logsigmoid

from bo_attention_smc.core.gp import (
    NOISE_LOC,
    NOISE_SCALE,
    TERM_CONCENTRATION,
    TERM_RATE,
)
from bo_attention_smc.core.posterior import (
    F64,
    Bank,
    Data,
    gamma_log_prob,
    lengthscale_log_prob,
    log_posterior,
    unpack,
)

LogDensity = Callable[[torch.Tensor], torch.Tensor]


@dataclass(frozen=True)
class Settings:
    """How the sampler runs.

    Attributes
    ----------
    particles
        Population size.
    mcmc_steps
        HMC transitions per particle at each tempering step.
    leapfrog_steps
        Leapfrog steps per HMC transition.
    target_ess
        Fraction of the population the effective sample size is kept at.
    max_tempering
        Tempering steps allowed per posterior.
    step_size
        The HMC step size a campaign starts with.

    """

    particles: int = 256
    mcmc_steps: int = 5
    leapfrog_steps: int = 10
    target_ess: float = 0.5
    max_tempering: int = 100
    step_size: float = 0.3


@dataclass(frozen=True)
class Health:
    """How one posterior's tempering went."""

    tempering_steps: int
    acceptance: float
    step_size: float
    log_evidence: float
    reached: bool


def log_reference(theta: torch.Tensor, bank: Bank, data: Data) -> torch.Tensor:
    """Return the first step's reference: the priors, N(0, 1) on the mean."""
    p = unpack(theta, bank, data.floor)
    log_noise = torch.log(p.noise)
    value = (
        lengthscale_log_prob(p.lengthscale, data.prior).sum(-1)
        + logsigmoid(p.raw_lengthscale).sum(-1)
        - log_noise
        - (log_noise - NOISE_LOC) ** 2 / (2 * NOISE_SCALE**2)
        + logsigmoid(p.u)
        - 0.5 * p.constant**2
        - 0.5 * ((p.head / bank.head_scale) ** 2).sum(-1)
    )
    if bank.n_reagents:
        value = value + (
            gamma_log_prob(p.weights, TERM_CONCENTRATION, TERM_RATE).sum(-1)
            + logsigmoid(p.raw_weights).sum(-1)
        )
    return value


def inverse_softplus(x: torch.Tensor) -> torch.Tensor:
    """Raw such that softplus(raw) = x."""
    return x + torch.log(-torch.expm1(-x))


def draw_reference(
    bank: Bank, data: Data, n: int, generator: torch.Generator
) -> torch.Tensor:
    """Draw n particles from the reference, shape (n, dimension)."""
    device = bank.device

    def normal(*shape: int) -> torch.Tensor:
        return torch.randn(
            *shape, generator=generator, device=device, dtype=F64
        )

    def gamma(concentration: float, rate: float, *shape: int) -> torch.Tensor:
        seed = int(
            torch.randint(
                2**31 - 1, (1,), generator=generator, device=device
            ).item()
        )
        rng = np.random.default_rng(seed)
        drawn = rng.gamma(concentration, 1.0 / rate, size=shape)
        return torch.as_tensor(drawn, device=device, dtype=F64)

    prior = data.prior
    if prior.lognormal is None:
        lengthscale = gamma(
            prior.concentration, prior.rate, n, bank.n_lengthscales
        )
    else:
        loc, scale = prior.lognormal
        lengthscale = torch.exp(loc + scale * normal(n, bank.n_lengthscales))
    noise = torch.exp(NOISE_LOC + NOISE_SCALE * normal(n))
    # the floor is where the noise stops; a draw below it starts just above
    excess = (noise - data.floor).clamp_min(1e-8)
    parts = [
        inverse_softplus(lengthscale),
        inverse_softplus(excess)[:, None],
        normal(n, 1),
        bank.head_scale * normal(n, bank.n_head),
    ]
    if bank.n_reagents:
        weights = gamma(TERM_CONCENTRATION, TERM_RATE, n, bank.n_reagents)
        parts.append(inverse_softplus(weights))
    return torch.cat(parts, dim=1)


def value_and_grad(
    log_density: LogDensity, theta: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return a batch's log densities and gradients; NaN made -inf, 0."""
    with torch.enable_grad():
        leaf = theta.detach().requires_grad_(True)
        value = log_density(leaf)
        finite = torch.isfinite(value)
        total = torch.where(finite, value, torch.zeros_like(value)).sum()
        (grad,) = torch.autograd.grad(total, leaf)
    grad = torch.where(finite[:, None], torch.nan_to_num(grad), 0.0)
    value = torch.where(finite, value, -torch.inf)
    return value.detach(), grad


def hmc(
    log_density: LogDensity,
    theta: torch.Tensor,
    step_size: float,
    inverse_mass: torch.Tensor,
    leapfrog_steps: int,
    transitions: int,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    """HMC transitions of every particle at once.

    Parameters
    ----------
    log_density
        Log density of a batch of theta.
    theta
        Shape (P, dimension).
    step_size
        Leapfrog step size.
    inverse_mass
        Diagonal inverse mass matrix, shape (dimension,).
    leapfrog_steps, transitions
        Leapfrog steps per transition, transitions per particle.
    generator
        Source of the momenta and the acceptance draws.

    Returns
    -------
    theta
        After the transitions.
    acceptance
        Each particle's mean acceptance probability, (P,).

    """
    value, grad = value_and_grad(log_density, theta)
    root = inverse_mass.sqrt()
    accepted = torch.zeros(len(theta), dtype=F64, device=theta.device)
    for _ in range(transitions):
        noise = torch.randn(
            theta.shape, generator=generator, device=theta.device, dtype=F64
        )
        momentum = noise / root
        kinetic = 0.5 * (momentum**2 * inverse_mass).sum(-1)
        x, new_value, new_grad = theta, value, grad
        p = momentum + 0.5 * step_size * grad
        for step in range(leapfrog_steps):
            x = x + step_size * inverse_mass * p
            new_value, new_grad = value_and_grad(log_density, x)
            last = step == leapfrog_steps - 1
            p = p + (0.5 if last else 1.0) * step_size * new_grad
        new_kinetic = 0.5 * (p**2 * inverse_mass).sum(-1)
        log_ratio = (new_value - new_kinetic) - (value - kinetic)
        log_ratio = torch.nan_to_num(log_ratio, nan=-torch.inf)
        probability = torch.exp(log_ratio.clamp(max=0.0))
        uniform = torch.rand(
            len(theta), generator=generator, device=theta.device, dtype=F64
        )
        accept = uniform < probability
        theta = torch.where(accept[:, None], x, theta)
        value = torch.where(accept, new_value, value)
        grad = torch.where(accept[:, None], new_grad, grad)
        accepted = accepted + probability
    return theta, accepted / transitions


def systematic(
    weights: torch.Tensor, generator: torch.Generator
) -> torch.Tensor:
    """Systematic resampling: indices of n particles by their weights."""
    n = len(weights)
    start = torch.rand(
        1, generator=generator, device=weights.device, dtype=F64
    )
    positions = (start + torch.arange(n, device=weights.device)) / n
    cumulative = torch.cumsum(weights, dim=0)
    cumulative[-1] = 1.0
    return torch.searchsorted(cumulative, positions).clamp(max=n - 1)


def next_increment(
    increment: torch.Tensor, room: float, wanted: float
) -> float:
    """Return the largest step in lambda, at most `room`, keeping the ESS."""

    def ess(delta: float) -> float:
        weights = torch.softmax(delta * increment, dim=0)
        return float(1.0 / (weights**2).sum())

    if ess(room) >= wanted:
        return room
    low, high = 0.0, room
    for _ in range(50):
        middle = 0.5 * (low + high)
        if ess(middle) >= wanted:
            low = middle
        else:
            high = middle
    # at least a small step, so that a lambda stuck by -inf still moves
    return max(low, 1e-6 * room)


def temper(
    particles: torch.Tensor,
    log_base: LogDensity,
    log_target: LogDensity,
    settings: Settings,
    step_size: float,
    generator: torch.Generator,
) -> tuple[torch.Tensor, Health]:
    """Move equally weighted particles of `base` to `target`.

    Returns
    -------
    particles
        Equally weighted, from target.
    health
        Tempering steps, mean HMC acceptance, the final step size, the log
        of target's normalising constant over base's, and whether lambda
        reached 1.

    """
    n = len(particles)
    lam, log_z, count, acceptances = 0.0, 0.0, 0, 0.0

    def gap(theta: torch.Tensor) -> torch.Tensor:
        value = log_target(theta) - log_base(theta)
        return torch.where(torch.isfinite(value), value, -torch.inf)

    while lam < 1.0 and count < settings.max_tempering:
        with torch.no_grad():
            increment = gap(particles)
        delta = next_increment(increment, 1.0 - lam, settings.target_ess * n)
        log_weights = delta * increment
        log_z += float(torch.logsumexp(log_weights, 0)) - math.log(n)
        index = systematic(torch.softmax(log_weights, dim=0), generator)
        particles = particles[index]
        lam = min(lam + delta, 1.0)
        inverse_mass = particles.var(dim=0) + 1e-12
        current = lam

        def log_density(
            theta: torch.Tensor, lam: float = current
        ) -> torch.Tensor:
            # at lambda 1, where most BO steps land at once, the base is
            # not needed: half the cost
            if lam >= 1.0:
                return log_target(theta)
            base = log_base(theta)
            return base + lam * (log_target(theta) - base)

        particles, rates = hmc(
            log_density,
            particles,
            step_size,
            inverse_mass,
            settings.leapfrog_steps,
            settings.mcmc_steps,
            generator,
        )
        acceptance = float(torch.nan_to_num(rates).mean())
        step_size = float(
            np.clip(step_size * math.exp(acceptance - 0.65), 1e-3, 2.0)
        )
        acceptances += acceptance
        count += 1
    health = Health(
        count, acceptances / max(count, 1), step_size, log_z, lam >= 1.0
    )
    return particles, health


def first(
    bank: Bank, data: Data, settings: Settings, generator: torch.Generator
) -> tuple[torch.Tensor, Health]:
    """Particles of a campaign's first posterior, from the reference."""
    particles = draw_reference(bank, data, settings.particles, generator)
    return temper(
        particles,
        lambda theta: log_reference(theta, bank, data),
        lambda theta: log_posterior(theta, bank, data),
        settings,
        settings.step_size,
        generator,
    )


def advance(
    particles: torch.Tensor,
    bank: Bank,
    before: Data,
    after: Data,
    settings: Settings,
    step_size: float,
    generator: torch.Generator,
) -> tuple[torch.Tensor, Health]:
    """Particles of the next posterior, from the last one's."""
    return temper(
        particles,
        lambda theta: log_posterior(theta, bank, before),
        lambda theta: log_posterior(theta, bank, after),
        settings,
        step_size,
        generator,
    )
