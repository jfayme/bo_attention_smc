# bo_smc: SMC in place of NUTS for the head posterior, on the GPU

A standalone package on `bo_hmc` and `bo_per_molecule`. It samples the same posterior as bo_hmc's `hmc_span_pca8` with sequential Monte Carlo (SMC) instead of NUTS. It can also sample that posterior with the per-molecule term added (`identity_head`).

## How it works

- **The model (`model.py`).** bo_hmc's `Posterior`, written for a batch of particles, in torch.
  - Its log posterior, gradients and predictions equal bo_hmc's, particle by particle: MACE with conformers, MACE with numeric settings, T5 tokens, and the per-molecule term (`tests/test_model.py`).
  - For speed, three things differ, none changing a value:
    - the measured reactions are padded to a fixed number;
    - kernels come from inner products;
    - predictions cover the whole pool.
- **The sampler (`smc.py`).** A population of particles is carried from one BO step's posterior to the next (IBIS, Chopin 2002).
  - Each move is adaptive tempering: bisection on the ESS, systematic resampling, and batched HMC with the particles' variance as inverse mass matrix.
  - The first step starts from the priors.
  - A Gaussian test checks moments and evidence.
- **The campaigns (`run.py`).** bo_per_molecule's `run_with` loop, the posterior's constants read from bo_hmc at every step, and LogEI through bo_hmc's `Prediction`.
- **`blackjax_cpu/`.** The first version, with BlackJAX's kernels on CPU JAX. It is tested against the same targets, but JAX has no GPU on native Windows, and it took 30–60 s per BO step.

## Running

    python -m bo_smc.run --out runs/bo_smc --workers 4     # GPU, 4 campaigns at once
    python -m bo_smc.run --out runs/bo_smc --report
    python -m unittest discover -s bo_smc/tests -t .       # 12 tests

- **Speed** (RTX 5090 laptop, float64, 256 particles):
  - about 1.2 s per tempering step alone, 1.1–1.2 tempering steps per BO step;
  - 2–3 minutes per campaign with 4 at once, against 125–217 for NUTS.
- **Two optimisations that mattered:**
  - the training kernel from inner products (the (P, n, n, D) array made the GPU compute-bound);
  - skipping the previous posterior when a step reaches λ = 1 at once.

`--rules` takes `chen`, `geom` or `default` (added 2026-09-30: bo_hmc's BoTorch LogNormal length-scale prior; the sampler's first draws and prior density follow it, `tests/test_model.py::test_default_rule`).

## Results (`runs/bo_smc`: MACE, geom, seeds 0–19, exploratory)

**SMC against NUTS, same model** (paired with C3's `hmc_span_pca8`):
- Lift is −0.039 (95 % CI −0.080 to −0.001; Wilcoxon p = 0.16; 25/60). Coverage is +0.002.
- It reproduces NUTS's pattern against the null: bh_1 +, bh_full ≈ 0, Shields −.

**The head with the term (`smc_identity_head`), mean lift:**

| | bh_full | bh_1 | Shields | average |
|---|---|---|---|---|
| null | 0.415 | 0.062 | 0.393 | 0.290 |
| identity (term, MAP) | 0.411 | 0.203 | 0.400 | 0.338 |
| hmc_span_pca8 (C3, NUTS) | 0.464 | 0.220 | 0.323 | 0.336 |
| smc_span_pca8 | 0.426 | 0.174 | 0.290 | 0.297 |
| **smc_identity_head** | **0.462** | **0.247** | **0.480** | **0.396** |

**Its contrasts on lift, over 60 pairs:**
- − identity: +0.058 (p = 0.015; positive in all three datasets);
- − smc_span_pca8: +0.100 (p = 0.015; Shields +0.19, p = 0.0007);
- − C3's head: +0.061 (p = 0.086);
- − null: +0.106 (p = 0.007).

Coverage barely moves. These are exploratory: one rule, MACE only, seeds the term was designed on, and unadjusted p.

## Sampler accuracy (`check_sampler.py`, `runs/bo_smc_check`)

At one state per dataset (C3's NUTS campaign, seed 0, cut at 30 reactions), each sampler's LogEI over the candidates is compared with a heavy reference: SMC from the priors, 1024 particles, ESS kept at 80 %, 10 moves.

| sampler | Spearman ρ (bh_1 / bh_full / Shields) | top-10 overlap |
|---|---|---|
| second reference run: the noise floor | 0.995 / 0.998 / 0.991 | 0.7 / 0.9 / 0.9 |
| SMC as run (256 particles, carried from n = 5) | 0.987 / 0.995 / 0.984 | 0.8 / 0.9 / 0.9 |
| SMC, 4× HMC moves | 0.989 / 0.995 / 0.986 | 0.9 / 0.9 / 0.9 |
| SMC, 1024 particles | 0.995 / 0.998 / 0.993 | 0.8 / 0.9 / 0.9 |
| NUTS as run (s500, 128 draws) | 0.912 / 0.945 / 0.917 | 0.7 / 0.5 / 0.6 |

- **SMC is close to the reference's own noise.** 1024 particles reach it; more moves hardly help.
- **NUTS is clearly further from the posterior it targets.** This fits P0.1: its trees hit the depth cap every iteration, and repeat chains disagree on the next pick.
- **So SMC's −0.04 lift against NUTS is not SMC sampling badly.** NUTS's approximation, which stays near its start at mean pooling, happens to pick about as well or slightly better.
- **Limits:** one seed and one n per dataset. The reference is SMC too, though a fresh one, not carried, and it agrees with the carried runs.

## Confirmation (`runs/bo_smc_confirm`, pre-registered in PREREGISTRATION.md)

MACE and T5, geom, seeds 20–39, 120 campaigns.

- **Against mean pooling:** +0.036 lift, Holm p = 0.087.
- **Against the term alone:** −0.023, Holm p = 0.34.
- **Shields:** the head still hurts there, −0.088 against the term alone.

Neither co-primary contrast holds. The exploratory advantage did not replicate, and the confirmed best model remains the per-molecule term alone (`bo_per_molecule`, MAP).

## Under the chen rule (`runs/bo_smc_chen`, pre-registered in PREREGISTRATION_CHEN.md)

MACE and T5, seeds 40–59.

- **Against the term alone:** −0.001 lift (p = 0.87).
- **Shields:** −0.066 against the term alone.

With the geom test, this is the second time the head fails to add anything to the per-molecule term.

**Exploratory screens of the head with the term** (seeds 0–19):
- a tight head prior (sd 0.5): −0.030 under geom, −0.020 under chen T5, against the default prior;
- under chen: MACE +0.007, T5 +0.028 against the term alone, neither significant.
