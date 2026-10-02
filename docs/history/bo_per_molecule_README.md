# bo_per_molecule: a per-molecule kernel term, a one-hot baseline, and best-case settings

A standalone experiment built on `bo_hmc`. It asks three questions about BO on the three MACE cells (bh_full, bh_1, Shields), each against bo_hmc's `null`:

1. **Does MACE's chemistry help?** `onehot` is the same GP on one-hot reagents. It knows which molecule each reaction uses and nothing about the molecules.
2. **Does a per-molecule term help?** `identity` adds one term per reagent to the null's kernel: k = Matérn(features) + Σ_c w_c · [same molecule of reagent c].
   - This lets the model learn each molecule's own effect, even where the features call two molecules alike (on bh_1 they put bromide closer to chloride than to iodide).
   - w_c is the reagent's weight. Its prior is Gamma(2, 4) and it is fitted with the rest by MAP.
3. **How far is BO from its model's best?** A `ceiling_*` arm fits a model once on the whole pool (at most 1,000 reactions), then runs BO with those settings held fixed.
   - The held settings are the yields' standardisation, the length-scales, noise, mean and weights, and for `ceiling_head` C3's PCA-8 span head at its posterior mode.
   - It cheats on purpose. It estimates the best each model could do if it learnt its settings perfectly.
   - It is not a strict bound: the settings that describe the whole pool best need not choose best.

## Running

    python -m bo_per_molecule.run --out runs/bo_per_molecule           # fit, run, report
    python -m bo_per_molecule.run --out runs/bo_per_molecule --report  # report only
    python -m unittest discover -s bo_per_molecule/tests -t .          # 8 tests, ~5 s

- **Morgan** (added 2026-09-30): `python -m bo_per_molecule.combine --fingerprint morgan` writes each dataset's Morgan cell with MACE's atoms attached, only to name each reaction's molecules for the term (it checks that both cells list the same reactions). `identity` then reads the Morgan features; `null` reads the plain Morgan cell (`FINGERPRINT`). There is no head on Morgan: it has no atoms.
- **Rules:** `--rules chen geom default`; `default` is bo_hmc's BoTorch LogNormal length-scale prior.
- **Design:** 3 datasets × 7 arms × seeds 0–19, geom rule, 5 random then 50 BO experiments.
- **Speed:** the refitted arms take minutes per campaign, the fixed ones seconds. The whole-pool head fit takes a few minutes per start.
- **Output:**
  - campaigns go to `<out>/campaigns.jsonl`, and a stopped run resumes;
  - the whole-pool fits are kept in `<out>/ceiling/`;
  - the report goes to `<out>/report.md`.
- **What the report adds:** C3's learned head (`hmc_span_pca8`, from `runs/fixes/benchmark07`) on the same seeds, and a check that `null` repeats C3's recorded nulls.

## How to read it

| contrast | if it is about zero | if it is clearly positive |
|---|---|---|
| `onehot` − `null` | MACE adds nothing over knowing the molecules' names | MACE's features mislead; the names do better |
| `identity` − `null` | the term is not the missing piece | own effects help: the features' similarity is part of the problem |
| `ceiling_X` − `X` | the model already learns about as well as it could: no better sampler, prior or head training will help X | learning from few data is the limit, and worth working on |
| `ceiling_head` − `ceiling_null` | even a perfectly trained head adds nothing on that dataset | the head has real value, if it could be learnt |

## What it reuses, and what is new

**From bo_hmc, called unchanged:**
- the stored cells, `null_features`, `make_prior` and `pool_bounds`;
- the null campaign itself: `null` and `onehot` run through `bo_hmc.runs.run.campaign_task`, the one-hot pool saved as a bo_hmc cell;
- LogEI, `SpanHead`, `Posterior` and `Prediction`;
- the metrics (`outcome`, `random_auc`) and the contrasts (`report.contrast`).

**New:**
- **The one-hot cell.**
- **The `SameMolecule` kernel term.**
- **`PoolStandardize`.** SingleTaskGP re-standardises its training yields; this keeps the pool's statistics fixed.
- **`build_gp` and `fit`.** bo_hmc's `fit_gp` builds and fits a fixed Matérn in one call, so it cannot hold the term or fixed settings. Without the term, `build_gp` + `fit` repeat `fit_gp` bit for bit; a test checks it.
- **The whole-pool fits.** A fit whose covariance breaks down on hundreds of reactions falls back to BoTorch's seeded restarts (status RETRIED). bo_hmc's `matern52` builds an (n × n × channels) tensor, so the head's whole-pool fit is capped at 1,000 reactions (about 4 GB each).
- **`run_with`.** bo_hmc's campaign loop, taking any model. With bo_hmc's null in it, it makes `run_campaign`'s choices exactly; a test checks it.

## Results

### Exploratory (`runs/bo_per_molecule`: MACE, geom, seeds 0–19)

- **Features vs one-hot.** MACE beats one-hot on bh_full (+0.19 lift) and Shields (+0.20). It loses on bh_1 (−0.24), where it calls bromide closer to chloride than to iodide.
- **The term.**
  - It helps on bh_1: +0.14 lift, +0.11 coverage.
  - It is neutral elsewhere.
  - It equals C3's HMC head on average (+0.002), at 0.5 min per campaign against 164.
- **Ceilings.** Settings fitted on 1,000 reactions do not beat the learned arms (null −0.002, head −0.016, term +0.044), so better learning of these models buys no lift.
- **Variants.** `identity_loose` and `identity_scaled` do not change lift (+0.009, +0.001).

### Confirmatory (`runs/bo_per_molecule_confirm`, pre-registered in PREREGISTRATION.md)

The design was fixed before any campaign ran: MACE and T5, geom and chen, seeds 20–39, 720 campaigns, none failed.

| test | result | verdict |
|---|---|---|
| H1: `identity` − `null`, lift, 240 pairs | +0.063 (95 % CI +0.034 to +0.091), p = 2.1e-5, wins 141/236 | **helps** |
| S1: the same on top-5 % coverage | +0.024, p = 9.2e-5 | helps |
| S2: no harm, per dataset (lower CI > −0.05) | bh_1 +0.092 (+0.044, +0.140); bh_full +0.086 (+0.036, +0.140); Shields +0.010 (−0.039, +0.059) | does not hurt on any |
| S3: `identity` − `onehot`, lift | +0.136, p = 1.8e-10 | better than no chemistry |

- **By rule and featuriser (descriptive):** +0.056 to +0.070 in each of the four combinations.
- **Mean lift, null → term:** bh_full 0.398 → 0.484; bh_1 0.073 → 0.165 (one-hot 0.173); Shields 0.556 → 0.567.
- **bh_1:** it motivated the term, so it is the least independent test. It shows the same direction on new seeds and on T5.

### Screens of the term's variants (2026-09-30, exploratory: seeds 0–19)

Each variant is compared with `identity`, the term alone by MAP, on the same featuriser, rule and seeds (60 pairs, lift; p unadjusted).

| variant | lift vs `identity` | note |
|---|---|---|
| `identity_pairs`: a weight per pair of reagents | −0.015 (p = 0.75) | bh_full −0.09 |
| `identity_additive`: each reagent's own Matérn | +0.004 (p = 0.73) | |
| `identity_cv05`: a wider length-scale prior | −0.032 (p = 0.10) | worse on all three datasets |
| `identity_long15`, `identity_long2`: chen's centre ×1.5, ×2 | MACE +0.008, +0.019; T5 −0.018, −0.010 | none significant |
| MACE + T5 combined features (`mace_t5`) | +0.004 (geom), +0.014 (chen) | |
| `identity_greedy`: the posterior mean alone | **−0.25** (p < 1e-4) | exploration matters |
| `identity_ucb1` / `identity_ucb` / `identity_ucb3` (UCB, β = 1, 2, 3; chen) | −0.05 to −0.11 / ≈ 0 / ≈ −0.01 | LogEI sits where β ≈ 2 does |

- **Only one change helps:** the rule and featuriser. T5 with chen beat MACE with geom on seeds 0–19 (+0.040) and 20–39 (+0.043).
- **It was confirmed** on seeds 40–59 (`PREREGISTRATION_CONFIG.md`): +0.053 lift, p = 0.016; coverage +0.019.

**Recommended configuration:** the per-molecule term, by MAP, on T5 (or MACE) features, with the chen rule and LogEI.
