# Pre-registration: the head with the per-molecule term, by SMC (2026-09-29)

Written before any campaign on the seeds below ran for this arm. Its SHA-256 is logged in `runs/bo_smc_confirm/preregistration.sha256`. A change after the first campaign starts is an amendment at the end, dated and hashed.

## Why

In the exploratory run (`runs/bo_smc`: MACE, geom, seeds 0–19), `smc_identity_head` had the highest mean lift so far: 0.462 / 0.247 / 0.480 on bh_full / bh_1 / Shields.
- The span head (PCA-8) with bo_per_molecule's per-molecule term, sampled by SMC on the GPU.
- Against the null: +0.106 lift (p = 0.007).
- Against the term alone (`identity`): +0.058 (p = 0.015), positive on all three datasets.

The term alone was confirmed on its own design (`bo_per_molecule/PREREGISTRATION.md`). This tests the combination on new seeds and on T5.

## Design

- **Arm:** `smc_identity_head` (bo_smc, commit of this file), unchanged:
  - 256 particles, 5 HMC transitions of 10 leapfrog steps per tempering step, ESS kept at 0.5;
  - head weights N(0, 1), reagent weights Gamma(2, 4).
- **Cells:** bh_full, bh_1 and Shields under MACE and under T5 (`t5_augm`), the stored decorr0.7 attention cells in `runs/bo_hmc/cells`. That is 6 cells.
- **Rule:** geom; cv 0.3, `match_concentration`.
- **Seeds:** 20–39, which no head campaign has run.
- **Campaigns:** 5 random then 50 BO experiments. 6 cells × 20 seeds = 120 campaigns.
- **References**, already recorded on the same cells, rule and seeds (`runs/bo_per_molecule_confirm`, the per-molecule confirmation):
  - `null`: mean pooling;
  - `identity`: the term alone;
  - `onehot`.

  Their outcomes are known; the new arm's are not. Every pair starts from the same five reactions.
- **Command:**

      python -m bo_smc.run --out runs/bo_smc_confirm --featurisers mace t5_augm --rules geom --first-seed 20 --seeds 20 --arms smc_identity_head --workers 4 --pair-with runs/bo_per_molecule_confirm/campaigns.jsonl

## Hypotheses and decision rules

Lift is each campaign's area under its best-so-far curve, rescaled so that random selection scores 0 and the ideal 1. A positive mean lift therefore beats random sampling.

**Primary, H1.** `smc_identity_head` − `null` (mean pooling) on lift, over all 120 pairs, by a two-sided Wilcoxon signed-rank test at α = 0.05. The combination *helps* if p < 0.05 and the mean difference is positive. It *hurts* if p < 0.05 and the mean is negative. Otherwise there is *no evidence either way*.

**Secondary** (reported with unadjusted p, no claims beyond the rule stated):
- **S1:** the same contrast on top-5 % coverage.
- **S2, no harm per dataset.** For each dataset, over its 40 pairs (both featurisers), the combination *does not hurt* if the lower end of the 95 % paired-bootstrap interval of `smc_identity_head` − `null` lift is above −0.05.
- **S3, does the head add to the term?** `smc_identity_head` − `identity` on lift, over all 120 pairs.
- **S4:** mean lift of every arm per cell (against random), and every contrast per cell and per dataset, descriptive.

**Reported alongside:**
- MACE and T5 separately;
- the sampler's health: tempering steps, acceptance, cells where λ did not reach 1.

## Analysis

    python -m bo_smc.run --out runs/bo_smc_confirm --report --pair-with runs/bo_per_molecule_confirm/campaigns.jsonl

This writes `report.md`, with each contrast per cell, per dataset (both featurisers) and over all campaigns: pairs, mean, 95 % paired bootstrap interval, Wilcoxon p and wins, by bo_hmc's `report.contrast`.
- H1 is the `smc_identity_head - null` "all" lift row, and S1 its coverage row.
- S2 is the per-dataset lift rows' intervals.
- S3 is the `smc_identity_head - identity` "all" lift row.

No outcome is read before all 120 campaigns exist, except a failed campaign's error, which is fixed and rerun. The smoke run before this file (one seed, two BO steps, in a scratch folder) was read only for errors.

## Amendments

**Amendment 1** (2026-09-29; before any campaign of this design ran, no outcome seen).
- *Co-primary contrasts.* At the user's request, the comparison with the term alone becomes co-primary.
  - H1: `smc_identity_head` − `null` on lift.
  - H2: `smc_identity_head` − `identity` on lift, which was S3.
  - Each is over all 120 pairs, by a two-sided Wilcoxon signed-rank test, with Holm–Bonferroni over the two at family-wise α = 0.05 (`bo_hmc.diagnostics_and_metrics.report.holm` on the report's two "all" lift p-values).
  - Each is decided as H1 was: *helps* if its adjusted p < 0.05 and the mean is positive, *hurts* if negative, otherwise *no evidence either way*.
  - S1, S2 and S4 are unchanged; S3 is now H2.
- *A separate exploratory run first.* `smc_span_pca8_tight` (the span head alone, every head weight N(0, 0.5²)) runs on MACE, geom, seeds 0–19 into `runs/bo_smc`, paired with the recorded `smc_span_pca8` and C3's NUTS `hmc_span_pca8`. It asks whether a more cautious head recovers NUTS's lift. It does not change this design. If it wins, it needs a confirmation of its own.

## Result (2026-09-29, all 120 campaigns, none failed; read after completion)

| test | lift difference, 120 pairs | 95 % CI | raw p | Holm p | verdict |
|---|---|---|---|---|---|
| H1: `smc_identity_head` − `null` | +0.036 | −0.011 to +0.083 | 0.043 | 0.087 | no evidence either way |
| H2: `smc_identity_head` − `identity` | −0.023 | −0.062 to +0.015 | 0.34 | 0.34 | no evidence either way |

- **S1:** coverage +0.005 against the null, not significant.
- **S2:** bh_1 +0.092 (+0.030, +0.156) does not hurt; bh_full +0.043 (−0.033, +0.116) does not hurt; **Shields −0.026 (−0.123, +0.070): harm not ruled out**.
- **Against the term alone, per dataset:** Shields −0.088 (p = 0.014), bh_1 +0.031, bh_full −0.014.
- **On the same 120 pairs (descriptive):** the term alone (`identity`, MAP) is +0.060 over the null (p = 0.002), about +0.06 in each dataset.
- **Every arm beats random:** mean lift > 0 in every cell.
- **The exploratory gain did not replicate.** The +0.058 over the term alone on seeds 0–19 was the best of several arms on the same seeds.
