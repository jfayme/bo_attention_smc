# Pre-registration: the term sampled by SMC against the term by MAP, under chen (2026-09-30)

Written before any campaign of this arm ran under chen. Its SHA-256 is logged in `runs/bo_smc_term/preregistration.sha256`. A change after the first campaign starts is an amendment at the end, dated and hashed.

## Why

- **The goal has changed.** Compute is no constraint (a reaction takes far longer than a fit). What matters is the quality of the predictions and of the chemical insight, balanced against BO performance.
- **The per-molecule term by MAP** (`identity`) is the confirmed best model for BO (`bo_per_molecule/PREREGISTRATION.md`), and it works best with the chen rule (`bo_per_molecule/PREREGISTRATION_CONFIG.md`).
- **What was seen with the term sampled by SMC** (`smc_identity`: the term with the head held at mean pooling), exploratory, MACE, geom, seeds 0–19, 60 pairs against `identity`:
  - lift +0.023 (p = 0.47), top-5 % coverage +0.015 (p = 0.05); Shields lift +0.082 (p = 0.06);
  - badly overconfident predictions (log score of the next measured yield below −10) were rarer: 0.7 % of steps against 1.8 %; the worst was −689 against −2,568.
- **Across all recorded campaigns,** MAP arms made such predictions on 1.8–4.8 % of steps and SMC arms on 0.5–2.3 %.

This test asks whether sampling the term makes the model's predictions more trustworthy without costing BO performance, under the rule the term is best with.

## Design

- **Arm:** `smc_identity` (bo_smc as of this file; the working tree's changes add the `default` rule and leave chen's path unchanged, checked by rerunning recorded chen campaigns choice for choice):
  - 256 particles, 5 HMC transitions of 10 leapfrog steps per tempering step, ESS kept at 0.5;
  - head weights N(0, 0.001²), so the molecules are mean-pooled; reagent weights Gamma(2, 4); the length-scale prior as `identity`'s.
- **Cells:** bh_full, bh_1 and Shields under MACE and under T5 (`t5_augm`): 6 cells. (Morgan has no atoms, which bo_smc's model needs.)
- **Rule:** chen; cv 0.3, `match_concentration`.
- **Seeds:** 20–59, confirmatory: 6 cells × 40 seeds = 240 campaigns. Seeds 0–19 (120 campaigns) run afterwards and are reported separately as exploratory.
- **Campaigns:** 5 random then 50 BO experiments.
- **Reference:** `identity` (MAP), recorded on the same cells, rule and seeds: `runs/bo_per_molecule_confirm` (seeds 20–39) and `runs/bo_per_molecule_config` (40–59); for seeds 0–19, `runs/bo_per_molecule`.
- **Commands:**

      python -m bo_smc.run --out runs/bo_smc_term --featurisers mace t5_augm --rules chen --first-seed 20 --seeds 40 --arms smc_identity --workers 6
      python -m bo_smc.run --out runs/bo_smc_term --featurisers mace t5_augm --rules chen --first-seed 0 --seeds 20 --arms smc_identity --workers 6

## Measures

- **Overconfidence share:** per campaign, the share of its 50 BO steps whose prequential log score (the model's log predictive density of the yield it then measured, `fits.log_score`) is below −10.
- **Lift** and **top-5 % coverage:** as recorded (`metrics`).

## Hypotheses and decision rule

All on the 240 confirmatory pairs (same cell, rule and seed), `smc_identity` − `identity`.

- **H1, calibration.** Overconfidence share, two-sided Wilcoxon signed-rank test at α = 0.05. SMC is *better calibrated* if p < 0.05 and the mean difference is negative.
- **H2, BO non-inferiority.** Mean lift difference with a 95 % paired-bootstrap interval (10,000 resamples of pairs, numpy seed 0). SMC is *non-inferior* if the lower end is above −0.03.
- **Decision.** Adopt the SMC term inside the BO loop if H2 holds and either H1 shows better calibration or lift is superior (Wilcoxon p < 0.05, mean positive). Otherwise keep the MAP term in the loop and use SMC only after the campaign.

**Secondary** (unadjusted, descriptive):
- **S1:** lift, Wilcoxon (superiority).
- **S2:** top-5 % coverage.
- **S3:** per dataset (80 pairs each). Stated expectation from the exploratory screen: a lift gain on Shields.
- **S4:** per featuriser (120 pairs each).
- **S5:** the median prequential log score per campaign. It is confounded by which reactions each arm chooses to measure, so it is not a clean measure of prediction quality.

## Analysis

    python -m bo_smc.term_test

It prints H1, H2 and S1–S5 for seeds 20–59, then the same for seeds 0–19 (exploratory). No outcome is read before all 240 confirmatory campaigns exist, except a failed campaign's error, which is fixed and rerun.

## Amendments

## Result (2026-10-01, all 240 confirmatory campaigns, none failed; read after completion)

`smc_identity` − `identity`, chen, seeds 20–59, 240 pairs (`python -m bo_smc.term_test`):

| test | difference | 95 % CI | Wilcoxon p | verdict |
|---|---|---|---|---|
| H1: overconfidence share | −0.028 (1.1 % against 4.0 % of steps) | −0.031 to −0.025 | 1e-31 | SMC better calibrated |
| H2: lift | −0.008 (0.389 against 0.398) | −0.0302 to +0.014 | 0.85 | non-inferiority not shown (lower end −0.0302, margin −0.03) |

**Decision, by the rule:** H2 fails, so the MAP term stays in the BO loop and SMC is used after the campaign. The miss is by 0.0002, and there is no evidence of a lift difference either way (S1, p = 0.85).

Secondary (unadjusted): S2 coverage −0.004 (p = 0.08). S3 per dataset: bh_full −0.006 (p = 0.80), bh_1 +0.035 (p = 0.044), Shields −0.054 (p = 0.015), the opposite of the expected Shields gain. S4: MACE −0.019 (p = 0.44), T5 +0.003 (p = 0.66). S5 median log score −0.095 (p = 6e-15): SMC's typical predictions are a little wider, its failures far rarer (overconfident steps: bh_full 0.8 % against 3.8 %, bh_1 1.2 % against 3.4 %, Shields 1.4 % against 4.7 %).

**Exploratory, seeds 0–19** (120 pairs, run after the confirmatory set, not part of the decision): overconfidence share −0.029 (p = 9e-18); lift +0.009 (95 % CI −0.023 to +0.042, p = 0.96); coverage −0.004 (p = 0.24); Shields −0.012 (p = 0.46), bh_1 +0.041, bh_full −0.003. Over all 360 pairs the lift difference is −0.003.
