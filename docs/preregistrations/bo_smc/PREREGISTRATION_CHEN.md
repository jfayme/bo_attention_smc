# Pre-registration: the head with the term, by SMC, under the chen rule (2026-09-30)

Written before any campaign on the seeds below ran for this arm. Its SHA-256 is logged in `runs/bo_smc_chen/preregistration.sha256`. A change after the first campaign starts is an amendment at the end, dated and hashed.

## Why

- **What earlier tests showed:**
  - The per-molecule term alone (`identity`, MAP) is the confirmed best model (`bo_per_molecule/PREREGISTRATION.md`).
  - It works best with the chen rule (`bo_per_molecule/PREREGISTRATION_CONFIG.md`: T5 with chen beat MACE with geom, +0.053, p = 0.016).
  - Under geom, the span head added to the term (`smc_identity_head`) did not beat the term alone (`PREREGISTRATION.md`: −0.023, Holm p = 0.34).
- **What was seen under chen** (exploratory, seeds 0–19): `smc_identity_head` − `identity` on lift was +0.007 on MACE (p = 0.97) and +0.028 on T5 (p = 0.56).

This test settles whether the head adds anything to the term under the rule the term is best with.

## Design

- **Arm:** `smc_identity_head` (bo_smc at the commit of this file), unchanged:
  - 256 particles, 5 HMC transitions of 10 leapfrog steps per tempering step, ESS kept at 0.5;
  - head weights N(0, 1), reagent weights Gamma(2, 4).
- **Cells:** bh_full, bh_1 and Shields under MACE and under T5 (`t5_augm`). That is 6 cells.
- **Rule:** chen; cv 0.3, `match_concentration`.
- **Seeds:** 40–59, which no head campaign has run.
- **Campaigns:** 5 random then 50 BO experiments. 6 cells × 20 seeds = 120 campaigns.
- **References:** already recorded on the same cells, rule and seeds (`runs/bo_per_molecule_config`, the configuration test):
  - `identity`: the term alone, MAP;
  - `null`.

  Their outcomes are known; this arm's are not.
- **Command:**

      python -m bo_smc.run --out runs/bo_smc_chen --featurisers mace t5_augm --rules chen --first-seed 40 --seeds 20 --arms smc_identity_head --workers 4 --pair-with runs/bo_per_molecule_config/campaigns.jsonl

## Hypothesis and decision rule

**Primary, H1.** `smc_identity_head` − `identity` on lift, over all 120 pairs, by a two-sided Wilcoxon signed-rank test at α = 0.05. The head *helps* if p < 0.05 and the mean difference is positive. It *hurts* if p < 0.05 and the mean is negative. Otherwise there is *no evidence either way*.

**Secondary** (unadjusted, descriptive):
- **S1:** the same contrast on top-5 % coverage.
- **S2:** per featuriser, T5 and MACE, 60 pairs each.
- **S3, no harm per dataset.** Over each dataset's 40 pairs, the head *does not hurt* if the lower end of the 95 % paired-bootstrap interval of lift is above −0.05.
- **S4:** `smc_identity_head` − `null`.

## Analysis

    python -m bo_smc.run --out runs/bo_smc_chen --report --pair-with runs/bo_per_molecule_config/campaigns.jsonl

- H1 is the `smc_identity_head - identity` "all" lift row of `report.md`, and S1 its coverage row.
- S2 is the per-cell rows, pooled per featuriser by bo_hmc's `report.contrast` on the records of that featuriser.
- S3 is the per-dataset lift rows.

No outcome is read before all 120 campaigns exist, except a failed campaign's error, which is fixed and rerun.

## Amendments

## Result (2026-09-30, all 120 campaigns, none failed; read after completion)

| test | lift difference | 95 % CI | Wilcoxon p | verdict |
|---|---|---|---|---|
| H1: `smc_identity_head` − `identity`, 120 pairs | −0.001 | −0.037 to +0.035 | 0.87 | no evidence either way |

- **S1:** coverage +0.004 (p = 0.25).
- **S2:** MACE +0.003 (p = 0.48), T5 −0.005 (p = 0.65).
- **S3:**
  - bh_1 +0.010 (−0.059, +0.079): harm not ruled out;
  - bh_full +0.054 (+0.006, +0.106): does not hurt;
  - **Shields −0.066 (−0.133, −0.007): hurts.**
- **S4:** against the null, +0.060 lift (p = 0.048) and +0.041 coverage (p = 0.0015). But Shields is −0.123 against the null, and there the term alone is also below the null under chen on these seeds.
- **Conclusion:** under both rules (this test, and `PREREGISTRATION.md` under geom), the span head adds nothing to the per-molecule term.
