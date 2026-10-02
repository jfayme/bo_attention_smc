# Pre-registration: T5 with the chen rule for the per-molecule term (2026-09-30)

Written before any campaign on the seeds below ran. Its SHA-256 is logged in `runs/bo_per_molecule_config/preregistration.sha256`. A change after the first campaign starts is an amendment at the end, dated and hashed.

## Why

The per-molecule term (`identity`, MAP) was confirmed on MACE and T5 under both rules (`PREREGISTRATION.md`), and it is the confirmed best model. Which featuriser and rule it works best with was not tested there. Descriptively, T5 with the chen rule beat MACE with the geom rule on both seed sets seen so far, by the same amount:

- seeds 0–19: +0.040 lift (p = 0.06), 60 pairs;
- seeds 20–39: +0.043 lift (p = 0.016), coverage +0.023 (p = 0.05), 60 pairs.

Both sets were seen before this test was designed, so it runs on new seeds.

## Design

- **Arms:** `identity` and `null` (bo_per_molecule, unchanged since the term's confirmation).
- **Featurisers:** MACE and T5 (`t5_augm`), the stored decorr0.7 attention cells.
- **Rules:** geom and chen; cv 0.3, `match_concentration`.
- **Datasets:** bh_full, bh_1 and Shields.
- **Seeds:** 40–59, which no campaign has run.
- **Campaigns:** 5 random then 50 BO experiments. 2 arms × 2 featurisers × 2 rules × 3 datasets × 20 seeds = 480 campaigns.
- **Command:**

      python -m bo_per_molecule.run --out runs/bo_per_molecule_config --featurisers mace t5_augm --rules geom chen --first-seed 40 --seeds 20 --arms null identity --workers 12

## Hypothesis and decision rule

Pairs are matched on dataset and seed; every pair starts from the same five reactions.

**Primary, H1.** `identity` on T5 with chen, minus `identity` on MACE with geom, on lift, over the 60 pairs. The test is a two-sided Wilcoxon signed-rank test at α = 0.05. T5 with chen is *better* if p < 0.05 and the mean difference is positive, *worse* if p < 0.05 and it is negative, and otherwise there is *no evidence either way*.

**Secondary** (unadjusted, descriptive):
- **S1:** the same contrast on top-5 % coverage.
- **S2:** per dataset (20 pairs each).
- **S3:** the other two configurations (MACE with chen, T5 with geom) against MACE with geom.
- **S4:** `identity` − `null` within each configuration, the term's own gain.

## Analysis

After all 480 campaigns exist:

    python -m bo_per_molecule.run --out runs/bo_per_molecule_config --report
    python -m bo_per_molecule.config_contrast runs/bo_per_molecule_config

`config_contrast` pairs `identity` across configurations by dataset and seed. It gives the mean, the 95 % paired-bootstrap interval (4000 resamples, seed 0) and the Wilcoxon p (bo_hmc's `wilcoxon_p`). No outcome is read before all 480 campaigns exist, except a failed campaign's error, which is fixed and rerun.

## Amendments

## Result (2026-09-30, all 480 campaigns, none failed; read after completion)

| contrast (`identity`, 60 pairs) | lift | 95 % CI | Wilcoxon p | wins | coverage |
|---|---|---|---|---|---|
| **H1: T5 + chen − MACE + geom** | **+0.053** | −0.012 to +0.112 | **0.016** | 38/59 | +0.019 (p = 0.029) |
| S3: MACE + chen − MACE + geom | +0.039 | −0.009 to +0.085 | 0.035 | 37/59 | +0.029 (p = 0.001) |
| S3: T5 + geom − MACE + geom | +0.021 | −0.037 to +0.081 | 0.45 | 33/60 | +0.001 |

- **Verdict:** H1 is *better* by the pre-registered rule: p < 0.05, positive mean. The bootstrap interval touches zero, so the effect is modest.
- **S2, per dataset:** bh_1 +0.12, Shields +0.055, bh_full −0.018.
- **S4, the term against the null in each configuration:**
  - MACE + geom: −0.011 (p = 0.65);
  - MACE + chen: +0.078 (p = 0.03);
  - T5 + geom: +0.026;
  - T5 + chen: +0.042.

  The term's gain varies between seed sets more than its confirmation suggested; it is positive in 3 of 4 configurations here.
