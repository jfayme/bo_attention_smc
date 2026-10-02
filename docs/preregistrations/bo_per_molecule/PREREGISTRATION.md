# Pre-registration: confirming the per-molecule term (2026-09-29)

Written before any campaign on the seeds below ran. Its SHA-256 is logged in `runs/bo_per_molecule_confirm/preregistration.sha256`. Any change after the first campaign starts is an amendment at the end, dated and hashed.

## Why

The per-molecule term (`identity`: the null's Matérn plus w_c · [same molecule of reagent c] for each reagent, w_c ~ Gamma(2, 4), fitted by MAP) was designed after bh_1 showed MACE features calling bromide closer to chloride than to iodide. Its first results are exploratory, from `runs/bo_per_molecule`: MACE only, geom only, seeds 0–19.
- identity − null lift was +0.14 on bh_1 and about zero on bh_full and Shields; +0.048 pooled (p = 0.48).
- identity equalled C3's HMC head on average (+0.002).

Two variants of the weight prior (`identity_loose`, `identity_scaled`) did not change lift (+0.009 and +0.001 pooled). The term is therefore confirmed as first designed, with nothing changed.

## Design

- **Cells:** bh_full, bh_1 and Shields under MACE and under T5 (`t5_augm`), the stored decorr0.7 attention cells in `runs/bo_hmc/cells`. That is 6 cells. T5 has never been run with the term.
- **Rules:** geom and chen, each centring the length-scale prior; cv 0.3, `match_concentration`.
- **Seeds:** 20–39, which no campaign of this package has run.
- **Campaigns:** 5 random then 50 BO experiments.
- **Arms:**
  - `null`: bo_hmc's null campaign, unchanged;
  - `onehot`: the null GP on one-hot reagents;
  - `identity`: as above, unchanged.
- **Scale:** 6 cells × 2 rules × 20 seeds × 3 arms = 720 campaigns.
- **Command:**

      python -m bo_per_molecule.run --out runs/bo_per_molecule_confirm --featurisers mace t5_augm --rules geom chen --first-seed 20 --seeds 20 --arms null onehot identity

## Hypotheses and decision rules

Pairs are matched on (cell, rule, seed); every arm of a pair starts from the same five reactions.

**Primary, H1.** `identity` − `null` on lift, over all 240 pairs, by a two-sided Wilcoxon signed-rank test at α = 0.05. The term *helps* if p < 0.05 and the mean difference is positive. It *hurts* if p < 0.05 and the mean is negative. Otherwise there is *no evidence either way*.

**Secondary** (reported with unadjusted p, no claims beyond the rule stated):
- **S1:** the same contrast on top-5 % coverage.
- **S2, no harm per dataset.** For each dataset, over its 80 pairs (both featurisers, both rules), the term *does not hurt* if the lower end of the 95 % paired-bootstrap interval of `identity` − `null` lift is above −0.05.
- **S3:** `identity` − `onehot` on lift, over all 240 pairs.
- **S4:** every contrast per cell and per dataset, descriptive.

**Reported alongside:**
- the bh_1 cells separately, since bh_1 motivated the term and so is the least independent test;
- whether the result differs between MACE and T5.

## Analysis

`python -m bo_per_molecule.run --out runs/bo_per_molecule_confirm --report` writes `report.md`. It gives each contrast per cell, per dataset (both featurisers) and over all campaigns: pairs, mean, 95 % paired bootstrap interval, Wilcoxon p and wins, computed by bo_hmc's `report.contrast`. H1 is its `identity - null` "all" lift row, S1 the coverage row, S2 the per-dataset lift rows' intervals, and S3 the `identity - onehot` "all" lift row.

No outcome is read before all 720 campaigns exist, except a failed campaign's error, which is fixed and rerun.

## Amendments
