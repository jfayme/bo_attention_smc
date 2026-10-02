# Consolidation into bo_attention_smc (2026-10-01)

`bo_hmc/`, `bo_per_molecule/` and `bo_smc/`, with the scripts of the
git-ignored `runs/fill/`, `runs/chemistry/` and `runs/compress/`, became one
package, `bo_attention_smc/`, following `consolidation_handover.md` (in this
folder) and the user's answers to its questions. SMC replaced NUTS as the
only sampler; MAP stayed.

## Where each piece went

| was | is |
|---|---|
| `bo_hmc/core/cells.py` | `core/cells.py`, unchanged but for its folders (`data/`, `conformers/` written in, not read from `settings.py`), standard logging, and `molecules()` |
| `bo_hmc/core/campaign.py`: rules, priors, `fit_gp` | `core/gp.py`: one builder, `build_gp`, with or without the term, and `fit` |
| `bo_per_molecule/run.py`: `SameMolecule`, `build_gp`, `fit`, `make_pool`, `run_with` | `core/gp.py`, `core/campaign.py` (`make_pool`, `run`) |
| `bo_hmc/core/hmc.py`: `Prediction` | `core/gp.py` |
| `bo_smc/model.py` | `core/posterior.py`, built from a cell, not read from bo_hmc's objects |
| `bo_smc/smc.py` | `core/smc.py` |
| `bo_smc/run.py`: `SMCStep`, `Mixture`, arms | `core/campaign.py`: `SMCStep`, `Sampled`, `MODELS` |
| `bo_hmc/core/hmc.py`: `Posterior`, `SpanHead`; `bo_per_molecule`'s `IdentityPosterior` | `tests/reference.py`, one class, NUTS removed: the plain version the fast one is tested against |
| `bo_hmc/diagnostics_and_metrics/metrics.py` | `diagnostics_and_metrics/metrics.py` |
| `bo_hmc` report contrasts; the report writers of `bo_per_molecule` and `bo_smc`; `runs/fill/table.py` | `diagnostics_and_metrics/report.py` (`run`, `table`) |
| `bo_hmc/runs/prepare.py` | `runs/prepare.py`, without `--convert` |
| `bo_hmc/runs/run.py`, `bo_per_molecule.run`, `bo_smc.run` (runners) | `runs/run.py` |
| `runs/fill/driver.py`, `start.cmd`, `stop.cmd` | `runs/queue.py`, `start.cmd`, `stop.cmd` |
| `runs/chemistry/predict.py`, `predict_summary.py` | `evaluation/predict.py` |
| `runs/chemistry/labels.py`, `relearn.py`, `prior_attention.py`, `summary.py`, `attention_summary.py`, `t5_head.py` | `evaluation/chemistry.py` |
| `runs/compress/diagnose.py` | `evaluation/representation.py` |
| the three packages' tests | `tests/` |
| `PREREGISTRATION*.md` | `docs/preregistrations/`, unchanged |
| the three READMEs (with their results) | `docs/history/`, unchanged |

## Changed on purpose

- **One record format.** The `null` model's fits are now named as every
  other model's (`log_score`, `z`, `noise`, `ell_ratio`); the package
  README gives the old names.
- **Molecules from the dataset.** The term reads each reaction's molecules
  from the dataset (`cells.molecules`), numbered as an attention cell's
  atoms number them: identical on every stored cell. Morgan no longer needs
  a cell combined with MACE's atoms.
- **Root modules.** The package imports only `conformer.py` and
  `featuriser.py` from the root. So its folders are written into `cells.py`,
  `prepare.py` logs to the console, and `prepare.py --convert`, which
  unpickled the study's cells through `reduce.py`, `attention.py` and
  `dataset.py`, is gone (the converted cells are in `runs/bo_hmc/cells`).
- **A head means SMC.** A model with a head is sampled, one without is
  fitted by MAP; SMC without a head is not offered (`smc_identity` keeps
  its head held at mean pooling, as its pre-registered test ran it).
- **The queue** runs modules only, logs beside its queue file, and rebuilds
  the table there.
- **The fill queue was cleaned** at the user's request: the unfinished
  blocks of the head alone (seeds 40–59, and the rule `default`) were
  removed from `runs/fill/queue_smc.txt`, and will not be run.

## Dropped

All of it is in git history, in the commit before the consolidation, except
what the next paragraph names.

- NUTS: `hmc.py`'s sampler and schedules, and the arms `hmc_null`,
  `hmc_span`, `hmc_span_pca8`, `identity_head`.
- One-hot (`onehot`), `combine.py` and its MACE+T5 cells.
- The ceilings (`ceiling_*`) and `PoolStandardize`.
- The term's variants: loose, scaled, pairs, additive, UCB (β 1, 2, 3),
  greedy, cv 0.5, longer length-scales; the tight head prior (`*_tight`);
  SMC's additive and UCB arms.
- The prior options `cv` and `match_parameterisation`: cv 0.3 under
  `match_concentration` is the only width.
- The BlackJAX CPU port (`bo_smc/blackjax_cpu/`).
- The analysis scripts of past pre-registrations: `bo_smc/term_test.py`,
  `bo_smc/summary.py`, `bo_smc/check_sampler.py`,
  `bo_per_molecule/config_contrast.py`.
- C3's report and the sweep's report (`bo_hmc/diagnostics_and_metrics/
  report.py`).

**Never committed.** The old packages' last changes were never committed
before they were deleted, at the user's choice: the rule `default`
(bo_hmc, bo_per_molecule, bo_smc), Morgan for the term
(`combine.py --fingerprint`), the head on every channel (`smc_span`), and
`bo_smc/term_test.py` with `PREREGISTRATION_TERM.md` (the latter kept in
`docs/preregistrations/`). The campaigns they recorded are reproduced by
this package (below). The scripts of `runs/` stay where they were,
untracked, and no longer run.

## Validation (2026-10-01)

On the RTX 5090 laptop, one CPU thread per campaign, as the campaigns were
recorded.

- **Tests:** the golden numbers of bo_fixes (the null campaigns, and the
  reference posterior at fixed vectors, `learned/hmc_null` now included);
  the fast posterior against the reference, particle for particle, on CPU
  and GPU; the term's GP against the reference; cells rebuilt from the
  data; molecules from the dataset against every stored attention cell.
- **Metrics:** lift, AUC, coverage, regret, best yield and first top-5 %
  hit recomputed from `sampled_indices` for all 8,820 recorded campaigns of
  12 run folders: all exact.
- **Full replays, 55 experiments each:** 32 MAP campaigns (null and term;
  chen, geom and default; MACE, T5 and Morgan) and 6 SMC campaigns on the
  GPU (`smc_identity_head` under chen and default, `smc_identity`,
  `smc_span_pca8` under geom and chen, `smc_span`): every choice, every
  log score and every metric as recorded.
- **Evaluation:** 66 fits of `runs/chemistry/predict.jsonl` (tests A, B
  and C; 36 MAP, 30 SMC) exactly; 24 MAP refits of `map.jsonl` exactly; the
  9 seed-0 SMC refits of `smc.jsonl` exactly at the thread count they were
  computed with (12), and different at one thread; the T5 head's
  `smc_t5.jsonl` exactly (with an empty `weights` field added);
  `prior_attention.json` exactly; the summaries, the representation
  diagnostic and the results table as the old scripts printed them.
