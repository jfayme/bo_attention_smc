# bo_attention_smc

Bayesian optimisation (BO) of reaction yields with a Gaussian process (GP),
on the study's three datasets (bh_full, bh_1: Buchwald–Hartwig; Shields:
direct arylation), each molecule described by MACE atoms, T5 tokens
(GT4SD `multitask-text-and-chemistry-t5-base-augm`, called `t5_augm`) or a
Morgan fingerprint. A model is fitted by MAP, or sampled by sequential
Monte Carlo (SMC) on the GPU when it has an attention head.

## The models

Each model is a combination of parts, named by the labels the study's
records use:

| label | kernel | head | inference |
|---|---|---|---|
| `null` | Matérn | none: mean pooling | MAP |
| `identity` | Matérn + per-molecule term | none | MAP |
| `smc_span_pca8` | Matérn | PCA-8 | SMC |
| `smc_span` | Matérn | every channel | SMC |
| `smc_identity_head` | Matérn + term | PCA-8 | SMC |
| `smc_identity` | Matérn + term | PCA-8 held at mean pooling (prior sd 0.001) | SMC |

- **Kernel.** Matérn-5/2 with one length-scale per column (ARD). The
  per-molecule term adds Σ_c w_c · [same molecule of reagent c], one
  weight per reagent, so the GP can learn each molecule's own effect.
- **Head.** Weights each molecule's atoms (MACE) or tokens (T5) instead of
  averaging them, within the span of the mean-pooled features. A head needs
  atoms, so there is none on Morgan.
- **Inference.** MAP: one L-BFGS-B fit at every BO step. SMC: 256
  particles carried from one BO step to the next, LogEI averaged over them.
- **Rule** (per campaign): the length-scale prior's centre, `chen`
  (0.4 √D + 4), `geom` (the pool's mean distance) or `default` (BoTorch's
  LogNormal).

The recommended configuration, from the pre-registered tests: the term by
MAP (`identity`), chen, T5 or MACE, LogEI.

## What each module does

**core**: the model

- `core/conformer.py` finds a molecule's conformers: RDKit embedding,
  force-field optimisation, clustering of the survivors, then relaxation
  with the MACE-MH1 potential. An ensemble is saved and read back, so the
  search, by far the slowest step, runs once per molecule.
- `core/featuriser.py` turns molecules into what a GP is given: Morgan
  fingerprints, T5 embeddings, and MACE or AIMNet2 descriptors of each atom,
  collapsed over atoms and conformers (kept, under attention, for the head
  to pool).
- `core/cells.py` reads a dataset, describes its molecules through
  `featuriser.py` (and `conformer.py` for MLIP atoms), and keeps the
  columns that matter: a *cell*. `molecules()` numbers each reaction's
  molecules for the term, from the dataset.
- `core/gp.py` holds the GP: the length-scale rules and priors, the GP with
  or without the term, its MAP fit, and `Prediction`, a mixture of
  Gaussians with its log density and LogEI.
- `core/posterior.py` holds what SMC samples: the head, and the GP's log
  posterior and predictions for a whole population of particles at once.
- `core/smc.py` is the sampler: adaptive tempering, systematic resampling
  and HMC moves.
- `core/campaign.py` names the models, and runs one campaign: 5 random
  reactions, then 50 chosen one at a time by LogEI.

**evaluation**: what a model knows, beyond BO lift

- `evaluation/predict.py`: prediction quality on common data (test A), and
  on molecules held out of BO data (B) or of random data (C).
- `evaluation/chemistry.py`: models fitted again on recorded campaigns,
  checked against known chemistry (molecule rankings, halide order, best
  ligands), and their attention against a random-head control.
- `evaluation/representation.py`: is the chemistry in a molecule's
  description at all? Leave-one-out prediction of each molecule's mean
  yield from the others.

**diagnostics_and_metrics**: what campaigns achieved

- `diagnostics_and_metrics/metrics.py` scores a campaign: its lift over
  random selection, AUC, top-5 % coverage, simple regret.
- `diagnostics_and_metrics/report.py` writes a run folder's report (means
  and paired contrasts) and the table of every model, rule, featuriser and
  seed set.

**runs**: the command lines

- `runs/prepare.py` builds cells from the datasets.
- `runs/run.py` runs campaigns of any model, resumable.
- `runs/queue.py`, `start.cmd`, `stop.cmd`: a queue of runs that can be
  stopped and resumed.

**tests** check the package against the numbers the study recorded.

## Running

From the repository root:

    # campaigns: the term and mean pooling by MAP (CPU), then the head with
    # the term by SMC (GPU)
    python -m bo_attention_smc.runs.run --out runs/x --arms null identity \
        --featurisers mace t5_augm morgan --rules chen --seeds 20 --workers 12
    python -m bo_attention_smc.runs.run --out runs/x_smc \
        --arms smc_identity_head --featurisers mace t5_augm --workers 6

    # a run's report; the table of the study's runs
    python -m bo_attention_smc.diagnostics_and_metrics.report run runs/x
    python -m bo_attention_smc.diagnostics_and_metrics.report table runs/table

    # the queue: one block per line of runs/queue/queue_smc.txt, queue_map.txt
    bo_attention_smc\runs\start.cmd
    bo_attention_smc\runs\stop.cmd

    # evaluation
    python -m bo_attention_smc.evaluation.predict --out runs/eval/predict.jsonl
    python -m bo_attention_smc.evaluation.chemistry refit --arm identity \
        --rule chen --records records/fill/map/campaigns.jsonl \
        --out runs/eval/map.jsonl
    python -m bo_attention_smc.evaluation.representation

    # tests: about 2 minutes, the SMC ones on the GPU
    python -m unittest discover -s bo_attention_smc/tests -t .

Each module's docstring gives its options. Campaigns go to
`<out>/campaigns.jsonl` as they end, and a stopped run resumes; a run folder
holds one design (`config.json`).

## Records

One line per campaign: `cell` (`dataset/featuriser`), `arm` (the label),
`rule`, `seed`, `status`, `sampled_indices`, `seconds`, `metrics` (`auc`,
`lift`, `coverage_top5`, `simple_regret`, `best_found`, `first_top5_hit`,
`fits_failed`, `log_score_total`, `seconds`) and `fits`, one list per
diagnostic with an entry per BO step:

- every model: `status`, `log_score` and `z` (how well the step predicted
  the yield it then measured), `seconds`;
- MAP: `ell_ratio` (fitted length-scale over the prior's centre), `noise`,
  and with the term `w[reagent]`;
- SMC: `smc_tempering`, `smc_acceptance`, `smc_step_size`,
  `smc_log_evidence`, `smc_seconds`, `predict_seconds`, and with the term
  `w[reagent]` (the particles' median).

The study's `null` records, made by `bo_hmc`, name some fields differently:
`log_score_null` and `z_null` are `log_score` and `z`, `noise_null` is
`noise`, and `ell_mean` over the top-level `ell_0` is `ell_ratio`; their
metric `log_score_null_total` is `log_score_total`.

## Reproducing recorded campaigns

- **One CPU thread per campaign,** as the runner sets. At another thread
  count a long fit can land elsewhere.
- **SMC campaigns pad every step to their last, 55 reactions.** A replay of
  part of a campaign must pad to the same (`make_step(..., n_max=55)`):
  another padding changes the last bits, and through resampling the
  choices. On the same GPU, SMC repeats itself exactly.
- **`records/chemistry/smc.jsonl` was computed in one process at torch's
  default thread count** (12 on the study's laptop), not one thread; at
  that count `evaluation.chemistry` gives its numbers exactly.

## What it needs beside it

- `settings.py`, `settings.toml` and `logging_config.py` at the root, which
  `core/conformer.py` and `core/featuriser.py` read; the rest of this
  package does not use them.
- `data/` holds the datasets and `conformers/` the conformer searches.
- `cells/` holds the study's cells, on which every recorded campaign ran,
  and `cells/full/` the MACE attention cells kept on every channel (for
  `evaluation.representation`). MACE on the GPU does not repeat itself to
  the last digit, so its cells cannot be rebuilt exactly.
- `records/` holds the study's recorded campaigns and evaluation outputs,
  which the tests, the table and the evaluation tools read.

This package was made from `bo_hmc`, `bo_per_molecule` and `bo_smc`.
