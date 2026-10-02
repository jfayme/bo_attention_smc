# bo_hmc

Bayesian optimisation of reactions with a Gaussian process (GP). Each molecule
is described by its atoms, which the GP pools with an attention head sampled
by NUTS. This package runs the length-scale study's sweep and experiment C3
(`bo_fixes/PREREGISTRATION.md`, section 7b).

## What each module does

**core**: the model

- `core/cells.py` reads a dataset, describes its molecules (fingerprints, T5
  tokens, or MACE/AIMNet2 atoms), and keeps the columns that matter: a *cell*.
- `core/campaign.py` fits the GP and runs one campaign: 5 random reactions,
  then 50 chosen one at a time by expected improvement. The length-scale
  prior's rule is `chen` or `geom` (a Gamma centred by the rule), or
  `default` (added 2026-09-30): BoTorch 0.18's own LogNormal(sqrt(2) +
  log(D)/2, sqrt(3)), each length-scale starting at its mode, the kernel
  and everything else unchanged. Under chen and geom the GP is built by the
  same operations as before, and recorded campaigns repeat choice for choice.
- `core/hmc.py` holds the attention head over the atoms, and samples it with
  the GP's length-scales, noise and mean (NUTS).

**diagnostics_and_metrics**: what the campaigns achieved

- `diagnostics_and_metrics/metrics.py` scores one campaign: its lift over
  random selection and its share of the best 5 % of reactions.
- `diagnostics_and_metrics/report.py` summarises a folder of campaigns: the
  sweep's tables and figures, or C3's pre-registered analysis.

**runs**: the command lines

- `runs/prepare.py` builds the cells from the datasets, or converts the
  study's stored cells.
- `runs/run.py` runs campaigns in parallel, resumes a stopped run, and shows
  no result until it ends.

**tests** checks bo_hmc against the numbers the study recorded.

## Running C3

```
python -m bo_hmc.runs.prepare --out runs/bo_hmc --convert runs/sweep/cells  # once
python -m bo_hmc.runs.run --out runs/fixes/benchmark07
python -m bo_hmc.diagnostics_and_metrics.report runs/fixes/benchmark07 --c3
python -m unittest discover -s bo_hmc/tests -t .
```

## What it needs beside it

`conformer.py` and `featuriser.py` describe the molecules; `settings.py`,
`settings.toml` and `logging_config.py` hold the settings and the log;
`data/` holds the datasets and `conformers/` the conformer searches. C3's
inputs are the study's cells, converted into `runs/bo_hmc/cells`: MACE on the
GPU does not repeat itself to the last digit, so its cells cannot be rebuilt
exactly.
