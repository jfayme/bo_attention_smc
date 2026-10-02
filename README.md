# bo_attention_smc

Pool-based Bayesian optimisation (BO) of reaction yields with a Gaussian
process (GP), each molecule described by MACE atoms, T5 tokens or a Morgan
fingerprint. A model either fits its settings by MAP or, when an attention
head weights each molecule's atoms or tokens, is sampled by sequential Monte
Carlo (SMC) on the GPU. Everything needed to rebuild the inputs, run the
campaigns and reproduce the study's recorded results is in this repository.

`bo_attention_smc/README.md` describes the models, the modules and the
commands; this file covers setup, the data and where it all came from.

## Layout

| path | what it holds |
|---|---|
| `bo_attention_smc/` | the package: core (cells, GP, SMC, campaigns), evaluation, metrics and reports, command lines, tests |
| `conformer.py`, `featuriser.py` | conformer search and the featurisers (Morgan, T5, MACE, AIMNet2) the cells are built with |
| `settings.py`, `settings.toml`, `logging_config.py` | the settings and logging those two modules read |
| `tests/` | tests of `conformer.py` and `featuriser.py` |
| `data/` | the three datasets |
| `conformers/` | the conformer searches of every molecule, cached (64 molecules) |
| `cells/` | the study's 38 cells (3 datasets × Morgan, two T5 models, MACE, AIMNet2 × mean, mean/std/max, attention; one cell, Shields under AIMNet2 with attention, cannot be built), and `cells.json`, their widths and prior centres as the study recorded them |
| `cells/full/` | the three MACE attention cells kept on all 1,024 channels, for the representation diagnostic (Git LFS) |
| `records/` | the study's recorded campaigns and evaluation outputs (below) |
| `docs/` | the pre-registrations of the confirmatory tests, and the history of the code |
| `runs/` | new outputs; ignored by git |

## Setup

    git clone https://github.com/jfayme/bo_attention_smc.git
    cd bo_attention_smc
    git lfs pull                      # the full-channel cells, 179 MB
    conda env create -f environment.yml
    conda activate bo-attention-smc

`environment.yml` pins what the code imports to the versions the recorded
results were produced with (Python 3.11, PyTorch 2.12 for CUDA 13.0,
BoTorch 0.18, GPyTorch 1.15, MACE 0.3.16, transformers 4.57, AIMNet2 at a
fixed commit). The MACE (mh-1) and T5 weights are downloaded on first use.
The devices are set in `settings.toml` (featurisers) and on the command
line (`--device`, SMC); everything runs from the repository root.

Check the installation (about 1.5 minutes; without a GPU, the SMC replays
and the T5 cell are skipped):

    python -m unittest discover -s bo_attention_smc/tests -t .
    python -m unittest discover -s tests -t .

## The data

- **Datasets** (`data/`):
  - `Buchwald_Hartwig.csv` (`bh_full`): 3,955 Buchwald–Hartwig aminations
    from the high-throughput screen of Ahneman et al. (Science, 2018), on a
    grid of 4 ligands, 22 additives, 3 bases and 15 aryl halides (3,960
    combinations);
  - `bh_reaction_1.csv` (`bh_1`): its 790 reactions of one aryl core,
    1-ethyl-4-chloro/bromo/iodobenzene (of 792 combinations);
  - `shields_dataset.csv` (`shields`): 1,728 direct arylations from
    Shields et al. (Nature, 2021): 4 solvents × 4 bases × 12 ligands × 3
    temperatures × 3 concentrations.
- **Cells** (`cells/`) are what `bo_attention_smc.runs.prepare` builds from
  the datasets and the conformer cache. Morgan and T5 cells rebuild array
  for array; MACE on the GPU does not repeat itself to the last digit (about
  5e-6), so its stored cells are the reference every recorded campaign ran
  on.
- **Recorded campaigns** (`records/`, one `campaigns.jsonl` per study
  folder, with its `config.json` and, for a pre-registered test, its
  `preregistration.sha256`): the 5,297 campaigns of the six models this
  code runs (`null`, `identity`, `smc_span_pca8`, `smc_span`,
  `smc_identity_head`, `smc_identity`) on the MACE, T5 and Morgan cells.
  Campaigns of models the study dropped (one-hot, ceilings, variants of the
  term, tight head priors, NUTS) were left out. The `null` records name some
  fields as the study's older code did; the package README gives the
  mapping.
- **Evaluation outputs** (`records/chemistry/`): the prediction study
  (`predict.jsonl`), the chemistry of refitted campaigns (`map.jsonl`,
  `smc.jsonl`, `smc_t5.jsonl`) and the random-head control
  (`prior_attention.json`).

## Where it came from

Copied on 2026-10-02 from `bo_attention_smc/` of
[jfayme/lengthscale_bo](https://github.com/jfayme/lengthscale_bo), commit
`f79b6f0`, where it was consolidated from three earlier packages
(`docs/history/consolidation_2026-10-01.md`). What differs from there:

- the cells are read from `cells/` and the recorded campaigns from
  `records/` (there: the git-ignored `runs/bo_hmc/cells` and `runs/...`);
  the queue's files live in `runs/queue/`;
- `test_metrics` reads the yields from the datasets, and checks the
  campaigns of `records/` only;
- `settings.toml` keeps only the sections `conformer.py` and `featuriser.py`
  read (values unchanged; the log file is renamed), with comments brought
  up to date;
- `environment.yml` is new;
- the `table` report creates its output folder (there it failed on a
  folder that did not exist yet).

`conformer.py`, `featuriser.py`, `settings.py`, `logging_config.py` and
their tests are copied unchanged.

## Checked in this repository (2026-10-02)

- Both test suites pass: 25 tests of the package, among them replays of
  the first steps of recorded MAP and SMC campaigns, cells rebuilt from the
  datasets, and the metrics of all 5,297 recorded campaigns recomputed
  exactly; 33 tests of `conformer.py` and `featuriser.py`.
- Ten full campaigns run here through `runs/run.py` (two SMC on the GPU,
  the head with the term and the term alone; eight MAP, mean pooling and
  the term on MACE and Morgan under the rule `default`) repeat their
  records choice for choice, with the same scores and metrics.
- From `records/`: the results table is the study's own, number for
  number; test A of the prediction study (seed 0, 30 fits, MAP and SMC)
  and its summary reproduce `records/chemistry/predict.jsonl` exactly; so
  do the random-head control and, on `cells/full`, the representation
  diagnostic.

## License

MIT (`LICENSE`).
