# Consolidation: one package, MAP and SMC (handover, 2026-10-01)

You are a new Claude Code session. Your job is to consolidate three packages of this repository into one, following the plan below. **Do not modify any file before the analysis step is done and the questions in section 4 are answered by the user.** If anything is unclear or you are unsure, ask. Do not guess.

---

## 1. Context

The study runs pool-based Bayesian optimisation (BO) of reaction yields on three datasets (bh_full, bh_1: Buchwald–Hartwig; Shields: direct arylation), with molecules described by MACE (MLIP atom descriptors), T5 (GT4SD `multitask-text-and-chemistry-t5-base-augm`, called `t5_augm`) or Morgan fingerprints. A Gaussian process (GP) scores candidates by LogEI.

Three packages hold the code today:

| package | what it adds | size |
|---|---|---|
| `bo_hmc/` | The whole pipeline: cells (`core/cells.py`), the GP fit and campaign loop (`core/campaign.py`), the attention span head and its NUTS sampler (`core/hmc.py`), metrics and reports (`diagnostics_and_metrics/`), cell building and campaign runner (`runs/`), tests | ~3,500 lines |
| `bo_per_molecule/` | The per-molecule kernel term (MAP), its variants, the one-hot baseline, ceiling arms, combined and Morgan cells, a generic campaign loop `run_with` | ~2,100 lines |
| `bo_smc/` | SMC (torch, GPU) in place of NUTS for the head and the term: batched target (`model.py`), sampler (`smc.py`), arms (`run.py`), a BlackJAX CPU port, analysis scripts | ~3,500 lines |

Exploratory and evaluation scripts also live in the git-ignored `runs/` folder: `runs/fill/` (pausable queue driver, results table), `runs/chemistry/` (attention read-out, chemistry checks, prediction and transfer study), `runs/compress/` (leave-one-out descriptor diagnostics).

**What the exploration established** (details in the READMEs, the `PREREGISTRATION*.md` files, `runs/fill/table.md` and `runs/chemistry/`):
- Length-scale rule: chen ≥ geom > BoTorch default, for every featuriser and arm.
- The per-molecule term (MAP) beats mean pooling in BO (pre-registered, confirmed twice), but it is a per-molecule lookup: it damages transfer to unseen molecules, and the MAP fit is overconfident.
- SMC versus MAP for the term (pre-registered): same lift, about 3.5× fewer overconfident predictions. By the pre-registered rule, MAP stays in the BO loop.
- The SMC head added to the term adds nothing (two pre-registered tests); its attention stays at its prior.
- The SMC head alone is the best-calibrated model and keeps transfer to unseen molecules; on T5 it matches the term in BO (exploratory, to be confirmed).
- NUTS (HMC) is too slow and no more accurate than SMC. It is no longer used.

So the code to keep is: MAP and SMC inference, the term, the head, three rules, three featurisers, and the evaluation tools that judge prediction quality and chemistry, not only BO lift.

## 2. Goal and priorities

The goal is **one clean, self-contained package** that implements the models still in use, built on `bo_hmc`'s organisation, with **SMC replacing NUTS as the only sampler** and **MAP kept** as the fast inference. The consolidation must simplify the code as much as possible while preserving all behaviour that is kept.

Priorities, in order:
1. Do not break existing mechanisms or behaviour (section 6). Recorded campaigns must be reproducible (section 7).
2. Simplicity first: the smallest amount of code needed.
3. Human readability: easy for a scientist or chemist to understand.
4. Consistency: match the existing style of `conformer.py` and `featuriser.py`.
5. No unnecessary abstraction: no helpers, classes, configuration, error handling or features unless genuinely required.

The result should feel like it belongs naturally in this repository, not like a new implementation in a different style.

## 3. Before coding

Do not modify any file yet. First inspect the repository and understand:
- `bo_hmc/`, `bo_per_molecule/`, `bo_smc/` and how they call each other;
- the scripts in `runs/fill/`, `runs/chemistry/`, `runs/compress/`;
- `conformer.py`, `featuriser.py`, `settings.py`, `logging_config.py` (used by `bo_hmc`; read-only unless the user agrees otherwise);
- how campaigns are launched, what inputs and outputs they expect (cells in `runs/bo_hmc/cells/`, `campaigns.jsonl` records, `config.json` design guards);
- what mechanisms must be preserved (section 6);
- the tests that cover current behaviour (section 7).

Then explain briefly, in simple terms:
1. How you understand the current implementation.
2. What it does conceptually.
3. What you propose to simplify, and the file layout you propose (section 5 is the starting point).
4. Which behaviours and mechanisms you consider essential to preserve.
5. Any assumptions you are making.

Then ask the questions of section 4 that are still open, plus any of your own that genuinely need an answer. Wait for the answers before coding.

## 4. Questions to settle with the user before coding

Recommended answers are given; the user decides.

1. **Package name and place.** Keep the folder name `bo_hmc/` (misleading once NUTS is gone) or create a new package (name?) and retire the three old ones? *Recommended: a new name, chosen by the user.*
2. **The old packages after validation.** Delete `bo_per_molecule/` and `bo_smc/` (and NUTS from `bo_hmc`) once the new package passes section 7, since they stay in the snapshot commit? Or keep them read-only? *Recommended: delete; git history keeps them.*
3. **Pre-registrations.** `bo_per_molecule/PREREGISTRATION*.md` and `bo_smc/PREREGISTRATION*.md` are hashed by path in `runs/*/preregistration.sha256`. Leave them where they are, or move them to one `docs/preregistrations/` folder (content unchanged, so the hashes still match)? *Recommended: move, keeping a note of the old paths.*
4. **Scripts in the git-ignored `runs/` folder.** The scripts that produced recent results (`runs/fill/*.py`, `start.cmd`, `stop.cmd`, `runs/chemistry/*.py`, `runs/compress/*.py`) are not tracked. Force-add them to the snapshot commit, or copy them into a tracked folder first? *Recommended: force-add the scripts only (not data) to the snapshot.*
5. **One-hot baseline and the combined MACE+T5 cell.** Keep or drop? *Recommended: drop both (one-hot was a one-off baseline, MACE+T5 screened at about 0); ask.*
6. **Molecule identity for the term.** Today the term reads each reaction's molecules from attention atoms, so Morgan needs a "combined cell" carrying MACE atoms. Derive the molecule index from the dataset for every cell instead, if it reproduces recorded choices exactly? *Recommended: yes.*
7. **Two implementations of one target density.** `bo_hmc/core/hmc.py`'s `Posterior` (per particle) and `bo_smc/model.py` (batched, GPU) compute the same log posterior and predictions; the tests check one against the other. Keep the batched one in the package and the per-particle one as the test reference? *Recommended: yes.*
8. **Exploratory diagnostics.** Promote only the generic leave-one-out representation diagnostic (`runs/compress/diagnose.py`) to the evaluation folder, and leave the Wasserstein (`sets.py`) and steric (`sterics.py`) experiments in the snapshot? *Recommended: yes.*
9. **Analysis scripts of past pre-registrations** (`bo_smc/term_test.py`, `bo_smc/summary.py`, `bo_smc/check_sampler.py`, `bo_per_molecule/config_contrast.py`). Keep in the new package, or leave them in the snapshot only? *Recommended: snapshot only; new pre-registrations get their own analysis.*
10. **Root modules.** `featuriser.py`, `conformer.py`, `settings.py`, `logging_config.py`: read-only this time? *Recommended: read-only.*
11. **Running jobs.** The fill queue (`runs/fill/start.cmd`) runs from the main working tree. Is anything running, and should it finish first? *Never edit code in a working tree a queue is running from; work in a separate git worktree.*

## 5. The consolidated design

### 5.1 Layout: four groups

Keep `bo_hmc`'s grouping, and add a fourth group for evaluation:

- **core**: the model and its inference. Cells (`cells.py`, as today); priors and rules; the GP (Matern ARD, optional per-molecule term); the span head; MAP fitting; the SMC target and sampler; the campaign loop.
- **evaluation** (new, its own folder): the tools that judge a model beyond BO lift:
  - prediction quality on common data (every model fitted on the same reactions, scored on the rest of the pool: log density, overconfident share, Spearman, RMSE, top-5 % recall) — today `runs/chemistry/predict.py` test A;
  - transfer to unseen molecules (a molecule held out of BO data, and of random data) — tests B and C;
  - attention read-out: per-atom or per-token weights, labelled by element (conformer cache) or token (T5 tokeniser), against a random-head prior control — today `labels.py`, `relearn.py`, `prior_attention.py`, `attention_summary.py`, `t5_head.py`;
  - molecule-level chemistry checks: predicted versus true molecule means, halide order within aryl cores, top ligands, the term's reagent weights — today `relearn.py`, `summary.py`;
  - the leave-one-out representation diagnostic — today `runs/compress/diagnose.py`.
- **diagnostics_and_metrics**: per-campaign metrics (lift, AUC, coverage, as today), reports and the results table (today `runs/fill/table.py`).
- **runs**: command lines: building cells (`prepare.py`), running campaigns (one runner for MAP and SMC, resumable), the pausable queue (`runs/fill/driver.py`, `start.cmd`, `stop.cmd`).

Collapse small, related scripts into one where it is logical. Do not duplicate functionality.

### 5.2 Models from parts, not a list of named arms

An arm becomes a combination of independent choices:

| part | options |
|---|---|
| features | MACE, T5 (`t5_augm`), Morgan |
| kernel | Matern ARD; optionally plus the per-molecule term |
| head | none, PCA-8, all channels |
| inference | MAP, SMC |
| rule | chen, geom, default |

Then mean pooling (`null`), the term (`identity`, MAP or SMC), the head alone (`smc_span_pca8`, `smc_span`) and head + term (`smc_identity_head`) exist without special code. Keep the old arm names as labels in the records, so new records compare with old ones in the same table. Design so that an extra feature block (for example physical ligand descriptors) could be added later, but **do not implement any new feature block now**.

### 5.3 Keep

- Cells and cell building, as today (all featurisers and aggregations `prepare.py` builds).
- Rules chen, geom, default; prior width cv 0.3, `match_concentration`.
- MAP fitting (one L-BFGS-B run, never retried, status recorded).
- The per-molecule term with its default prior only (`Term()`: weights Gamma(2, 4)).
- The span head: PCA-8 and all channels; head prior N(0, head_scale²) with head_scale 1.
- SMC, as in `bo_smc` (section 6).
- LogEI as the only acquisition.
- The evaluation tools of 5.1.

### 5.4 Drop

- NUTS (`hmc.py`'s sampler and schedules) and the NUTS arms (`hmc_null`, `hmc_span`, `hmc_span_pca8`, `identity_head`). Keep the posterior definition as the reference the SMC target is tested against (question 7).
- Ceiling arms and `PoolStandardize`.
- The tight head prior (`*_tight`).
- Term variants: loose, scaled, pairs, additive, UCB (beta 1, 2, 3), greedy, cv05, long length-scales.
- The BlackJAX CPU port (`bo_smc/blackjax_cpu/`).
- Pending answers to section 4: one-hot, MACE+T5, old analysis scripts.
- Never touch `bo.py`, `attention.py`, `bo_fixes/`: they are historical references.

## 6. Behaviour that must be preserved

**Campaign.** 5 random reactions (numpy, seed 1337 + seed), then 50 chosen by LogEI. The torch stream is seeded with the same number. The first index wins a tie. Each choice is scored before it joins the data (prequential log score and z). Campaigns run on one torch thread per worker; replays need the same thread count to match.

**GP.**
- Matern-5/2, one length-scale per column (ARD), no amplitude.
- Inputs mapped to the unit box by the pool's feature range; yields standardised by the measured ones.
- Noise LogNormal(−4, 1), floored at 1e-4, starting at its mode (BoTorch 0.18's default, written out).
- The null's inputs are the head's pooling with equal scores (`null_features`); the stored features set the prior and the input range.

**Rules.**
- chen: ℓ₀ = 0.4·√D + 4.
- geom: the pool's mean pairwise distance in the unit box over u* (≤ 2,000 rows, seed 0).
- Both use Gamma(1/cv², 1/(cv²·ℓ₀)) with cv 0.3, length-scales starting at ℓ₀.
- default: LogNormal(√2 + ½·log D, √3), starting at its mode (`Prior.lognormal`, added 2026-09-30).

**Term.** k = Matern(features) + Σ_c w_c·[same molecule of reagent c], w_c ~ Gamma(2, 4), starting at its mode; the Matern reads the features only.

**Head.**
- Atom scores w·tanh(V·x + b) on the 8 whitened principal components of the atoms (PCA-8), or on every channel with the hidden size capped by `MAX_WEIGHTS_PER_MOLECULE`.
- Softmax over a molecule's atoms, then conformer pooling.
- Each reagent's pooled columns mapped onto the null's span.

**SMC.**
- 256 particles, carried from one BO step to the next (resample-move).
- Adaptive tempering to ESS 0.5 (at most 100 steps); 5 HMC transitions of 10 leapfrog steps per tempering step, step size adapted from 0.3.
- Systematic resampling.
- The generator is seeded 1337 + seed. Gamma draws are made by numpy, seeded from the torch generator.
- LogEI is averaged over the particles; predictions are taken for the whole pool at every step.

**Metrics.** lift, AUC, top-5 % coverage, simple regret, as `bo_hmc/diagnostics_and_metrics/metrics.py` computes them (random AUC: 400 draws, seed 1337 + 90210).

**Records and runs.**
- `campaigns.jsonl` fields and arm names unchanged, so old and new records share one table.
- Runs append as campaigns end and skip finished ones (resumable).
- `config.json` refuses a folder of another design.
- The queue can be stopped and resumed.

## 7. Validation: what must still pass

Run on one torch thread, as recorded. MAP must match exactly; SMC on the GPU must match choice for choice where it did before, and any departure must be reported with the step where it starts.

1. **Existing tests that do not need NUTS:**
   - `bo_hmc/tests/test_cells.py`, `test_metrics.py`, and the null parts of `test_golden.py`;
   - `bo_per_molecule/tests/test_run.py`: the builder repeats `fit_gp`, the loop repeats the null campaign, the same-molecule kernel, the identity posterior;
   - `bo_smc/tests/test_model.py`: the target particle for particle, the default rule, the full head, the Gaussian sampler test.
2. **Recorded campaigns, replayed choice for choice:**
   - MAP null and term, chen: `runs/bo_per_molecule_config/` (for example shields/mace and bh_1/t5_augm, seed 40);
   - MAP default rule and Morgan: `runs/fill/map/`;
   - SMC head + term, chen: `runs/bo_smc_chen/` (shields/mace, seed 40);
   - SMC head alone (PCA-8): `runs/fill/smc/`;
   - SMC head alone (all channels): `runs/compress/smc/`;
   - SMC term: `runs/bo_smc_term/`.
3. **Metrics:** recompute lift, AUC and coverage from `sampled_indices` for every recorded campaign; all must match.
4. **Evaluation:** reproduce a sample of `runs/chemistry/predict.jsonl` (MAP fits exactly; SMC within a stated tolerance) and of the attention numbers in `runs/chemistry/smc.jsonl`.

No long campaign runs. A replay of a few campaigns is enough.

## 8. Process

0. **Snapshot** (with the user's go): commit the current state, including the decision of question 4, so every result points to the code that produced it. Treat old run folders as read-only.
1. **Analysis** (section 3) and the questions (section 4). Stop and wait.
2. **Layout proposal**: the files of the new package, what moves where, what is dropped. Stop and wait for approval.
3. **Implement** on a branch, in a separate git worktree.
4. **Validate** (section 7) and report the results plainly, failures included.
5. **Simplify**: review the code asking "Can this be made simpler without changing its behaviour?" If yes, simplify, then rerun section 7.
6. **Document**: a README for the package (what each module does, how to run), and a dated note of what was consolidated and what was dropped. Commit only when the user asks.

## 9. Refactoring approach

- Produce simpler and clearer code where possible.
- Do not duplicate functionality.
- Collapse small, scattered but related scripts into one where it is logical.
- Reuse the repository functionality that is already correct (`featuriser.py`, `conformer.py`, `cells.py`) rather than rewriting it.

## 10. Coding style

Match the coding style of `D:\Git\supramol_explorer` as closely as possible, and of `conformer.py` and `featuriser.py` in this repository:
- naming conventions, function and class structure, import style, type-hint style;
- docstring style (numpydoc, as in `bo_hmc`), comment style, terminology;
- level of abstraction, and the general length and density of the code.

Prefer straightforward code over clever Python. If something can be expressed clearly with a few normal lines, do not replace it with complicated comprehensions, unnecessary lambdas, excessive helper functions, abstractions, configuration or defensive programming. Comments explain *why*, not what the code says, in simple clear language.

## 11. Simplicity

Write the smallest amount of code that correctly implements the kept behaviour.
- Do not add functionality that was not requested.
- Do not "improve" unrelated parts of the project.
- Do not introduce new dependencies.
- Do not refactor for style alone if that makes the code more complicated.

## 12. Final requirement

Before finishing, review the resulting code asking: **"Can this be made simpler without changing its behaviour?"** If yes, simplify it. The final code should be minimal, human readable, scientifically correct, consistent with the repository's style, and commented and documented in clear, simple language that helps a human reviewer.
