r"""How well each model predicts: on common data, and on unseen molecules.

BO lift says how fast a model finds good reactions, not how well it
predicts them. Here every model is fitted on the same reactions, those a
recorded campaign measured (by default the 55 of each `identity` campaign
under chen), and scored on the reactions it has not seen:

- Test A, common data (seeds 0-9): predict every other reaction of the
  pool. Scores: the mean log predictive density, the share of reactions
  predicted overconfidently (log density below -10), Spearman's rho, RMSE,
  and top-5 % recall (of the pool's top 5 % among the unseen reactions,
  the share the model ranks in its own top k, k their number).
- Test B, an unseen molecule in BO data (seeds 0-4): for each molecule of
  the dominant reagent (the aryl halide of bh_1 and bh_full, the ligand of
  Shields), drop every training reaction that uses it, refit, and predict
  every reaction of the pool that uses it. Skipped, and recorded as such,
  when fewer than MIN_TRAIN reactions remain: BO piles onto the best
  molecule.
- Test C, an unseen molecule in random data (seeds 0-4): the same, trained
  on as many reactions drawn at random from those not using the molecule,
  so that BO's choices play no part: transfer alone.

Scores of B and C: the mean log density, RMSE, rho within the molecule,
and the molecule's predicted and true mean yield; the summary ranks the
unseen molecules by them.

    python -m bo_attention_smc.evaluation.predict --out runs/eval/predict.jsonl
    python -m bo_attention_smc.evaluation.predict \
        --out runs/eval/predict.jsonl --summary

One line per fit is appended to --out, and a stopped run resumes. MAP fits
take seconds on one CPU thread each; SMC fits run on --device.

"""

import argparse
import json
import os
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr, wilcoxon

from bo_attention_smc.core.campaign import (
    MODELS,
    Fitted,
    Sampled,
    Step,
    make_pool,
    make_step,
)
from bo_attention_smc.core.cells import Cell, load_cell, molecules, reagents
from bo_attention_smc.core.gp import Prediction
from bo_attention_smc.runs.run import cell_path

MODELS_COMPARED = (
    "null",
    "identity",
    "smc_identity",
    "smc_identity_head",
    "smc_span_pca8",
)
NAMES = {
    "null": "pooling",
    "identity": "term MAP",
    "smc_identity": "term SMC",
    "smc_identity_head": "head+term SMC",
    "smc_span_pca8": "head SMC",
}
DATASETS = ("bh_1", "bh_full", "shields")
FEATURISERS = ("mace", "t5_augm")
DOMINANT = {
    "bh_1": "aryl halide",
    "bh_full": "aryl halide",
    "shields": "ligand",
}
RULE = "chen"
MIN_TRAIN = 30
# a log density below this is an overconfident prediction
BAD = -10.0
# the recorded `identity` campaigns whose reactions every model is fitted on
RECORDS = (
    "records/bo_per_molecule/campaigns.jsonl",
    "records/bo_per_molecule_confirm/campaigns.jsonl",
    "records/bo_per_molecule_config/campaigns.jsonl",
)


def recorded(
    paths: tuple[str, ...], model: str, rule: str, cell: str, seed: int
) -> list[int]:
    """Return the reactions a recorded campaign measured, in order.

    The first of `paths` that holds the campaign gives it.
    """
    for path in paths:
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            key = (r["arm"], r["rule"], r["cell"], r["seed"], r["status"])
            if key == (model, rule, cell, seed, "ok"):
                return r["sampled_indices"]
    raise KeyError((model, rule, cell, seed))


def refit(
    model: str,
    cell: Cell,
    dataset: str,
    rule: str,
    seed: int,
    rows: list[int],
    device: str,
) -> tuple[Step, Fitted | Sampled, dict]:
    """Fit a model on some reactions of a cell, as a campaign's last step.

    Returns
    -------
    step, fitted, record
        The step (an SMC step holds its particles), the fitted model, and
        what the campaign would record of it. An SMC model starts from its
        priors on these reactions at once, seeded by `seed`.

    """
    pool = make_pool(cell, rule, dataset if MODELS[model].term else None)
    step = make_step(model, cell, pool, seed, len(rows), device)
    fitted, record = step(rows)
    return step, fitted, record


def scores(prediction: Prediction, y: np.ndarray, test: np.ndarray) -> dict:
    """Score the predictions of some reactions against their yields."""
    mean, _ = prediction.moments()
    mean = mean.numpy()
    log_p = prediction.log_density(torch.as_tensor(y)).numpy()
    found = {
        "log_density": float(log_p[test].mean()),
        "bad_share": float((log_p[test] < BAD).mean()),
        "rmse": float(np.sqrt(((mean[test] - y[test]) ** 2).mean())),
        "rho": float(spearmanr(mean[test], y[test])[0]),
        "predicted_mean": float(mean[test].mean()),
        "true_mean": float(y[test].mean()),
    }
    threshold = np.quantile(y, 0.95)
    top = test[y[test] >= threshold]
    if len(top):
        ranked = test[np.argsort(-mean[test])][: len(top)]
        found["top5_recall"] = float(len(set(ranked) & set(top)) / len(top))
    return found


def run_task(task: dict, records: tuple[str, ...], device: str) -> dict:
    """Fit one model for one test, and score it."""
    torch.set_num_threads(1)
    dataset, featuriser = task["dataset"], task["featuriser"]
    seed = task["seed"]
    cell = load_cell(cell_path(dataset, featuriser))
    y = cell.objective
    rows = recorded(records, "identity", RULE, f"{dataset}/{featuriser}", seed)
    if task["test"] == "A":
        train = rows
        test = np.setdiff1d(np.arange(len(y)), train)
    else:
        c = reagents(dataset).index(DOMINANT[dataset])
        numbers = molecules(dataset)[:, c]
        held = task["molecule"]
        test = np.flatnonzero(numbers == held)
        if task["test"] == "B":
            train = [r for r in rows if numbers[r] != held]
            if len(train) < MIN_TRAIN:
                return {**task, "n_train": len(train), "skipped": True}
        else:
            others = np.flatnonzero(numbers != held)
            rng = np.random.default_rng([seed, held])
            train = rng.choice(others, len(rows), replace=False).tolist()
    _, fitted, _ = refit(task["arm"], cell, dataset, RULE, seed, train, device)
    prediction = fitted.predict(np.arange(len(y)))
    return {
        **task,
        "n_train": len(train),
        "n_test": len(test),
        **scores(prediction, y, test),
    }


def tasks(part: str) -> list[dict]:
    """Every fit of the tests asked for, the MAP fits first."""
    found: list[dict] = []
    for dataset in DATASETS:
        c = reagents(dataset).index(DOMINANT[dataset])
        held = np.unique(molecules(dataset)[:, c]).tolist()
        for featuriser in FEATURISERS:
            for model in MODELS_COMPARED:
                base = dict(arm=model, dataset=dataset, featuriser=featuriser)
                if "A" in part:
                    for seed in range(10):
                        found.append(dict(base, seed=seed, test="A"))
                for test in ("B", "C"):
                    if test not in part:
                        continue
                    for seed in range(5):
                        for m in held:
                            found.append(
                                dict(base, seed=seed, test=test, molecule=m)
                            )
    found.sort(key=lambda t: (t["test"], MODELS[t["arm"]].head is not None))
    return found


def key(task: dict) -> str:
    """Return what identifies a fit, for resuming."""
    fields = ("arm", "dataset", "featuriser", "seed", "test", "molecule")
    return json.dumps([task.get(k) for k in fields])


def p_value(differences: list[float]) -> float:
    """Wilcoxon p of paired differences; NaN for 5 or fewer, or all zero."""
    d = np.asarray(differences)
    if len(d) > 5 and np.any(d != 0):
        return float(wilcoxon(d).pvalue)
    return float("nan")


def summary(path: Path) -> None:
    """Print the tests' summary: each model, and its difference to pooling."""
    text = path.read_text(encoding="utf-8")
    rows = [json.loads(line) for line in text.splitlines()]
    rows = [r for r in rows if not r.get("skipped")]

    print(
        "== Test A: fitted on a term campaign's reactions, predicting the "
        "rest of the pool (seeds 0-9)"
    )
    by: dict = defaultdict(dict)
    for r in rows:
        if r["test"] == "A":
            by[(r["dataset"], r["featuriser"], r["seed"])][r["arm"]] = r
    for metric, better in (
        ("log_density", "+"),
        ("bad_share", "-"),
        ("rho", "+"),
        ("rmse", "-"),
        ("top5_recall", "+"),
    ):
        print(
            f"  {metric} ({better} is better); per dataset over MACE and "
            "T5, and the difference to pooling (p)"
        )
        for dataset in (*DATASETS, "all"):
            keys = [k for k in by if dataset in ("all", k[0])]
            cells = []
            for model in MODELS_COMPARED:
                v = [by[k][model][metric] for k in keys]
                d = [
                    by[k][model][metric] - by[k]["null"][metric] for k in keys
                ]
                extra = ""
                if model != "null":
                    extra = f" ({np.mean(d):+.3f}, p {p_value(d):.2g})"
                cells.append(f"{NAMES[model]} {np.mean(v):.3f}{extra}")
            print(f"    {dataset:8s} " + " | ".join(cells))

    for test, what in (
        ("B", "unseen molecule, BO data without it"),
        ("C", "unseen molecule, random reactions without it"),
    ):
        print(f"== Test {test}: {what} (seeds 0-4)")
        # (dataset, featuriser, seed) -> model -> molecule -> row
        unseen: dict = defaultdict(lambda: defaultdict(dict))
        for r in rows:
            if r["test"] == test:
                where = (r["dataset"], r["featuriser"], r["seed"])
                unseen[where][r["arm"]][r["molecule"]] = r
        for dataset in (*DATASETS, "all"):
            keys = [k for k in unseen if dataset in ("all", k[0])]
            print(f"  {dataset}")
            for model in MODELS_COMPARED:
                rank, log_d, rmse, within, d_log, d_rank = (
                    [],
                    [],
                    [],
                    [],
                    [],
                    [],
                )
                for k in keys:
                    mine, base = unseen[k][model], unseen[k]["null"]
                    held = sorted(m for m in mine if m in base)
                    if len(held) >= 3:
                        rho = spearmanr(
                            [mine[m]["true_mean"] for m in held],
                            [mine[m]["predicted_mean"] for m in held],
                        )[0]
                        rho_0 = spearmanr(
                            [base[m]["true_mean"] for m in held],
                            [base[m]["predicted_mean"] for m in held],
                        )[0]
                        rank.append(rho)
                        d_rank.append(rho - rho_0)
                    for m in held:
                        log_d.append(mine[m]["log_density"])
                        rmse.append(mine[m]["rmse"])
                        within.append(mine[m]["rho"])
                        d_log.append(
                            mine[m]["log_density"] - base[m]["log_density"]
                        )
                extra = ""
                if model != "null":
                    extra = (
                        f" [vs pooling: rank {np.mean(d_rank):+.2f} p "
                        f"{p_value(d_rank):.2g}; logdens {np.mean(d_log):+.2f}"
                        f" p {p_value(d_log):.2g}]"
                    )
                print(
                    f"     {NAMES[model]}: rank unseen {np.nanmean(rank):+.2f}"
                    f", logdens {np.mean(log_d):.2f}, rmse {np.mean(rmse):.1f}"
                    f", within {np.nanmean(within):+.2f}{extra}"
                )


def main(argv: list[str] | None = None) -> None:
    """Run the fits not yet in --out, or print the summary."""
    parser = argparse.ArgumentParser(
        prog="python -m bo_attention_smc.evaluation.predict",
        description=__doc__.split("\n")[0],
    )
    parser.add_argument("--out", type=Path, required=True, help="a .jsonl")
    parser.add_argument("--part", default="ABC", help="the tests to run")
    parser.add_argument(
        "--records", nargs="+", default=list(RECORDS), help="campaigns.jsonl"
    )
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="where the SMC particles live",
    )
    parser.add_argument("--summary", action="store_true")
    args = parser.parse_args(argv)
    if args.summary:
        summary(args.out)
        return
    done = set()
    if args.out.is_file():
        text = args.out.read_text(encoding="utf-8")
        done = {key(json.loads(line)) for line in text.splitlines()}
    todo = [t for t in tasks(args.part) if key(t) not in done]
    print(f"{len(done)} fits done; {len(todo)} to run", flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    # one CPU thread per worker; the SMC particles live on the device
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    with ProcessPoolExecutor(args.workers) as executor:
        futures = [
            executor.submit(run_task, t, tuple(args.records), args.device)
            for t in todo
        ]
        for count, future in enumerate(as_completed(futures), 1):
            try:
                found = future.result()
            except Exception as error:  # noqa: BLE001 -- printed; goes on
                print(f"FAILED: {type(error).__name__}: {error}", flush=True)
                continue
            with args.out.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(found) + "\n")
            if count % 20 == 0:
                print(f"[{count}/{len(todo)}]", flush=True)


if __name__ == "__main__":
    main()
