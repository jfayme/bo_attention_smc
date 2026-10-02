"""Reports of run folders: one run's report, and the table of every run.

Both read the campaigns runs/run.py recorded. Two campaigns pair when they
share a cell, a rule and a seed, hence a random start, and every contrast
is the mean of their paired differences with a two-sided Wilcoxon
signed-rank test.

- `run`: <folder>/report.md, a run folder's mean lift, AUC and top-5 %
  coverage per cell, rule and model, its contrasts between models (each
  with a 95 % paired-bootstrap interval) per cell, per dataset and over
  all, and how the SMC sampler went. Campaigns of other folders join where
  they pair with the run's (`--pair-with`).
- `table`: <out>/table.md and table.csv, every model, rule, featuriser,
  dataset and seed set of the study's run folders (TABLE_RUNS): mean
  pooling, the term, the SMC head with the term and the SMC head alone, and
  their paired differences. The queue rebuilds it after every block.

    python -m bo_attention_smc.diagnostics_and_metrics.report run runs/x
    python -m bo_attention_smc.diagnostics_and_metrics.report table runs/table

"""

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

# Resamples of a contrast's paired-bootstrap interval.
BOOTSTRAP = 10_000

# The contrasts the run report gives, model minus reference, and what each
# asks; a contrast whose models did not both run is left out.
CONTRASTS = (
    ("identity", "null", "the per-molecule term"),
    ("smc_span_pca8", "null", "the head, by SMC, against mean pooling"),
    ("smc_span", "smc_span_pca8", "every channel against 8 components"),
    ("smc_identity_head", "identity", "the head added to the term"),
    ("smc_identity_head", "smc_span_pca8", "the term added to the head"),
    ("smc_identity_head", "null", "term and head against mean pooling"),
    ("smc_identity", "identity", "the term sampled by SMC against MAP"),
)

# The run folders the table reads, in order: a campaign found twice is
# counted once, from the first.
TABLE_RUNS = (
    "records/bo_per_molecule",
    "records/bo_per_molecule_confirm",
    "records/bo_per_molecule_config",
    "records/bo_smc",
    "records/bo_smc_confirm",
    "records/bo_smc_chen",
    "records/fill/map",
    "records/fill/smc",
)
TABLE_MODELS = {
    "null": "mean pooling",
    "identity": "term",
    "smc_identity_head": "SMC head+term",
    "smc_span_pca8": "SMC head alone",
}
HEADS = ("smc_identity_head", "smc_span_pca8")
RULES = ("chen", "geom", "default")
FEATURISERS = {"mace": "MACE", "t5_augm": "T5", "morgan": "Morgan"}
DATASETS = ("bh_full", "bh_1", "shields")
SEED_SETS = {
    "0-19": range(0, 20),
    "20-39": range(20, 40),
    "40-59": range(40, 60),
}
METRICS = {
    "lift": "Lift (AUC rescaled: random selection 0, best reaction first 1)",
    "coverage_top5": "Top-5 % coverage (share of the pool's top 5 % found)",
    "auc": "AUC (area under the scaled best-so-far curve, not rescaled)",
}
DIFFERENCES = (
    ("smc_identity_head", "identity"),
    ("smc_identity_head", "null"),
    ("identity", "null"),
    ("smc_span_pca8", "null"),
)


def se(values: list[float] | np.ndarray) -> float:
    """Compute the standard error of the mean, NaN left out; 0 below two."""
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if len(finite) < 2:
        return 0.0
    return float(finite.std(ddof=1) / np.sqrt(len(finite)))


def wilcoxon_p(differences: np.ndarray) -> float:
    """Two-sided Wilcoxon signed-rank p of paired differences.

    NaN for a single pair, which scipy cannot test.
    """
    if len(differences) < 2:
        return float("nan")
    return float(wilcoxon(differences).pvalue)


def read_campaigns(folder: Path) -> list[dict]:
    """Read the successful campaigns a run folder recorded."""
    text = (folder / "campaigns.jsonl").read_text(encoding="utf-8")
    records = [json.loads(line) for line in text.splitlines()]
    return [r for r in records if r["status"] == "ok"]


def contrast(
    records: list[dict],
    model: str,
    reference: str,
    metric: str,
    rng: np.random.Generator,
) -> dict:
    """Compare two models on one metric, campaign by campaign.

    Parameters
    ----------
    records
        The campaigns compared.
    model, reference
        The contrast is model minus reference.
    metric
        A key of the campaigns' metrics, such as `lift`.
    rng
        Source of the bootstrap resamples.

    Returns
    -------
    found
        `pairs`, the `mean` difference, its 95 % bootstrap interval (`low`,
        `high`), the two-sided Wilcoxon `p` and the `wins`; `pairs` alone
        when no campaign pairs.

    """
    mine, theirs = {}, {}
    for r in records:
        key = (r["cell"], r["rule"], r["seed"])
        if r["arm"] == model:
            mine[key] = r["metrics"][metric]
        if r["arm"] == reference:
            theirs[key] = r["metrics"][metric]
    d = np.array([mine[key] - theirs[key] for key in mine if key in theirs])
    if len(d) == 0:
        return {"pairs": 0}
    means = d[rng.integers(len(d), size=(BOOTSTRAP, len(d)))].mean(axis=1)
    low, high = np.percentile(means, [2.5, 97.5])
    return {
        "pairs": len(d),
        "mean": d.mean(),
        "low": low,
        "high": high,
        "p": wilcoxon_p(d),
        "wins": f"{(d > 0).sum()}/{(d != 0).sum()}",
    }


def contrast_row(name: str, metric: str, found: dict) -> str:
    """Format one contrast as a row of a markdown table."""
    if not found["pairs"]:
        return f"| {name} | {metric} | 0 | | | | |"
    return (
        f"| {name} | {metric} | {found['pairs']} | {found['mean']:+.4f} | "
        f"{found['low']:+.4f} to {found['high']:+.4f} | {found['p']:.3g} | "
        f"{found['wins']} |"
    )


def run_report(folder: Path, pair_with: list[Path]) -> str:
    """Write <folder>/report.md: means, contrasts and the sampler's health.

    Parameters
    ----------
    folder
        The run folder.
    pair_with
        Other run folders, whose campaigns join where they share a cell,
        rule and seed with the run's.

    """
    records = read_campaigns(folder)
    mine = {(r["cell"], r["rule"], r["seed"]) for r in records}
    for other in pair_with:
        for r in read_campaigns(other):
            if (r["cell"], r["rule"], r["seed"]) in mine:
                records.append(r)
    cells = sorted({r["cell"] for r in records})
    rules = sorted({r["rule"] for r in records})
    models = sorted({r["arm"] for r in records})
    lines = [
        f"# Report of {folder.as_posix()}",
        "",
        f"Written by the run report of bo_attention_smc on "
        f"{time.strftime('%Y-%m-%d %H:%M')}. Campaigns joined from "
        f"{', '.join(p.as_posix() for p in pair_with) or 'no other folder'}"
        " where they share a cell, rule and seed. Intervals: 95 % paired "
        "bootstrap; p: two-sided Wilcoxon, unadjusted.",
        "",
        "## Mean per model",
        "",
        "| cell | rule | model | n | lift | AUC | top-5 % coverage | "
        "minutes per campaign |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for cell in cells:
        for rule in rules:
            for model in models:
                metrics = [
                    r["metrics"]
                    for r in records
                    if (r["cell"], r["rule"], r["arm"]) == (cell, rule, model)
                ]
                if not metrics:
                    continue
                lift = [m["lift"] for m in metrics]
                auc = [m["auc"] for m in metrics]
                coverage = [m["coverage_top5"] for m in metrics]
                minutes = np.mean([m["seconds"] for m in metrics]) / 60
                lines.append(
                    f"| {cell} | {rule} | {model} | {len(metrics)} | "
                    f"{np.mean(lift):.3f} ± {se(lift):.3f} | "
                    f"{np.mean(auc):.4f} | {np.mean(coverage):.3f} ± "
                    f"{se(coverage):.3f} | {minutes:.1f} |"
                )

    rng = np.random.default_rng(0)
    groups = [(c, [r for r in records if r["cell"] == c]) for c in cells]
    datasets = sorted({cell.split("/")[0] for cell in cells})
    if len(datasets) < len(cells):
        for dataset in datasets:
            chosen = [
                r for r in records if r["cell"].startswith(f"{dataset}/")
            ]
            groups.append((f"{dataset} (all featurisers)", chosen))
    groups.append(("all", records))
    lines += ["", "## Contrasts", ""]
    for model, reference, question in CONTRASTS:
        if model not in models or reference not in models:
            continue
        lines += [
            f"**{model} - {reference}**: {question}",
            "",
            "| within | metric | pairs | mean difference | 95 % interval "
            "| p | wins |",
            "|---|---|---|---|---|---|---|",
        ]
        for within, chosen in groups:
            for metric in ("lift", "coverage_top5"):
                found = contrast(chosen, model, reference, metric, rng)
                if found["pairs"]:
                    lines.append(contrast_row(within, metric, found))
        lines.append("")

    sampled = sorted({r["arm"] for r in records if "smc_seconds" in r["fits"]})
    if sampled:
        lines += [
            "## How the sampler went",
            "",
            "| model | tempering steps per BO step | acceptance | "
            "seconds sampling | seconds predicting | unreached |",
            "|---|---|---|---|---|---|",
        ]
        for model in sampled:
            fits = [r["fits"] for r in records if r["arm"] == model]
            mean = {
                name: float(np.mean([np.mean(f[name]) for f in fits]))
                for name in (
                    "smc_tempering",
                    "smc_acceptance",
                    "smc_seconds",
                    "predict_seconds",
                )
            }
            unreached = sum(
                s == "UNREACHED" for f in fits for s in f["status"]
            )
            lines.append(
                f"| {model} | {mean['smc_tempering']:.1f} | "
                f"{mean['smc_acceptance']:.2f} | {mean['smc_seconds']:.2f} | "
                f"{mean['predict_seconds']:.2f} | {unreached} |"
            )
    text = "\n".join(lines) + "\n"
    (folder / "report.md").write_text(text, encoding="utf-8")
    return text


def table_records(folders: tuple[str, ...]) -> tuple[dict, int]:
    """Read the table's campaigns, by (model, rule, featuriser, dataset, seed).

    Returns
    -------
    records, duplicates
        Each campaign's metrics, from the first folder holding it, and how
        many campaigns were found again in a later folder.

    """
    records, duplicates = {}, 0
    for folder in folders:
        if not (Path(folder) / "campaigns.jsonl").is_file():
            continue
        for r in read_campaigns(Path(folder)):
            if r["arm"] not in TABLE_MODELS:
                continue
            dataset, featuriser = r["cell"].split("/")[:2]
            key = (r["arm"], r["rule"], featuriser, dataset, r["seed"])
            if key in records:
                duplicates += 1
                continue
            records[key] = r["metrics"]
    return records, duplicates


def values(
    records: dict,
    model: str,
    rule: str,
    featuriser: str,
    dataset: str,
    seeds: range,
    metric: str = "lift",
) -> dict:
    """One model's metric, by (dataset, seed); `all` takes every dataset."""
    datasets = DATASETS if dataset == "all" else (dataset,)
    return {
        (d, s): records[(model, rule, featuriser, d, s)][metric]
        for d in datasets
        for s in seeds
        if (model, rule, featuriser, d, s) in records
    }


def table_entry(found: dict, model: str, featuriser: str) -> str:
    """Format one model's mean ± se (n) for the table."""
    if featuriser == "morgan" and model in HEADS:
        return "n/a"
    if not found:
        return "—"
    v = np.array(list(found.values()))
    spread = v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else float("nan")
    return f"{v.mean():.3f} ±{spread:.3f} ({len(v)})"


def paired(a: dict, b: dict) -> str:
    """Format the paired difference a - b, from 5 pairs on."""
    keys = [k for k in a if k in b]
    if len(keys) < 5:
        return ""
    d = np.array([a[k] - b[k] for k in keys])
    p = wilcoxon(d).pvalue if np.any(d != 0) else 1.0
    return f"{d.mean():+.3f} (p {p:.2g}, n {len(d)})"


def table_rows(
    records: dict, seeds: range, detail: bool, metric: str
) -> list[str]:
    """One markdown table: every rule and featuriser, by dataset if asked."""
    heads = ["rule", "featuriser", "dataset", *TABLE_MODELS.values()]
    heads += [f"{TABLE_MODELS[a]} − {TABLE_MODELS[b]}" for a, b in DIFFERENCES]
    lines = ["| " + " | ".join(heads) + " |", "|" + "---|" * len(heads)]
    for rule in RULES:
        for featuriser, name in FEATURISERS.items():
            for dataset in (*DATASETS, "all") if detail else ("all",):
                found = {
                    model: values(
                        records,
                        model,
                        rule,
                        featuriser,
                        dataset,
                        seeds,
                        metric,
                    )
                    for model in TABLE_MODELS
                }
                row = [
                    rule,
                    name,
                    "all three" if dataset == "all" else dataset,
                ]
                row += [
                    table_entry(found[model], model, featuriser)
                    for model in TABLE_MODELS
                ]
                row += [paired(found[a], found[b]) for a, b in DIFFERENCES]
                if dataset == "all" and detail:
                    row = [f"**{c}**" if c else c for c in row]
                lines.append("| " + " | ".join(row) + " |")
    return lines


def write_table(out: Path, folders: tuple[str, ...] = TABLE_RUNS) -> None:
    """Write <out>/table.md and <out>/table.csv from the study's runs."""
    out.mkdir(parents=True, exist_ok=True)
    records, duplicates = table_records(folders)
    everything = range(0, 60)
    lines = [
        "# Every arm, rule and featuriser",
        "",
        f"Rebuilt {time.strftime('%Y-%m-%d %H:%M')} by bo_attention_smc's "
        "report (`table`). "
        "Mean ± se (n campaigns) of each metric; differences are paired on "
        "the same dataset and seed (Wilcoxon). Lift is the headline: AUC "
        "mostly says how easy a dataset is, so pooled AUC mixes scales. T5 "
        "= GT4SD multitask-text-and-chemistry-t5-base-augm. `default` = "
        "BoTorch 0.18's LogNormal length-scale prior, started at its mode. "
        "n/a: the head pools atoms, Morgan has none. — : not run yet.",
        "",
        "## Progress",
        "",
        "| arm | done | of |",
        "|---|---|---|",
    ]
    for model, name in TABLE_MODELS.items():
        featurisers = ("mace", "t5_augm") if model in HEADS else FEATURISERS
        total = len(RULES) * len(featurisers) * len(DATASETS) * 60
        done = sum(
            1
            for k in records
            if k[0] == model and k[1] in RULES and k[2] in featurisers
        )
        lines.append(f"| {name} | {done} | {total} |")
    if duplicates:
        lines.append(f"\n{duplicates} duplicate campaigns ignored.")
    for metric, title in METRICS.items():
        lines += ["", f"# {title}", ""]
        lines += ["", "## All seeds (0-59), all three datasets pooled", ""]
        lines += table_rows(records, everything, False, metric)
        lines += ["", "## All seeds (0-59), by dataset", ""]
        lines += table_rows(records, everything, True, metric)
    for metric, title in METRICS.items():
        for name, seeds in SEED_SETS.items():
            lines += ["", f"## {title.split(' (')[0]}, seeds {name}", ""]
            lines += table_rows(records, seeds, True, metric)
    (out / "table.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    rows = []
    seed_sets = {**SEED_SETS, "0-59": everything}
    for rule in RULES:
        for featuriser in FEATURISERS:
            for dataset in (*DATASETS, "all"):
                for set_name, seeds in seed_sets.items():
                    for model in TABLE_MODELS:
                        where = (model, rule, featuriser, dataset, seeds)
                        if not values(records, *where):
                            continue
                        row: dict = {"seeds": set_name, "rule": rule}
                        row.update(featuriser=featuriser, dataset=dataset)
                        row["arm"] = model
                        for metric, short in (
                            ("lift", "lift"),
                            ("coverage_top5", "coverage"),
                            ("auc", "auc"),
                        ):
                            found = values(records, *where, metric)
                            v = np.array(list(found.values()))
                            row["n"] = len(v)
                            row[metric] = round(float(v.mean()), 4)
                            row[f"{short}_se"] = ""
                            if len(v) > 1:
                                spread = v.std(ddof=1) / np.sqrt(len(v))
                                row[f"{short}_se"] = round(float(spread), 4)
                        rows.append(row)
    with (out / "table.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main(argv: list[str] | None = None) -> None:
    """Write a run folder's report, or the table of every run."""
    parser = argparse.ArgumentParser(
        prog="python -m bo_attention_smc.diagnostics_and_metrics.report",
        description=__doc__.split("\n")[0],
    )
    parser.add_argument("what", choices=("run", "table"))
    parser.add_argument(
        "folder", type=Path, help="the run folder, or where the table goes"
    )
    parser.add_argument(
        "--pair-with",
        type=Path,
        nargs="*",
        default=[],
        help="run folders whose campaigns join the run's where they pair",
    )
    args = parser.parse_args(argv)
    if args.what == "run":
        print(run_report(args.folder, args.pair_with))
    else:
        write_table(args.folder)


if __name__ == "__main__":
    main()
