r"""Run BO campaigns of any model on the study's cells.

Every (dataset, featuriser, model, rule, seed) asked for that the run
folder does not hold yet is run, each campaign 5 random then 50 BO
experiments. The models are the labels of campaign.MODELS.

    # the per-molecule term and mean pooling, by MAP, on the CPU
    python -m bo_attention_smc.runs.run --out runs/x --arms null identity \
        --featurisers mace t5_augm morgan --rules chen --seeds 20

    # the SMC head with the term, 6 campaigns at once on the GPU
    python -m bo_attention_smc.runs.run --out runs/y \
        --arms smc_identity_head --featurisers mace --workers 6

    # a smoke test: one seed, two BO steps
    python -m bo_attention_smc.runs.run --out <scratch>/smoke --seeds 1 \
        --bo-steps 2 --datasets shields --featurisers mace --arms identity

The cells are the study's, in cells/: a featuriser's attention
cell where it has one (MACE, T5), else its own cell (Morgan). A model with
a head needs atoms, so it fails, recorded as such, on Morgan.

Each worker runs one campaign on one CPU thread, as the recorded campaigns
were run: at another thread count a long fit can land elsewhere. SMC
models keep their particles on --device. A campaign is appended to
<out>/campaigns.jsonl as it ends, so a stopped run resumes where it was,
and a failed one is recorded with its error and run again next time.
<out>/config.json holds the design, and a folder of another design is
refused.

"""

import argparse
import dataclasses
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import torch

from bo_attention_smc.core import smc
from bo_attention_smc.core.campaign import (
    HELD,
    MODELS,
    N_BO,
    N_INIT,
    SEED_BASE,
    run_campaign,
)
from bo_attention_smc.core.cells import DATASETS, load_cell
from bo_attention_smc.core.gp import CV, RULES, TERM_CONCENTRATION, TERM_RATE
from bo_attention_smc.diagnostics_and_metrics.metrics import (
    random_auc,
    summarise,
)

CELLS = Path("cells")


def cell_path(dataset: str, featuriser: str) -> Path:
    """Return the study's cell of a dataset under a featuriser."""
    path = CELLS / f"{dataset}__{featuriser}__attention.npz"
    if path.is_file():
        return path
    return CELLS / f"{dataset}__{featuriser}.npz"


def campaign_task(task: dict) -> dict:
    """Run one campaign in a worker process.

    Parameters
    ----------
    task
        The cell's name and path, the model, rule and seed, the number of
        BO steps, the device and the pool's random baseline.

    Returns
    -------
    record
        The cell, model (`arm`), rule and seed, then `status` ok with the
        campaign and its metrics, or `status` failed with the error: one
        failed campaign does not stop the others.

    """
    record = {key: task[key] for key in ("cell", "arm", "rule", "seed")}
    try:
        cell = load_cell(task["path"])
        found = run_campaign(
            cell,
            task["cell"].split("/")[0],
            task["arm"],
            task["rule"],
            task["seed"],
            task["n_bo"],
            task["device"],
        )
        found["metrics"] = summarise(found, cell.objective, task["auc_random"])
    except Exception as error:  # noqa: BLE001 -- recorded; the run goes on
        reason = f"{type(error).__name__}: {error}"
        return {**record, "status": "failed", "error": reason}
    return {**record, "status": "ok", **found}


def main(argv: list[str] | None = None) -> None:
    """Run every campaign asked for that the run folder does not hold."""
    parser = argparse.ArgumentParser(
        prog="python -m bo_attention_smc.runs.run",
        description=__doc__.split("\n")[0],
    )
    parser.add_argument("--out", type=Path, required=True, help="run folder")
    parser.add_argument(
        "--datasets", nargs="+", choices=list(DATASETS), default=list(DATASETS)
    )
    parser.add_argument(
        "--featurisers", nargs="+", default=["mace"], help="e.g. mace t5_augm"
    )
    parser.add_argument(
        "--arms", nargs="+", choices=list(MODELS), required=True
    )
    parser.add_argument("--rules", nargs="+", choices=RULES, default=["chen"])
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--first-seed", type=int, default=0)
    parser.add_argument("--bo-steps", type=int, default=N_BO)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="where the SMC particles live",
    )
    args = parser.parse_args(argv)

    # a run folder holds campaigns of one design only
    config = {
        "n_init": N_INIT,
        "n_bo": args.bo_steps,
        "seed_base": SEED_BASE,
        "cv": CV,
        "term_prior": [TERM_CONCENTRATION, TERM_RATE],
        "held_head_scale": HELD,
        "sampler": dataclasses.asdict(smc.Settings()),
    }
    args.out.mkdir(parents=True, exist_ok=True)
    stored = args.out / "config.json"
    before = stored.is_file() and json.loads(
        stored.read_text(encoding="utf-8")
    )
    if before and before != config:
        raise SystemExit(f"{args.out} holds campaigns of another design")
    stored.write_text(json.dumps(config, indent=1), encoding="utf-8")

    records = args.out / "campaigns.jsonl"
    done = set()
    if records.is_file():
        for line in records.read_text(encoding="utf-8").splitlines():
            record = json.loads(line)
            if record["status"] == "ok":
                keys = ("cell", "arm", "rule", "seed")
                done.add(tuple(record[key] for key in keys))

    tasks = []
    seeds = range(args.first_seed, args.first_seed + args.seeds)
    for dataset in args.datasets:
        for featuriser in args.featurisers:
            name = f"{dataset}/{featuriser}"
            path = cell_path(dataset, featuriser)
            # lift's zero depends on the pool and the budget alone
            auc_random = random_auc(
                load_cell(path).objective, N_INIT + args.bo_steps
            )
            for arm in args.arms:
                for rule in args.rules:
                    for seed in seeds:
                        if (name, arm, rule, seed) in done:
                            continue
                        task = {"cell": name, "arm": arm, "rule": rule}
                        task.update(seed=seed, path=str(path))
                        task.update(n_bo=args.bo_steps, device=args.device)
                        task["auc_random"] = auc_random
                        tasks.append(task)
    # the SMC models first: they take minutes, MAP ones less
    tasks.sort(key=lambda task: MODELS[task["arm"]].head is None)
    print(f"{len(done)} campaigns in {records}; {len(tasks)} to run")

    # one thread per worker, for torch and for the BLAS under numpy
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    start = time.perf_counter()
    with ProcessPoolExecutor(
        args.workers, initializer=torch.set_num_threads, initargs=(1,)
    ) as executor:
        futures = [executor.submit(campaign_task, task) for task in tasks]
        for count, future in enumerate(as_completed(futures), 1):
            record = future.result()
            with records.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record) + "\n")
            line = (
                f"[{count}/{len(tasks)}] {record['cell']:16s} "
                f"{record['arm']:18s} {record['rule']:7s} "
                f"seed {record['seed']:<3d}"
            )
            if record["status"] == "ok":
                metrics = record["metrics"]
                line += (
                    f" lift {metrics['lift']:+.3f}  "
                    f"{metrics['seconds'] / 60:5.1f} min"
                )
            else:
                line += f" FAILED {record['error'][:100]}"
            elapsed = (time.perf_counter() - start) / 60
            print(f"{line}  (elapsed {elapsed:.0f} min)", flush=True)


if __name__ == "__main__":
    main()
