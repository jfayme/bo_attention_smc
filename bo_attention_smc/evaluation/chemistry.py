r"""Whether a model relearns known chemistry.

For recorded campaigns, the model is fitted again on the reactions its
campaign measured, every reaction of the pool is predicted, and the
predictions are compared with what the pool's yields say (`refit`):

- molecules: per reagent, the predicted mean yield of each molecule (over
  the pool's reactions that use it) against the true one, by Spearman's
  rho, and the known chemistry: whether the best molecule is found, the
  halide order I > Br > Cl within each aryl core (bh_1, bh_full), and the
  best two and worst four Shields ligands;
- attention, for a model with a head: where each molecule's attention
  sits, averaged over 64 of the particles: its entropy (1 is uniform), its
  top three atoms (MACE, the lowest conformer) or tokens (T5), and the
  attention on the key atoms (the halogen of an aryl halide, the
  phosphorus of a ligand) over a uniform share;
- the control (`prior`): the attention a head drawn from its prior gives,
  over 4000 draws, since random heads already favour some atoms.

    python -m bo_attention_smc.evaluation.chemistry refit --arm identity \
        --rule chen --seeds 60 --records records/fill/map/campaigns.jsonl \
        --out runs/eval/map.jsonl
    python -m bo_attention_smc.evaluation.chemistry refit \
        --arm smc_span_pca8 --rule geom --featurisers mace --seeds 10 \
        --records records/bo_smc/campaigns.jsonl --out runs/eval/smc.jsonl
    python -m bo_attention_smc.evaluation.chemistry prior \
        --out runs/eval/prior_attention.json
    python -m bo_attention_smc.evaluation.chemistry molecules \
        runs/eval/map.jsonl runs/eval/smc.jsonl
    python -m bo_attention_smc.evaluation.chemistry attention \
        runs/eval/smc.jsonl --prior runs/eval/prior_attention.json

`refit` appends one line per campaign to --out and resumes; MAP fits run on
one CPU thread per worker, SMC fits keep their particles on --device.

"""

import argparse
import functools
import json
import os
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch
from rdkit import Chem
from scipy.stats import spearmanr
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from bo_attention_smc.core.campaign import MODELS, SMCStep
from bo_attention_smc.core.cells import (
    DATASETS,
    Cell,
    cached,
    load_cell,
    molecules,
    read_reactions,
    reagents,
)
from bo_attention_smc.core.conformer import ConformerEnsemble
from bo_attention_smc.core.featuriser import T5Featurizer
from bo_attention_smc.core.gp import pool_bounds
from bo_attention_smc.core.posterior import Bank, unpack
from bo_attention_smc.evaluation.predict import DOMINANT, recorded, refit
from bo_attention_smc.runs.run import cell_path

F64 = torch.float64

# the key atoms of a dominant reagent (predict.DOMINANT)
KEY_ATOMS = {"aryl halide": ("Cl", "Br", "I"), "ligand": ("P",)}
# the particles the attention is averaged over, and the prior's draws
PARTICLES = 64
PRIOR_DRAWS, PRIOR_BATCH = 40, 100


def named_molecules(dataset: str) -> list[dict[int, str]]:
    """Per reagent, each molecule's number (cells.molecules) and SMILES."""
    reactions = read_reactions(dataset)
    numbers = molecules(dataset)
    names = []
    for c, compound in enumerate(DATASETS[dataset].compounds):
        pairs = zip(numbers[:, c].tolist(), reactions[compound], strict=True)
        names.append(dict(pairs))
    return names


@functools.cache
def tokeniser(featuriser: str) -> PreTrainedTokenizerBase:
    """Load a T5 featuriser's tokeniser, once."""
    return T5Featurizer(t5=featuriser[3:]).tokeniser


def atom_labels(smiles: str, featuriser: str) -> list[str]:
    """Name what a head attends to in a molecule: its atoms, or tokens.

    MACE's atoms are the cached conformers' elements, in order; T5's
    tokens are its tokeniser's pieces of the SMILES.
    """
    if featuriser.startswith("t5_"):
        pieces = tokeniser(featuriser)(smiles)["input_ids"]
        return tokeniser(featuriser).convert_ids_to_tokens(pieces)
    table = Chem.GetPeriodicTable()
    return [table.GetElementSymbol(int(z)) for z in conformers(smiles).numbers]


def conformers(smiles: str) -> ConformerEnsemble:
    """Read a molecule's relaxed conformers from the cache (cells.py)."""
    ensemble = ConformerEnsemble.load(cached(smiles), smiles)
    if ensemble is None:
        raise FileNotFoundError(f"no cached conformers for {smiles}")
    return ensemble


def is_key(label: str, key: tuple[str, ...], featuriser: str) -> bool:
    """Whether an atom (MACE) or a token (T5) holds one of the key atoms."""
    if featuriser == "mace":
        return label in key
    return any(x in label.replace("▁", "") for x in key)


def truths(y: np.ndarray, dataset: str) -> list[dict[int, float]]:
    """Per reagent, each molecule's mean yield over the pool's reactions."""
    numbers = molecules(dataset)
    return [
        {int(m): float(y[numbers[:, c] == m].mean()) for m in np.unique(col)}
        for c, col in enumerate(numbers.T)
    ]


def molecule_scores(
    predicted: np.ndarray, y: np.ndarray, dataset: str
) -> dict:
    """Compare predicted molecule means with the true ones, per reagent.

    Parameters
    ----------
    predicted
        The model's mean prediction of every reaction of the pool.
    y
        Every reaction's yield.
    dataset
        The pool's dataset.

    Returns
    -------
    scores
        Per reagent: `rho`, `top1` and each molecule's `predicted` mean;
        for an aryl halide `chloride_lowest` and `order_I_Br_Cl` (the share
        of aryl cores where they hold); for the Shields ligand
        `top2_overlap` and `bottom4_overlap`.

    """
    numbers = molecules(dataset)
    smiles = named_molecules(dataset)
    found = {}
    for c, (name, truth) in enumerate(
        zip(reagents(dataset), truths(y, dataset), strict=True)
    ):
        rows = sorted(truth)
        mean = {m: float(predicted[numbers[:, c] == m].mean()) for m in rows}
        t = np.array([truth[m] for m in rows])
        p = np.array([mean[m] for m in rows])
        entry = {
            "rho": float(spearmanr(t, p)[0]) if len(rows) > 2 else None,
            "top1": bool(np.argmax(p) == np.argmax(t)),
            "predicted": {smiles[c][m]: round(mean[m], 2) for m in rows},
        }
        if name == "aryl halide":
            # each aryl core's halides, by the halogen its SMILES starts with
            cores: dict[str, dict[str, float]] = {}
            for m in rows:
                s = smiles[c][m]
                x = "I"
                if s.startswith("Cl") or s.startswith("Br"):
                    x = s[:2]
                cores.setdefault(s[len(x) :], {})[x] = mean[m]
            entry["chloride_lowest"] = float(
                np.mean(
                    [min(v, key=v.__getitem__) == "Cl" for v in cores.values()]
                )
            )
            entry["order_I_Br_Cl"] = float(
                np.mean([v["I"] > v["Br"] > v["Cl"] for v in cores.values()])
            )
        if name == "ligand" and dataset == "shields":
            order_t, order_p = np.argsort(-t), np.argsort(-p)
            entry["top2_overlap"] = (
                len(set(order_t[:2]) & set(order_p[:2])) / 2
            )
            entry["bottom4_overlap"] = (
                len(set(order_t[-4:]) & set(order_p[-4:])) / 4
            )
        found[name] = entry
    return found


def lowest_conformer(bank: Bank) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the head's inputs and mask, each molecule's lowest conformer."""
    inputs, mask = bank.inputs.cpu(), bank.mask.cpu()
    if inputs.dim() == 4:
        inputs, mask = inputs[:, 0], mask[:, 0]
    return inputs, mask


def attention(
    step: SMCStep, cell: Cell, dataset: str, featuriser: str
) -> dict:
    """Where a sampled head's attention sits, per molecule of every reagent.

    Parameters
    ----------
    step
        The SMC step, after its fit: its particles and its bank.
    cell, dataset, featuriser
        What it was fitted on.

    Returns
    -------
    attention
        Per reagent and molecule (SMILES): its `atoms`, the `entropy` of
        its mean attention over 64 particles (1 is uniform), the `top`
        three atoms with their weights, and `key_ratio`, the attention on
        the key atoms over their uniform share (None without key atoms).

    """
    if step.particles is None or step.data is None:
        raise ValueError("the step has not been fitted yet")
    inputs, mask = lowest_conformer(step.bank)
    particles = step.particles.cpu()
    pick = torch.linspace(0, len(particles) - 1, PARTICLES).long()
    p = unpack(particles[pick], step.bank, step.data.floor)
    weights = []
    for k in range(PARTICLES):
        hidden = torch.tanh(inputs @ p.V[k].T + p.b[k])
        scores = (hidden @ p.w[k].unsqueeze(-1)).squeeze(-1)
        scores = scores.masked_fill(~mask, float("-inf"))
        weights.append((torch.softmax(scores, dim=-1) * mask).numpy())
    mean = np.mean(weights, axis=0)
    found = {}
    for name, own in zip(
        reagents(dataset), named_molecules(dataset), strict=True
    ):
        per = {}
        for m, smiles in own.items():
            n = int(mask[m].sum())
            a = mean[m, :n]
            labels = atom_labels(smiles, featuriser)
            entropy = 1.0
            if n > 1:
                entropy = float(-(a * np.log(a + 1e-300)).sum() / np.log(n))
            ratio = None
            if name in KEY_ATOMS:
                hit = [
                    i
                    for i, label in enumerate(labels)
                    if is_key(label, KEY_ATOMS[name], featuriser)
                ]
                if hit:
                    ratio = round(float(a[hit].sum() / (len(hit) / n)), 2)
            per[smiles] = {
                "atoms": n,
                "entropy": round(entropy, 3),
                "top": [
                    (labels[i], round(float(a[i]), 3))
                    for i in np.argsort(-a)[:3]
                ],
                "key_ratio": ratio,
            }
        found[name] = per
    return found


def refit_task(task: dict, records: tuple[str, ...], device: str) -> dict:
    """Fit one recorded campaign's model again and check its chemistry."""
    torch.set_num_threads(1)
    dataset, featuriser = task["dataset"], task["featuriser"]
    cell = load_cell(cell_path(dataset, featuriser))
    name = f"{dataset}/{featuriser}"
    rows = recorded(records, task["arm"], task["rule"], name, task["seed"])
    step, fitted, record = refit(
        task["arm"], cell, dataset, task["rule"], task["seed"], rows, device
    )
    everything = np.arange(len(cell.objective))
    mean, _ = fitted.predict(everything).moments()
    found = {**task, "n": len(rows)}
    if isinstance(step, SMCStep):
        found["status"] = record["status"]
        found["weights"] = {
            k[2:-1]: v for k, v in record.items() if k.startswith("w[")
        }
    found["molecules"] = molecule_scores(mean.numpy(), cell.objective, dataset)
    if isinstance(step, SMCStep):
        found["attention"] = attention(step, cell, dataset, featuriser)
    return found


def prior_attention() -> dict:
    """Attention on the key atoms under heads drawn from their prior.

    Returns
    -------
    ratios
        By `dataset|featuriser|SMILES`, for the dominant reagent's
        molecules: the attention on the key atoms over their uniform share,
        averaged over 4000 PCA-8 heads with every weight drawn N(0, 1), the
        head prior (generator seeded 0).

    """
    found = {}
    generator = torch.Generator().manual_seed(0)
    for dataset in ("bh_1", "bh_full", "shields"):
        for featuriser in ("mace", "t5_augm"):
            cell = load_cell(cell_path(dataset, featuriser))
            bounds = torch.tensor(pool_bounds(cell.features))
            bank = Bank.of(cell, bounds, "pca8")
            inputs, mask = lowest_conformer(bank)
            h, i = bank.n_hidden, bank.n_inputs
            total = torch.zeros(mask.shape, dtype=F64)
            n = PRIOR_BATCH
            for _ in range(PRIOR_DRAWS):
                v = torch.randn(n, h, i, generator=generator, dtype=F64)
                b = torch.randn(n, h, 1, 1, generator=generator, dtype=F64)
                w = torch.randn(n, 1, 1, h, generator=generator, dtype=F64)
                hidden = torch.tanh(
                    torch.einsum("mni,khi->kmnh", inputs, v)
                    + b.view(n, 1, 1, h)
                )
                scores = (hidden * w).sum(-1).masked_fill(~mask, float("-inf"))
                total = total + (torch.softmax(scores, dim=-1) * mask).sum(0)
            mean = (total / (PRIOR_DRAWS * PRIOR_BATCH)).numpy()
            dominant = DOMINANT[dataset]
            key = KEY_ATOMS[dominant]
            c = reagents(dataset).index(dominant)
            for m, smiles in named_molecules(dataset)[c].items():
                atoms = int(mask[m].sum())
                labels = atom_labels(smiles, featuriser)
                hit = [
                    j
                    for j, label in enumerate(labels)
                    if is_key(label, key, featuriser)
                ]
                ratio = None
                if hit:
                    share = mean[m, hit].sum() / (len(hit) / atoms)
                    ratio = round(float(share), 2)
                found[f"{dataset}|{featuriser}|{smiles}"] = ratio
    return found


def read_lines(paths: list[Path]) -> list[dict]:
    """Read every line of some .jsonl files."""
    rows = []
    for path in paths:
        text = path.read_text(encoding="utf-8")
        rows += [json.loads(line) for line in text.splitlines()]
    return rows


def molecules_summary(paths: list[Path], seeds: int | None) -> None:
    """Print each model's molecule-level chemistry, per dataset.

    The mean Spearman rho of predicted against true molecule means per
    reagent, and the checks of the dominant reagent; `seeds` keeps the
    campaigns of the first seeds only.
    """
    groups = defaultdict(list)
    for r in read_lines(paths):
        if seeds is None or r["seed"] < seeds:
            key = (r["dataset"], r["featuriser"], r["rule"], r["arm"])
            groups[key].append(r)
    order = list(MODELS)
    for dataset in ("bh_1", "bh_full", "shields"):
        print(
            f"== {dataset}: Spearman rho of predicted against true molecule "
            "means (mean over campaigns); checks"
        )
        dominant = DOMINANT[dataset]
        names = [dominant] + [n for n in reagents(dataset) if n != dominant]
        keys = [k for k in groups if k[0] == dataset]
        for key in sorted(keys, key=lambda k: (k[2], k[1], order.index(k[3]))):
            runs = groups[key]
            parts = []
            for name in names:
                rho = [
                    r["molecules"][name]["rho"]
                    for r in runs
                    if r["molecules"][name]["rho"] is not None
                    and not np.isnan(r["molecules"][name]["rho"])
                ]
                parts.append(f"{name} {np.mean(rho):+.2f}")
            first = [r["molecules"][dominant] for r in runs]
            checks = [f"top1 {np.mean([e['top1'] for e in first]):.0%}"]
            if dataset == "shields":
                top2 = np.mean([e["top2_overlap"] for e in first])
                bottom4 = np.mean([e["bottom4_overlap"] for e in first])
                checks += [f"top2 {top2:.0%}", f"bottom4 {bottom4:.0%}"]
            else:
                lowest = np.mean([e["chloride_lowest"] for e in first])
                ordered = np.mean([e["order_I_Br_Cl"] for e in first])
                checks += [f"Cl lowest {lowest:.0%}", f"I>Br>Cl {ordered:.0%}"]
            print(
                f"   {key[2]:7s} {key[1]:7s} {key[3]:17s} n={len(runs):2d}  "
                + "  ".join(parts)
                + "   | "
                + "  ".join(checks)
            )


def attention_summary(paths: list[Path], prior_path: Path) -> None:
    """Print the key-atom attention of the dominant reagent's molecules.

    Learned against the prior's, the entropy, and how often a key atom is
    the top one, the molecules ordered by their true mean yield.
    """
    prior = json.loads(prior_path.read_text(encoding="utf-8"))
    groups = defaultdict(list)
    for r in read_lines(paths):
        groups[(r["dataset"], r["featuriser"], r["arm"])].append(r)
    for (dataset, featuriser, model), runs in sorted(groups.items()):
        dominant = DOMINANT[dataset]
        c = reagents(dataset).index(dominant)
        y = load_cell(cell_path(dataset, featuriser)).objective
        smiles = named_molecules(dataset)[c]
        truth = {smiles[m]: v for m, v in truths(y, dataset)[c].items()}
        ratio, entropy, top = (defaultdict(list) for _ in range(3))
        for r in runs:
            for s, a in r["attention"][dominant].items():
                if a["key_ratio"] is not None:
                    ratio[s].append(a["key_ratio"])
                entropy[s].append(a["entropy"])
                top[s].append(a["top"][0][0])
        print(
            f"== {dataset} {featuriser} {model} (n={len(runs)}): key-atom "
            "attention over uniform, learned against prior; entropy; how "
            "often a key atom is the top one"
        )
        learned, priors = [], []
        for s in sorted(ratio, key=lambda s: -truth[s]):
            p = prior.get(f"{dataset}|{featuriser}|{s}")
            on_top = np.mean(
                [is_key(t, KEY_ATOMS[dominant], featuriser) for t in top[s]]
            )
            learned.append(np.mean(ratio[s]))
            priors.append(p)
            print(
                f"   yield {truth[s]:5.1f}  {s[:40]:40s} learned "
                f"{np.mean(ratio[s]):4.2f}  prior {p}  entropy "
                f"{np.mean(entropy[s]):.2f}  key atom on top {on_top:.0%}"
            )
        print(
            "   mean learned/prior ratio "
            f"{np.mean(np.array(learned) / np.array(priors)):.2f}"
        )


def main(argv: list[str] | None = None) -> None:
    """Refit recorded campaigns, draw the prior control, or summarise."""
    parser = argparse.ArgumentParser(
        prog="python -m bo_attention_smc.evaluation.chemistry",
        description=__doc__.split("\n")[0],
    )
    commands = parser.add_subparsers(dest="command", required=True)
    fits = commands.add_parser("refit", help="refit recorded campaigns")
    fits.add_argument("--arm", choices=list(MODELS), required=True)
    fits.add_argument("--rule", required=True)
    fits.add_argument("--records", nargs="+", required=True)
    fits.add_argument("--out", type=Path, required=True)
    fits.add_argument(
        "--datasets", nargs="+", default=["bh_1", "bh_full", "shields"]
    )
    fits.add_argument("--featurisers", nargs="+", default=["mace", "t5_augm"])
    fits.add_argument("--seeds", type=int, default=10)
    fits.add_argument("--first-seed", type=int, default=0)
    fits.add_argument("--workers", type=int, default=6)
    fits.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    control = commands.add_parser("prior", help="the random-head control")
    control.add_argument("--out", type=Path, required=True)
    by_molecule = commands.add_parser("molecules", help="molecule summary")
    by_molecule.add_argument("files", type=Path, nargs="+")
    by_molecule.add_argument("--seeds", type=int, help="the first seeds only")
    by_atom = commands.add_parser("attention", help="attention summary")
    by_atom.add_argument("files", type=Path, nargs="+")
    by_atom.add_argument("--prior", type=Path, required=True)
    args = parser.parse_args(argv)

    if args.command == "prior":
        found = prior_attention()
        args.out.write_text(json.dumps(found, indent=1), encoding="utf-8")
    elif args.command == "molecules":
        molecules_summary(args.files, args.seeds)
    elif args.command == "attention":
        attention_summary(args.files, args.prior)
    else:
        fields = ("arm", "rule", "dataset", "featuriser", "seed")
        done = set()
        if args.out.is_file():
            for r in read_lines([args.out]):
                done.add(tuple(r[k] for k in fields))
        todo = []
        for dataset in args.datasets:
            for featuriser in args.featurisers:
                for seed in range(
                    args.first_seed, args.first_seed + args.seeds
                ):
                    task = {"arm": args.arm, "rule": args.rule}
                    task.update(dataset=dataset, featuriser=featuriser)
                    task["seed"] = seed
                    if tuple(task[k] for k in fields) not in done:
                        todo.append(task)
        print(f"{len(done)} campaigns done; {len(todo)} to refit", flush=True)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        os.environ["OMP_NUM_THREADS"] = "1"
        os.environ["MKL_NUM_THREADS"] = "1"
        records = tuple(args.records)
        with ProcessPoolExecutor(args.workers) as executor:
            futures = [
                executor.submit(refit_task, t, records, args.device)
                for t in todo
            ]
            for future in futures:
                found = future.result()
                with args.out.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(found) + "\n")


if __name__ == "__main__":
    main()
