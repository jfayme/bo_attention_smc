r"""Build the cells of a sweep from the datasets.

A cell is one dataset under one featuriser and one aggregation (cells.py).
By default every cell of the study's sweep is built: bh_full, bh_1 and
Shields under Morgan, both T5 encoders, MACE and AIMNet2, each under
`mean`, `mean_std_max` and `attention`, at decorrelation 0.7 and the lowest
conformer. One cannot be built: Shields under AIMNet2 with attention, whose
bases fall back to MACE, since one head cannot score two potentials. It is
recorded as skipped, with the reason.

    python -m bo_attention_smc.runs.prepare --out runs/cells_new
    python -m bo_attention_smc.runs.prepare --out runs/cells_new \
        --datasets shields --featurisers mace --aggregations attention

Cells are written to <out>/cells/, one .npz each, named as the study named
them (shields__mace__attention.npz), and <out>/cells.json records each one:
its width D, the mean distance of its pool in the unit box, both centres
of the length-scale prior, and its blocks of columns. A cell already built
is not built again, so a stopped preparation resumes.

The models run on the device settings.toml names; the conformer searches
are cached in conformers/. The campaigns read the study's own cells, in
cells/: MACE on the GPU does not repeat itself to the last
digit, so its cells cannot be rebuilt exactly.

"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np

from bo_attention_smc.core.cells import (
    AGGREGATIONS,
    DATASETS,
    FEATURISERS,
    POOLINGS,
    REDUCTIONS,
    Cell,
    build_attention,
    build_fixed,
    describe,
    read_reactions,
    save_cell,
)
from bo_attention_smc.core.gp import make_prior, mean_distance


def summary(cell: Cell, dataset: str, used: list[str]) -> dict:
    """Describe a built cell for cells.json.

    Parameters
    ----------
    cell
        The cell.
    dataset
        Its dataset.
    used
        The featuriser that described each component, after fallbacks.

    Returns
    -------
    entry
        `d`, `D_bar` (the mean distance in the unit box), `ell_chen` and
        `ell_geom` (the two centres of the prior), and `blocks`, each
        component's featuriser and width, then the numeric settings.

    """
    blocks = [
        f"{name}: {featuriser} {block.stop - block.start}"
        for name, featuriser, block in zip(
            DATASETS[dataset].compounds, used, cell.blocks, strict=True
        )
    ]
    blocks += [f"{name}: numeric 1" for name in DATASETS[dataset].numeric]
    entry = {
        "status": "ok",
        "d": cell.features.shape[1],
        "D_bar": mean_distance(cell.features),
        "ell_chen": make_prior(cell.features, "chen").ell_0,
        "ell_geom": make_prior(cell.features, "geom").ell_0,
        "blocks": ", ".join(blocks),
    }
    if cell.atoms is not None:
        values = cell.atoms.values
        n_channels, n_molecules = values.shape[-1], len(values)
        entry["atoms"] = f"{n_channels} channels, {n_molecules} molecules"
    return entry


def same(a: Cell, b: Cell) -> bool:
    """Whether two cells hold the same arrays, dtypes included."""
    if a.blocks != b.blocks or (a.atoms is None) != (b.atoms is None):
        return False
    pairs = [(a.objective, b.objective), (a.features, b.features)]
    if a.atoms is not None and b.atoms is not None:
        if a.atoms.pooling != b.atoms.pooling:
            return False
        if len(a.atoms.columns) != len(b.atoms.columns):
            return False
        for name in ("values", "mask", "weights", "index", "settings"):
            pairs.append((getattr(a.atoms, name), getattr(b.atoms, name)))
        pairs += list(zip(a.atoms.columns, b.atoms.columns, strict=True))
    for x, y in pairs:
        if (x is None) != (y is None):
            return False
        if x is not None and not (
            x.dtype == y.dtype and np.array_equal(np.asarray(x), np.asarray(y))
        ):
            return False
    return True


def main(argv: list[str] | None = None) -> None:
    """Build every cell asked for that is not built yet."""
    parser = argparse.ArgumentParser(
        prog="python -m bo_attention_smc.runs.prepare",
        description=__doc__.split("\n")[0],
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--datasets", nargs="+", choices=list(DATASETS), default=list(DATASETS)
    )
    parser.add_argument(
        "--featurisers", nargs="+", choices=FEATURISERS, default=FEATURISERS
    )
    parser.add_argument(
        "--aggregations", nargs="+", choices=AGGREGATIONS, default=AGGREGATIONS
    )
    parser.add_argument("--reduction", choices=REDUCTIONS, default="decorr0.7")
    parser.add_argument("--pooling", choices=POOLINGS, default="lowest")
    parser.add_argument("--device", help="torch device of the models")
    args = parser.parse_args(argv)
    # the featurisers' progress, on the console
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s (%(name)s)",
        datefmt="%d-%b-%y %H:%M:%S",
    )

    path = args.out / "cells.json"
    status = {}
    if path.is_file():
        status = json.loads(path.read_text(encoding="utf-8"))
    # a folder holds cells of one reduction and one pooling only
    design = {"reduction": args.reduction, "pooling": args.pooling}
    for name, entry in status.items():
        if entry.get("design", design) != design:
            raise SystemExit(f"{name} in {path} was built with {entry}")
    for dataset in args.datasets:
        reactions = read_reactions(dataset)
        for featuriser in args.featurisers:
            # fingerprints have no atoms, so one cell and no aggregation
            aggregations = args.aggregations
            if featuriser == "morgan":
                aggregations = ["mean"]
            for aggregation in aggregations:
                parts = [dataset, featuriser]
                if featuriser != "morgan":
                    parts.append(aggregation)
                name = "/".join(parts)
                if status.get(name, {}).get("status") == "ok":
                    continue
                try:
                    described = describe(
                        reactions,
                        dataset,
                        featuriser,
                        aggregation,
                        args.pooling,
                        args.device,
                    )
                    if aggregation == "attention" and featuriser != "morgan":
                        cell = build_attention(
                            reactions,
                            dataset,
                            described,
                            args.reduction,
                            args.pooling,
                        )
                    else:
                        cell = build_fixed(
                            reactions, dataset, described, args.reduction
                        )
                except Exception as error:  # noqa: BLE001 -- recorded
                    reason = f"{type(error).__name__}: {error}"
                    status[name] = {"status": "skipped", "reason": reason}
                    print(f"skipped {name}: {reason}", flush=True)
                else:
                    save_cell(
                        cell, args.out / "cells" / f"{'__'.join(parts)}.npz"
                    )
                    used = [
                        described[c].used for c in DATASETS[dataset].compounds
                    ]
                    status[name] = {
                        **summary(cell, dataset, used),
                        "design": design,
                    }
                    print(
                        f"built {name:34s} D={cell.features.shape[1]}",
                        flush=True,
                    )
                # written after every cell, so a stopped run keeps its work
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(status, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
