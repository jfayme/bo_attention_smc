"""What a campaign achieved: lift, coverage, and how its fits went.

The outcome of a campaign is read from its best-so-far curve: after each
experiment, the best yield found so far, scaled so that the pool's worst
reaction is 0 and its best is 1.

- `auc`: the area under that curve, by trapezoids, over its width.
- `lift`: the AUC rescaled so that random selection scores 0 and a
  campaign that finds the best reaction first scores 1. Random selection's
  AUC depends on the pool and the budget alone (`random_auc`), and a raw
  AUC mostly says how easy a dataset is, so lift is the study's headline.
- `coverage_top5`: the share of the pool's top 5 % of reactions found.

These are bo.metrics' numbers, computed the same way, so they compare with
every campaign the study recorded (tests/test_metrics.py).

"""

import numpy as np

# The random baseline: 400 random campaigns, from a stream no campaign uses
# (seed 1337 + 90210), as the earlier sweep drew it.
RANDOM_AUC_DRAWS = 400
RANDOM_AUC_SEED = 1337 + 90210


def auc_and_coverage(
    sampled_objective: np.ndarray, pool_objective: np.ndarray
) -> tuple[float, float]:
    """Area under a campaign's best-so-far curve, and its top-5 % coverage.

    Parameters
    ----------
    sampled_objective
        Yield of every experiment, in the order they were run.
    pool_objective
        Yield of every reaction of the pool.

    Returns
    -------
    auc
        Area under the scaled best-so-far curve over its width.
    coverage
        Share of the pool's top 5 % found.

    """
    low, high = pool_objective.min(), pool_objective.max()
    best = (np.maximum.accumulate(sampled_objective) - low) / (high - low)
    # a running sum, as bo.trajectory: np.sum adds in another order
    area = np.cumsum((best[1:] + best[:-1]) / 2.0)[-1]
    threshold = np.quantile(pool_objective, 0.95)
    found = np.sum(sampled_objective >= threshold)
    coverage = min(found / np.sum(pool_objective >= threshold), 1.0)
    return float(area / (len(best) - 1)), float(coverage)


def random_auc(pool_objective: np.ndarray, n_iter: int) -> float:
    """Mean AUC of random campaigns of `n_iter` experiments: lift's zero."""
    rng = np.random.default_rng(RANDOM_AUC_SEED)
    budget = min(n_iter, len(pool_objective))
    aucs = []
    for _ in range(RANDOM_AUC_DRAWS):
        rows = rng.choice(len(pool_objective), budget, replace=False)
        aucs.append(auc_and_coverage(pool_objective[rows], pool_objective)[0])
    return float(np.mean(aucs))


def outcome(
    sampled_objective: np.ndarray,
    pool_objective: np.ndarray,
    auc_random: float,
) -> dict:
    """Summarise what a campaign found, as bo.metrics does.

    Parameters
    ----------
    sampled_objective
        Yield of every experiment, in order.
    pool_objective
        Yield of every reaction of the pool.
    auc_random
        The pool's random baseline, from `random_auc`.

    Returns
    -------
    found
        `auc`, `lift`, `coverage_top5`, `simple_regret` (the best yield
        missed, as a share of the pool's range), `best_found` and
        `first_top5_hit` (1-based; NaN if the top 5 % was never reached).

    """
    auc, coverage = auc_and_coverage(sampled_objective, pool_objective)
    low, high = pool_objective.min(), pool_objective.max()
    threshold = np.quantile(pool_objective, 0.95)
    hits = np.flatnonzero(sampled_objective >= threshold)
    return {
        "auc": auc,
        "lift": (auc - auc_random) / (1.0 - auc_random),
        "coverage_top5": coverage,
        "simple_regret": float(
            (high - sampled_objective.max()) / (high - low)
        ),
        "best_found": float(sampled_objective.max()),
        "first_top5_hit": int(hits[0]) + 1 if hits.size else float("nan"),
    }


def summarise(
    record: dict, pool_objective: np.ndarray, auc_random: float
) -> dict:
    """Summarise a campaign for its record.

    Parameters
    ----------
    record
        What campaign.run_campaign returned.
    pool_objective
        Yield of every reaction of the pool.
    auc_random
        The pool's random baseline, from `random_auc`.

    Returns
    -------
    metrics
        The `outcome`, then how the fits went: `fits_failed`, the MAP fits
        whose L-BFGS run failed; `log_score_total`, the prequential log
        score, higher where the model predicted each experiment better
        before measuring it; and `seconds`.

    """
    sampled = pool_objective[record["sampled_indices"]]
    fits = record["fits"]
    metrics = outcome(sampled, pool_objective, auc_random)
    failed = [status in ("FAILURE", "ERROR") for status in fits["status"]]
    metrics["fits_failed"] = int(np.sum(failed))
    metrics["log_score_total"] = float(np.sum(fits["log_score"]))
    metrics["seconds"] = record["seconds"]
    return metrics
