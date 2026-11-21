from __future__ import annotations

import math
from statistics import NormalDist

import numpy as np


def wilson_interval(successes: int, total: int, confidence: float = 0.95):
    if total <= 0 or not 0 <= successes <= total:
        raise ValueError("Invalid binomial counts")
    z = NormalDist().inv_cdf((1 + confidence) / 2)
    p = successes / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    radius = z / denominator * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total))
    return [max(0.0, center - radius), min(1.0, center + radius)]


def group_outcomes(predictions):
    if not predictions:
        raise ValueError("No predictions")
    ids = [row["id"] for row in predictions]
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate prediction IDs")
    original, para_correct, para_count, consistency, worst = [], [], [], [], []
    for row in predictions:
        gold = row["label"]
        pred = row["original"]["prediction"]
        paraphrases = [p["prediction"] for p in row["paraphrases"]]
        if not paraphrases:
            raise ValueError("Every evaluation group needs at least one paraphrase")
        original.append(pred == gold)
        para_correct.append(sum(p == gold for p in paraphrases))
        para_count.append(len(paraphrases))
        consistency.append(all(p == pred for p in paraphrases))
        worst.append(pred == gold and all(p == gold for p in paraphrases))
    return {"original_accuracy": np.asarray(original, dtype=float),
            "paraphrase_correct": np.asarray(para_correct, dtype=float),
            "paraphrase_count": np.asarray(para_count, dtype=float),
            "consistency": np.asarray(consistency, dtype=float),
            "worst_case_accuracy": np.asarray(worst, dtype=float)}


def metrics_from_outcomes(outcomes, indices=None):
    if indices is None:
        indices = np.arange(len(outcomes["original_accuracy"]))
    return {"original_accuracy": float(outcomes["original_accuracy"][indices].mean()),
            "paraphrase_accuracy": float(outcomes["paraphrase_correct"][indices].sum() / outcomes["paraphrase_count"][indices].sum()),
            "consistency": float(outcomes["consistency"][indices].mean()),
            "worst_case_accuracy": float(outcomes["worst_case_accuracy"][indices].mean())}


def metric_report(predictions, bootstrap_samples=2000, seed=42):
    outcomes = group_outcomes(predictions)
    metrics = metrics_from_outcomes(outcomes)
    n = len(predictions)
    counts = {key: [int(outcomes[key].sum()), n]
              for key in ["original_accuracy", "consistency", "worst_case_accuracy"]}
    counts["paraphrase_accuracy"] = [int(outcomes["paraphrase_correct"].sum()), int(outcomes["paraphrase_count"].sum())]
    rng = np.random.default_rng(seed)
    draws = {key: [] for key in metrics}
    for _ in range(bootstrap_samples):
        boot = metrics_from_outcomes(outcomes, rng.integers(0, n, n))
        for key, value in boot.items():
            draws[key].append(value)
    return {
        "groups": n, "paraphrase_realizations": counts["paraphrase_accuracy"][1], "metrics": metrics,
        "counts": counts,
        "wilson_95_ci": {key: wilson_interval(*value) for key, value in counts.items()},
        "cluster_bootstrap_95_ci": {key: np.quantile(value, [0.025, 0.975]).tolist() for key, value in draws.items()},
        "bootstrap_samples": bootstrap_samples, "bootstrap_seed": seed,
        "interval_scope": "Finite evaluation groups, conditional on one trained model. Cluster bootstrap resamples whole question groups; pooled paraphrase Wilson intervals ignore within-group dependence.",
    }


def exact_mcnemar(left, right):
    left, right = np.asarray(left, dtype=bool), np.asarray(right, dtype=bool)
    b = int(np.sum(left & ~right))
    c = int(np.sum(~left & right))
    n = b + c
    if n == 0:
        p = 1.0
    else:
        k = min(b, c)
        log_terms = [math.lgamma(n + 1) - math.lgamma(i + 1) - math.lgamma(n - i + 1) - n * math.log(2)
                     for i in range(k + 1)]
        peak = max(log_terms)
        p = min(1.0, 2 * math.exp(peak) * sum(math.exp(t - peak) for t in log_terms))
    return {"left_only_correct": b, "right_only_correct": c, "discordant_groups": n, "two_sided_p": p}


def paired_comparison(left_predictions, right_predictions, bootstrap_samples=10000, seed=42):
    left_by_id = {row["id"]: row for row in left_predictions}
    right_by_id = {row["id"]: row for row in right_predictions}
    if set(left_by_id) != set(right_by_id):
        raise ValueError("Paired tests require identical question IDs")
    ids = sorted(left_by_id)
    left = [left_by_id[i] for i in ids]
    right = [right_by_id[i] for i in ids]
    for a, b in zip(left, right):
        if a["group_hash"] != b["group_hash"] or a["label"] != b["label"]:
            raise ValueError("Paired tests require identical stems, options, labels and paraphrases")
    a, b = group_outcomes(left), group_outcomes(right)
    original_a, original_b = metrics_from_outcomes(a), metrics_from_outcomes(b)
    rng = np.random.default_rng(seed)
    draws = {key: [] for key in original_a}
    for _ in range(bootstrap_samples):
        indices = rng.integers(0, len(ids), len(ids))
        ma, mb = metrics_from_outcomes(a, indices), metrics_from_outcomes(b, indices)
        for key in draws:
            draws[key].append(ma[key] - mb[key])
    return {
        "difference_direction": "left minus right",
        "metric_difference": {key: original_a[key] - original_b[key] for key in draws},
        "paired_cluster_bootstrap_95_ci": {key: np.quantile(value, [0.025, 0.975]).tolist() for key, value in draws.items()},
        "exact_mcnemar": {key: exact_mcnemar(a[key], b[key]) for key in ["original_accuracy", "consistency", "worst_case_accuracy"]},
        "scope": "Paired test-group uncertainty; multiple-testing correction and training-seed inference are not implied.",
    }
