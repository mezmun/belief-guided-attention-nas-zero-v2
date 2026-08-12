"""Dependency-free ranking metrics for zero-cost monitoring."""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence


def average_ranks(values: Sequence[float]) -> List[float]:
    """Return ascending average ranks with tie handling."""

    indexed = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0] * len(values)
    position = 0
    while position < len(indexed):
        end = position + 1
        while end < len(indexed) and indexed[end][1] == indexed[position][1]:
            end += 1
        average = ((position + 1) + end) / 2.0
        for offset in range(position, end):
            ranks[indexed[offset][0]] = average
        position = end
    return ranks


def pearson(left: Sequence[float], right: Sequence[float]) -> Optional[float]:
    if len(left) != len(right):
        raise ValueError("Metric inputs must have the same length")
    if len(left) < 2:
        return None
    left_mean = sum(left) / len(left)
    right_mean = sum(right) / len(right)
    left_centered = [value - left_mean for value in left]
    right_centered = [value - right_mean for value in right]
    denominator = math.sqrt(sum(value * value for value in left_centered)) * math.sqrt(
        sum(value * value for value in right_centered)
    )
    if denominator <= 0:
        return None
    return float(
        sum(a * b for a, b in zip(left_centered, right_centered)) / denominator
    )


def spearman(left: Sequence[float], right: Sequence[float]) -> Optional[float]:
    if len(left) != len(right):
        raise ValueError("Metric inputs must have the same length")
    if len(left) < 2:
        return None
    return pearson(average_ranks(left), average_ranks(right))


def kendall_tau_b(left: Sequence[float], right: Sequence[float]) -> Optional[float]:
    """Return Kendall tau-b, accounting for ties in either sequence."""

    if len(left) != len(right):
        raise ValueError("Metric inputs must have the same length")
    if len(left) < 2:
        return None
    concordant = discordant = tie_left = tie_right = 0
    for first in range(len(left) - 1):
        for second in range(first + 1, len(left)):
            delta_left = left[first] - left[second]
            delta_right = right[first] - right[second]
            if delta_left == 0 and delta_right == 0:
                continue
            if delta_left == 0:
                tie_left += 1
            elif delta_right == 0:
                tie_right += 1
            elif delta_left * delta_right > 0:
                concordant += 1
            else:
                discordant += 1
    denominator = math.sqrt(
        (concordant + discordant + tie_left)
        * (concordant + discordant + tie_right)
    )
    if denominator <= 0:
        return None
    return float((concordant - discordant) / denominator)


def pairwise_accuracy(scores: Sequence[float], truth: Sequence[float]) -> Optional[float]:
    """Return correct pair ordering; prediction ties receive half credit."""

    if len(scores) != len(truth):
        raise ValueError("Metric inputs must have the same length")
    correct = 0.0
    total = 0
    for first in range(len(scores) - 1):
        for second in range(first + 1, len(scores)):
            truth_delta = truth[first] - truth[second]
            if truth_delta == 0:
                continue
            score_delta = scores[first] - scores[second]
            total += 1
            if score_delta == 0:
                correct += 0.5
            elif score_delta * truth_delta > 0:
                correct += 1.0
    if total == 0:
        return None
    return float(correct / total)


def method_metrics(
    scores: Sequence[float],
    truth: Sequence[float],
    top_k: int,
) -> Dict[str, Optional[float] | int]:
    """Calculate the cycle-level metrics used for method comparison."""

    if len(scores) != len(truth):
        raise ValueError("Metric inputs must have the same length")
    if not scores:
        raise ValueError("At least one score is required")
    effective_k = min(max(1, top_k), len(scores))
    score_order = sorted(range(len(scores)), key=lambda index: scores[index], reverse=True)
    truth_order = sorted(range(len(truth)), key=lambda index: truth[index], reverse=True)
    score_top = set(score_order[:effective_k])
    truth_top = set(truth_order[:effective_k])
    best_truth_index = truth_order[0]
    descending_ranks = average_ranks([-value for value in scores])
    return {
        "n": len(scores),
        "spearman": spearman(scores, truth),
        "kendall_tau_b": kendall_tau_b(scores, truth),
        "pairwise_accuracy": pairwise_accuracy(scores, truth),
        "top_k": effective_k,
        "top_k_recall": float(len(score_top.intersection(truth_top)) / effective_k),
        "best_model_rank": float(descending_ranks[best_truth_index]),
    }
