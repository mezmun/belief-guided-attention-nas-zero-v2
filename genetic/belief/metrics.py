"""
Evaluate how pre-training belief scores match real fitness.

Ranking quality is the main target because the method prioritizes candidates
instead of predicting exact accuracy. These cycle metrics are computed only on
the search-evaluated subset. Independent audit evaluations are stored and
reported separately and are never fed back into the search.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Dict, Iterable, List, Optional, Sequence

from .monitoring import EvaluatedBeliefRecord


@dataclass(frozen=True)
class CycleMetrics:
    """Store validation metrics for one search-evaluated offspring cycle."""

    run_id: str
    cycle: int
    evaluated_count: int
    spearman_correlation: Optional[float]
    kendall_tau_b: Optional[float]
    pairwise_accuracy: Optional[float]
    best_true_model_belief_rank: Optional[float]
    pearson_correlation: Optional[float]
    mae: float
    rmse: float
    top_k: int
    top_k_hit_count: int
    top_k_recall: float
    belief_top_k_mean_fitness: float
    true_top_k_mean_fitness: float
    top_k_mean_regret: float
    overall_mean_fitness: float
    best_true_fitness: float
    uncertainty_error_correlation: Optional[float]

    def to_dict(self) -> Dict[str, object]:
        """Return the metrics as a plain dictionary."""

        return asdict(self)


class BeliefMetricsCalculator:
    """Calculate cycle-wise ranking, selection-quality, and diagnostic metrics."""

    VERSION = "2.1"

    def calculate_cycle(
        self,
        records: Iterable[EvaluatedBeliefRecord],
        top_k: int = 5,
    ) -> CycleMetrics:
        """Calculate metrics from completed search rows belonging to one cycle."""

        items = list(records)
        if not items:
            raise ValueError("At least one evaluated belief record is required")
        if top_k < 1:
            raise ValueError("top_k must be at least 1")

        run_ids = {item.run_id for item in items}
        cycles = {item.cycle for item in items}
        if len(run_ids) != 1 or len(cycles) != 1:
            raise ValueError("All records must belong to the same run and cycle")

        beliefs = [item.belief_mean for item in items]
        fitness = [item.true_fitness for item in items]
        errors = [item.absolute_error for item in items]
        squared_errors = [item.squared_error for item in items]

        effective_k = min(top_k, len(items))
        belief_order = sorted(
            range(len(items)), key=lambda index: beliefs[index], reverse=True
        )
        true_order = sorted(
            range(len(items)), key=lambda index: fitness[index], reverse=True
        )
        belief_top = set(belief_order[:effective_k])
        true_top = set(true_order[:effective_k])
        hit_count = len(belief_top.intersection(true_top))

        belief_top_k_mean = self._mean(
            [fitness[index] for index in belief_order[:effective_k]]
        )
        true_top_k_mean = self._mean(
            [fitness[index] for index in true_order[:effective_k]]
        )
        best_true_index = true_order[0]
        descending_belief_ranks = self._average_ranks([-value for value in beliefs])

        uncertainty_pairs = [
            (item.belief_uncertainty, item.absolute_error)
            for item in items
            if item.belief_uncertainty is not None
        ]
        uncertainty_error_correlation = None
        if len(uncertainty_pairs) >= 2:
            uncertainty_error_correlation = self._pearson(
                [float(pair[0]) for pair in uncertainty_pairs],
                [pair[1] for pair in uncertainty_pairs],
            )

        return CycleMetrics(
            run_id=next(iter(run_ids)),
            cycle=next(iter(cycles)),
            evaluated_count=len(items),
            spearman_correlation=self._spearman(beliefs, fitness),
            kendall_tau_b=self._kendall_tau_b(beliefs, fitness),
            pairwise_accuracy=self._pairwise_accuracy(beliefs, fitness),
            best_true_model_belief_rank=float(descending_belief_ranks[best_true_index]),
            pearson_correlation=self._pearson(beliefs, fitness),
            mae=self._mean(errors),
            rmse=float(math.sqrt(self._mean(squared_errors))),
            top_k=effective_k,
            top_k_hit_count=hit_count,
            top_k_recall=float(hit_count / effective_k),
            belief_top_k_mean_fitness=belief_top_k_mean,
            true_top_k_mean_fitness=true_top_k_mean,
            top_k_mean_regret=float(true_top_k_mean - belief_top_k_mean),
            overall_mean_fitness=self._mean(fitness),
            best_true_fitness=max(fitness),
            uncertainty_error_correlation=uncertainty_error_correlation,
        )

    @classmethod
    def _spearman(
        cls,
        left: Sequence[float],
        right: Sequence[float],
    ) -> Optional[float]:
        """Return tie-aware Spearman correlation or None when undefined."""

        if len(left) != len(right):
            raise ValueError("Correlation inputs must have the same length")
        if len(left) < 2:
            return None
        return cls._pearson(cls._average_ranks(left), cls._average_ranks(right))

    @staticmethod
    def _kendall_tau_b(
        left: Sequence[float],
        right: Sequence[float],
    ) -> Optional[float]:
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

    @staticmethod
    def _pairwise_accuracy(
        scores: Sequence[float],
        truth: Sequence[float],
    ) -> Optional[float]:
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

    @staticmethod
    def _pearson(
        left: Sequence[float],
        right: Sequence[float],
    ) -> Optional[float]:
        """Return Pearson correlation or None for constant inputs."""

        if len(left) != len(right):
            raise ValueError("Correlation inputs must have the same length")
        if len(left) < 2:
            return None
        left_mean = sum(left) / len(left)
        right_mean = sum(right) / len(right)
        left_centered = [value - left_mean for value in left]
        right_centered = [value - right_mean for value in right]
        denominator = math.sqrt(sum(value**2 for value in left_centered)) * math.sqrt(
            sum(value**2 for value in right_centered)
        )
        if denominator <= 0:
            return None
        return float(
            sum(a * b for a, b in zip(left_centered, right_centered)) / denominator
        )

    @staticmethod
    def _average_ranks(values: Sequence[float]) -> List[float]:
        """Return ascending average ranks with correct tie handling."""

        indexed = sorted(enumerate(values), key=lambda item: item[1])
        ranks = [0.0] * len(values)
        position = 0
        while position < len(indexed):
            end = position + 1
            while end < len(indexed) and indexed[end][1] == indexed[position][1]:
                end += 1
            average_rank = ((position + 1) + end) / 2.0
            for offset in range(position, end):
                ranks[indexed[offset][0]] = average_rank
            position = end
        return ranks

    @staticmethod
    def _mean(values: Sequence[float]) -> float:
        """Return a finite arithmetic mean."""

        if not values:
            raise ValueError("Cannot calculate the mean of an empty sequence")
        result = float(sum(values) / len(values))
        if not math.isfinite(result):
            raise ValueError("Metric calculation produced a non-finite mean")
        return result
