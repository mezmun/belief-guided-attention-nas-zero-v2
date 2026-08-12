"""Selection policies for belief-guided candidate evaluation."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Sequence

import numpy as np

from .encoder import ArchitectureEncoding
from .estimator import BeliefEstimate
from .novelty import NoveltyEstimate
from .uncertainty import UncertaintyEstimate


@dataclass(frozen=True)
class CandidateAssessment:
    individual: Any
    encoding: ArchitectureEncoding
    belief: BeliefEstimate
    uncertainty: UncertaintyEstimate
    novelty: NoveltyEstimate

    @property
    def ucb_score(self) -> float:
        return self.belief.belief_mean + self.uncertainty.uncertainty


@dataclass(frozen=True)
class SelectionDecision:
    assessment: CandidateAssessment
    reason: str
    selection_score: float


class BeliefSelector:
    """Select the 20 search evaluations plus an independent audit sample."""

    def __init__(self, random_seed: int = 2312390) -> None:
        self.random_seed = int(random_seed)

    def select(
        self,
        assessments: Iterable[CandidateAssessment],
        budget: int,
        policy: str,
        ucb_kappa: float,
        quotas: Sequence[float],
        bounded_novelty_quantile: float = 0.75,
        bounded_belief_quantile: float = 0.50,
    ) -> List[SelectionDecision]:
        items = self._unique(list(assessments))
        if budget < 1:
            raise ValueError("budget must be at least 1")
        if not items:
            return []
        budget = min(int(budget), len(items))

        if policy == "mean_topk":
            return self._ranked(items, budget, "mean_topk", lambda x: x.belief.belief_mean)
        if policy == "ucb":
            return self._ranked(
                items,
                budget,
                "ucb",
                lambda x: x.belief.belief_mean + ucb_kappa * x.uncertainty.uncertainty,
            )
        if policy == "novelty":
            return self._ranked(items, budget, "novelty", lambda x: x.novelty.novelty)
        if policy != "quota":
            raise ValueError(f"Unsupported selection policy: {policy}")

        return self._quota_select(
            items=items,
            budget=budget,
            ucb_kappa=ucb_kappa,
            quotas=quotas,
            bounded_novelty_quantile=bounded_novelty_quantile,
            bounded_belief_quantile=bounded_belief_quantile,
        )

    def select_audit(
        self,
        assessments: Iterable[CandidateAssessment],
        excluded_architecture_ids: Iterable[str],
        cycle: int,
        count: int = 1,
    ) -> List[SelectionDecision]:
        """Uniformly sample audit candidates from candidates rejected by search."""

        if count <= 0:
            return []
        excluded = {str(value) for value in excluded_architecture_ids}
        remaining = [
            item
            for item in self._unique(list(assessments))
            if item.encoding.architecture_id not in excluded
        ]
        if not remaining:
            return []
        count = min(int(count), len(remaining))
        rng = random.Random(self.random_seed + int(cycle) * 100003 + 17)
        chosen = rng.sample(remaining, count)
        return [SelectionDecision(item, "random_audit", 0.0) for item in chosen]

    def _quota_select(
        self,
        items: List[CandidateAssessment],
        budget: int,
        ucb_kappa: float,
        quotas: Sequence[float],
        bounded_novelty_quantile: float,
        bounded_belief_quantile: float,
    ) -> List[SelectionDecision]:
        if len(quotas) != 3:
            raise ValueError("quota selection requires mean, UCB, and novelty quotas")
        counts = self._quota_counts(budget, quotas)
        selected: Dict[str, SelectionDecision] = {}

        mean_order = sorted(
            items,
            key=lambda item: (item.belief.belief_mean, item.encoding.architecture_id),
            reverse=True,
        )
        self._take_ranked(
            selected,
            mean_order,
            counts[0],
            "mean_topk",
            lambda item: item.belief.belief_mean,
        )

        ucb_order = sorted(
            items,
            key=lambda item: (
                item.belief.belief_mean + ucb_kappa * item.uncertainty.uncertainty,
                item.encoding.architecture_id,
            ),
            reverse=True,
        )
        self._take_ranked(
            selected,
            ucb_order,
            counts[1],
            "ucb",
            lambda item: item.belief.belief_mean
            + ucb_kappa * item.uncertainty.uncertainty,
        )

        if counts[2] > 0:
            novelty_values = np.asarray(
                [item.novelty.novelty for item in items], dtype=np.float64
            )
            belief_values = np.asarray(
                [item.belief.belief_mean for item in items], dtype=np.float64
            )
            novelty_floor = float(np.quantile(novelty_values, bounded_novelty_quantile))
            belief_floor = float(np.quantile(belief_values, bounded_belief_quantile))

            candidates = [
                item
                for item in items
                if item.encoding.architecture_id not in selected
                and item.novelty.novelty >= novelty_floor
                and item.belief.belief_mean >= belief_floor
            ]
            candidates.sort(
                key=lambda item: (item.novelty.novelty, item.belief.belief_mean),
                reverse=True,
            )

            if not candidates:
                candidates = [
                    item
                    for item in items
                    if item.encoding.architecture_id not in selected
                    and item.novelty.novelty >= novelty_floor
                ]
                candidates.sort(
                    key=lambda item: (item.belief.belief_mean, item.novelty.novelty),
                    reverse=True,
                )

            self._take_ranked(
                selected,
                candidates,
                counts[2],
                "bounded_novelty",
                lambda item: item.novelty.novelty,
            )

        if len(selected) < budget:
            remaining = [
                item for item in mean_order if item.encoding.architecture_id not in selected
            ]
            self._take_ranked(
                selected,
                remaining,
                budget - len(selected),
                "mean_fill",
                lambda item: item.belief.belief_mean,
            )

        return list(selected.values())[:budget]

    @staticmethod
    def _take_ranked(
        selected: Dict[str, SelectionDecision],
        ordered: List[CandidateAssessment],
        needed: int,
        reason: str,
        score_fn: Any,
    ) -> None:
        if needed <= 0:
            return
        count = 0
        for item in ordered:
            key = item.encoding.architecture_id
            if key in selected:
                continue
            selected[key] = SelectionDecision(item, reason, float(score_fn(item)))
            count += 1
            if count >= needed:
                break

    @staticmethod
    def _ranked(
        items: List[CandidateAssessment],
        budget: int,
        reason: str,
        score_fn: Any,
    ) -> List[SelectionDecision]:
        ordered = sorted(items, key=score_fn, reverse=True)[:budget]
        return [SelectionDecision(item, reason, float(score_fn(item))) for item in ordered]

    @staticmethod
    def _unique(items: List[CandidateAssessment]) -> List[CandidateAssessment]:
        result: List[CandidateAssessment] = []
        seen = set()
        for item in items:
            key = item.encoding.architecture_id
            if key not in seen:
                seen.add(key)
                result.append(item)
        return result

    @staticmethod
    def _quota_counts(budget: int, quotas: Sequence[float]) -> List[int]:
        total = float(sum(quotas))
        if total <= 0 or any(value < 0 for value in quotas):
            raise ValueError("quotas must be non-negative and have a positive total")
        raw = [budget * value / total for value in quotas]
        counts = [int(value) for value in raw]
        remaining = budget - sum(counts)
        order = sorted(
            range(len(raw)),
            key=lambda index: raw[index] - counts[index],
            reverse=True,
        )
        for index in order[:remaining]:
            counts[index] += 1
        return counts
