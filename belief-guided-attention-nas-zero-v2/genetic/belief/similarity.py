"""Interpretable structural similarity for attention-aware ENAS."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from math import isfinite
from typing import Dict, Iterable, List, Mapping, Sequence

from .encoder import ArchitectureEncoding


@dataclass(frozen=True)
class SimilarityWeights:
    """Weights used to combine similarity components.

    V2 deliberately starts from a neutral cold-start prior: all components have
    the same weight. Online calibration may change these weights later using
    only fitness observations from the current run.
    """

    module_sequence: float = 0.125
    base_sequence: float = 0.125
    attention_sequence: float = 0.125
    count_similarity: float = 0.125
    module_bigram: float = 0.125
    position_similarity: float = 0.125
    structural_numeric_similarity: float = 0.125
    capacity_numeric_similarity: float = 0.125

    def normalized(self) -> "SimilarityWeights":
        values = asdict(self)
        if any(not isfinite(value) or value < 0 for value in values.values()):
            raise ValueError("Similarity weights must be finite and non-negative")
        total = sum(values.values())
        if total <= 0:
            raise ValueError("At least one similarity weight must be positive")
        return SimilarityWeights(**{key: value / total for key, value in values.items()})


@dataclass(frozen=True)
class SimilarityBreakdown:
    """Final similarity and each interpretable component."""

    total: float
    module_sequence: float
    base_sequence: float
    attention_sequence: float
    count_similarity: float
    module_bigram: float
    position_similarity: float
    structural_numeric_similarity: float
    capacity_numeric_similarity: float

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)


@dataclass(frozen=True)
class SimilarityMatrix:
    candidate_ids: List[str]
    reference_ids: List[str]
    values: List[List[float]]

    @property
    def shape(self) -> tuple[int, int]:
        return len(self.candidate_ids), len(self.reference_ids)


class ArchitectureSimilarity:
    """Calculate detailed similarity between deterministic encodings."""

    VERSION = "2.0"

    def __init__(self, weights: SimilarityWeights | None = None) -> None:
        self.weights = (weights or SimilarityWeights()).normalized()

    def compare(
        self,
        left: ArchitectureEncoding,
        right: ArchitectureEncoding,
    ) -> SimilarityBreakdown:
        module_sequence = self._sequence_similarity(
            left.module_sequence, right.module_sequence
        )
        base_sequence = self._sequence_similarity(left.base_sequence, right.base_sequence)
        attention_sequence = self._sequence_similarity(
            left.attention_sequence, right.attention_sequence
        )

        # base_attention_pairs is intentionally not included as a separate count
        # feature: it is equivalent to the module label in the current encoding.
        count_similarity = self._mean(
            [
                self._counter_similarity(left.module_counts, right.module_counts),
                self._counter_similarity(left.base_counts, right.base_counts),
                self._counter_similarity(left.attention_counts, right.attention_counts),
            ]
        )

        module_bigram = self._counter_similarity(
            Counter(left.module_bigrams), Counter(right.module_bigrams)
        )

        position_similarity = self._mean(
            [
                self._relative_position_similarity(
                    left.module_sequence, right.module_sequence
                ),
                self._relative_position_similarity(left.base_sequence, right.base_sequence),
                self._relative_position_similarity(
                    left.attention_sequence, right.attention_sequence
                ),
            ]
        )

        structural_numeric_similarity = self._numeric_similarity(
            left.structural_numeric_summary,
            right.structural_numeric_summary,
        )
        capacity_numeric_similarity = self._numeric_similarity(
            left.capacity_numeric_summary,
            right.capacity_numeric_summary,
        )

        weighted_total = (
            self.weights.module_sequence * module_sequence
            + self.weights.base_sequence * base_sequence
            + self.weights.attention_sequence * attention_sequence
            + self.weights.count_similarity * count_similarity
            + self.weights.module_bigram * module_bigram
            + self.weights.position_similarity * position_similarity
            + self.weights.structural_numeric_similarity * structural_numeric_similarity
            + self.weights.capacity_numeric_similarity * capacity_numeric_similarity
        )

        return SimilarityBreakdown(
            total=self._clip(weighted_total),
            module_sequence=module_sequence,
            base_sequence=base_sequence,
            attention_sequence=attention_sequence,
            count_similarity=count_similarity,
            module_bigram=module_bigram,
            position_similarity=position_similarity,
            structural_numeric_similarity=structural_numeric_similarity,
            capacity_numeric_similarity=capacity_numeric_similarity,
        )

    def build_matrix(
        self,
        candidates: Iterable[ArchitectureEncoding],
        references: Iterable[ArchitectureEncoding],
    ) -> SimilarityMatrix:
        candidate_list = list(candidates)
        reference_list = list(references)
        values = [
            [self.compare(candidate, reference).total for reference in reference_list]
            for candidate in candidate_list
        ]
        return SimilarityMatrix(
            candidate_ids=[item.architecture_id for item in candidate_list],
            reference_ids=[item.architecture_id for item in reference_list],
            values=values,
        )

    @classmethod
    def _sequence_similarity(cls, left: Sequence[str], right: Sequence[str]) -> float:
        """Normalized longest-common-subsequence similarity."""

        if not left and not right:
            return 1.0
        if not left or not right:
            return 0.0

        previous = [0] * (len(right) + 1)
        for left_value in left:
            current = [0]
            for index, right_value in enumerate(right, start=1):
                if left_value == right_value:
                    current.append(previous[index - 1] + 1)
                else:
                    current.append(max(previous[index], current[-1]))
            previous = current
        return cls._clip(previous[-1] / max(len(left), len(right)))

    @classmethod
    def _relative_position_similarity(
        cls, left: Sequence[str], right: Sequence[str]
    ) -> float:
        """Compare where matching labels occur on a normalized 0..1 axis.

        Occurrences of the same label are paired in positional order. Unmatched
        occurrences contribute zero. This keeps the score symmetric and works
        for variable-length architectures without introducing hand-crafted
        module semantics.
        """

        if not left and not right:
            return 1.0
        if not left or not right:
            return 0.0

        def normalized_positions(values: Sequence[str]) -> Dict[str, List[float]]:
            grouped: Dict[str, List[float]] = defaultdict(list)
            denominator = max(1, len(values) - 1)
            for index, value in enumerate(values):
                position = 0.5 if len(values) == 1 else index / denominator
                grouped[value].append(float(position))
            return grouped

        left_positions = normalized_positions(left)
        right_positions = normalized_positions(right)
        labels = set(left_positions).union(right_positions)
        numerator = 0.0
        denominator = 0

        for label in labels:
            lvals = left_positions.get(label, [])
            rvals = right_positions.get(label, [])
            denominator += max(len(lvals), len(rvals))
            for lpos, rpos in zip(lvals, rvals):
                numerator += 1.0 - abs(lpos - rpos)

        if denominator == 0:
            return 1.0
        return cls._clip(numerator / denominator)

    @classmethod
    def _counter_similarity(
        cls,
        left: Mapping[str, int],
        right: Mapping[str, int],
    ) -> float:
        keys = set(left).union(right)
        if not keys:
            return 1.0
        intersection = sum(min(left.get(key, 0), right.get(key, 0)) for key in keys)
        union = sum(max(left.get(key, 0), right.get(key, 0)) for key in keys)
        return 1.0 if union == 0 else cls._clip(intersection / union)

    @classmethod
    def _numeric_similarity(
        cls,
        left: Mapping[str, float],
        right: Mapping[str, float],
    ) -> float:
        keys = set(left).intersection(right)
        if not keys:
            return 1.0
        similarities: List[float] = []
        for key in sorted(keys):
            left_value = float(left[key])
            right_value = float(right[key])
            scale = max(abs(left_value), abs(right_value), 1e-12)
            similarities.append(
                cls._clip(1.0 - abs(left_value - right_value) / scale)
            )
        return cls._mean(similarities)

    @staticmethod
    def _mean(values: Sequence[float]) -> float:
        return 1.0 if not values else float(sum(values) / len(values))

    @staticmethod
    def _clip(value: float) -> float:
        return float(min(1.0, max(0.0, value)))


def _make_encoding(
    architecture_id: str,
    modules: List[str],
    bases: List[str],
    attentions: List[str],
) -> ArchitectureEncoding:
    pairs = [f"{attention}-{base}" for attention, base in zip(attentions, bases)]
    return ArchitectureEncoding(
        architecture_id=architecture_id,
        architecture_string="-".join(modules),
        individual_id=architecture_id,
        length=len(modules),
        module_sequence=modules,
        base_sequence=bases,
        attention_sequence=attentions,
        base_attention_pairs=pairs,
        module_counts=dict(Counter(modules)),
        base_counts=dict(Counter(bases)),
        attention_counts=dict(Counter(attentions)),
        module_bigrams=[f"{a}->{b}" for a, b in zip(modules, modules[1:])],
        structural_numeric_summary={
            "length": float(len(modules)),
            "attention_density": 0.5,
        },
        capacity_numeric_summary={"out_channel_mean": 128.0},
        unit_records=[],
    )


def _run_self_test() -> None:
    first = _make_encoding(
        "a",
        ["ca-densenet", "pool", "cbam-resnet"],
        ["densenet", "pool", "resnet"],
        ["ca", "none", "cbam"],
    )
    second = _make_encoding(
        "b",
        ["ca-densenet", "pool", "se-resnet"],
        ["densenet", "pool", "resnet"],
        ["ca", "none", "se"],
    )

    calculator = ArchitectureSimilarity()
    identity = calculator.compare(first, first)
    forward = calculator.compare(first, second)
    backward = calculator.compare(second, first)
    matrix = calculator.build_matrix([first, second], [first])

    assert abs(identity.total - 1.0) < 1e-12
    assert 0.0 <= forward.total <= 1.0
    assert abs(forward.total - backward.total) < 1e-12
    assert matrix.shape == (2, 1)
    assert matrix.values[0][0] == 1.0
    assert abs(sum(asdict(calculator.weights).values()) - 1.0) < 1e-12
    print("Architecture similarity self-test passed.")
    print(forward.to_dict())


if __name__ == "__main__":
    _run_self_test()
