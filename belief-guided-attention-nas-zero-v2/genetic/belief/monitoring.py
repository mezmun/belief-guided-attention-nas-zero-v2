"""Idempotent pre/post evaluation monitoring for belief-guided NAS."""

from __future__ import annotations

import csv
import math
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

from .encoder import ArchitectureEncoding
from .estimator import BeliefEstimate


@dataclass(frozen=True)
class PreEvaluationRecord:
    record_id: str
    run_id: str
    cycle: int
    architecture_id: str
    individual_id: str
    architecture_string: str
    candidate_source: str
    evaluation_role: str
    belief_mean: float
    evidence_strength: float
    effective_neighbour_count: float
    neighbour_disagreement: float
    max_similarity: float
    eligible_neighbour_count: int
    used_neighbour_count: int
    used_prior_only: bool
    model_variance: Optional[float]
    belief_uncertainty: Optional[float]
    raw_uncertainty: Optional[float]
    novelty: Optional[float]
    selected_for_evaluation: bool
    selection_reason: str
    selection_score: Optional[float]
    created_at_utc: str

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class EvaluatedBeliefRecord:
    record_id: str
    run_id: str
    cycle: int
    architecture_id: str
    individual_id: str
    architecture_string: str
    candidate_source: str
    evaluation_role: str
    belief_mean: float
    evidence_strength: float
    effective_neighbour_count: float
    neighbour_disagreement: float
    max_similarity: float
    eligible_neighbour_count: int
    used_neighbour_count: int
    used_prior_only: bool
    model_variance: Optional[float]
    belief_uncertainty: Optional[float]
    raw_uncertainty: Optional[float]
    novelty: Optional[float]
    selected_for_evaluation: bool
    selection_reason: str
    selection_score: Optional[float]
    true_fitness: float
    absolute_error: float
    squared_error: float
    evaluation_source: str
    created_at_utc: str
    evaluated_at_utc: str

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


class BeliefCycleMonitor:
    """Track beliefs before training and attach truth later without duplicates."""

    VERSION = "3.0"

    def __init__(
        self,
        pre_evaluation_csv: Optional[Path] = None,
        evaluated_csv: Optional[Path] = None,
    ) -> None:
        self.pre_evaluation_csv = Path(pre_evaluation_csv) if pre_evaluation_csv else None
        self.evaluated_csv = Path(evaluated_csv) if evaluated_csv else None
        self._pre_records: Dict[str, PreEvaluationRecord] = {}
        self._evaluated_records: Dict[str, EvaluatedBeliefRecord] = {}
        self._existing_pre_ids = self._read_ids(self.pre_evaluation_csv, "record_id")
        self._existing_evaluated_ids = self._read_ids(self.evaluated_csv, "record_id")

    def register_pre_evaluation(
        self,
        encoding: ArchitectureEncoding,
        estimate: BeliefEstimate,
        run_id: str,
        cycle: int,
        selected_for_evaluation: bool,
        selection_reason: str,
        belief_uncertainty: Optional[float] = None,
        raw_uncertainty: Optional[float] = None,
        novelty: Optional[float] = None,
        selection_score: Optional[float] = None,
        candidate_source: str = "unknown",
        evaluation_role: str = "none",
    ) -> PreEvaluationRecord:
        if encoding.architecture_id != estimate.architecture_id:
            raise ValueError("Encoding and belief estimate refer to different architectures")
        record_id = self.make_record_id(
            run_id=run_id,
            cycle=cycle,
            individual_id=encoding.individual_id,
            architecture_id=encoding.architecture_id,
        )

        record = PreEvaluationRecord(
            record_id=record_id,
            run_id=str(run_id),
            cycle=int(cycle),
            architecture_id=encoding.architecture_id,
            individual_id=encoding.individual_id,
            architecture_string=encoding.architecture_string,
            candidate_source=str(candidate_source),
            evaluation_role=str(evaluation_role),
            belief_mean=self._finite(estimate.belief_mean),
            evidence_strength=self._non_negative(estimate.evidence_strength),
            effective_neighbour_count=self._non_negative(estimate.effective_neighbour_count),
            neighbour_disagreement=self._non_negative(estimate.neighbour_disagreement),
            max_similarity=self._unit_interval(estimate.max_similarity),
            eligible_neighbour_count=int(estimate.eligible_neighbour_count),
            used_neighbour_count=int(estimate.used_neighbour_count),
            used_prior_only=bool(estimate.used_prior_only),
            model_variance=self._optional_non_negative(estimate.model_variance),
            belief_uncertainty=self._optional_non_negative(belief_uncertainty),
            raw_uncertainty=self._optional_non_negative(raw_uncertainty),
            novelty=self._optional_unit_interval(novelty),
            selected_for_evaluation=bool(selected_for_evaluation),
            selection_reason=str(selection_reason),
            selection_score=self._optional_finite(selection_score),
            created_at_utc=self._utc_now(),
        )
        self._pre_records[record_id] = record
        if self.pre_evaluation_csv is not None and record_id not in self._existing_pre_ids:
            self.append_csv(self.pre_evaluation_csv, record.to_dict())
            self._existing_pre_ids.add(record_id)
        return record

    def register_evaluation(
        self,
        record_id: str,
        true_fitness: float,
        evaluation_source: str = "training",
    ) -> EvaluatedBeliefRecord:
        if record_id not in self._pre_records:
            raise KeyError(f"Pre-evaluation record was not found: {record_id}")
        if record_id in self._evaluated_records:
            return self._evaluated_records[record_id]

        pre = self._pre_records[record_id]
        if not pre.selected_for_evaluation:
            raise ValueError("Cannot attach fitness to a candidate marked as not selected")
        fitness = self._finite(true_fitness)
        error = fitness - pre.belief_mean
        record = EvaluatedBeliefRecord(
            **pre.to_dict(),
            true_fitness=fitness,
            absolute_error=abs(error),
            squared_error=error**2,
            evaluation_source=str(evaluation_source),
            evaluated_at_utc=self._utc_now(),
        )
        self._evaluated_records[record_id] = record
        if self.evaluated_csv is not None and record_id not in self._existing_evaluated_ids:
            self.append_csv(self.evaluated_csv, record.to_dict())
            self._existing_evaluated_ids.add(record_id)
        return record

    def pre_records(self, cycle: Optional[int] = None) -> List[PreEvaluationRecord]:
        records = list(self._pre_records.values())
        return records if cycle is None else [item for item in records if item.cycle == cycle]

    def evaluated_records(self, cycle: Optional[int] = None) -> List[EvaluatedBeliefRecord]:
        records = list(self._evaluated_records.values())
        return records if cycle is None else [item for item in records if item.cycle == cycle]

    @staticmethod
    def make_record_id(run_id: str, cycle: int, individual_id: str, architecture_id: str) -> str:
        return f"{run_id}:{int(cycle)}:{individual_id}:{architecture_id}"

    @staticmethod
    def append_csv(path: Path, row: Dict[str, object]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        write_header = not path.exists() or path.stat().st_size == 0
        with path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
            if write_header:
                writer.writeheader()
            writer.writerow(row)

    @staticmethod
    def replace_cycle_csv(path: Path, cycle: int, rows: List[Dict[str, object]]) -> None:
        """Replace all rows for one cycle while preserving other cycles."""
        path.parent.mkdir(parents=True, exist_ok=True)
        existing: List[Dict[str, object]] = []
        fields: List[str] = []
        if path.exists() and path.stat().st_size:
            with path.open("r", newline="", encoding="utf-8") as handle:
                reader = csv.DictReader(handle)
                fields = list(reader.fieldnames or [])
                for row in reader:
                    try:
                        row_cycle = int(row.get("cycle", -1))
                    except (TypeError, ValueError):
                        row_cycle = -1
                    if row_cycle != int(cycle):
                        existing.append(dict(row))
        if rows:
            fields = list(rows[0].keys())
        if not fields:
            return
        temporary = path.with_suffix(path.suffix + ".tmp")
        with temporary.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            for row in existing:
                writer.writerow({field: row.get(field) for field in fields})
            for row in rows:
                writer.writerow({field: row.get(field) for field in fields})
            handle.flush()
        temporary.replace(path)

    @staticmethod
    def _read_ids(path: Optional[Path], field: str) -> set[str]:
        if path is None or not path.exists() or path.stat().st_size == 0:
            return set()
        with path.open("r", newline="", encoding="utf-8") as handle:
            return {
                str(row.get(field))
                for row in csv.DictReader(handle)
                if row.get(field)
            }

    @staticmethod
    def _utc_now() -> str:
        return datetime.now(timezone.utc).isoformat()

    @staticmethod
    def _finite(value: float) -> float:
        clean = float(value)
        if not math.isfinite(clean):
            raise ValueError("A recorded numeric value must be finite")
        return clean

    @classmethod
    def _non_negative(cls, value: float) -> float:
        clean = cls._finite(value)
        if clean < 0:
            raise ValueError("A recorded value must be non-negative")
        return clean

    @classmethod
    def _unit_interval(cls, value: float) -> float:
        clean = cls._finite(value)
        if not 0.0 <= clean <= 1.0:
            raise ValueError("A recorded value must be inside [0, 1]")
        return clean

    @classmethod
    def _optional_non_negative(cls, value: Optional[float]) -> Optional[float]:
        return None if value is None else cls._non_negative(value)

    @classmethod
    def _optional_unit_interval(cls, value: Optional[float]) -> Optional[float]:
        return None if value is None else cls._unit_interval(value)

    @classmethod
    def _optional_finite(cls, value: Optional[float]) -> Optional[float]:
        return None if value is None else cls._finite(value)
