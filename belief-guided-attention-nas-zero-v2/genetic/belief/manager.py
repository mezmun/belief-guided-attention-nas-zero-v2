"""Single integration point for belief-guided candidate ranking and monitoring."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from .archive import EvaluatedArchitectureArchive
from .calibration import SimilarityWeightCalibrator, UncertaintyCalibrator
from .config import BeliefConfig
from .encoder import ArchitectureEncoder
from .estimator import SimilarityBeliefEstimator
from .metrics import BeliefMetricsCalculator, CycleMetrics
from .monitoring import BeliefCycleMonitor, EvaluatedBeliefRecord
from .novelty import ArchitectureNovelty
from .selector import BeliefSelector, CandidateAssessment, SelectionDecision
from .similarity import ArchitectureSimilarity, SimilarityWeights
from .storage import ArchiveStorage
from .uncertainty import BeliefUncertaintyEstimator


@dataclass
class CyclePreparation:
    cycle: int
    original_candidate_count: int
    unique_candidate_count: int
    known_candidate_count: int
    assessed_candidate_count: int
    selected_individuals: List[Any]
    audit_individuals: List[Any]
    selected_record_ids: Dict[str, str]
    audit_record_ids: Dict[str, str]
    selection_reasons: Dict[str, str]


class BeliefManager:
    """Coordinate similarity, belief, uncertainty, selection, archive, and logging."""

    VERSION = "2.0"

    def __init__(
        self,
        config: Optional[BeliefConfig] = None,
        log: Optional[Any] = None,
        restore_existing: bool = False,
    ) -> None:
        self.config = config or BeliefConfig.from_ini()
        self.log = log
        self.base_output_directory = self.config.output_path()
        self.base_output_directory.mkdir(parents=True, exist_ok=True)
        self.active_run_path = self.base_output_directory / "active_run.txt"
        self.run_id = self._resolve_run_id(restore_existing)
        self.output_directory = self.base_output_directory / self.run_id
        self.output_directory.mkdir(parents=True, exist_ok=True)

        self.archive_path = self.output_directory / "belief_archive.json"
        self.state_path = self.output_directory / "belief_state.json"
        self.pre_csv = self.output_directory / "candidate_pre_evaluation.csv"
        self.evaluated_csv = self.output_directory / "evaluated_offspring.csv"
        self.metrics_csv = self.output_directory / "cycle_metrics.csv"
        self.selection_csv = self.output_directory / "selection_summary.csv"
        self.weights_csv = self.output_directory / "similarity_weights_history.csv"

        self.encoder = ArchitectureEncoder()
        self.archive = EvaluatedArchitectureArchive(self.encoder)
        self.similarity = ArchitectureSimilarity()
        self.estimator = SimilarityBeliefEstimator(
            similarity=self.similarity,
            kernel_bandwidth=self.config.kernel_bandwidth,
            method=self.config.belief_method,
        )
        self.uncertainty_calibrator = UncertaintyCalibrator(
            ridge_alpha=self.config.calibration_ridge_alpha
        )
        self.uncertainty_estimator = BeliefUncertaintyEstimator(
            calibrator=self.uncertainty_calibrator
        )
        self.novelty = ArchitectureNovelty(
            similarity=self.similarity,
            top_k=self.config.novelty_neighbours,
        )
        self.selector = BeliefSelector(self.config.random_seed)
        self.monitor = BeliefCycleMonitor(self.pre_csv, self.evaluated_csv)
        self.metrics = BeliefMetricsCalculator()
        self.similarity_calibrator = SimilarityWeightCalibrator(
            target_tau=self.config.similarity_target_tau,
            ridge_alpha=self.config.similarity_ridge_alpha,
            max_pairs=self.config.similarity_max_pairs,
            random_seed=self.config.random_seed,
        )

        self.last_cycle = -1
        self.calibration_samples: List[Dict[str, float]] = []
        self.current_preparation: Optional[CyclePreparation] = None
        self._selected_prediction_snapshot: Dict[str, Dict[str, Any]] = {}
        self._loaded_archive = False
        if restore_existing:
            self._restore_state()

    @property
    def is_enabled(self) -> bool:
        return self.config.enabled

    @property
    def is_monitoring(self) -> bool:
        return self.is_enabled and self.config.mode == "monitor"

    @property
    def is_guided(self) -> bool:
        return self.is_enabled and self.config.mode == "guided"

    def guided_active(self, cycle: int) -> bool:
        return (
            self.is_guided
            and int(cycle) > self.config.warmup_generations
            and len(self.archive) >= self.config.minimum_archive_size
        )

    def candidate_target_size(self, population_size: int, cycle: int) -> int:
        return (
            int(population_size * self.config.candidate_multiplier)
            if self.guided_active(cycle)
            else int(population_size)
        )

    def selected_prediction_snapshot(self, cycle: int) -> Dict[str, Dict[str, Any]]:
        if self.current_preparation is None or self.current_preparation.cycle != int(cycle):
            return {}
        return {key: dict(value) for key, value in self._selected_prediction_snapshot.items()}

    def excluded_architecture_ids(self, cache_map: Optional[Dict[str, Any]] = None) -> set[str]:
        excluded = set(self.archive.architecture_ids())
        if cache_map:
            excluded.update(str(key) for key in cache_map.keys())
        return excluded

    def bootstrap_population(self, individuals: Iterable[Any], generation: int) -> None:
        if not self.is_enabled:
            return
        count_as_measurement = not self._loaded_archive
        for individual in individuals:
            if float(getattr(individual, "acc", -1.0)) < 0:
                continue
            self.archive.add_individual(
                individual=individual,
                generation=generation,
                run_id=self.run_id,
                source="bootstrap",
                count_as_new_measurement=count_as_measurement,
            )
        self.last_cycle = max(self.last_cycle, int(generation))
        self._save_state(generation)
        self._write_weights_history(generation, "bootstrap")
        self._info("Belief archive bootstrapped with %d unique architectures" % len(self.archive))

    def prepare_cycle(
        self,
        candidates: Iterable[Any],
        cycle: int,
        cache_map: Optional[Dict[str, Any]] = None,
    ) -> CyclePreparation:
        if not self.is_enabled:
            raise RuntimeError("prepare_cycle was called while the belief system is disabled")

        cycle = int(cycle)
        candidate_list = list(candidates)
        cache = cache_map or {}
        self._selected_prediction_snapshot = {}
        encoded_candidates = [(individual, self.encoder.encode(individual)) for individual in candidate_list]

        unique_by_architecture: Dict[str, tuple[Any, Any]] = {}
        for individual, encoding in encoded_candidates:
            unique_by_architecture.setdefault(encoding.architecture_id, (individual, encoding))
        unique = list(unique_by_architecture.values())

        unknown: List[tuple[Any, Any]] = []
        known_fitness: Dict[str, float] = {}
        known_count = 0
        for individual, encoding in unique:
            architecture_id = encoding.architecture_id
            if self.archive.contains(architecture_id):
                known_fitness[architecture_id] = self.archive.get(architecture_id).fitness_mean
                known_count += 1
                continue
            if architecture_id in cache:
                cached_fitness = float(cache[architecture_id])
                individual.acc = cached_fitness
                self.archive.add_individual(
                    individual=individual,
                    generation=cycle,
                    run_id=self.run_id,
                    source="cache",
                    count_as_new_measurement=False,
                )
                known_fitness[architecture_id] = cached_fitness
                known_count += 1
                continue
            unknown.append((individual, encoding))

        for individual, encoding in encoded_candidates:
            if encoding.architecture_id in known_fitness:
                individual.acc = known_fitness[encoding.architecture_id]

        assessments: List[CandidateAssessment] = []
        if len(self.archive) > 0:
            for individual, encoding in unknown:
                belief = self.estimator.estimate_one(
                    candidate=encoding,
                    archive=self.archive,
                    top_neighbours=self.config.top_neighbours,
                    exclude_exact_match=self.config.exclude_exact_matches,
                )
                uncertainty = self.uncertainty_estimator.estimate(belief, self.archive)
                novelty = self.novelty.estimate(encoding, self.archive)
                assessments.append(
                    CandidateAssessment(
                        individual=individual,
                        encoding=encoding,
                        belief=belief,
                        uncertainty=uncertainty,
                        novelty=novelty,
                    )
                )

        assessment_by_architecture = {item.encoding.architecture_id: item for item in assessments}
        search_decisions = self._select_assessments(assessments, cycle)
        search_architecture_ids = {
            item.assessment.encoding.architecture_id for item in search_decisions
        }
        audit_decisions: List[SelectionDecision] = []
        if self.guided_active(cycle) and self.config.audit_count > 0:
            audit_decisions = self.selector.select_audit(
                assessments=assessments,
                excluded_architecture_ids=search_architecture_ids,
                cycle=cycle,
                count=self.config.audit_count,
            )
        search_by_architecture = {
            item.assessment.encoding.architecture_id: item for item in search_decisions
        }
        audit_by_architecture = {
            item.assessment.encoding.architecture_id: item for item in audit_decisions
        }

        guided_now = self.guided_active(cycle)
        selected_individuals = (
            [item.assessment.individual for item in search_decisions]
            if guided_now
            else candidate_list
        )
        audit_individuals = [item.assessment.individual for item in audit_decisions]

        selected_record_ids: Dict[str, str] = {}
        audit_record_ids: Dict[str, str] = {}
        selection_reasons: Dict[str, str] = {}

        for individual, encoding in encoded_candidates:
            assessment = assessment_by_architecture.get(encoding.architecture_id)
            if assessment is None:
                continue
            individual_id = str(getattr(individual, "id", "unknown"))
            search_decision = search_by_architecture.get(encoding.architecture_id)
            audit_decision = audit_by_architecture.get(encoding.architecture_id)
            selected = False
            role = "none"
            reason = "not_selected"
            score = None

            if search_decision is not None:
                is_representative = individual_id == search_decision.assessment.encoding.individual_id
                if is_representative:
                    selected = True
                    role = "search"
                    reason = search_decision.reason
                    score = search_decision.selection_score
                else:
                    reason = "duplicate_not_selected"
                    score = search_decision.selection_score
            elif audit_decision is not None:
                is_representative = individual_id == audit_decision.assessment.encoding.individual_id
                if is_representative:
                    selected = True
                    role = "audit"
                    reason = "random_audit"
                    score = audit_decision.selection_score
                else:
                    reason = "duplicate_not_selected"

            setattr(individual, "evaluation_role", role)
            candidate_source = str(getattr(individual, "candidate_source", "unknown"))
            record = self.monitor.register_pre_evaluation(
                encoding=encoding,
                estimate=assessment.belief,
                run_id=self.run_id,
                cycle=cycle,
                selected_for_evaluation=selected,
                selection_reason=reason,
                belief_uncertainty=assessment.uncertainty.uncertainty,
                raw_uncertainty=assessment.uncertainty.raw_uncertainty,
                novelty=assessment.novelty.novelty,
                selection_score=score,
                candidate_source=candidate_source,
                evaluation_role=role,
            )
            if selected:
                setattr(individual, "selection_reason", reason)
                setattr(individual, "selection_score", score)
                selection_reasons[individual_id] = reason
                snapshot = {
                    "architecture_id": encoding.architecture_id,
                    "belief_mean": assessment.belief.belief_mean,
                    "belief_uncertainty": assessment.uncertainty.uncertainty,
                    "raw_uncertainty": assessment.uncertainty.raw_uncertainty,
                    "novelty": assessment.novelty.novelty,
                    "selection_reason": reason,
                    "selection_score": score,
                    "evaluation_role": role,
                    "candidate_source": candidate_source,
                }
                self._selected_prediction_snapshot[individual_id] = snapshot
                if role == "search":
                    selected_record_ids[individual_id] = record.record_id
                elif role == "audit":
                    audit_record_ids[individual_id] = record.record_id

        for individual in selected_individuals:
            setattr(individual, "evaluation_role", "search")
            if not hasattr(individual, "selection_reason"):
                setattr(individual, "selection_reason", "known_or_warmup")
        for individual in audit_individuals:
            setattr(individual, "evaluation_role", "audit")
            setattr(individual, "selection_reason", "random_audit")

        preparation = CyclePreparation(
            cycle=cycle,
            original_candidate_count=len(candidate_list),
            unique_candidate_count=len(unique),
            known_candidate_count=known_count,
            assessed_candidate_count=len(assessments),
            selected_individuals=selected_individuals,
            audit_individuals=audit_individuals,
            selected_record_ids=selected_record_ids,
            audit_record_ids=audit_record_ids,
            selection_reasons=selection_reasons,
        )
        self.current_preparation = preparation
        self._write_selection_summary(preparation)
        self._info(
            "Belief cycle %d: candidates=%d, unique=%d, known=%d, search=%d, audit=%d"
            % (
                cycle,
                len(candidate_list),
                len(unique),
                known_count,
                len(selected_individuals),
                len(audit_individuals),
            )
        )
        return preparation

    def post_evaluate(
        self,
        evaluated_individuals: Iterable[Any],
        cycle: int,
        audit_individuals: Optional[Iterable[Any]] = None,
    ) -> Optional[CycleMetrics]:
        if not self.is_enabled:
            return None
        cycle = int(cycle)
        if self.current_preparation is None or self.current_preparation.cycle != cycle:
            raise RuntimeError("No matching belief preparation exists for this cycle")

        completed_search: List[EvaluatedBeliefRecord] = []
        for individual in evaluated_individuals:
            individual_id = str(getattr(individual, "id", "unknown"))
            fitness = float(getattr(individual, "acc", -1.0))
            if fitness < 0:
                raise ValueError(f"Search-selected individual has no real fitness: {individual_id}")
            record_id = self.current_preparation.selected_record_ids.get(individual_id)
            if record_id is not None:
                record = self.monitor.register_evaluation(
                    record_id=record_id,
                    true_fitness=fitness,
                    evaluation_source="training",
                )
                completed_search.append(record)
                if self._use_for_uncertainty_calibration(record):
                    self.calibration_samples.append(
                        {
                            "evidence_strength": record.evidence_strength,
                            "neighbour_disagreement": record.neighbour_disagreement,
                            "effective_neighbour_count": record.effective_neighbour_count,
                            "absolute_error": record.absolute_error,
                        }
                    )
            self.archive.add_individual(
                individual=individual,
                generation=cycle,
                run_id=self.run_id,
                source="training" if record_id is not None else "inherited_or_duplicate",
                count_as_new_measurement=record_id is not None,
            )

        for individual in list(audit_individuals or []):
            individual_id = str(getattr(individual, "id", "unknown"))
            fitness = float(getattr(individual, "acc", -1.0))
            if fitness < 0:
                raise ValueError(f"Audit individual has no real fitness: {individual_id}")
            record_id = self.current_preparation.audit_record_ids.get(individual_id)
            if record_id is not None:
                self.monitor.register_evaluation(
                    record_id=record_id,
                    true_fitness=fitness,
                    evaluation_source="audit_training",
                )
            # Audit truth is deliberately not added to the belief archive and is not
            # used for uncertainty or similarity calibration.

        cycle_metrics = None
        if completed_search:
            cycle_metrics = self.metrics.calculate_cycle(
                completed_search, top_k=min(5, len(completed_search))
            )
            BeliefCycleMonitor.replace_cycle_csv(
                self.metrics_csv, cycle, [cycle_metrics.to_dict()]
            )

        self._update_calibration(cycle)
        self.last_cycle = cycle
        self._save_state(cycle)
        self._write_weights_history(cycle, "post_update")
        self.current_preparation = None
        return cycle_metrics

    def describe(self) -> Dict[str, Any]:
        return {
            "enabled": self.is_enabled,
            "mode": self.config.mode,
            "version": self.VERSION,
            "run_id": self.run_id,
            "archive_size": len(self.archive),
            "belief_method": self.config.belief_method,
            "top_neighbours": self.config.top_neighbours,
            "similarity_weights": asdict(self.similarity.weights),
            "uncertainty_calibrated": self.uncertainty_calibrator.state.fitted,
        }

    def _select_assessments(
        self, assessments: List[CandidateAssessment], cycle: int
    ) -> List[SelectionDecision]:
        if not assessments:
            return []
        if not self.guided_active(cycle):
            reason = "monitor_all" if self.is_monitoring else "warmup_all"
            return [
                SelectionDecision(
                    assessment=item,
                    reason=reason,
                    selection_score=item.belief.belief_mean,
                )
                for item in assessments
            ]
        return self.selector.select(
            assessments=assessments,
            budget=self.config.evaluation_budget,
            policy=self.config.selection_policy,
            ucb_kappa=self.config.ucb_kappa,
            quotas=(
                self.config.mean_quota,
                self.config.ucb_quota,
                self.config.novelty_quota,
            ),
            bounded_novelty_quantile=self.config.bounded_novelty_quantile,
            bounded_belief_quantile=self.config.bounded_belief_quantile,
        )

    def _use_for_uncertainty_calibration(self, record: EvaluatedBeliefRecord) -> bool:
        # The independent audit is never fed back into search. By default the
        # warm-up calibration is frozen once guided selection starts.
        if record.cycle <= self.config.warmup_generations:
            return True
        if self.config.freeze_uncertainty_after_warmup:
            return False
        return record.evaluation_role == "search"

    def _update_calibration(self, cycle: int) -> None:
        if cycle % self.config.calibration_update_frequency != 0:
            return
        if (
            self.config.calibration_method == "ridge"
            and len(self.calibration_samples) >= self.config.calibration_min_samples
        ):
            self.uncertainty_calibrator.fit_samples(self.calibration_samples)

        pair_count = len(self.archive) * (len(self.archive) - 1) // 2
        if self.config.learn_similarity_weights and pair_count >= self.config.similarity_min_pairs:
            learned = self.similarity_calibrator.fit(self.archive, self.similarity)
            if learned is not None:
                self.similarity.weights = learned.normalized()

    def _restore_state(self) -> None:
        if self.archive_path.exists():
            self.archive = ArchiveStorage.load_archive(self.archive_path, self.encoder)
            self._loaded_archive = True
        if not self.state_path.exists():
            return
        state = ArchiveStorage.load_state(self.state_path)
        if str(state.get("run_id", self.run_id)) != self.run_id:
            raise ValueError("Belief state run_id does not match the active run")
        self.last_cycle = int(state.get("last_cycle", -1))
        self.calibration_samples = [
            {key: float(value) for key, value in item.items()}
            for item in state.get("calibration_samples", [])
        ]
        uncertainty_state = state.get("uncertainty_calibration")
        if isinstance(uncertainty_state, dict):
            self.uncertainty_calibrator.load_state(uncertainty_state)
        weight_data = state.get("similarity_weights")
        if isinstance(weight_data, dict):
            current_fields = set(asdict(SimilarityWeights()).keys())
            if set(weight_data.keys()) == current_fields:
                self.similarity.weights = SimilarityWeights(
                    **{key: float(value) for key, value in weight_data.items()}
                ).normalized()

    def _save_state(self, cycle: int) -> None:
        ArchiveStorage.save_archive(self.archive, self.archive_path)
        ArchiveStorage.save_state(
            {
                "run_id": self.run_id,
                "last_cycle": int(cycle),
                "uncertainty_calibration": self.uncertainty_calibrator.state.to_dict(),
                "calibration_samples": self.calibration_samples[-10000:],
                "similarity_weights": asdict(self.similarity.weights),
                "similarity_pair_count": self.similarity_calibrator.last_pair_count,
                "config": self.config.as_dict(),
            },
            self.state_path,
        )

    def _write_selection_summary(self, preparation: CyclePreparation) -> None:
        reason_counts = Counter(preparation.selection_reasons.values())
        row = {
            "run_id": self.run_id,
            "cycle": preparation.cycle,
            "guided_active": self.guided_active(preparation.cycle),
            "original_candidate_count": preparation.original_candidate_count,
            "unique_candidate_count": preparation.unique_candidate_count,
            "known_candidate_count": preparation.known_candidate_count,
            "assessed_candidate_count": preparation.assessed_candidate_count,
            "search_selected_count": len(preparation.selected_individuals),
            "audit_count": len(preparation.audit_individuals),
            "mean_topk_count": reason_counts.get("mean_topk", 0) + reason_counts.get("mean_fill", 0),
            "ucb_count": reason_counts.get("ucb", 0),
            "bounded_novelty_count": reason_counts.get("bounded_novelty", 0),
        }
        BeliefCycleMonitor.replace_cycle_csv(self.selection_csv, preparation.cycle, [row])

    def _write_weights_history(self, cycle: int, phase: str) -> None:
        row = {
            "run_id": self.run_id,
            "cycle": int(cycle),
            "phase": str(phase),
            **asdict(self.similarity.weights),
            "pair_count": self.similarity_calibrator.last_pair_count,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        # One final weight vector per cycle is enough for convergence analysis.
        BeliefCycleMonitor.replace_cycle_csv(self.weights_csv, int(cycle), [row])

    def _info(self, message: str) -> None:
        if self.log is not None and hasattr(self.log, "info"):
            self.log.info(message)

    def _resolve_run_id(self, restore_existing: bool) -> str:
        if restore_existing and self.active_run_path.exists():
            run_id = self.active_run_path.read_text(encoding="utf-8").strip()
            if run_id:
                return run_id
        run_id = self._new_run_id()
        self.active_run_path.write_text(run_id, encoding="utf-8")
        return run_id

    @staticmethod
    def _new_run_id() -> str:
        return datetime.now(timezone.utc).strftime("run_%Y%m%d_%H%M%S")
