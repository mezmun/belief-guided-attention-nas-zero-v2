"""Passive collection and reporting of zero-cost proxy scores."""

from __future__ import annotations

import csv
import importlib
import json
import math
import random
import statistics
import sys
from itertools import combinations
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional

import numpy as np
import torch

from utils import StatusUpdateTool

from .config import ZeroCostConfig
from .metrics import average_ranks, method_metrics, spearman
from .proxies import build_proxy, parameter_count, proxy_metadata


class ZeroCostManager:
    """Run zero-cost methods without influencing NAS selection or fitness."""

    VERSION = "2.1-passive-audit"

    def __init__(
        self,
        config: ZeroCostConfig,
        run_id: str,
        output_directory: str | Path,
        log: Optional[Any] = None,
    ) -> None:
        config.validate()
        self.config = config
        self.run_id = str(run_id)
        self.log = log
        self.output_directory = Path(output_directory)
        self.output_directory.mkdir(parents=True, exist_ok=True)
        self.candidate_csv = self.output_directory / "candidate_scores.csv"
        self.evaluated_csv = self.output_directory / "evaluated_scores.csv"
        self.metrics_csv = self.output_directory / "cycle_metrics.csv"
        self.pairwise_csv = self.output_directory / "proxy_pairwise_correlations.csv"
        self.health_csv = self.output_directory / "proxy_health.csv"
        self.failures_csv = self.output_directory / "proxy_failures.csv"
        self.config_snapshot = self.output_directory / "config_snapshot.json"
        self.device = self._resolve_device(config.device)
        self._probe_inputs: Optional[torch.Tensor] = None
        self._probe_targets: Optional[torch.Tensor] = None
        self._cycle_rows: Dict[int, List[Dict[str, Any]]] = {}
        self._write_config_snapshot()

    @property
    def is_enabled(self) -> bool:
        return self.config.enabled

    def describe(self) -> Dict[str, Any]:
        return {
            "enabled": self.is_enabled,
            "version": self.VERSION,
            "proxies": list(self.config.proxies),
            "device": str(self.device),
            "output_directory": str(self.output_directory),
        }

    def prepare_cycle(
        self,
        individuals: Iterable[Any],
        cycle: int,
        belief_predictions: Optional[Mapping[str, Mapping[str, Any]]] = None,
    ) -> List[Dict[str, Any]]:
        """Calculate scores without changing the main run's RNG state."""

        if not self.is_enabled:
            return []
        random_state = self._capture_random_state()
        try:
            return self._prepare_cycle_impl(
                individuals=individuals,
                cycle=cycle,
                belief_predictions=belief_predictions,
            )
        finally:
            self._restore_random_state(random_state)

    def _prepare_cycle_impl(
        self,
        individuals: Iterable[Any],
        cycle: int,
        belief_predictions: Optional[Mapping[str, Mapping[str, Any]]] = None,
    ) -> List[Dict[str, Any]]:
        """Internal implementation for one pre-training collection cycle."""

        prediction_map = belief_predictions or {}
        candidates = [
            individual
            for individual in individuals
            if float(getattr(individual, "acc", -1.0)) < 0
        ]
        if not candidates:
            self._info("Zero-cost cycle %d: no unevaluated candidates" % cycle)
            self._cycle_rows[cycle] = []
            return []

        try:
            inputs, targets = self._get_probe_batch()
        except Exception as exc:
            self._record_failure(cycle, "*", "*", "probe_batch", exc)
            self._warning(
                "Zero-cost cycle %d skipped because the probe batch failed: %s"
                % (cycle, exc)
            )
            self._cycle_rows[cycle] = []
            return []

        rows: List[Dict[str, Any]] = []
        architecture_cache: Dict[str, Dict[str, Any]] = {}
        for individual in candidates:
            individual_id = str(getattr(individual, "id", "unknown"))
            architecture_id = str(individual.uuid()[0])
            belief = dict(prediction_map.get(individual_id, {}))
            row = self._base_row(
                cycle=cycle,
                individual_id=individual_id,
                architecture_id=architecture_id,
                belief=belief,
            )
            if architecture_id in architecture_cache:
                cached = architecture_cache[architecture_id]
                for key, value in cached.items():
                    row[key] = value
                row["zero_cost_cache_hit"] = 1
                for proxy_name in self.config.proxies:
                    row[proxy_name + "_seconds"] = 0.0
                rows.append(row)
                continue

            scores = self._score_individual(
                individual_id=individual_id,
                architecture_id=architecture_id,
                cycle=cycle,
                inputs=inputs,
                targets=targets,
            )
            row.update(scores)
            architecture_cache[architecture_id] = dict(scores)
            rows.append(row)

        self._add_ranks_and_ensembles(rows)
        self._replace_cycle_rows(self.candidate_csv, cycle, rows, self._candidate_fields())
        self._cycle_rows[cycle] = [dict(row) for row in rows]
        self._info(
            "Zero-cost cycle %d: scored %d candidates with %s"
            % (cycle, len(rows), ", ".join(self.config.proxies))
        )
        return rows

    def post_evaluate(
        self,
        evaluated_individuals: Iterable[Any],
        cycle: int,
    ) -> List[Dict[str, Any]]:
        """Attach true fitness and compare belief/proxy ranking quality."""

        if not self.is_enabled:
            return []
        rows = self._cycle_rows.pop(cycle, [])
        if not rows:
            self._warning("Zero-cost cycle %d has no pre-evaluation rows" % cycle)
            return []
        fitness_by_id = {
            str(getattr(individual, "id", "unknown")): float(
                getattr(individual, "acc", -1.0)
            )
            for individual in evaluated_individuals
        }
        completed: List[Dict[str, Any]] = []
        for row in rows:
            fitness = fitness_by_id.get(str(row["individual_id"]), -1.0)
            if fitness < 0:
                self._record_failure(
                    cycle,
                    str(row["individual_id"]),
                    str(row["architecture_id"]),
                    "post_evaluate",
                    ValueError("No real fitness was available after training"),
                )
                continue
            evaluated = dict(row)
            evaluated["true_fitness"] = fitness
            evaluated["evaluated_at_utc"] = self._utc_now()
            completed.append(evaluated)
        if not completed:
            return []

        self._replace_cycle_rows(
            self.evaluated_csv,
            cycle,
            completed,
            self._evaluated_fields(),
        )
        metric_rows = self._calculate_cycle_metrics(completed, cycle)
        self._replace_cycle_rows(self.metrics_csv, cycle, metric_rows, self._metric_fields())
        pairwise_rows = self._calculate_pairwise_correlations(completed, cycle)
        self._replace_cycle_rows(self.pairwise_csv, cycle, pairwise_rows, self._pairwise_fields())
        health_rows = self._calculate_proxy_health(completed, cycle)
        self._replace_cycle_rows(self.health_csv, cycle, health_rows, self._health_fields())
        winner = self._best_spearman_method(metric_rows, subset="search")
        if winner is not None:
            self._info(
                "Zero-cost cycle %d best Spearman: %s=%s"
                % (cycle, winner[0], self._format_metric(winner[1]))
            )
        return metric_rows

    def _score_individual(
        self,
        individual_id: str,
        architecture_id: str,
        cycle: int,
        inputs: torch.Tensor,
        targets: torch.Tensor,
    ) -> Dict[str, Any]:
        result: Dict[str, Any] = {"zero_cost_cache_hit": 0}
        try:
            model_class = self._load_model_class(individual_id)
        except Exception as exc:
            self._record_failure(cycle, individual_id, architecture_id, "model_import", exc)
            for proxy_name in self.config.proxies:
                result[proxy_name + "_raw"] = None
                result[proxy_name + "_seconds"] = None
            result["parameter_count"] = None
            return result

        architecture_seed = self._architecture_seed(architecture_id)
        if self.config.include_parameter_count:
            model = None
            try:
                self._set_seed(architecture_seed)
                model = model_class().to(self.device)
                result["parameter_count"] = parameter_count(model)
            except Exception as exc:
                result["parameter_count"] = None
                self._record_failure(
                    cycle, individual_id, architecture_id, "parameter_count", exc
                )
            finally:
                self._release_model(model)
        else:
            result["parameter_count"] = None

        for proxy_name in self.config.proxies:
            model = None
            started = time.perf_counter()
            try:
                self._set_seed(architecture_seed)
                model = model_class().to(self.device)
                proxy = build_proxy(proxy_name)
                score = proxy.calculate(model, inputs, targets)
                if not math.isfinite(score):
                    raise FloatingPointError("Proxy returned a non-finite value")
                result[proxy_name + "_raw"] = float(score)
                result[proxy_name + "_seconds"] = float(time.perf_counter() - started)
            except Exception as exc:
                result[proxy_name + "_raw"] = None
                result[proxy_name + "_seconds"] = float(time.perf_counter() - started)
                self._record_failure(cycle, individual_id, architecture_id, proxy_name, exc)
            finally:
                self._release_model(model)
        return result

    def _load_model_class(self, individual_id: str):
        module_name = "scripts.%s" % individual_id
        importlib.invalidate_caches()
        if module_name in sys.modules:
            del sys.modules[module_name]
        module = importlib.import_module(module_name, ".")
        return getattr(module, "EvoCNNModel")

    def _get_probe_batch(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self._probe_inputs is not None and self._probe_targets is not None:
            return self._probe_inputs, self._probe_targets
        import data_loader

        dataset_dir = StatusUpdateTool.is_dataset_directory_set()
        _, validation_loader = data_loader.get_train_valid_loader(
            dataset_dir,
            batch_size=self.config.batch_size,
            augment=False,
            random_seed=self.config.random_seed,
            valid_size=self.config.valid_size,
            shuffle=False,
            show_sample=False,
            num_workers=self.config.num_workers,
            pin_memory=self.device.type == "cuda",
        )
        input_batches: List[torch.Tensor] = []
        target_batches: List[torch.Tensor] = []
        for batch_index, (inputs, targets) in enumerate(validation_loader):
            input_batches.append(inputs)
            target_batches.append(targets)
            if batch_index + 1 >= self.config.num_batches:
                break
        if not input_batches:
            raise RuntimeError("Validation loader returned no probe batches")
        self._probe_inputs = torch.cat(input_batches, dim=0).to(
            self.device, non_blocking=self.device.type == "cuda"
        )
        self._probe_targets = torch.cat(target_batches, dim=0).long().to(
            self.device, non_blocking=self.device.type == "cuda"
        )
        return self._probe_inputs, self._probe_targets

    def _add_ranks_and_ensembles(self, rows: List[Dict[str, Any]]) -> None:
        for proxy_name in self.config.proxies:
            self._assign_normalized_ranks(rows, proxy_name + "_raw", proxy_name + "_rank")
        if self.config.include_parameter_count:
            self._assign_normalized_ranks(rows, "parameter_count", "parameter_count_rank")

        family_members: Dict[str, List[str]] = {}
        for proxy_name in self.config.proxies:
            family_members.setdefault(proxy_metadata(proxy_name).family, []).append(proxy_name)

        for row in rows:
            proxy_ranks = [
                float(row[proxy_name + "_rank"])
                for proxy_name in self.config.proxies
                if self._is_finite(row.get(proxy_name + "_rank"))
            ]
            row["ensemble_proxy_count"] = len(proxy_ranks)
            row["zc_mean_rank"] = (
                float(sum(proxy_ranks) / len(proxy_ranks))
                if self.config.mean_rank_ensemble and proxy_ranks
                else None
            )
            row["zc_median_rank"] = (
                float(statistics.median(proxy_ranks))
                if self.config.median_rank_ensemble and proxy_ranks
                else None
            )
            row["zc_complete_mean_rank"] = (
                float(sum(proxy_ranks) / len(proxy_ranks))
                if self.config.complete_case_ensemble
                and len(proxy_ranks) == len(self.config.proxies)
                else None
            )

            family_scores: List[float] = []
            for proxy_names in family_members.values():
                values = [
                    float(row[name + "_rank"])
                    for name in proxy_names
                    if self._is_finite(row.get(name + "_rank"))
                ]
                if values:
                    family_scores.append(float(sum(values) / len(values)))
            row["ensemble_family_count"] = len(family_scores)
            row["zc_family_mean_rank"] = (
                float(sum(family_scores) / len(family_scores))
                if self.config.family_rank_ensemble and family_scores
                else None
            )

    @staticmethod
    def _assign_normalized_ranks(
        rows: List[Dict[str, Any]], source_field: str, target_field: str
    ) -> None:
        valid = [
            (index, float(row[source_field]))
            for index, row in enumerate(rows)
            if ZeroCostManager._is_finite(row.get(source_field))
        ]
        for row in rows:
            row[target_field] = None
        if not valid:
            return
        ranks = average_ranks([value for _, value in valid])
        denominator = max(1, len(valid) - 1)
        for (row_index, _), rank in zip(valid, ranks):
            rows[row_index][target_field] = (
                1.0 if len(valid) == 1 else float((rank - 1.0) / denominator)
            )

    def _calculate_cycle_metrics(
        self, completed: List[Dict[str, Any]], cycle: int
    ) -> List[Dict[str, Any]]:
        methods = [("belief", "belief_mean")]
        methods.extend((proxy_name, proxy_name + "_raw") for proxy_name in self.config.proxies)
        if self.config.mean_rank_ensemble:
            methods.append(("zc_mean_rank", "zc_mean_rank"))
        if self.config.median_rank_ensemble:
            methods.append(("zc_median_rank", "zc_median_rank"))
        if self.config.family_rank_ensemble:
            methods.append(("zc_family_mean_rank", "zc_family_mean_rank"))
        if self.config.complete_case_ensemble:
            methods.append(("zc_complete_mean_rank", "zc_complete_mean_rank"))
        if self.config.include_parameter_count:
            methods.append(("parameter_count", "parameter_count"))

        metric_rows: List[Dict[str, Any]] = []
        subsets = {
            "search": [row for row in completed if row.get("evaluation_role") == "search"],
            "audit": [row for row in completed if row.get("evaluation_role") == "audit"],
            "all_observed": completed,
        }
        for subset_name, subset_rows in subsets.items():
            for method_name, score_field in methods:
                usable = [
                    row
                    for row in subset_rows
                    if self._is_finite(row.get(score_field))
                    and self._is_finite(row.get("true_fitness"))
                ]
                if not usable:
                    continue
                scores = [float(row[score_field]) for row in usable]
                truth = [float(row["true_fitness"]) for row in usable]
                calculated = method_metrics(scores, truth, self.config.top_k)
                metric_rows.append(
                    {
                        "run_id": self.run_id,
                        "cycle": cycle,
                        "subset": subset_name,
                        "method": method_name,
                        **calculated,
                        "created_at_utc": self._utc_now(),
                    }
                )
        return metric_rows

    def _method_fields(self) -> List[tuple[str, str]]:
        methods = [("belief", "belief_mean")]
        methods.extend((name, name + "_raw") for name in self.config.proxies)
        if self.config.include_parameter_count:
            methods.append(("parameter_count", "parameter_count"))
        if self.config.mean_rank_ensemble:
            methods.append(("zc_mean_rank", "zc_mean_rank"))
        if self.config.median_rank_ensemble:
            methods.append(("zc_median_rank", "zc_median_rank"))
        if self.config.family_rank_ensemble:
            methods.append(("zc_family_mean_rank", "zc_family_mean_rank"))
        if self.config.complete_case_ensemble:
            methods.append(("zc_complete_mean_rank", "zc_complete_mean_rank"))
        return methods

    def _calculate_pairwise_correlations(
        self, completed: List[Dict[str, Any]], cycle: int
    ) -> List[Dict[str, Any]]:
        result: List[Dict[str, Any]] = []
        for (name_a, field_a), (name_b, field_b) in combinations(self._method_fields(), 2):
            usable = [
                row for row in completed
                if self._is_finite(row.get(field_a)) and self._is_finite(row.get(field_b))
            ]
            if len(usable) < 2:
                continue
            correlation = spearman(
                [float(row[field_a]) for row in usable],
                [float(row[field_b]) for row in usable],
            )
            result.append({
                "run_id": self.run_id,
                "cycle": cycle,
                "method_a": name_a,
                "method_b": name_b,
                "n": len(usable),
                "spearman": correlation,
                "created_at_utc": self._utc_now(),
            })
        return result

    def _calculate_proxy_health(
        self, completed: List[Dict[str, Any]], cycle: int
    ) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        for proxy_name in self.config.proxies:
            attempted = len(completed)
            successful_rows = [
                row for row in completed if self._is_finite(row.get(proxy_name + "_raw"))
            ]
            times = [
                float(row[proxy_name + "_seconds"])
                for row in completed
                if self._is_finite(row.get(proxy_name + "_seconds"))
            ]
            successful = len(successful_rows)
            rows.append({
                "run_id": self.run_id,
                "cycle": cycle,
                "proxy": proxy_name,
                "family": proxy_metadata(proxy_name).family,
                "attempted": attempted,
                "successful": successful,
                "failed": attempted - successful,
                "success_rate": (float(successful) / attempted) if attempted else None,
                "mean_seconds": (float(sum(times) / len(times))) if times else None,
                "created_at_utc": self._utc_now(),
            })
        return rows

    def _base_row(
        self,
        cycle: int,
        individual_id: str,
        architecture_id: str,
        belief: Mapping[str, Any],
    ) -> Dict[str, Any]:
        reason = belief.get("selection_reason")
        role = str(belief.get("evaluation_role") or "search")
        return {
            "run_id": self.run_id,
            "cycle": int(cycle),
            "individual_id": individual_id,
            "architecture_id": architecture_id,
            "candidate_source": belief.get("candidate_source", "unknown"),
            "evaluation_role": role,
            "belief_mean": belief.get("belief_mean"),
            "belief_uncertainty": belief.get("belief_uncertainty"),
            "raw_uncertainty": belief.get("raw_uncertainty"),
            "novelty": belief.get("novelty"),
            "selection_reason": reason,
            "selection_score": belief.get("selection_score"),
            "random_audit": int(role == "audit"),
            "created_at_utc": self._utc_now(),
        }

    def _candidate_fields(self) -> List[str]:
        fields = [
            "run_id",
            "cycle",
            "individual_id",
            "architecture_id",
            "candidate_source",
            "evaluation_role",
            "belief_mean",
            "belief_uncertainty",
            "raw_uncertainty",
            "novelty",
            "selection_reason",
            "selection_score",
            "random_audit",
            "zero_cost_cache_hit",
            "parameter_count",
            "parameter_count_rank",
        ]
        for proxy_name in self.config.proxies:
            fields.extend(
                [proxy_name + "_raw", proxy_name + "_rank", proxy_name + "_seconds"]
            )
        fields.extend(
            [
                "ensemble_proxy_count",
                "ensemble_family_count",
                "zc_mean_rank",
                "zc_median_rank",
                "zc_family_mean_rank",
                "zc_complete_mean_rank",
                "created_at_utc",
            ]
        )
        return fields

    def _evaluated_fields(self) -> List[str]:
        return self._candidate_fields() + ["true_fitness", "evaluated_at_utc"]

    @staticmethod
    def _metric_fields() -> List[str]:
        return [
            "run_id",
            "cycle",
            "subset",
            "method",
            "n",
            "spearman",
            "kendall_tau_b",
            "pairwise_accuracy",
            "top_k",
            "top_k_recall",
            "best_model_rank",
            "created_at_utc",
        ]

    @staticmethod
    def _pairwise_fields() -> List[str]:
        return [
            "run_id",
            "cycle",
            "method_a",
            "method_b",
            "n",
            "spearman",
            "created_at_utc",
        ]

    @staticmethod
    def _health_fields() -> List[str]:
        return [
            "run_id",
            "cycle",
            "proxy",
            "family",
            "attempted",
            "successful",
            "failed",
            "success_rate",
            "mean_seconds",
            "created_at_utc",
        ]

    def _record_failure(
        self,
        cycle: int,
        individual_id: str,
        architecture_id: str,
        stage: str,
        exc: Exception,
    ) -> None:
        self._append_rows(
            self.failures_csv,
            [
                {
                    "run_id": self.run_id,
                    "cycle": cycle,
                    "individual_id": individual_id,
                    "architecture_id": architecture_id,
                    "stage": stage,
                    "exception_type": type(exc).__name__,
                    "exception_message": str(exc),
                    "created_at_utc": self._utc_now(),
                }
            ],
            [
                "run_id",
                "cycle",
                "individual_id",
                "architecture_id",
                "stage",
                "exception_type",
                "exception_message",
                "created_at_utc",
            ],
        )
        self._warning(
            "Zero-cost failure cycle=%s individual=%s stage=%s: %s"
            % (cycle, individual_id, stage, exc)
        )

    @staticmethod
    def _replace_cycle_rows(
        path: Path,
        cycle: int,
        rows: List[Dict[str, Any]],
        fields: List[str],
    ) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        existing: List[Dict[str, Any]] = []
        if path.exists() and path.stat().st_size:
            with path.open("r", newline="", encoding="utf-8") as handle:
                reader = csv.DictReader(handle)
                for row in reader:
                    try:
                        row_cycle = int(row.get("cycle", -1))
                    except (TypeError, ValueError):
                        row_cycle = -1
                    if row_cycle != int(cycle):
                        existing.append(dict(row))
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
    def _append_rows(path: Path, rows: List[Dict[str, Any]], fields: List[str]) -> None:
        if not rows:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        write_header = not path.exists() or path.stat().st_size == 0
        with path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
            if write_header:
                writer.writeheader()
            for row in rows:
                writer.writerow({field: row.get(field) for field in fields})

    def _write_config_snapshot(self) -> None:
        payload = {
            "run_id": self.run_id,
            "version": self.VERSION,
            "resolved_device": str(self.device),
            "config": self.config.as_dict(),
            "created_at_utc": self._utc_now(),
        }
        self.config_snapshot.write_text(
            json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
        )

    @staticmethod
    def _resolve_device(value: str) -> torch.device:
        if value == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda", torch.cuda.current_device())
            return torch.device("cpu")
        device = torch.device(value)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("zero_cost.device requests CUDA but CUDA is unavailable")
        return device

    def _architecture_seed(self, architecture_id: str) -> int:
        # Every architecture is instantiated from the same configured seed.
        # Different tensor shapes still consume different random streams, but no
        # architecture receives a deliberately privileged seed.
        del architecture_id
        return int(self.config.random_seed)

    @staticmethod
    def _capture_random_state() -> Dict[str, Any]:
        state: Dict[str, Any] = {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.random.get_rng_state(),
        }
        if torch.cuda.is_available():
            state["cuda"] = torch.cuda.get_rng_state_all()
        return state

    @staticmethod
    def _restore_random_state(state: Mapping[str, Any]) -> None:
        random.setstate(state["python"])
        np.random.set_state(state["numpy"])
        torch.random.set_rng_state(state["torch"])
        if torch.cuda.is_available() and "cuda" in state:
            torch.cuda.set_rng_state_all(state["cuda"])

    @staticmethod
    def _set_seed(seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    @staticmethod
    def _release_model(model: Optional[torch.nn.Module]) -> None:
        if model is not None:
            try:
                model.zero_grad(set_to_none=True)
            except Exception:
                pass
            del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    @staticmethod
    def _is_finite(value: Any) -> bool:
        try:
            return value is not None and math.isfinite(float(value))
        except (TypeError, ValueError):
            return False

    @staticmethod
    def _best_spearman_method(
        rows: List[Dict[str, Any]], subset: str
    ) -> Optional[tuple[str, float]]:
        candidates = [
            (str(row["method"]), float(row["spearman"]))
            for row in rows
            if row.get("subset") == subset and ZeroCostManager._is_finite(row.get("spearman"))
        ]
        return max(candidates, key=lambda item: item[1]) if candidates else None

    @staticmethod
    def _format_metric(value: float) -> str:
        return "%.4f" % value

    @staticmethod
    def _utc_now() -> str:
        return datetime.now(timezone.utc).isoformat()

    def _info(self, message: str) -> None:
        if self.log is not None and hasattr(self.log, "info"):
            self.log.info(message)

    def _warning(self, message: str) -> None:
        if self.log is not None:
            if hasattr(self.log, "warning"):
                self.log.warning(message)
                return
            if hasattr(self.log, "warn"):
                self.log.warn(message)
