from __future__ import annotations

import copy
import csv
import os
import pickle
import random
import sys
from pathlib import Path

import numpy as np
import torch

from utils import Log, StatusUpdateTool, Utils
from genetic.population import Individual, Population
from genetic.evaluate import FitnessEvaluate
from genetic.crossover_and_mutation import CrossoverAndMutation
from genetic.selection_operator import Selection
from genetic.belief import BeliefConfig, BeliefManager
from genetic.zero_cost import ZeroCostConfig, ZeroCostManager

try:
    import horovod.torch as hvd
except ImportError:
    hvd = None


class FlushStdout:
    def write(self, message):
        sys.__stdout__.write(message)
        sys.__stdout__.flush()

    def flush(self):
        sys.__stdout__.flush()


class EvolveCNN(object):
    def __init__(self, params):
        self.params = params
        self.pops = None
        self.parent_pops = None
        self.horovod_enabled = StatusUpdateTool.is_horovod_enabled()
        self.belief_manager = None
        self.belief_enabled = False
        self.zero_cost_manager = None
        self.zero_cost_enabled = False
        self.rank = 0
        self.size = 1
        self.max_gen = 20

    def sync_individuals(self, value):
        if not self.horovod_enabled:
            return value
        hvd.barrier()
        if hvd.rank() == 0:
            pickled_data = pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)
            data_size = torch.IntTensor([len(pickled_data)])
        else:
            pickled_data = None
            data_size = torch.IntTensor([0])
        data_size = hvd.broadcast(data_size, root_rank=0)
        if hvd.rank() == 0:
            tensor = torch.ByteTensor(list(pickled_data))
        else:
            tensor = torch.empty(data_size.item(), dtype=torch.uint8)
        tensor = hvd.broadcast(tensor, root_rank=0)
        return pickle.loads(tensor.cpu().numpy().tobytes())

    def initialize_population(self):
        if (not self.horovod_enabled) or self.rank == 0:
            StatusUpdateTool.begin_evolution()
    
        if self.horovod_enabled:
            hvd.barrier()
    
        pops = Population(self.params, 0)
        pops.initialize()
        self.pops = pops
    
        if self.horovod_enabled:
            self.pops = self.sync_individuals(self.pops)
            hvd.barrier()



    def setup_belief_manager(self, restore_existing):
        belief_config = BeliefConfig.from_ini()
        self.belief_enabled = belief_config.enabled
        if (not self.horovod_enabled) or self.rank == 0:
            self.belief_manager = BeliefManager(
                config=belief_config,
                log=Log,
                restore_existing=restore_existing,
            )
            Log.info("Belief manager status: %s" % self.belief_manager.describe())

    def setup_zero_cost_manager(self):
        zero_cost_config = ZeroCostConfig.from_ini()
        self.zero_cost_enabled = zero_cost_config.enabled
        if not self.zero_cost_enabled:
            if (not self.horovod_enabled) or self.rank == 0:
                Log.info("Zero-cost monitoring is disabled")
            return
        if (not self.horovod_enabled) or self.rank == 0:
            if self.belief_manager is None:
                raise RuntimeError("Zero-cost manager requires the belief manager")
            output_directory = (
                self.belief_manager.output_directory / zero_cost_config.output_subdirectory
            )
            self.zero_cost_manager = ZeroCostManager(
                config=zero_cost_config,
                run_id=self.belief_manager.run_id,
                output_directory=output_directory,
                log=Log,
            )
            Log.info("Zero-cost manager status: %s" % self.zero_cost_manager.describe())

    def fitness_evaluate(self, cycle=None, collect_zero_cost=False):
        if self.horovod_enabled:
            self.pops = self.sync_individuals(self.pops)
            hvd.barrier()

        evaluator = FitnessEvaluate(self.pops.individuals, Log)
        evaluator.generate_to_python_file()

        if (
            collect_zero_cost
            and self.zero_cost_enabled
            and self.zero_cost_manager is not None
            and cycle is not None
            and ((not self.horovod_enabled) or self.rank == 0)
        ):
            belief_predictions = {}
            if self.belief_manager is not None and self.belief_manager.is_enabled:
                belief_predictions = self.belief_manager.selected_prediction_snapshot(cycle)
            self.zero_cost_manager.prepare_cycle(
                individuals=self.pops.individuals,
                cycle=cycle,
                belief_predictions=belief_predictions,
            )
        if self.horovod_enabled:
            hvd.barrier()

        evaluator.evaluate()
        self.pops.individuals = evaluator.individuals

        if (
            collect_zero_cost
            and self.zero_cost_enabled
            and self.zero_cost_manager is not None
            and cycle is not None
            and ((not self.horovod_enabled) or self.rank == 0)
        ):
            self.zero_cost_manager.post_evaluate(
                evaluated_individuals=self.pops.individuals,
                cycle=cycle,
            )
        if self.horovod_enabled:
            hvd.barrier()

    def belief_prepare_cycle(self, cycle, cache_map=None):
        preparation = None
        if (not self.horovod_enabled) or self.rank == 0:
            if self.belief_manager is not None and self.belief_manager.is_enabled:
                preparation = self.belief_manager.prepare_cycle(
                    candidates=self.pops.individuals,
                    cycle=cycle,
                    cache_map=cache_map,
                )
                evaluation_individuals = (
                    copy.deepcopy(preparation.selected_individuals)
                    + copy.deepcopy(preparation.audit_individuals)
                )
                self.pops.individuals = evaluation_individuals
        if self.horovod_enabled:
            self.pops = self.sync_individuals(self.pops)
            hvd.barrier()
        return preparation

    def belief_post_evaluate(self, cycle, search_individuals, audit_individuals):
        if (not self.horovod_enabled) or self.rank == 0:
            if self.belief_manager is not None and self.belief_manager.is_enabled:
                metrics = self.belief_manager.post_evaluate(
                    evaluated_individuals=search_individuals,
                    audit_individuals=audit_individuals,
                    cycle=cycle,
                )
                if metrics is not None:
                    Log.info("Belief cycle metrics: %s" % metrics.to_dict())

    def _generate_global_candidates(
        self,
        generation,
        evolutionary_candidates,
        excluded_architecture_ids,
    ):
        balanced_count = 15
        novelty_count = 5
        novelty_pool_count = 25
        excluded = set(excluded_architecture_ids or set())
        excluded.update(individual.uuid()[0] for individual in evolutionary_candidates)
        seen = set(excluded)

        python_random_state = random.getstate()
        numpy_random_state = np.random.get_state()

        def generate_balanced_group(required_count, seed_offset):
            accepted = []
            for attempt in range(10):
                if len(accepted) >= required_count:
                    break
                remaining = required_count - len(accepted)
                batch_size = max(remaining, required_count)
                seed = 2312390 + int(generation) * 100 + seed_offset + attempt
                dummy = Individual(self.params, indi_no="temp")
                raw_population, name_to_id = dummy.population_with_all_positions(
                    pop_size=batch_size, seed=seed
                )
                balanced_positions = Individual.balance_population_arrays(
                    raw_population, name_to_id, tolerance=1
                )
                for positions in balanced_positions:
                    individual = Individual(self.params, indi_no="temp")
                    individual.initialize(positions)
                    architecture_id = individual.uuid()[0]
                    if architecture_id in seen:
                        continue
                    seen.add(architecture_id)
                    accepted.append(individual)
                    if len(accepted) >= required_count:
                        break
            if len(accepted) < required_count:
                raise RuntimeError(
                    "Could not generate %d unique balanced global candidates."
                    % required_count
                )
            return accepted

        try:
            balanced_candidates = generate_balanced_group(balanced_count, 0)
            novelty_pool = generate_balanced_group(novelty_pool_count, 1000)
        finally:
            random.setstate(python_random_state)
            np.random.set_state(numpy_random_state)

        for individual in balanced_candidates:
            individual.candidate_source = "balanced"
            individual.evaluation_role = "none"

        reference_encodings = (
            self.belief_manager.archive.encodings()
            + [
                self.belief_manager.encoder.encode(individual)
                for individual in evolutionary_candidates + balanced_candidates
            ]
        )
        novelty_scores = []
        for individual in novelty_pool:
            encoding = self.belief_manager.encoder.encode(individual)
            max_similarity = max(
                self.belief_manager.similarity.compare(encoding, reference).total
                for reference in reference_encodings
            )
            novelty_scores.append((max_similarity, encoding.architecture_id, individual))
        novelty_scores.sort(key=lambda item: (item[0], item[1]))
        novelty_candidates = [item[2] for item in novelty_scores[:novelty_count]]
        for individual in novelty_candidates:
            individual.candidate_source = "novelty_injected"
            individual.evaluation_role = "none"
        return balanced_candidates + novelty_candidates

    def crossover_and_mutation(
        self,
        generation,
        target_size=None,
        legacy_mode=True,
        excluded_architecture_ids=None,
        minimum_required_size=None,
    ):
        if (not self.horovod_enabled) or self.rank == 0:
            generation = int(generation)
            self.parent_pops = copy.deepcopy(self.pops)

            # If a full guided mutation file was written just before an interruption,
            # restore it rather than generating a different global pool.
            restored_full = None
            mutation_path = Path("./populations/mutation_%02d.txt" % generation)
            if not legacy_mode and mutation_path.exists():
                try:
                    restored = Utils.load_population("mutation", generation)
                    usable = []
                    seen = set(excluded_architecture_ids or set())
                    for individual in restored.individuals:
                        architecture_id = individual.uuid()[0]
                        if architecture_id in seen:
                            continue
                        seen.add(architecture_id)
                        usable.append(individual)
                    if len(usable) >= 100:
                        restored_full = usable[:100]
                        for index, individual in enumerate(restored_full):
                            individual.id = f"indi{generation:02d}{index:03d}"
                            individual.acc = -1.0
                            if index < 80:
                                individual.candidate_source = "evolutionary"
                            elif index < 95:
                                individual.candidate_source = "balanced"
                            else:
                                individual.candidate_source = "novelty_injected"
                            individual.evaluation_role = "none"
                except Exception as exc:
                    Log.warn("Stored full mutation pool could not be restored: %s" % exc)

            if restored_full is None:
                cm = CrossoverAndMutation(
                    self.params["genetic_prob"][0],
                    self.params["genetic_prob"][1],
                    Log,
                    self.parent_pops.individuals,
                    _params={
                        "gen_no": generation,
                        "target_size": target_size or self.params["pop_size"],
                        "legacy_mode": legacy_mode,
                        "excluded_architecture_ids": excluded_architecture_ids or set(),
                        "minimum_required_size": minimum_required_size or self.params["pop_size"],
                    },
                )
                offspring = cm.process()
                if not legacy_mode:
                    global_candidates = self._generate_global_candidates(
                        generation=generation,
                        evolutionary_candidates=offspring,
                        excluded_architecture_ids=excluded_architecture_ids or set(),
                    )
                    offspring.extend(global_candidates)
                    if len(offspring) != 100:
                        raise RuntimeError(
                            "Guided candidate pool must contain exactly 100 candidates, got %d"
                            % len(offspring)
                        )
                    seen_architectures = set()
                    for index, individual in enumerate(offspring):
                        architecture_id = individual.uuid()[0]
                        if architecture_id in seen_architectures:
                            raise RuntimeError("Guided candidate pool contains a duplicate architecture")
                        seen_architectures.add(architecture_id)
                        individual.id = f"indi{generation:02d}{index:03d}"
                        individual.acc = -1.0
                        individual.evaluation_role = "none"
                    cm.offspring = offspring
                    Utils.save_population_after_mutation(cm.individuals_to_string(), generation)
            else:
                offspring = restored_full

            candidate_pops = Population(self.params, generation)
            candidate_pops.individuals = copy.deepcopy(offspring)
            candidate_pops.number_id = len(candidate_pops.individuals)
            self.pops = candidate_pops

        if self.horovod_enabled:
            self.parent_pops = self.sync_individuals(self.parent_pops)
            self.pops = self.sync_individuals(self.pops)
            hvd.barrier()

    def environment_selection(self, cycle, search_individuals=None):
        stats = None
        if (not self.horovod_enabled) or self.rank == 0:
            cycle = int(cycle)
            offspring = list(search_individuals or self.pops.individuals)
            combined = offspring + list(self.parent_pops.individuals)

            representative_by_architecture = {}
            for individual in combined:
                architecture_id = individual.uuid()[0]
                current = representative_by_architecture.get(architecture_id)
                if current is None or float(individual.acc) > float(current.acc):
                    representative_by_architecture[architecture_id] = individual
            unique_individuals = list(representative_by_architecture.values())
            if len(unique_individuals) < self.params["pop_size"]:
                raise RuntimeError(
                    "Environment selection has only %d unique architectures for population size %d"
                    % (len(unique_individuals), self.params["pop_size"])
                )

            fitness_values = [float(individual.acc) for individual in unique_individuals]
            if any(value < 0 for value in fitness_values):
                raise ValueError("Environment selection received an unevaluated individual")

            log_rows = []
            offspring_ids = {id(item) for item in offspring}
            offspring_arch = {item.uuid()[0] for item in offspring}
            for individual in unique_individuals:
                prefix = "Indi" if individual.uuid()[0] in offspring_arch else "Pare"
                log_rows.append(
                    "%s-%s-%.5f-%s"
                    % (prefix, individual.id, individual.acc, individual.uuid()[0])
                )

            selection = Selection()
            pseudo_fitness = selection.GetGeometricPseudoFitness(
                fitness_values, cycle, self.max_gen
            )
            selection_seed = 2312390 + cycle * 100003 + 41
            python_rng = random.Random(selection_seed)
            numpy_rng = np.random.RandomState(selection_seed % (2**32 - 1))
            if python_rng.random() < 0.5:
                selected_indices = selection.RouletteSelection(
                    pseudo_fitness,
                    k=self.params["pop_size"],
                    replace=False,
                    rng=numpy_rng,
                )
                method = "roulette_without_replacement"
            else:
                selected_indices = selection.WheelSelection(
                    pseudo_fitness,
                    k=self.params["pop_size"],
                    replace=False,
                    rng=numpy_rng,
                )
                method = "wheel_without_replacement"

            max_index = int(np.argmax(fitness_values))
            if max_index not in selected_indices:
                selected_fitness = [fitness_values[index] for index in selected_indices]
                replace_position = int(np.argmin(selected_fitness))
                selected_indices[replace_position] = max_index
            if len(set(selected_indices)) != self.params["pop_size"]:
                raise RuntimeError("Environment selection produced repeated survivor indices")

            next_individuals = [unique_individuals[index] for index in selected_indices]
            next_population = Population(self.params, cycle)
            next_population.create_from_offspring(next_individuals, preserve_ids=True)
            self.pops = next_population

            survivor_ids = [individual.uuid()[0] for individual in self.pops.individuals]
            if len(set(survivor_ids)) != self.params["pop_size"]:
                raise RuntimeError("Survivor population contains duplicate architectures")

            for individual in self.pops.individuals:
                log_rows.append(
                    "new -%s-%.5f-%s" % (individual.id, individual.acc, individual.uuid()[0])
                )
            Utils.write_to_file(
                "\n".join(log_rows), "./populations/ENVI_%02d.txt" % cycle
            )
            Utils.save_population_at_begin(str(self.pops), cycle)

            stats = {
                "selection_method": method,
                "combined_count": len(combined),
                "combined_unique_count": len(unique_individuals),
                "survivor_unique_count": len(set(survivor_ids)),
                "survivor_mean_fitness": float(
                    np.mean([individual.acc for individual in self.pops.individuals])
                ),
                "survivor_best_fitness": float(
                    np.max([individual.acc for individual in self.pops.individuals])
                ),
            }

        if self.horovod_enabled:
            self.pops = self.sync_individuals(self.pops)
            hvd.barrier()
        return stats

    @staticmethod
    def _partition_evaluations(individuals):
        search = []
        audit = []
        for individual in individuals:
            if str(getattr(individual, "evaluation_role", "search")) == "audit":
                audit.append(individual)
            else:
                search.append(individual)
        return search, audit

    def _save_candidates_checkpoint(self, cycle):
        if (not self.horovod_enabled) or self.rank == 0:
            Utils.save_cycle_checkpoint(
                cycle,
                "candidates",
                {
                    "parent_pops": copy.deepcopy(self.parent_pops),
                    "candidate_pops": copy.deepcopy(self.pops),
                },
            )

    def _save_selected_checkpoint(self, cycle, candidate_pops, preparation):
        if (not self.horovod_enabled) or self.rank == 0:
            Utils.save_cycle_checkpoint(
                cycle,
                "selected",
                {
                    "parent_pops": copy.deepcopy(self.parent_pops),
                    "candidate_pops": copy.deepcopy(candidate_pops),
                    "search_ids": [item.id for item in preparation.selected_individuals],
                    "audit_ids": [item.id for item in preparation.audit_individuals],
                },
            )

    def _save_evaluated_checkpoint(
        self, cycle, candidate_pops, search_individuals, audit_individuals
    ):
        if (not self.horovod_enabled) or self.rank == 0:
            Utils.save_cycle_checkpoint(
                cycle,
                "evaluated",
                {
                    "parent_pops": copy.deepcopy(self.parent_pops),
                    "candidate_pops": copy.deepcopy(candidate_pops),
                    "search_individuals": copy.deepcopy(search_individuals),
                    "audit_individuals": copy.deepcopy(audit_individuals),
                },
            )

    def _write_generation_metrics(
        self,
        cycle,
        candidate_pops,
        search_individuals,
        audit_individuals,
        environment_stats,
    ):
        if self.horovod_enabled and self.rank != 0:
            return
        search_fitness = [float(item.acc) for item in search_individuals]
        reason_groups = {}
        source_counts = {}
        for item in candidate_pops.individuals:
            source = str(getattr(item, "candidate_source", "unknown"))
            source_counts[source] = source_counts.get(source, 0) + 1
        for item in search_individuals:
            reason = str(getattr(item, "selection_reason", "unknown"))
            reason_groups.setdefault(reason, []).append(float(item.acc))

        best_so_far = None
        if self.belief_manager is not None and len(self.belief_manager.archive) > 0:
            best_so_far = max(self.belief_manager.archive.fitness_values())

        row = {
            "cycle": int(cycle),
            "parent_unique_count": len({item.uuid()[0] for item in self.parent_pops.individuals}),
            "candidate_count": len(candidate_pops.individuals),
            "candidate_unique_count": len({item.uuid()[0] for item in candidate_pops.individuals}),
            "candidate_evolutionary_count": source_counts.get("evolutionary", 0),
            "candidate_balanced_count": source_counts.get("balanced", 0),
            "candidate_novelty_injected_count": source_counts.get("novelty_injected", 0),
            "search_count": len(search_individuals),
            "search_mean_fitness": float(np.mean(search_fitness)) if search_fitness else None,
            "search_best_fitness": float(np.max(search_fitness)) if search_fitness else None,
            "mean_topk_mean_fitness": self._group_mean(reason_groups, {"mean_topk", "mean_fill"}),
            "ucb_mean_fitness": self._group_mean(reason_groups, {"ucb"}),
            "bounded_novelty_fitness": self._group_mean(reason_groups, {"bounded_novelty"}),
            "audit_count": len(audit_individuals),
            "audit_fitness": float(audit_individuals[0].acc) if audit_individuals else None,
            "survivor_mean_fitness": environment_stats.get("survivor_mean_fitness"),
            "survivor_best_fitness": environment_stats.get("survivor_best_fitness"),
            "survivor_unique_count": environment_stats.get("survivor_unique_count"),
            "combined_unique_count": environment_stats.get("combined_unique_count"),
            "best_so_far": best_so_far,
        }
        self._replace_generation_metrics_row(row)

    def _write_initial_generation_metrics(self):
        if self.horovod_enabled and self.rank != 0:
            return
        fitness = [float(item.acc) for item in self.pops.individuals]
        row = {
            "cycle": 0,
            "parent_unique_count": None,
            "candidate_count": len(self.pops.individuals),
            "candidate_unique_count": len({item.uuid()[0] for item in self.pops.individuals}),
            "candidate_evolutionary_count": 0,
            "candidate_balanced_count": 0,
            "candidate_novelty_injected_count": 0,
            "search_count": len(self.pops.individuals),
            "search_mean_fitness": float(np.mean(fitness)) if fitness else None,
            "search_best_fitness": float(np.max(fitness)) if fitness else None,
            "mean_topk_mean_fitness": None,
            "ucb_mean_fitness": None,
            "bounded_novelty_fitness": None,
            "audit_count": 0,
            "audit_fitness": None,
            "survivor_mean_fitness": float(np.mean(fitness)) if fitness else None,
            "survivor_best_fitness": float(np.max(fitness)) if fitness else None,
            "survivor_unique_count": len({item.uuid()[0] for item in self.pops.individuals}),
            "combined_unique_count": len({item.uuid()[0] for item in self.pops.individuals}),
            "best_so_far": float(np.max(fitness)) if fitness else None,
        }
        self._replace_generation_metrics_row(row)

    @staticmethod
    def _replace_generation_metrics_row(row):
        path = Path("./populations/generation_metrics.csv")
        existing = []
        fields = list(row.keys())
        if path.exists() and path.stat().st_size:
            with path.open("r", newline="", encoding="utf-8") as handle:
                for existing_row in csv.DictReader(handle):
                    if int(existing_row.get("cycle", -1)) != int(row["cycle"]):
                        existing.append(dict(existing_row))
        temporary = path.with_suffix(path.suffix + ".tmp")
        with temporary.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for existing_row in existing:
                writer.writerow({field: existing_row.get(field) for field in fields})
            writer.writerow(row)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)

    @staticmethod
    def _group_mean(reason_groups, reasons):
        values = []
        for reason in reasons:
            values.extend(reason_groups.get(reason, []))
        return float(np.mean(values)) if values else None

    def _prepare_new_cycle(self, cycle):
        target_size = self.params["pop_size"]
        minimum_required_size = self.params["pop_size"]
        excluded_architecture_ids = set()
        legacy_mode = True
        if self.belief_manager is not None and self.belief_manager.is_enabled:
            target_size = self.belief_manager.candidate_target_size(
                self.params["pop_size"], cycle
            )
            legacy_mode = not self.belief_manager.guided_active(cycle)
            if not legacy_mode:
                target_size = 80
                minimum_required_size = 80
                excluded_architecture_ids = self.belief_manager.excluded_architecture_ids(
                    Utils.load_cache_data("search")
                )
        # Candidate generation uses a deterministic cycle-local random stream.
        # This makes a restarted cycle reproduce the same candidates without
        # serializing global RNG state.
        python_state = random.getstate()
        numpy_state = np.random.get_state()
        cycle_seed = 2312390 + int(cycle) * 100003
        random.seed(cycle_seed)
        np.random.seed(cycle_seed % (2**32 - 1))
        try:
            self.crossover_and_mutation(
                generation=cycle,
                target_size=target_size,
                legacy_mode=legacy_mode,
                excluded_architecture_ids=excluded_architecture_ids,
                minimum_required_size=minimum_required_size,
            )
        finally:
            random.setstate(python_state)
            np.random.set_state(numpy_state)
        self._save_candidates_checkpoint(cycle)

    def _continue_cycle(self, cycle, stage, payload):
        cycle = int(cycle)
        self.parent_pops = copy.deepcopy(payload["parent_pops"])
        candidate_pops = copy.deepcopy(payload.get("candidate_pops"))

        if stage == "evaluated":
            search_individuals = copy.deepcopy(payload["search_individuals"])
            audit_individuals = copy.deepcopy(payload.get("audit_individuals", []))
            self.pops = Population(self.params, cycle)
            self.pops.individuals = copy.deepcopy(search_individuals)
            environment_stats = self.environment_selection(cycle, search_individuals)
            self._write_generation_metrics(
                cycle,
                candidate_pops,
                search_individuals,
                audit_individuals,
                environment_stats,
            )
            if (not self.horovod_enabled) or self.rank == 0:
                Utils.complete_cycle_checkpoint(cycle)
            return

        if stage == "candidates":
            self.pops = copy.deepcopy(candidate_pops)
            preparation = self.belief_prepare_cycle(cycle, cache_map={})
            if (not self.horovod_enabled) or self.rank == 0:
                self._save_selected_checkpoint(cycle, candidate_pops, preparation)
        elif stage == "selected":
            belief_cycle_complete = bool(
                self.belief_manager is not None
                and self.belief_manager.last_cycle >= cycle
            ) if ((not self.horovod_enabled) or self.rank == 0) else False
            if self.horovod_enabled:
                belief_cycle_complete = self.sync_individuals(belief_cycle_complete)
            if belief_cycle_complete:
                search_ids = set(payload.get("search_ids", []))
                audit_ids = set(payload.get("audit_ids", []))
                selected = []
                for item in candidate_pops.individuals:
                    if item.id in search_ids:
                        item.evaluation_role = "search"
                        selected.append(item)
                    elif item.id in audit_ids:
                        item.evaluation_role = "audit"
                        selected.append(item)
                self.pops = Population(self.params, cycle)
                self.pops.individuals = copy.deepcopy(selected)
                evaluator = FitnessEvaluate(self.pops.individuals, Log)
                if (not self.horovod_enabled) or self.rank == 0:
                    evaluator._restore_completed_results()
                if self.horovod_enabled:
                    evaluator.individuals = evaluator.sync_individuals_acc(evaluator.individuals)
                self.pops.individuals = evaluator.individuals
                search_individuals, audit_individuals = self._partition_evaluations(
                    self.pops.individuals
                )
                if any(item.acc < 0 for item in search_individuals + audit_individuals):
                    raise RuntimeError("Belief state says cycle is complete but fitness cache is incomplete")
                self._save_evaluated_checkpoint(
                    cycle, candidate_pops, search_individuals, audit_individuals
                )
                environment_stats = self.environment_selection(cycle, search_individuals)
                self._write_generation_metrics(
                    cycle,
                    candidate_pops,
                    search_individuals,
                    audit_individuals,
                    environment_stats,
                )
                if (not self.horovod_enabled) or self.rank == 0:
                    Utils.complete_cycle_checkpoint(cycle)
                if self.horovod_enabled:
                    hvd.barrier()
                return

            self.pops = copy.deepcopy(candidate_pops)
            preparation = self.belief_prepare_cycle(cycle, cache_map={})
            if (not self.horovod_enabled) or self.rank == 0:
                expected_search = list(payload.get("search_ids", []))
                expected_audit = list(payload.get("audit_ids", []))
                actual_search = [item.id for item in preparation.selected_individuals]
                actual_audit = [item.id for item in preparation.audit_individuals]
                if actual_search != expected_search or actual_audit != expected_audit:
                    raise RuntimeError(
                        "Restarted belief selection does not match the saved selection checkpoint"
                    )
        else:
            raise ValueError("Unknown checkpoint stage: %s" % stage)

        self.fitness_evaluate(cycle=cycle, collect_zero_cost=True)
        search_individuals, audit_individuals = self._partition_evaluations(self.pops.individuals)
        self.belief_post_evaluate(cycle, search_individuals, audit_individuals)
        self._save_evaluated_checkpoint(
            cycle, candidate_pops, search_individuals, audit_individuals
        )
        environment_stats = self.environment_selection(cycle, search_individuals)
        self._write_generation_metrics(
            cycle,
            candidate_pops,
            search_individuals,
            audit_individuals,
            environment_stats,
        )
        if (not self.horovod_enabled) or self.rank == 0:
            Utils.complete_cycle_checkpoint(cycle)

    def do_work(self, max_gen):
        self.max_gen = int(max_gen)
        if self.horovod_enabled:
            if hvd is None:
                raise RuntimeError("Horovod is enabled but horovod.torch is unavailable")
            self.rank = hvd.rank()
            self.size = hvd.size()

        resume_requested = StatusUpdateTool.is_evolution_running()
        if not resume_requested and ((not self.horovod_enabled) or self.rank == 0):
            Utils.prepare_new_run_runtime()
        if self.horovod_enabled:
            hvd.barrier()

        self.setup_belief_manager(restore_existing=resume_requested)
        self.setup_zero_cost_manager()
        Log.info("*" * 25)

        checkpoint = None
        if resume_requested and ((not self.horovod_enabled) or self.rank == 0):
            checkpoint = Utils.latest_incomplete_cycle_checkpoint()
        if self.horovod_enabled:
            checkpoint = self.sync_individuals(checkpoint)

        if checkpoint is not None:
            cycle = int(checkpoint["cycle"])
            Log.info(
                "Resume from cycle %d stage=%s" % (cycle, checkpoint["stage"])
            )
            self._continue_cycle(cycle, checkpoint["stage"], checkpoint["payload"])
            next_cycle = cycle + 1
        elif resume_requested:
            gen_no = Utils.get_newest_file_based_on_prefix("begin")
            if gen_no is None:
                raise ValueError(
                    "Evolution is marked running but no completed population or checkpoint exists"
                )
            if (not self.horovod_enabled) or self.rank == 0:
                self.pops = Utils.load_population("begin", gen_no)
            if self.horovod_enabled:
                self.pops = self.sync_individuals(self.pops)
            Log.info("Resume from completed generation %d" % gen_no)
            next_cycle = int(gen_no) + 1
        else:
            Log.info("Initialize fresh run")
            self.initialize_population()
            Log.info("EVOLVE[0-gen]-Begin to evaluate the fitness")
            self.fitness_evaluate()
            if (not self.horovod_enabled) or self.rank == 0:
                Utils.save_population_at_begin(str(self.pops), 0)
                if self.belief_manager is not None and self.belief_manager.is_enabled:
                    self.belief_manager.bootstrap_population(self.pops.individuals, generation=0)
                self._write_initial_generation_metrics()
            if self.horovod_enabled:
                hvd.barrier()
            Log.info("EVOLVE[0-gen]-Finish the evaluation")
            next_cycle = 1

        for curr_gen in range(next_cycle, self.max_gen):
            self.params["gen_no"] = curr_gen
            if (not self.horovod_enabled) or self.rank == 0:
                Log.info("EVOLVE[%d-gen]-Begin to crossover and mutation" % curr_gen)
            self._prepare_new_cycle(curr_gen)
            candidate_pops = copy.deepcopy(self.pops) if ((not self.horovod_enabled) or self.rank == 0) else None
            if (not self.horovod_enabled) or self.rank == 0:
                Log.info("EVOLVE[%d-gen]-Finish crossover and mutation" % curr_gen)

            preparation = self.belief_prepare_cycle(curr_gen, cache_map=Utils.load_cache_data("search") if ((not self.horovod_enabled) or self.rank == 0) else {})
            if (not self.horovod_enabled) or self.rank == 0:
                self._save_selected_checkpoint(curr_gen, candidate_pops, preparation)
                Log.info("EVOLVE[%d-gen]-Begin to evaluate the fitness" % curr_gen)
            if self.horovod_enabled:
                hvd.barrier()

            self.fitness_evaluate(cycle=curr_gen, collect_zero_cost=True)
            search_individuals, audit_individuals = self._partition_evaluations(self.pops.individuals)
            self.belief_post_evaluate(curr_gen, search_individuals, audit_individuals)
            self._save_evaluated_checkpoint(
                curr_gen, candidate_pops, search_individuals, audit_individuals
            )
            if (not self.horovod_enabled) or self.rank == 0:
                Log.info("EVOLVE[%d-gen]-Finish the evaluation" % curr_gen)
                Log.info("EVOLVE[%d-gen]-Begin to environment selection" % curr_gen)
            environment_stats = self.environment_selection(curr_gen, search_individuals)
            if (not self.horovod_enabled) or self.rank == 0:
                self._write_generation_metrics(
                    curr_gen,
                    candidate_pops,
                    search_individuals,
                    audit_individuals,
                    environment_stats,
                )
                Utils.complete_cycle_checkpoint(curr_gen)
                Log.info("EVOLVE[%d-gen]-Finish the environment selection" % curr_gen)
            if self.horovod_enabled:
                hvd.barrier()

        if (not self.horovod_enabled) or self.rank == 0:
            StatusUpdateTool.end_evolution()
        if self.horovod_enabled:
            hvd.barrier()


if StatusUpdateTool.is_horovod_enabled():
    if hvd is None:
        raise ImportError("global.ini enables Horovod, but horovod.torch cannot be imported")
    hvd.init()
    print("Horovod initialized. Total ranks: %d" % hvd.size())
    torch.cuda.set_device(hvd.local_rank())
    print(
        "Rank %d assigned to GPU %d (%s)"
        % (hvd.rank(), hvd.local_rank(), torch.cuda.get_device_name(hvd.local_rank()))
    )
    if __name__ == "__main__":
        params = StatusUpdateTool.get_init_params()
        hvd.barrier()
        EvolveCNN(params).do_work(max_gen=20)
        hvd.barrier()
else:
    if __name__ == "__main__":
        params = StatusUpdateTool.get_init_params()
        EvolveCNN(params).do_work(max_gen=20)
