"""Generate, train, resume, and cache candidate fitness evaluations."""

from __future__ import annotations

import importlib
import multiprocessing
import pickle
import sys
import time
from typing import Iterable

import numpy as np
import torch

from utils import GPUTools, StatusUpdateTool, Utils

try:
    import horovod.torch as hvd
except ImportError:
    hvd = None


def _run_model_worker(module_name, file_name, gpu_id, evaluation_role):
    module = importlib.import_module(module_name, ".")
    run_model_class = getattr(module, "RunModel")
    run_model = run_model_class()
    run_model.do_work(file_name, str(gpu_id), evaluation_role=evaluation_role)


class FitnessEvaluate(object):
    """Evaluate candidates with model-level restart support."""

    def __init__(self, individuals, log):
        self.individuals = individuals
        self.log = log
        self.horovod_enabled = StatusUpdateTool.is_horovod_enabled()
        if self.horovod_enabled:
            if hvd is None:
                raise RuntimeError("Horovod is enabled in global.ini but horovod.torch is unavailable")
            self.rank = hvd.rank()
            self.size = hvd.size()
        else:
            self.rank = 0
            self.size = 1

    def generate_to_python_file(self):
        if (not self.horovod_enabled) or self.rank == 0:
            self.log.info("Begin to generate python files")
            for indi in self.individuals:
                Utils.generate_pytorch_file(indi)
            self.log.info("Finish the generation of python files")
        if self.horovod_enabled:
            hvd.barrier()

    def sync_individuals_acc(self, individuals):
        hvd.barrier()
        if hvd.rank() == 0:
            pickled_data = pickle.dumps(
                [float(indi.acc) for indi in individuals],
                protocol=pickle.HIGHEST_PROTOCOL,
            )
            data_size = torch.IntTensor([len(pickled_data)])
        else:
            pickled_data = None
            data_size = torch.IntTensor([0])
        data_size = hvd.broadcast(data_size, root_rank=0)
        if hvd.rank() == 0:
            pickled_tensor = torch.ByteTensor(list(pickled_data))
        else:
            pickled_tensor = torch.empty(data_size.item(), dtype=torch.uint8)
        pickled_tensor = hvd.broadcast(pickled_tensor, root_rank=0)
        synced_acc_list = pickle.loads(pickled_tensor.cpu().numpy().tobytes())
        for index, indi in enumerate(individuals):
            indi.acc = float(synced_acc_list[index])
        return individuals

    @staticmethod
    def _role(individual):
        role = str(getattr(individual, "evaluation_role", "search"))
        return "audit" if role == "audit" else "search"

    def _restore_completed_results(self):
        search_cache = Utils.load_cache_data("search")
        audit_cache = Utils.load_cache_data("audit")
        cache_by_role = {"search": search_cache, "audit": audit_cache}
        cache_hits = 0
        result_hits = 0
        for indi in self.individuals:
            role = self._role(indi)
            key, _ = indi.uuid()
            if float(getattr(indi, "acc", -1.0)) >= 0:
                continue
            if key in cache_by_role[role]:
                indi.acc = float(cache_by_role[role][key])
                cache_hits += 1
                continue
            completed = Utils.load_completed_fitness(indi.id, role=role)
            if completed is not None:
                indi.acc = float(completed)
                Utils.save_fitness_to_cache([indi], role=role)
                result_hits += 1
        self.log.info(
            "Restored completed fitness: cache_hits=%d result_file_hits=%d"
            % (cache_hits, result_hits)
        )

    def _wait_for_gpu(self):
        start = time.monotonic()
        gpu_id = GPUTools.detect_availabel_gpu_id()
        while gpu_id is None:
            if time.monotonic() - start >= 1800:
                raise TimeoutError("30 dakika içinde kullanılabilir GPU bulunamadı.")
            time.sleep(2)
            gpu_id = GPUTools.detect_availabel_gpu_id()
        return gpu_id

    def _train_one(self, indi):
        role = self._role(indi)
        file_name = str(indi.id)
        module_name = "scripts.%s" % file_name
        if (not self.horovod_enabled) or self.rank == 0:
            Utils.remove_partial_model_artifacts(file_name)
        if self.horovod_enabled:
            hvd.barrier()
        if (not self.horovod_enabled) or self.rank == 0:
            self.log.info("Begin to train %s role=%s" % (file_name, role))

        if self.horovod_enabled:
            if module_name in sys.modules:
                del sys.modules[module_name]
            module = importlib.import_module(module_name, ".")
            run_model_class = getattr(module, "RunModel")
            run_model = run_model_class()
            run_model.do_work(file_name, evaluation_role=role)
            hvd.barrier()
        else:
            gpu_id = self._wait_for_gpu()
            process_context = multiprocessing.get_context("spawn")
            process = process_context.Process(
                target=_run_model_worker,
                args=(module_name, file_name, str(gpu_id), role),
            )
            process.start()
            process.join()
            if process.exitcode != 0:
                raise RuntimeError(
                    "Model training process failed for %s with exit code %s"
                    % (file_name, process.exitcode)
                )

        if (not self.horovod_enabled) or self.rank == 0:
            completed = Utils.load_completed_fitness(file_name, role=role)
            if completed is None:
                raise RuntimeError(
                    "Training returned without a completed fitness record for %s" % file_name
                )
            indi.acc = float(completed)
            Utils.save_fitness_to_cache([indi], role=role)
            self.log.info(
                "Completed %s role=%s acc=%.5f" % (file_name, role, float(indi.acc))
            )
        if self.horovod_enabled:
            self.individuals = self.sync_individuals_acc(self.individuals)
            hvd.barrier()

    def evaluate(self):
        if (not self.horovod_enabled) or self.rank == 0:
            self._restore_completed_results()
            # Persist already-known search fitness in the cycle result file as an
            # idempotent record. Audits are always newly trained in guided mode.
            for indi in self.individuals:
                if float(getattr(indi, "acc", -1.0)) >= 0:
                    role = self._role(indi)
                    Utils.write_completed_fitness(indi.id, indi.acc, role=role)
                    Utils.save_fitness_to_cache([indi], role=role)

        if self.horovod_enabled:
            self.individuals = self.sync_individuals_acc(self.individuals)
            hvd.barrier()

        for indi in self.individuals:
            if float(getattr(indi, "acc", -1.0)) < 0:
                # Re-check the role-specific cache immediately before training.
                # This also prevents a duplicate architecture later in the same
                # warm-up batch from being trained a second time.
                if (not self.horovod_enabled) or self.rank == 0:
                    role = self._role(indi)
                    key, _ = indi.uuid()
                    live_cache = Utils.load_cache_data(role)
                    if key in live_cache:
                        indi.acc = float(live_cache[key])
                        Utils.write_completed_fitness(indi.id, indi.acc, role=role)
                if self.horovod_enabled:
                    self.individuals = self.sync_individuals_acc(self.individuals)
                    hvd.barrier()
                if float(getattr(indi, "acc", -1.0)) < 0:
                    self._train_one(indi)
                elif (not self.horovod_enabled) or self.rank == 0:
                    self.log.info(
                        "%s restored from live cache with fitness %.5f"
                        % (indi.id, float(indi.acc))
                    )
            elif (not self.horovod_enabled) or self.rank == 0:
                self.log.info(
                    "%s already has fitness %.5f; training skipped"
                    % (indi.id, float(indi.acc))
                )

        if self.horovod_enabled:
            self.individuals = self.sync_individuals_acc(self.individuals)
            hvd.barrier()

        if any(float(getattr(indi, "acc", -1.0)) < 0 for indi in self.individuals):
            raise RuntimeError("Evaluation ended with at least one unfinished individual")
        return self.individuals
