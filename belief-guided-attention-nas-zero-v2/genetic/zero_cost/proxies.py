"""Training-free proxy implementations used only for passive monitoring.

The implementations are dependency-light adaptations of public reference code.
They never update model parameters and every proxy is run on a fresh model
instance by :mod:`genetic.zero_cost.manager`.
"""

from __future__ import annotations

import contextlib
import math
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


@dataclass(frozen=True)
class ProxyMetadata:
    name: str
    family: str
    source: str
    max_samples: Optional[int] = None
    higher_is_better: bool = True


class ZeroCostProxy(ABC):
    """Common interface for one zero-cost proxy."""

    metadata: ProxyMetadata

    @property
    def name(self) -> str:
        return self.metadata.name

    @abstractmethod
    def calculate(
        self,
        model: nn.Module,
        inputs: torch.Tensor,
        targets: torch.Tensor,
    ) -> float:
        """Return one scalar score; larger values are normalized as better."""

    def _limit_batch(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        limit = self.metadata.max_samples
        if limit is None or inputs.size(0) <= limit:
            return inputs, targets
        return inputs[:limit], targets[:limit]


def _trainable_parameters(model: nn.Module) -> Iterator[nn.Parameter]:
    return (parameter for parameter in model.parameters() if parameter.requires_grad)


def _weight_modules(model: nn.Module) -> Iterator[nn.Module]:
    for module in model.modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            yield module


def _finite_float(value: torch.Tensor | float | np.floating) -> float:
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise ValueError("Zero-cost proxy must return a scalar tensor")
        scalar = float(value.detach().cpu().item())
    else:
        scalar = float(value)
    if not math.isfinite(scalar):
        raise FloatingPointError("Zero-cost proxy produced a non-finite score")
    return scalar


def _extract_tensor(output: object) -> torch.Tensor:
    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, (tuple, list)):
        for item in output:
            if isinstance(item, torch.Tensor):
                return item
    if isinstance(output, Mapping):
        for item in output.values():
            if isinstance(item, torch.Tensor):
                return item
    raise TypeError("Model output did not contain a tensor")


def _cross_entropy(model: nn.Module, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    logits = _extract_tensor(model(inputs))
    return F.cross_entropy(logits, targets)


def _safe_zero_grad(model: nn.Module) -> None:
    model.zero_grad(set_to_none=True)


class SynFlowProxy(ZeroCostProxy):
    """SynFlow with the reference implementation's float64 stabilization."""

    metadata = ProxyMetadata(
        name="synflow",
        family="data_free_gradient",
        source="mohsaied/zero-cost-nas",
        max_samples=1,
    )

    def calculate(self, model: nn.Module, inputs: torch.Tensor, targets: torch.Tensor) -> float:
        del targets
        model.eval()
        model.double()
        signs: Dict[str, torch.Tensor] = {}
        with torch.no_grad():
            for name, parameter in model.state_dict().items():
                if not torch.is_floating_point(parameter):
                    continue
                signs[name] = torch.sign(parameter)
                parameter.abs_()
        try:
            _safe_zero_grad(model)
            input_shape = list(inputs[0].shape)
            ones = torch.ones(
                [1] + input_shape,
                device=inputs.device,
                dtype=torch.float64,
            )
            output = _extract_tensor(model(ones))
            if not torch.isfinite(output).all():
                raise FloatingPointError("SynFlow forward output is non-finite")
            output.sum().backward()
            score = torch.zeros((), device=inputs.device, dtype=torch.float64)
            for module in _weight_modules(model):
                weight = module.weight
                if weight.grad is None:
                    continue
                if not torch.isfinite(weight.grad).all():
                    raise FloatingPointError("SynFlow gradient is non-finite")
                score = score + torch.sum(torch.abs(weight * weight.grad))
            return _finite_float(score)
        finally:
            with torch.no_grad():
                state = model.state_dict()
                for name, sign in signs.items():
                    if name in state:
                        state[name].mul_(sign)
            _safe_zero_grad(model)


class SnipProxy(ZeroCostProxy):
    """SNIP connection sensitivity, equivalent to mask-gradient saliency."""

    metadata = ProxyMetadata(
        name="snip",
        family="saliency",
        source="mohsaied/zero-cost-nas",
        max_samples=32,
    )

    def calculate(self, model: nn.Module, inputs: torch.Tensor, targets: torch.Tensor) -> float:
        inputs, targets = self._limit_batch(inputs, targets)
        model.train()
        _safe_zero_grad(model)
        loss = _cross_entropy(model, inputs, targets)
        loss.backward()
        score = torch.zeros((), device=inputs.device)
        for module in _weight_modules(model):
            if module.weight.grad is not None:
                score = score + torch.sum(torch.abs(module.weight * module.weight.grad))
        _safe_zero_grad(model)
        return _finite_float(score)


class GradNormProxy(ZeroCostProxy):
    """Sum of per-layer weight-gradient L2 norms."""

    metadata = ProxyMetadata(
        name="gradnorm",
        family="gradient_magnitude",
        source="mohsaied/zero-cost-nas",
        max_samples=32,
    )

    def calculate(self, model: nn.Module, inputs: torch.Tensor, targets: torch.Tensor) -> float:
        inputs, targets = self._limit_batch(inputs, targets)
        model.train()
        _safe_zero_grad(model)
        _cross_entropy(model, inputs, targets).backward()
        score = torch.zeros((), device=inputs.device)
        for module in _weight_modules(model):
            if module.weight.grad is not None:
                score = score + module.weight.grad.norm(p=2)
        _safe_zero_grad(model)
        return _finite_float(score)


class PlainProxy(ZeroCostProxy):
    """Signed weight-gradient product from the original proxy bank."""

    metadata = ProxyMetadata(
        name="plain",
        family="saliency",
        source="mohsaied/zero-cost-nas",
        max_samples=32,
    )

    def calculate(self, model: nn.Module, inputs: torch.Tensor, targets: torch.Tensor) -> float:
        inputs, targets = self._limit_batch(inputs, targets)
        model.train()
        _safe_zero_grad(model)
        _cross_entropy(model, inputs, targets).backward()
        score = torch.zeros((), device=inputs.device)
        for module in _weight_modules(model):
            if module.weight.grad is not None:
                score = score + torch.sum(module.weight * module.weight.grad)
        _safe_zero_grad(model)
        return _finite_float(score)


class L2NormProxy(ZeroCostProxy):
    """Sum of per-layer weight L2 norms."""

    metadata = ProxyMetadata(
        name="l2norm",
        family="weight_statistics",
        source="mohsaied/zero-cost-nas",
    )

    def calculate(self, model: nn.Module, inputs: torch.Tensor, targets: torch.Tensor) -> float:
        del inputs, targets
        score = sum(float(module.weight.detach().norm(p=2).cpu()) for module in _weight_modules(model))
        return _finite_float(score)


class FisherProxy(ZeroCostProxy):
    """Channel-wise Fisher information using activation-gradient products."""

    metadata = ProxyMetadata(
        name="fisher",
        family="second_order_saliency",
        source="mohsaied/zero-cost-nas",
        max_samples=24,
    )

    def calculate(self, model: nn.Module, inputs: torch.Tensor, targets: torch.Tensor) -> float:
        inputs, targets = self._limit_batch(inputs, targets)
        model.train()
        _safe_zero_grad(model)
        channel_scores: List[torch.Tensor] = []
        handles: List[torch.utils.hooks.RemovableHandle] = []

        def forward_hook(module: nn.Module, module_inputs: tuple[object, ...], output: object) -> None:
            del module, module_inputs
            tensor = _extract_tensor(output)
            if not tensor.requires_grad:
                return
            activation = tensor.detach()

            def output_grad_hook(gradient: torch.Tensor) -> None:
                if activation.ndim > 2:
                    reduce_dims = tuple(range(2, activation.ndim))
                    g_nk = (activation * gradient.detach()).sum(dim=reduce_dims)
                else:
                    g_nk = activation * gradient.detach()
                channel_scores.append(0.5 * g_nk.pow(2).mean(dim=0).sum())

            tensor.register_hook(output_grad_hook)

        try:
            for module in _weight_modules(model):
                handles.append(module.register_forward_hook(forward_hook))
            _cross_entropy(model, inputs, targets).backward()
            if not channel_scores:
                raise RuntimeError("Fisher did not capture any Conv2d/Linear activations")
            score = torch.stack([item.to(inputs.device) for item in channel_scores]).sum()
            return _finite_float(score)
        finally:
            for handle in handles:
                handle.remove()
            _safe_zero_grad(model)


class GraSPProxy(ZeroCostProxy):
    """GraSP gradient-signal-preservation score."""

    metadata = ProxyMetadata(
        name="grasp",
        family="second_order_saliency",
        source="mohsaied/zero-cost-nas",
        max_samples=16,
    )

    def calculate(self, model: nn.Module, inputs: torch.Tensor, targets: torch.Tensor) -> float:
        inputs, targets = self._limit_batch(inputs, targets)
        model.train()
        weights = [module.weight for module in _weight_modules(model)]
        if not weights:
            raise RuntimeError("GraSP found no Conv2d/Linear weights")
        _safe_zero_grad(model)

        first_loss = _cross_entropy(model, inputs, targets)
        first_grads = torch.autograd.grad(
            first_loss,
            weights,
            allow_unused=True,
            retain_graph=False,
        )

        second_loss = _cross_entropy(model, inputs, targets)
        second_grads = torch.autograd.grad(
            second_loss,
            weights,
            create_graph=True,
            allow_unused=True,
        )
        z = torch.zeros((), device=inputs.device)
        for first, second in zip(first_grads, second_grads):
            if first is not None and second is not None:
                z = z + (first.detach() * second).sum()
        if not z.requires_grad:
            raise RuntimeError("GraSP could not construct a second-order graph")
        z.backward()

        score = torch.zeros((), device=inputs.device)
        for weight in weights:
            if weight.grad is not None:
                score = score + torch.sum(-weight.detach() * weight.grad.detach())
        _safe_zero_grad(model)
        return _finite_float(score)


class JacobCovProxy(ZeroCostProxy):
    """Jacobian covariance score from Zero-Cost Proxies for Lightweight NAS."""

    metadata = ProxyMetadata(
        name="jacov",
        family="input_jacobian",
        source="mohsaied/zero-cost-nas",
        max_samples=16,
    )

    def calculate(self, model: nn.Module, inputs: torch.Tensor, targets: torch.Tensor) -> float:
        inputs, targets = self._limit_batch(inputs, targets)
        del targets
        model.train()
        _safe_zero_grad(model)
        probe = inputs.detach().clone().requires_grad_(True)
        output = _extract_tensor(model(probe))
        output.backward(torch.ones_like(output))
        if probe.grad is None:
            raise RuntimeError("JacobCov input gradient was not produced")
        jacobian = probe.grad.detach().reshape(probe.size(0), -1).double().cpu().numpy()
        correlations = np.corrcoef(jacobian)
        correlations = np.nan_to_num(correlations, nan=0.0, posinf=0.0, neginf=0.0)
        eigenvalues = np.linalg.eigvalsh(correlations)
        eigenvalues = np.clip(np.real(eigenvalues), a_min=0.0, a_max=None)
        k = 1e-5
        score = -np.sum(np.log(eigenvalues + k) + 1.0 / (eigenvalues + k))
        _safe_zero_grad(model)
        return _finite_float(score)


class ZiCoProxy(ZeroCostProxy):
    """Inverse coefficient of variation on Conv2d/Linear gradients."""

    metadata = ProxyMetadata(
        name="zico",
        family="gradient_consistency",
        source="SLDGroup/ZiCo",
        max_samples=32,
    )

    def calculate(self, model: nn.Module, inputs: torch.Tensor, targets: torch.Tensor) -> float:
        inputs, targets = self._limit_batch(inputs, targets)
        if inputs.size(0) < 4:
            raise ValueError("ZiCo requires at least four probe samples")
        model.train()
        modules = [(name, module) for name, module in model.named_modules() if isinstance(module, (nn.Conv2d, nn.Linear))]
        if not modules:
            raise RuntimeError("ZiCo found no Conv2d/Linear modules")

        split_count = min(4, inputs.size(0))
        gradient_samples: Dict[str, List[np.ndarray]] = defaultdict(list)
        input_chunks = torch.tensor_split(inputs, split_count)
        target_chunks = torch.tensor_split(targets, split_count)
        for input_chunk, target_chunk in zip(input_chunks, target_chunks):
            if input_chunk.numel() == 0:
                continue
            _safe_zero_grad(model)
            _cross_entropy(model, input_chunk, target_chunk).backward()
            for name, module in modules:
                if module.weight.grad is not None:
                    gradient_samples[name].append(
                        module.weight.grad.detach().float().reshape(-1).cpu().numpy()
                    )

        score = 0.0
        used_layers = 0
        for samples in gradient_samples.values():
            if len(samples) < 2:
                continue
            array = np.stack(samples, axis=0)
            std = np.std(array, axis=0)
            mean_abs = np.mean(np.abs(array), axis=0)
            valid = std > 1e-12
            if not np.any(valid):
                continue
            layer_sum = float(np.sum(mean_abs[valid] / std[valid]))
            if layer_sum > 0 and math.isfinite(layer_sum):
                score += math.log(layer_sum)
                used_layers += 1
        _safe_zero_grad(model)
        if used_layers == 0:
            raise FloatingPointError("ZiCo had no layers with non-zero gradient variance")
        return _finite_float(score)


def _gaussian_initialize(model: nn.Module) -> None:
    with torch.no_grad():
        for module in model.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.normal_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.GroupNorm)):
                if module.weight is not None:
                    nn.init.ones_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)


class _LastFeatureCollector:
    def __init__(self) -> None:
        self.last: Optional[torch.Tensor] = None
        self.handles: List[torch.utils.hooks.RemovableHandle] = []

    def install(self, model: nn.Module) -> None:
        def hook(module: nn.Module, module_inputs: tuple[object, ...], output: object) -> None:
            del module, module_inputs
            try:
                tensor = _extract_tensor(output)
            except TypeError:
                return
            if tensor.ndim == 4:
                self.last = tensor

        for module in model.modules():
            if isinstance(module, nn.Conv2d):
                self.handles.append(module.register_forward_hook(hook))

    def remove(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


class ZenProxy(ZeroCostProxy):
    """Generic Zen-Score adaptation using the last convolutional feature map."""

    metadata = ProxyMetadata(
        name="zen",
        family="forward_expressivity",
        source="idstcv/ZenNAS",
        max_samples=16,
    )

    def calculate(self, model: nn.Module, inputs: torch.Tensor, targets: torch.Tensor) -> float:
        inputs, targets = self._limit_batch(inputs, targets)
        del targets
        model.train()
        _gaussian_initialize(model)
        gamma = 1e-2
        collector = _LastFeatureCollector()
        collector.install(model)
        try:
            with torch.no_grad():
                random_input = torch.randn_like(inputs)
                random_input_2 = torch.randn_like(inputs)
                mixed_input = random_input + gamma * random_input_2

                collector.last = None
                logits_1 = _extract_tensor(model(random_input))
                feature_1 = collector.last if collector.last is not None else logits_1

                collector.last = None
                logits_2 = _extract_tensor(model(mixed_input))
                feature_2 = collector.last if collector.last is not None else logits_2

                reduce_dims = tuple(range(1, feature_1.ndim))
                difference = torch.abs(feature_1 - feature_2).sum(dim=reduce_dims).mean()
                difference = torch.clamp(difference, min=1e-12)
                log_bn_scaling = torch.zeros((), device=inputs.device)
                for module in model.modules():
                    if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):
                        variance = torch.clamp(module.running_var.float().mean(), min=1e-12)
                        log_bn_scaling = log_bn_scaling + torch.log(torch.sqrt(variance))
                score = torch.log(difference.float()) + log_bn_scaling
                return _finite_float(score)
        finally:
            collector.remove()


class _FunctionalReluCollector:
    """Temporarily instrument torch.nn.functional.relu without editing templates."""

    def __init__(self, callback: Callable[[torch.Tensor, torch.Tensor], None]) -> None:
        self.callback = callback
        self._original: Optional[Callable[..., torch.Tensor]] = None

    def __enter__(self) -> "_FunctionalReluCollector":
        self._original = F.relu

        def wrapped(input_tensor: torch.Tensor, inplace: bool = False) -> torch.Tensor:
            assert self._original is not None
            callback_input = input_tensor.detach().clone() if inplace else input_tensor
            output = self._original(input_tensor, inplace=inplace)
            self.callback(callback_input, output)
            return output

        F.relu = wrapped  # type: ignore[assignment]
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        if self._original is not None:
            F.relu = self._original  # type: ignore[assignment]


class NASWOTProxy(ZeroCostProxy):
    """NASWOT log-determinant score with functional-ReLU instrumentation."""

    metadata = ProxyMetadata(
        name="naswot",
        family="activation_kernel",
        source="BayesWatch/nas-without-training",
        max_samples=32,
    )

    def calculate(self, model: nn.Module, inputs: torch.Tensor, targets: torch.Tensor) -> float:
        inputs, targets = self._limit_batch(inputs, targets)
        del targets
        model.eval()
        batch_size = inputs.size(0)
        kernel = torch.zeros(batch_size, batch_size, dtype=torch.float64, device="cpu")
        activation_count = 0

        def collect(input_tensor: torch.Tensor, output: torch.Tensor) -> None:
            del output
            nonlocal kernel, activation_count
            if input_tensor.ndim < 2 or input_tensor.size(0) != batch_size:
                return
            binary = (input_tensor.detach() > 0).reshape(batch_size, -1).to(torch.float64).cpu()
            kernel += binary @ binary.t() + (1.0 - binary) @ (1.0 - binary).t()
            activation_count += 1

        with torch.no_grad(), _FunctionalReluCollector(collect):
            model(inputs)
        if activation_count == 0:
            raise RuntimeError("NASWOT observed no torch.nn.functional.relu activations")
        kernel += torch.eye(batch_size, dtype=torch.float64) * 1e-6
        sign, logdet = torch.linalg.slogdet(kernel)
        if sign <= 0:
            raise FloatingPointError("NASWOT kernel was not positive definite")
        return _finite_float(logdet)


class SWAPProxy(ZeroCostProxy):
    """Sample-Wise Activation Patterns with memory-safe pattern hashing."""

    metadata = ProxyMetadata(
        name="swap",
        family="activation_patterns",
        source="pym1024/SWAP",
        max_samples=32,
    )

    def calculate(self, model: nn.Module, inputs: torch.Tensor, targets: torch.Tensor) -> float:
        inputs, targets = self._limit_batch(inputs, targets)
        del targets
        model.eval()
        batch_size = inputs.size(0)
        unique_patterns: set[bytes] = set()
        activation_count = 0

        def collect(input_tensor: torch.Tensor, output: torch.Tensor) -> None:
            del input_tensor
            nonlocal activation_count
            if output.ndim < 2 or output.size(0) != batch_size:
                return
            patterns = (output.detach() > 0).reshape(batch_size, -1).t().cpu().numpy().astype(np.uint8)
            packed = np.packbits(patterns, axis=1)
            unique_patterns.update(row.tobytes() for row in packed)
            activation_count += 1

        with torch.no_grad(), _FunctionalReluCollector(collect):
            model(inputs)
        if activation_count == 0:
            raise RuntimeError("SWAP observed no torch.nn.functional.relu activations")
        return _finite_float(float(len(unique_patterns)))


class MeCoProxy(ZeroCostProxy):
    """Minimum eigenvalue of feature-map correlations on one input sample."""

    metadata = ProxyMetadata(
        name="meco",
        family="feature_correlation",
        source="HamsterMimi/MeCo",
        max_samples=1,
    )

    def calculate(self, model: nn.Module, inputs: torch.Tensor, targets: torch.Tensor) -> float:
        inputs, targets = self._limit_batch(inputs, targets)
        del targets
        model.eval()
        layer_scores: List[torch.Tensor] = []
        handles: List[torch.utils.hooks.RemovableHandle] = []

        def hook(module: nn.Module, module_inputs: tuple[object, ...], output: object) -> None:
            del module, module_inputs
            tensor = _extract_tensor(output).detach()
            if tensor.ndim != 4 or tensor.size(0) == 0:
                return
            feature = tensor[0].reshape(tensor.size(1), -1).double().cpu()
            if feature.size(0) < 2 or feature.size(1) < 2:
                return
            centered = feature - feature.mean(dim=1, keepdim=True)
            std = centered.std(dim=1, unbiased=False, keepdim=True)
            valid = std.squeeze(1) > 1e-12
            centered = centered[valid]
            std = std[valid]
            if centered.size(0) < 2:
                return
            normalized = centered / std
            correlation = normalized @ normalized.t() / max(1, normalized.size(1))
            correlation = torch.nan_to_num(correlation, nan=0.0, posinf=0.0, neginf=0.0)
            eigenvalues = torch.linalg.eigvalsh(correlation)
            layer_scores.append(eigenvalues.min())

        try:
            for module in model.modules():
                if isinstance(module, nn.Conv2d):
                    handles.append(module.register_forward_hook(hook))
            with torch.no_grad():
                model(inputs)
            if not layer_scores:
                raise RuntimeError("MeCo found no valid convolutional feature maps")
            return _finite_float(torch.stack(layer_scores).sum())
        finally:
            for handle in handles:
                handle.remove()


class MACsProxy(ZeroCostProxy):
    """Multiply-accumulate count per input sample, used as a complexity baseline."""

    metadata = ProxyMetadata(
        name="macs",
        family="complexity",
        source="local deterministic hook counter",
        max_samples=1,
    )

    def calculate(self, model: nn.Module, inputs: torch.Tensor, targets: torch.Tensor) -> float:
        inputs, targets = self._limit_batch(inputs, targets)
        del targets
        model.eval()
        total_macs = 0.0
        handles: List[torch.utils.hooks.RemovableHandle] = []

        def conv_hook(module: nn.Module, module_inputs: tuple[object, ...], output: object) -> None:
            del module_inputs
            nonlocal total_macs
            assert isinstance(module, nn.Conv2d)
            tensor = _extract_tensor(output)
            output_elements = tensor[0].numel()
            kernel_ops = module.kernel_size[0] * module.kernel_size[1] * (module.in_channels / module.groups)
            total_macs += float(output_elements * kernel_ops)

        def linear_hook(module: nn.Module, module_inputs: tuple[object, ...], output: object) -> None:
            del module_inputs, output
            nonlocal total_macs
            assert isinstance(module, nn.Linear)
            total_macs += float(module.in_features * module.out_features)

        try:
            for module in model.modules():
                if isinstance(module, nn.Conv2d):
                    handles.append(module.register_forward_hook(conv_hook))
                elif isinstance(module, nn.Linear):
                    handles.append(module.register_forward_hook(linear_hook))
            with torch.no_grad():
                model(inputs)
            return _finite_float(total_macs)
        finally:
            for handle in handles:
                handle.remove()


_PROXY_REGISTRY: Dict[str, type[ZeroCostProxy]] = {
    proxy.metadata.name: proxy
    for proxy in (
        SynFlowProxy,
        SnipProxy,
        GradNormProxy,
        PlainProxy,
        L2NormProxy,
        FisherProxy,
        GraSPProxy,
        JacobCovProxy,
        ZiCoProxy,
        ZenProxy,
        NASWOTProxy,
        SWAPProxy,
        MeCoProxy,
        MACsProxy,
    )
}


def build_proxy(name: str) -> ZeroCostProxy:
    """Build one configured proxy by name."""

    normalized = name.strip().lower()
    try:
        return _PROXY_REGISTRY[normalized]()
    except KeyError as exc:
        raise ValueError("Unknown zero-cost proxy: %s" % name) from exc


def proxy_metadata(name: str) -> ProxyMetadata:
    """Return immutable metadata without instantiating a model."""

    return build_proxy(name).metadata


def supported_proxy_names() -> tuple[str, ...]:
    return tuple(_PROXY_REGISTRY)


def parameter_count(model: nn.Module) -> float:
    """Return trainable parameter count as a simple complexity baseline."""

    return float(sum(parameter.numel() for parameter in _trainable_parameters(model)))
