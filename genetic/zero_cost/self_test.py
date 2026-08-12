"""Dataset-free self-tests for the extended zero-cost proxy bank."""

from __future__ import annotations

import math
import tempfile
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

from .config import DEFAULT_PROXIES, ZeroCostConfig
from .metrics import method_metrics, spearman
from .proxies import build_proxy, parameter_count, proxy_metadata


class TinyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 8, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(8)
        self.conv2 = nn.Conv2d(8, 12, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(12)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(12, 10)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        output = F.relu(self.bn1(self.conv1(inputs)))
        output = F.relu(self.bn2(self.conv2(output)))
        output = self.pool(output).flatten(1)
        return self.classifier(output)


def run() -> None:
    torch.set_num_threads(1)
    inputs = torch.randn(16, 3, 16, 16)
    targets = torch.randint(0, 10, (16,))

    results = {}
    for proxy_name in DEFAULT_PROXIES:
        torch.manual_seed(7)
        model = TinyModel()
        score = build_proxy(proxy_name).calculate(model, inputs, targets)
        assert math.isfinite(score), (proxy_name, score)
        results[proxy_name] = score
        assert proxy_metadata(proxy_name).name == proxy_name

    assert parameter_count(TinyModel()) > 0
    assert results["macs"] > 0
    assert results["swap"] > 0
    assert math.isclose(spearman([1, 2, 3], [10, 20, 30]), 1.0)
    metrics = method_metrics([0.9, 0.1, 0.5], [0.8, 0.2, 0.6], top_k=2)
    assert metrics["top_k_recall"] == 1.0

    with tempfile.TemporaryDirectory() as directory:
        ini = Path(directory) / "global.ini"
        ini.write_text(
            "[zero_cost]\n"
            "enabled = 1\n"
            "proxies = " + ", ".join(DEFAULT_PROXIES) + "\n"
            "batch_size = 16\n"
            "num_batches = 2\n",
            encoding="utf-8",
        )
        config = ZeroCostConfig.from_ini(ini)
        assert config.enabled
        assert config.proxies == DEFAULT_PROXIES

    print("zero_cost extended self-test passed")
    for name, score in results.items():
        print("%-10s %.6g" % (name, score))


if __name__ == "__main__":
    run()
