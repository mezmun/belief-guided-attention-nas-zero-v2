"""Configuration reader for passive zero-cost monitoring."""

from __future__ import annotations

import configparser
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Tuple


DEFAULT_PROXIES = (
    "synflow",
    "snip",
    "gradnorm",
    "plain",
    "l2norm",
    "fisher",
    "grasp",
    "jacov",
    "zico",
    "zen",
    "naswot",
    "swap",
    "meco",
    "macs",
)


@dataclass(frozen=True)
class ZeroCostConfig:
    """Runtime settings read from the ``[zero_cost]`` INI section."""

    enabled: bool = False
    proxies: Tuple[str, ...] = DEFAULT_PROXIES
    batch_size: int = 32
    num_batches: int = 2
    valid_size: float = 0.10
    random_seed: int = 2312390
    device: str = "auto"
    num_workers: int = 0
    mean_rank_ensemble: bool = True
    median_rank_ensemble: bool = True
    family_rank_ensemble: bool = True
    complete_case_ensemble: bool = True
    include_parameter_count: bool = True
    top_k: int = 5
    output_subdirectory: str = "zero_cost"

    SUPPORTED_PROXIES = DEFAULT_PROXIES

    @classmethod
    def from_ini(cls, ini_path: str | Path | None = None) -> "ZeroCostConfig":
        """Load configuration; a missing section means the feature is disabled."""

        path = Path(ini_path) if ini_path is not None else cls.default_ini_path()
        parser = configparser.ConfigParser()
        parser.read(path, encoding="utf-8")
        if not parser.has_section("zero_cost"):
            return cls(enabled=False)

        section = parser["zero_cost"]
        default_proxy_text = ", ".join(DEFAULT_PROXIES)
        proxies = tuple(
            item.strip().lower()
            for item in section.get("proxies", default_proxy_text).split(",")
            if item.strip()
        )
        config = cls(
            enabled=section.getboolean("enabled", fallback=False),
            proxies=proxies,
            batch_size=section.getint("batch_size", fallback=32),
            num_batches=section.getint("num_batches", fallback=2),
            valid_size=section.getfloat("valid_size", fallback=0.10),
            random_seed=section.getint("random_seed", fallback=2312390),
            device=section.get("device", fallback="auto").strip().lower(),
            num_workers=section.getint("num_workers", fallback=0),
            mean_rank_ensemble=section.getboolean(
                "mean_rank_ensemble", fallback=True
            ),
            median_rank_ensemble=section.getboolean(
                "median_rank_ensemble", fallback=True
            ),
            family_rank_ensemble=section.getboolean(
                "family_rank_ensemble", fallback=True
            ),
            complete_case_ensemble=section.getboolean(
                "complete_case_ensemble", fallback=True
            ),
            include_parameter_count=section.getboolean(
                "include_parameter_count", fallback=True
            ),
            top_k=section.getint("top_k", fallback=5),
            output_subdirectory=section.get(
                "output_subdirectory", fallback="zero_cost"
            ).strip(),
        )
        config.validate()
        return config

    @staticmethod
    def default_ini_path() -> Path:
        """Return the repository-level ``global.ini`` path."""

        return Path(__file__).resolve().parents[2] / "global.ini"

    def validate(self) -> None:
        """Raise a clear error for invalid settings."""

        unsupported = sorted(set(self.proxies) - set(self.SUPPORTED_PROXIES))
        if unsupported:
            raise ValueError(
                "Unsupported zero-cost proxies: %s. Supported: %s"
                % (unsupported, list(self.SUPPORTED_PROXIES))
            )
        if not self.proxies:
            raise ValueError("At least one zero-cost proxy must be configured")
        if len(set(self.proxies)) != len(self.proxies):
            raise ValueError("zero_cost.proxies cannot contain duplicate names")
        if self.batch_size < 2:
            raise ValueError("zero_cost.batch_size must be at least 2")
        if self.num_batches < 1:
            raise ValueError("zero_cost.num_batches must be at least 1")
        if not 0 < self.valid_size <= 1:
            raise ValueError("zero_cost.valid_size must be in (0, 1]")
        if self.num_workers < 0:
            raise ValueError("zero_cost.num_workers cannot be negative")
        if self.top_k < 1:
            raise ValueError("zero_cost.top_k must be at least 1")
        if (
            self.device != "auto"
            and self.device != "cpu"
            and not self.device.startswith("cuda")
        ):
            raise ValueError("zero_cost.device must be auto, cpu, cuda, or cuda:<index>")
        if not self.output_subdirectory:
            raise ValueError("zero_cost.output_subdirectory cannot be empty")

    def as_dict(self) -> Dict[str, Any]:
        """Return a JSON-serializable configuration snapshot."""

        data = asdict(self)
        data["proxies"] = list(self.proxies)
        return data
