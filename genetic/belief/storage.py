"""Persistent JSON storage for the belief archive and learned state."""

from __future__ import annotations

import json
import os
from dataclasses import fields
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

from .archive import ArchiveEntry, EvaluatedArchitectureArchive
from .encoder import ArchitectureEncoder, ArchitectureEncoding


class ArchiveStorage:
    """Save and restore an EvaluatedArchitectureArchive object."""

    SCHEMA_VERSION = 2
    SUPPORTED_SCHEMA_VERSIONS = {1, 2}

    @classmethod
    def save_archive(cls, archive: EvaluatedArchitectureArchive, path: Path | str) -> Path:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": cls.SCHEMA_VERSION,
            "saved_at_utc": datetime.now(timezone.utc).isoformat(),
            "archive_summary": archive.summary(),
            "entries": archive.to_records(),
        }
        cls._atomic_json_write(output_path, payload)
        return output_path

    @classmethod
    def load_archive(
        cls,
        path: Path | str,
        encoder: Optional[ArchitectureEncoder] = None,
    ) -> EvaluatedArchitectureArchive:
        input_path = Path(path)
        if not input_path.exists():
            raise FileNotFoundError(f"Archive file was not found: {input_path}")
        with input_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        cls._validate_payload(payload)
        schema_version = int(payload.get("schema_version", 1))
        archive = EvaluatedArchitectureArchive(encoder=encoder)
        for raw_entry in payload["entries"]:
            entry = cls._entry_from_dict(raw_entry, schema_version=schema_version)
            if archive.contains(entry.architecture_id):
                raise ValueError(
                    "The archive file contains a repeated architecture_id: "
                    f"{entry.architecture_id}"
                )
            archive._entries[entry.architecture_id] = entry
        return archive

    @classmethod
    def save_state(cls, state: Dict[str, Any], path: Path | str) -> Path:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": cls.SCHEMA_VERSION,
            "saved_at_utc": datetime.now(timezone.utc).isoformat(),
            "state": state,
        }
        cls._atomic_json_write(output_path, payload)
        return output_path

    @classmethod
    def load_state(cls, path: Path | str) -> Dict[str, Any]:
        input_path = Path(path)
        if not input_path.exists():
            raise FileNotFoundError(f"Belief state file was not found: {input_path}")
        with input_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        version = int(payload.get("schema_version", 1))
        if version not in cls.SUPPORTED_SCHEMA_VERSIONS:
            raise ValueError(f"Unsupported belief state schema version: {version}")
        state = payload.get("state")
        if not isinstance(state, dict):
            raise TypeError("Belief state field must be a dictionary")
        return state

    @classmethod
    def _validate_payload(cls, payload: Any) -> None:
        if not isinstance(payload, dict):
            raise TypeError("Archive JSON must contain an object at the top level")
        version = int(payload.get("schema_version", 1))
        if version not in cls.SUPPORTED_SCHEMA_VERSIONS:
            raise ValueError(f"Unsupported archive schema version: {version}")
        if not isinstance(payload.get("entries"), list):
            raise TypeError("Archive JSON field 'entries' must be a list")

    @classmethod
    def _entry_from_dict(cls, data: Dict[str, Any], schema_version: int) -> ArchiveEntry:
        if not isinstance(data, dict):
            raise TypeError("Each archive entry must be a dictionary")
        encoding_data = data.get("encoding")
        if not isinstance(encoding_data, dict):
            raise TypeError("Archive entry field 'encoding' must be a dictionary")
        migrated = cls._migrate_encoding(dict(encoding_data), schema_version)
        encoding_field_names = {field.name for field in fields(ArchitectureEncoding)}
        missing = encoding_field_names.difference(migrated)
        if missing:
            raise ValueError(f"Stored architecture encoding is missing fields: {sorted(missing)}")
        encoding = ArchitectureEncoding(**{name: migrated[name] for name in encoding_field_names})
        return ArchiveEntry(
            architecture_id=str(data["architecture_id"]),
            architecture_string=str(data["architecture_string"]),
            encoding=encoding,
            fitness_mean=float(data["fitness_mean"]),
            fitness_last=float(data["fitness_last"]),
            fitness_best=float(data["fitness_best"]),
            fitness_worst=float(data["fitness_worst"]),
            evaluation_count=int(data["evaluation_count"]),
            first_generation=int(data["first_generation"]),
            last_generation=int(data["last_generation"]),
            first_individual_id=str(data["first_individual_id"]),
            last_individual_id=str(data["last_individual_id"]),
            run_ids=cls._string_list(data.get("run_ids", [])),
            sources=cls._string_list(data.get("sources", [])),
            fitness_history=[float(value) for value in data.get("fitness_history", [])],
        )

    @staticmethod
    def _migrate_encoding(data: Dict[str, Any], schema_version: int) -> Dict[str, Any]:
        if schema_version >= 2:
            return data
        numeric = dict(data.get("numeric_summary") or {})
        structural_keys = {
            "length",
            "attention_count",
            "attention_density",
            "pool_count",
            "max_pool_count",
            "avg_pool_count",
        }
        structural = {key: float(value) for key, value in numeric.items() if key in structural_keys}
        capacity = {key: float(value) for key, value in numeric.items() if key not in structural_keys}
        data["structural_numeric_summary"] = structural
        data["capacity_numeric_summary"] = capacity
        data.pop("numeric_summary", None)
        data.pop("pair_bigrams", None)
        return data

    @staticmethod
    def _atomic_json_write(path: Path, payload: Dict[str, Any]) -> None:
        temporary = path.with_suffix(path.suffix + ".tmp")
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)

    @staticmethod
    def _string_list(values: Iterable[Any]) -> list[str]:
        return [str(value) for value in values]


def _run_self_test() -> None:
    from tempfile import TemporaryDirectory

    encoding = ArchitectureEncoding(
        architecture_id="arch_1",
        architecture_string="[ca-densenet]-[pool]",
        individual_id="indi0001",
        length=2,
        module_sequence=["ca-densenet", "pool"],
        base_sequence=["densenet", "pool"],
        attention_sequence=["ca", "none"],
        base_attention_pairs=["ca-densenet", "none-pool"],
        module_counts={"ca-densenet": 1, "pool": 1},
        base_counts={"densenet": 1, "pool": 1},
        attention_counts={"ca": 1, "none": 1},
        module_bigrams=["ca-densenet->pool"],
        structural_numeric_summary={"length": 2.0, "attention_density": 0.5},
        capacity_numeric_summary={"channel_ratio_mean": 1.0},
        unit_records=[],
    )
    archive = EvaluatedArchitectureArchive()
    archive.add_encoding(encoding, fitness=0.82, generation=0)
    with TemporaryDirectory() as directory:
        path = Path(directory) / "archive.json"
        ArchiveStorage.save_archive(archive, path)
        restored = ArchiveStorage.load_archive(path)
    assert len(restored) == 1
    assert abs(restored.get("arch_1").fitness_mean - 0.82) < 1e-12
    print("Archive storage self-test passed.")


if __name__ == "__main__":
    _run_self_test()
