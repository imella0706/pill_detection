from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def as_project_path(project_root: Path, path_value: str | Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return project_root / path


def infer_dataset_identity(data_yaml: Path) -> dict[str, str | None]:
    dataset_yaml = Path(data_yaml)
    dataset_root = dataset_yaml.parent
    dataset_family = dataset_root.parent.name if dataset_root.parent != dataset_root else None
    variant_id = dataset_root.name if dataset_root.name else None
    return {
        "dataset_family": dataset_family,
        "variant_id": variant_id,
    }


def resolve_manifest_path(project_root: Path, processed_root: str | Path, data_yaml: Path) -> Path | None:
    processed_dir = as_project_path(project_root, processed_root)
    identity = infer_dataset_identity(data_yaml)
    dataset_family = identity["dataset_family"]
    variant_id = identity["variant_id"]

    candidates: list[Path] = []
    if dataset_family and variant_id:
        candidates.append(processed_dir / "offline" / dataset_family / variant_id / "manifest.json")
    if variant_id:
        candidates.append(processed_dir / "offline" / variant_id / "manifest.json")

    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def load_data_lineage(project_root: Path, processed_root: str | Path, data_yaml: Path) -> dict[str, Any]:
    identity = infer_dataset_identity(data_yaml)
    manifest_path = resolve_manifest_path(project_root, processed_root, data_yaml)

    lineage: dict[str, Any] = {
        "dataset_family": identity["dataset_family"],
        "variant_id": identity["variant_id"],
        "manifest_path": str(manifest_path) if manifest_path else None,
        "data_version": None,
        "dataset_hash": None,
        "input_paths": {},
        "annotation_source_dir": None,
    }
    if manifest_path is None:
        return lineage

    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    input_paths = payload.get("input_paths") or {}
    if not isinstance(input_paths, dict):
        input_paths = {}

    lineage.update(
        {
            "data_version": payload.get("data_version"),
            "dataset_hash": payload.get("dataset_hash"),
            "input_paths": input_paths,
            "annotation_source_dir": input_paths.get("annotation_source_dir")
            or payload.get("input_annotations_dir"),
        }
    )
    return lineage
