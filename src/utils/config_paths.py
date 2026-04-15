from __future__ import annotations

from pathlib import Path
import re
from typing import Any

from src.utils.project_paths import (
    CURATED_DATA_DIR,
    CURATED_TRAIN_ANNOTATIONS_DIR,
    DATASETS_DIR,
    DATA_DIR,
    PROCESSED_DATA_DIR,
    RAW_DATA_DIR,
    TEST_IMAGES_DIR,
    TRAIN_IMAGES_DIR,
)


_PATH_REF_PATTERN = re.compile(r"\$\{paths\.([a-zA-Z0-9_]+)\}")


def default_paths_config() -> dict[str, str]:
    return {
        "data_root": _to_posix_relative(DATA_DIR),
        "image_root": _to_posix_relative(RAW_DATA_DIR),
        "annotation_root": _to_posix_relative(CURATED_DATA_DIR),
        "processed_root": _to_posix_relative(PROCESSED_DATA_DIR),
        "datasets_root": _to_posix_relative(DATASETS_DIR),
        "train_images_dir": _to_posix_relative(TRAIN_IMAGES_DIR),
        "test_images_dir": _to_posix_relative(TEST_IMAGES_DIR),
        "annotation_source_dir": _to_posix_relative(CURATED_TRAIN_ANNOTATIONS_DIR),
    }


def resolve_config_paths(config: dict[str, Any]) -> dict[str, str]:
    paths = default_paths_config()
    configured_paths = config.get("paths") or {}
    if not isinstance(configured_paths, dict):
        raise TypeError("config.paths must be a mapping/dict")

    for key, value in configured_paths.items():
        paths[key] = str(value)

    # Resolve ${paths.<key>} references in a few passes.
    for _ in range(5):
        changed = False
        for key, value in list(paths.items()):
            resolved = _PATH_REF_PATTERN.sub(lambda m: paths.get(m.group(1), m.group(0)), str(value))
            if resolved != value:
                paths[key] = resolved
                changed = True
        if not changed:
            break

    return paths


def apply_train_config_paths(config: dict[str, Any]) -> dict[str, Any]:
    resolved = dict(config)
    resolved_paths = resolve_config_paths(config)
    resolved["paths"] = resolved_paths
    return resolved


def apply_inference_config_paths(config: dict[str, Any]) -> dict[str, Any]:
    resolved = dict(config)
    resolved_paths = resolve_config_paths(config)
    resolved["paths"] = resolved_paths

    if not resolved.get("test_images"):
        resolved["test_images"] = resolved_paths["test_images_dir"]
    if not resolved.get("json_dir"):
        resolved["json_dir"] = resolved_paths["annotation_source_dir"]

    return resolved


def merge_runtime_overrides(
    config: dict[str, Any],
    cli: dict[str, Any],
    schema: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """
    Merge runtime options using precedence: CLI > config aliases > default.

    schema example:
    {
      "seed": {"aliases": ("data_seed", "seed"), "default": 42, "cast": int},
      "use_copy_paste": {"aliases": ("use_copy_paste",), "default": True, "cast": bool},
    }
    """
    merged: dict[str, Any] = {}

    for key, rule in schema.items():
        aliases = tuple(rule.get("aliases") or (key,))
        default = rule.get("default")
        cast = rule.get("cast")

        cli_value = cli.get(key)
        if cli_value is not None:
            value = cli_value
        else:
            value = default
            for alias in aliases:
                if alias in config and config[alias] is not None:
                    value = config[alias]
                    break

        if cast is not None:
            value = _cast_runtime_value(value, cast)
        merged[key] = value

    return merged


def _cast_runtime_value(value: Any, cast: Any) -> Any:
    if value is None:
        return None
    if cast is bool:
        return _to_bool(value)
    return cast(value)


def _to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)

    text = str(value).strip().lower()
    if text in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise ValueError(f"invalid boolean value: {value}")


def _to_posix_relative(path: Path) -> str:
    return path.relative_to(path.parents[1]).as_posix()
