#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from urllib.parse import urlparse, unquote

import mlflow
from mlflow.tracking import MlflowClient


def parse_models_uri(model_uri: str) -> tuple[str, str, bool]:
    # models:/<name>@<alias> or models:/<name>/<version>
    remainder = model_uri[len("models:/") :].strip("/")
    if "@" in remainder:
        model_name, alias = remainder.rsplit("@", 1)
        if not model_name or not alias:
            raise ValueError(f"invalid registry alias URI: {model_uri}")
        return model_name, alias, True
    if "/" in remainder:
        model_name, version = remainder.rsplit("/", 1)
        if not model_name or not version:
            raise ValueError(f"invalid registry version URI: {model_uri}")
        return model_name, version, False
    raise ValueError(f"unsupported model URI format: {model_uri}")


def file_uri_to_path(uri: str) -> Path | None:
    if not uri.startswith("file://"):
        return None
    parsed = urlparse(uri)
    return Path(unquote(parsed.path))


def select_checkpoint(root: Path) -> Path:
    if root.is_file():
        return root

    for candidate in (root / "best.pt", root / "model.pt"):
        if candidate.exists():
            return candidate

    sibling_raw_best = root.parent / "model_raw" / "best.pt"
    if sibling_raw_best.exists():
        return sibling_raw_best

    pt_files = sorted(root.rglob("*.pt"))
    if not pt_files:
        raise FileNotFoundError(f"No .pt checkpoint found under: {root}")

    for candidate in pt_files:
        if candidate.name == "best.pt":
            return candidate
    return pt_files[0]


def resolve_model_uri(model_uri: str, tracking_uri: str | None) -> Path:
    if not model_uri.startswith("models:/"):
        return Path(model_uri)

    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)

    model_name, alias_or_version, is_alias = parse_models_uri(model_uri)
    client = MlflowClient()
    mv = (
        client.get_model_version_by_alias(model_name, alias_or_version)
        if is_alias
        else client.get_model_version(model_name, alias_or_version)
    )
    source_uri = getattr(mv, "source", None)
    if not source_uri:
        raise RuntimeError(f"source URI not found: {model_uri}")

    local_source = file_uri_to_path(source_uri)
    if local_source is not None:
        return select_checkpoint(local_source)

    downloaded_path = Path(mlflow.artifacts.download_artifacts(artifact_uri=source_uri))
    return select_checkpoint(downloaded_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-uri", required=True)
    parser.add_argument("--tracking-uri", default=None)
    args = parser.parse_args()
    resolved = resolve_model_uri(args.model_uri, args.tracking_uri)
    print(str(resolved))


if __name__ == "__main__":
    main()
