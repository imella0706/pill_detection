#!/usr/bin/env python3
"""
Registry operations for candidate registration, promotion, and rollback.

Designed for the flow:
1) register candidate -> set @staging
2) deploy green with @staging and validate
3) promote @staging -> @production on success
4) rollback @production to previous version if needed
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
from typing import Any

import mlflow
from mlflow.tracking import MlflowClient


def parse_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"invalid boolean value: {value}")


def utc_now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def build_client(tracking_uri: str | None) -> MlflowClient:
    uri = tracking_uri or os.getenv("MLFLOW_TRACKING_URI")
    if uri:
        mlflow.set_tracking_uri(uri)
    return MlflowClient(tracking_uri=uri)


def safe_get_alias_version(client: MlflowClient, model_name: str, alias: str) -> str | None:
    try:
        mv = client.get_model_version_by_alias(model_name, alias)
    except Exception:
        return None
    version = getattr(mv, "version", None)
    return str(version) if version is not None else None


def clear_archived_tags(client: MlflowClient, model_name: str, version: str) -> None:
    for key in ("status", "archived_at_utc", "archived_reason"):
        try:
            client.delete_model_version_tag(model_name, version, key)
        except Exception:
            pass


def archive_version(
    client: MlflowClient,
    model_name: str,
    version: str,
    tag_key: str,
    tag_value: str,
    reason: str,
) -> None:
    client.set_model_version_tag(model_name, version, tag_key, tag_value)
    client.set_model_version_tag(model_name, version, "archived_at_utc", utc_now_iso())
    client.set_model_version_tag(model_name, version, "archived_reason", reason)


def cmd_register(args: argparse.Namespace) -> dict[str, Any]:
    client = build_client(args.tracking_uri)
    model_uri = f"runs:/{args.run_id}/{args.model_artifact_path.strip('/')}"
    registration = mlflow.register_model(model_uri=model_uri, name=args.model_name)
    registered_version = str(getattr(registration, "version"))

    previous_staging = None
    if args.set_staging:
        previous_staging = safe_get_alias_version(client, args.model_name, args.staging_alias)
        client.set_registered_model_alias(args.model_name, args.staging_alias, registered_version)

    result = {
        "action": "register",
        "model_name": args.model_name,
        "run_id": args.run_id,
        "model_uri": model_uri,
        "registered_version": registered_version,
        "set_staging": bool(args.set_staging),
        "staging_alias": args.staging_alias if args.set_staging else None,
        "previous_staging_version": previous_staging,
    }
    return result


def cmd_promote(args: argparse.Namespace) -> dict[str, Any]:
    client = build_client(args.tracking_uri)

    source_version = safe_get_alias_version(client, args.model_name, args.source_alias)
    if source_version is None:
        raise RuntimeError(
            f"source alias not found: model={args.model_name}, alias={args.source_alias}"
        )

    previous_production = safe_get_alias_version(client, args.model_name, args.production_alias)
    client.set_registered_model_alias(args.model_name, args.production_alias, source_version)

    staging_cleared = False
    if args.clear_staging_on_promote:
        try:
            client.delete_registered_model_alias(args.model_name, args.source_alias)
            staging_cleared = True
        except Exception:
            staging_cleared = False

    if previous_production and previous_production != source_version:
        archive_version(
            client=client,
            model_name=args.model_name,
            version=previous_production,
            tag_key=args.archive_tag_key,
            tag_value=args.archive_tag_value,
            reason=f"replaced_by_{source_version}",
        )

    if args.clear_archived_on_new:
        clear_archived_tags(client, args.model_name, source_version)

    result = {
        "action": "promote",
        "model_name": args.model_name,
        "source_alias": args.source_alias,
        "source_version": source_version,
        "production_alias": args.production_alias,
        "previous_production_version": previous_production,
        "clear_staging_on_promote": bool(args.clear_staging_on_promote),
        "staging_cleared": staging_cleared,
    }
    return result


def cmd_rollback(args: argparse.Namespace) -> dict[str, Any]:
    client = build_client(args.tracking_uri)
    to_version = str(args.to_version)

    previous_production = safe_get_alias_version(client, args.model_name, args.production_alias)
    client.set_registered_model_alias(args.model_name, args.production_alias, to_version)

    if args.archive_current and previous_production and previous_production != to_version:
        archive_version(
            client=client,
            model_name=args.model_name,
            version=previous_production,
            tag_key=args.archive_tag_key,
            tag_value=args.archive_tag_value,
            reason=f"rollback_to_{to_version}",
        )

    if args.clear_archived_on_new:
        clear_archived_tags(client, args.model_name, to_version)

    result = {
        "action": "rollback",
        "model_name": args.model_name,
        "to_version": to_version,
        "production_alias": args.production_alias,
        "previous_production_version": previous_production,
    }
    return result


def cmd_status(args: argparse.Namespace) -> dict[str, Any]:
    client = build_client(args.tracking_uri)

    production_version = safe_get_alias_version(client, args.model_name, args.production_alias)
    staging_version = safe_get_alias_version(client, args.model_name, args.staging_alias)
    versions = []
    for mv in client.search_model_versions(f"name='{args.model_name}'"):
        versions.append(
            {
                "version": str(mv.version),
                "aliases": list(getattr(mv, "aliases", []) or []),
                "run_id": getattr(mv, "run_id", None),
                "source": getattr(mv, "source", None),
            }
        )
    versions.sort(key=lambda x: int(x["version"]), reverse=True)
    return {
        "action": "status",
        "model_name": args.model_name,
        "production_alias": args.production_alias,
        "production_version": production_version,
        "staging_alias": args.staging_alias,
        "staging_version": staging_version,
        "versions": versions,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="MLflow registry operations")
    parser.add_argument("--tracking-uri", type=str, default=None, help="MLflow tracking URI")

    sub = parser.add_subparsers(dest="command", required=True)

    p_register = sub.add_parser("register", help="register model from run artifact")
    p_register.add_argument("--run-id", required=True, type=str)
    p_register.add_argument("--model-name", required=True, type=str)
    p_register.add_argument("--model-artifact-path", default="model", type=str)
    p_register.add_argument("--set-staging", type=parse_bool, default=True)
    p_register.add_argument("--staging-alias", type=str, default="staging")

    p_promote = sub.add_parser("promote", help="promote source alias to production")
    p_promote.add_argument("--model-name", required=True, type=str)
    p_promote.add_argument("--source-alias", type=str, default="staging")
    p_promote.add_argument("--production-alias", type=str, default="production")
    p_promote.add_argument("--archive-tag-key", type=str, default="status")
    p_promote.add_argument("--archive-tag-value", type=str, default="archived")
    p_promote.add_argument("--clear-archived-on-new", type=parse_bool, default=True)
    p_promote.add_argument("--clear-staging-on-promote", type=parse_bool, default=True)

    p_rollback = sub.add_parser("rollback", help="rollback production alias to target version")
    p_rollback.add_argument("--model-name", required=True, type=str)
    p_rollback.add_argument("--to-version", required=True, type=str)
    p_rollback.add_argument("--production-alias", type=str, default="production")
    p_rollback.add_argument("--archive-current", type=parse_bool, default=True)
    p_rollback.add_argument("--archive-tag-key", type=str, default="status")
    p_rollback.add_argument("--archive-tag-value", type=str, default="archived")
    p_rollback.add_argument("--clear-archived-on-new", type=parse_bool, default=True)

    p_status = sub.add_parser("status", help="show alias pointer and versions")
    p_status.add_argument("--model-name", required=True, type=str)
    p_status.add_argument("--production-alias", type=str, default="production")
    p_status.add_argument("--staging-alias", type=str, default="staging")

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "register":
        result = cmd_register(args)
    elif args.command == "promote":
        result = cmd_promote(args)
    elif args.command == "rollback":
        result = cmd_rollback(args)
    elif args.command == "status":
        result = cmd_status(args)
    else:
        raise RuntimeError(f"unsupported command: {args.command}")

    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
