from __future__ import annotations

"""
YOLO 학습과 validation 평가를 한 번에 수행하는 학습 스크립트.

설정 파일 또는 CLI 인자를 받아 모델을 학습한 뒤, 같은 실험의 `validation` 성능을
평가해 mAP 지표를 출력한다. 최종 성능 결과는 metrics JSON으로 저장해
`experiments.md` 기록 및 실험 비교의 기준값으로 사용한다.
"""

from pathlib import Path
import argparse
import csv
import datetime
import getpass
import json
import os
import re
import sys
from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()

import torch
import yaml
from ultralytics import YOLO

from src.utils.config_paths import apply_train_config_paths, default_paths_config
from src.utils.data_lineage_utils import load_data_lineage
from src.utils.logging_utils import start_run_logging
from src.utils.metrics_utils import collect_runtime_env
from src.utils.project_paths import CURATED_TRAIN_ANNOTATIONS_DIR, PROJECT_ROOT, TEST_IMAGES_DIR


DEFAULT_DATA_YAML_PRIMARY = PROJECT_ROOT / "data" / "yolo_dataset" / "dataset.yaml"
METRICS_DIR = PROJECT_ROOT / "metrics"
DEFAULT_INFER_CONFIG_DIR = PROJECT_ROOT / "configs" / "inference"
DEFAULT_TEST_IMAGES_DIR = TEST_IMAGES_DIR
DEFAULT_JSON_DIR = CURATED_TRAIN_ANNOTATIONS_DIR

# 👉 runs 경로를 절대경로로 고정 (핵심)
RUNS_DIR = PROJECT_ROOT / "runs"


def get_device() -> str:
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "0"
    return "cpu"


def find_default_dataset_yaml() -> Path:
    # Current default training dataset path in this project.
    if DEFAULT_DATA_YAML_PRIMARY.exists():
        return DEFAULT_DATA_YAML_PRIMARY

    candidates = sorted(PROJECT_ROOT.glob("data/**/dataset.yaml"))
    hint = ""
    if candidates:
        hint = "\n\nFound these candidates:\n" + "\n".join(f"- {p}" for p in candidates)

    raise FileNotFoundError(
        "dataset.yaml not found.\n"
        f"- looked for: {DEFAULT_DATA_YAML_PRIMARY}"
        f"{hint}\n\n"
        "If your dataset lives elsewhere, pass it explicitly with:\n"
        "  python train_yolov11.py --data <path/to/dataset.yaml>\n"
    )


def load_config(config_path: Path) -> dict:
    if not config_path.exists():
        raise FileNotFoundError(f"config not found: {config_path}")

    suffix = config_path.suffix.lower()
    if suffix in {".yaml", ".yml"}:
        with config_path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
    elif suffix == ".json":
        with config_path.open("r", encoding="utf-8") as f:
            data = json.load(f) or {}
    else:
        raise ValueError(f"Unsupported config type: {config_path} (use .yaml/.yml or .json)")

    if not isinstance(data, dict):
        raise TypeError(f"Config must be a mapping/dict, got: {type(data).__name__}")

    return apply_train_config_paths(data)


def parse_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value

    text = str(value).strip().lower()
    if text in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"invalid boolean value: {value}")


def build_parser(defaults: dict) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=defaults.get("config"),
        help="Path to config file (.yaml/.yml/.json). CLI args override config values.",
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=Path(defaults["data"]) if defaults.get("data") else None,
        help="Path to Ultralytics dataset.yaml (default: auto-detect).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=defaults.get("model", "yolo11s.pt"),
        help="Model path (e.g., yolo11s.pt, yolov8n.pt)",
    )
    parser.add_argument("--epochs", type=int, default=int(defaults.get("epochs", 50)), help="number of epochs")
    parser.add_argument("--imgsz", type=int, default=int(defaults.get("imgsz", 640)), help="image size")
    parser.add_argument("--batch", type=int, default=int(defaults.get("batch", 32)), help="batch size")
    parser.add_argument(
        "--name",
        type=str,
        default=defaults.get("name", "pill_exp2_yolo11s"),
        help="experiment name",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=defaults.get("device", "0"),
        help="cuda device, i.e. 0 or 0,1,2,3 or cpu",
    )
    default_save_log = bool(defaults.get("save_log", True))
    parser.add_argument(
        "--save-log",
        dest="save_log",
        action="store_true",
        help="save runtime console log to logs/train",
    )
    parser.add_argument(
        "--no-save-log",
        dest="save_log",
        action="store_false",
        help="disable runtime console log file saving",
    )
    parser.set_defaults(save_log=default_save_log)
    parser.add_argument(
        "--workers",
        type=int,
        default=int(defaults.get("workers", 8)),
        help="number of dataloader workers",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=int(defaults.get("patience", 100)),
        help="early stopping patience",
    )
    parser.add_argument(
        "--amp",
        type=parse_bool,
        default=bool(defaults.get("amp", True)),
        help="use mixed precision (AMP)",
    )
    parser.add_argument(
        "--close_mosaic",
        type=int,
        default=int(defaults.get("close_mosaic", 10)),
        help="disable mosaic augmentation in final epochs",
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        default=str(defaults.get("optimizer", "auto")),
        help="optimizer (e.g., auto, SGD, AdamW)",
    )
    parser.add_argument("--lr0", type=float, default=float(defaults.get("lr0", 0.01)), help="initial learning rate")
    parser.add_argument("--lrf", type=float, default=float(defaults.get("lrf", 0.01)), help="final LR factor")
    parser.add_argument(
        "--momentum",
        type=float,
        default=float(defaults.get("momentum", 0.937)),
        help="SGD momentum / Adam beta1",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=float(defaults.get("weight_decay", 0.0005)),
        help="optimizer weight decay",
    )
    parser.add_argument(
        "--warmup_epochs",
        type=float,
        default=float(defaults.get("warmup_epochs", 3.0)),
        help="warmup epochs",
    )
    parser.add_argument(
        "--warmup_momentum",
        type=float,
        default=float(defaults.get("warmup_momentum", 0.8)),
        help="warmup momentum",
    )
    parser.add_argument(
        "--warmup_bias_lr",
        type=float,
        default=float(defaults.get("warmup_bias_lr", 0.1)),
        help="warmup bias learning rate",
    )
    parser.add_argument(
        "--cos_lr",
        type=parse_bool,
        default=bool(defaults.get("cos_lr", True)),
        help="use cosine LR scheduler",
    )
    parser.add_argument(
        "--pretrained",
        type=parse_bool,
        default=bool(defaults.get("pretrained", True)),
        help="use pretrained weights",
    )
    parser.add_argument(
        "--resume",
        type=parse_bool,
        default=bool(defaults.get("resume", False)),
        help="resume previous training run",
    )
    parser.add_argument(
        "--box",
        type=float,
        default=float(defaults.get("box", 7.5)),
        help="box loss gain weight",
    )
    parser.add_argument(
        "--cls",
        type=float,
        default=float(defaults.get("cls", 0.5)),
        help="classification loss gain weight",
    )
    parser.add_argument(
        "--dfl",
        type=float,
        default=float(defaults.get("dfl", 1.5)),
        help="distribution focal loss gain weight",
    )
    parser.add_argument("--hsv_h", type=float, default=float(defaults.get("hsv_h", 0.015)), help="hsv_h augmentation")
    parser.add_argument("--hsv_s", type=float, default=float(defaults.get("hsv_s", 0.7)), help="hsv_s augmentation")
    parser.add_argument("--hsv_v", type=float, default=float(defaults.get("hsv_v", 0.4)), help="hsv_v augmentation")
    parser.add_argument(
        "--translate",
        type=float,
        default=float(defaults.get("translate", 0.1)),
        help="image translation augmentation",
    )
    parser.add_argument("--scale", type=float, default=float(defaults.get("scale", 0.5)), help="image scale augmentation")
    parser.add_argument("--shear", type=float, default=float(defaults.get("shear", 0.0)), help="image shear augmentation")
    parser.add_argument(
        "--perspective",
        type=float,
        default=float(defaults.get("perspective", 0.0)),
        help="perspective augmentation",
    )
    parser.add_argument(
        "--erasing",
        type=float,
        default=float(defaults.get("erasing", 0.4)),
        help="random erasing augmentation",
    )
    parser.add_argument(
        "--auto_augment",
        type=str,
        default=str(defaults.get("auto_augment", "randaugment")),
        help="auto augmentation strategy",
    )
    # --- 증강(Augmentation) 제어용 인자 --- #
    parser.add_argument(
        "--fliplr",
        type=float,
        default=float(defaults.get("fliplr", 0.5)),
        help="horizontal flip probability",
    )
    parser.add_argument(
        "--flipud",
        type=float,
        default=float(defaults.get("flipud", 0.0)),
        help="vertical flip probability",
    )
    parser.add_argument(
        "--degrees",
        type=float,
        default=float(defaults.get("degrees", 0.0)),
        help="image rotation degrees",
    )
    parser.add_argument(
        "--mosaic",
        type=float,
        default=float(defaults.get("mosaic", 1.0)),
        help="mosaic augmentation probability",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=int(defaults.get("seed", 0)),
        help="random seed for reproducibility",
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        default=bool(defaults.get("deterministic", True)),
        help="enable deterministic mode",
    )

    parser.add_argument(
        "--copy_paste",
        type=float,
        default=float(defaults.get("copy_paste", 0.0)),
        help="copy-paste augmentation probability",
    )
    parser.add_argument(
        "--mixup",
        type=float,
        default=float(defaults.get("mixup", 0.0)),
        help="mixup augmentation probability",
    )
    default_enable_mlflow = bool(defaults.get("enable_mlflow", True))
    parser.add_argument(
        "--enable-mlflow",
        dest="enable_mlflow",
        action="store_true",
        help="enable MLflow run logging",
    )
    parser.add_argument(
        "--disable-mlflow",
        dest="enable_mlflow",
        action="store_false",
        help="disable MLflow run logging",
    )
    parser.set_defaults(enable_mlflow=default_enable_mlflow)
    parser.add_argument(
        "--mlflow-tracking-uri",
        type=str,
        default=defaults.get("mlflow_tracking_uri"),
        help="MLflow tracking URI. Falls back to MLFLOW_TRACKING_URI env var.",
    )
    parser.add_argument(
        "--mlflow-experiment",
        type=str,
        default=defaults.get("mlflow_experiment", "pill_detection_v2_train"),
        help="MLflow experiment name",
    )
    default_registered_model_name = (
        defaults.get("mlflow_registered_model_name")
        or os.getenv("MLFLOW_REGISTERED_MODEL_NAME")
        or "pill_detection_v2_yolo"
    )
    parser.add_argument(
        "--mlflow-registered-model-name",
        type=str,
        default=default_registered_model_name,
        help="MLflow registered model name for alias automation",
    )
    parser.add_argument(
        "--mlflow-staging-alias",
        type=str,
        default=defaults.get("mlflow_staging_alias", "staging"),
        help="Alias name for staging candidate",
    )
    parser.add_argument(
        "--mlflow-production-alias",
        type=str,
        default=defaults.get("mlflow_production_alias", "production"),
        help="Alias name for production serving target",
    )
    parser.add_argument(
        "--mlflow-set-staging-alias",
        type=parse_bool,
        default=bool(defaults.get("mlflow_set_staging_alias", True)),
        help="set staging alias to current model version after registration",
    )
    parser.add_argument(
        "--mlflow-set-production-alias",
        type=parse_bool,
        default=bool(defaults.get("mlflow_set_production_alias", True)),
        help="set production alias to current model version after registration",
    )
    parser.add_argument(
        "--mlflow-registry-strict",
        type=parse_bool,
        default=bool(defaults.get("mlflow_registry_strict", False)),
        help="fail training when registry registration/alias update fails",
    )

    return parser


def infer_run_name_from_config(config_path: Path | None, seed: int) -> str | None:
    if config_path is None:
        return None
    match = re.search(r"^(exp\d+[A-Za-z0-9]*)", config_path.stem, flags=re.IGNORECASE)
    if not match:
        return None
    exp_id = match.group(1)
    return f"{exp_id}_s{seed}"


def parse_args() -> argparse.Namespace:
    base = argparse.ArgumentParser(add_help=False)
    base.add_argument("--config", type=Path, default=None)
    known, _ = base.parse_known_args()

    defaults: dict = {}
    if known.config:
        defaults = load_config(known.config)
        defaults["config"] = known.config

        allowed_keys = {
            "config",
            "data",
            "model",
            "epochs",
            "imgsz",
            "batch",
            "name",
            "device",
            "save_log",
            "workers",
            "patience",
            "amp",
            "close_mosaic",
            "optimizer",
            "lr0",
            "lrf",
            "momentum",
            "weight_decay",
            "warmup_epochs",
            "warmup_momentum",
            "warmup_bias_lr",
            "cos_lr",
            "pretrained",
            "resume",
            "box",
            "cls",
            "dfl",
            "hsv_h",
            "hsv_s",
            "hsv_v",
            "translate",
            "scale",
            "shear",
            "perspective",
            "erasing",
            "auto_augment",
            "fliplr",
            "flipud",
            "degrees",
            "mosaic",
            "copy_paste",
            "mixup",
            "seed",
            "deterministic",
            "enable_mlflow",
            "mlflow_tracking_uri",
            "mlflow_experiment",
            "mlflow_registered_model_name",
            "mlflow_staging_alias",
            "mlflow_production_alias",
            "mlflow_set_staging_alias",
            "mlflow_set_production_alias",
            "mlflow_registry_strict",
            "paths",
        }
        unknown_keys = sorted(set(defaults.keys()) - allowed_keys)
        if unknown_keys:
            print(
                f"Warning: ignoring unknown config keys: {unknown_keys}",
                file=sys.stderr,
            )

    parser = build_parser(defaults)
    args = parser.parse_args()
    explicit_cli_name = "--name" in sys.argv
    if not explicit_cli_name:
        inferred_run_name = infer_run_name_from_config(args.config, int(args.seed))
        if inferred_run_name:
            args.name = inferred_run_name
    return args


def infer_source_experiment(model_path: str) -> str | None:
    match = re.search(r"(?:^|[/_])exp[_-]?(\d+)(?:[^0-9]|$)", model_path, flags=re.IGNORECASE)
    if not match:
        match = re.search(r"pill_exp(\d+)", model_path, flags=re.IGNORECASE)
    return f"exp{match.group(1)}" if match else None


def infer_model_name(model_path: str) -> str:
    match = re.search(r"(yolo\d+[a-z]+|yolov\d+[a-z]+|rtdetr[\w-]*)", model_path, flags=re.IGNORECASE)
    return match.group(1) if match else model_path


def to_project_relative(path_value: str | Path) -> str:
    p = Path(path_value)
    if not p.is_absolute():
        p = (PROJECT_ROOT / p).resolve()
    else:
        p = p.resolve()
    try:
        return str(p.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path_value)


def normalize_inference_stem(stem: str) -> str:
    safe = re.sub(r"\s+", "_", stem.strip())
    safe = safe.replace("_train_", "_inference_")
    if safe.startswith("train_"):
        safe = "inference_" + safe[len("train_"):]
    if safe.endswith("_train"):
        safe = safe[:-6] + "_inference"
    return safe or "inference"


def ensure_inference_token(stem: str) -> str:
    safe = normalize_inference_stem(stem)
    if "_inference_" in safe or safe.startswith("inference_") or safe.endswith("_inference"):
        return safe
    match = re.match(r"^(exp\d+)(.*)$", safe, flags=re.IGNORECASE)
    if match:
        suffix = match.group(2) or ""
        return f"{match.group(1)}_inference{suffix}"
    return f"{safe}_inference"


def infer_default_submission_name(inference_stem: str) -> str:
    return inference_stem.replace("_inference_", "_")


def _is_mlflow_scalar(value: object) -> bool:
    return isinstance(value, (str, int, float, bool)) or value is None


def _normalize_mlflow_param_value(value: object) -> str | int | float | bool:
    if isinstance(value, Path):
        return str(value)
    if value is None:
        return "null"
    return value  # str/int/float/bool


def flatten_mlflow_params(payload: dict, prefix: str = "") -> dict[str, str | int | float | bool]:
    flattened: dict[str, str | int | float | bool] = {}
    for key, value in payload.items():
        flat_key = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, dict):
            flattened.update(flatten_mlflow_params(value, prefix=flat_key))
            continue
        if _is_mlflow_scalar(value) or isinstance(value, Path):
            flattened[flat_key] = _normalize_mlflow_param_value(value)
    return flattened


def build_resolved_mlflow_params(
    args: argparse.Namespace,
    resolved_paths: dict[str, str],
    data_yaml: Path,
    device: str,
) -> dict[str, str | int | float | bool]:
    args_payload = {
        key: (str(value) if isinstance(value, Path) else value)
        for key, value in vars(args).items()
    }
    payload = {
        "args": args_payload,
        "paths": resolved_paths,
        "runtime": {
            "data_yaml": to_project_relative(data_yaml),
            "device_resolved": device,
        },
    }
    return flatten_mlflow_params(payload)


def save_resolved_train_config(
    train_save_dir: Path,
    args: argparse.Namespace,
    config: dict,
    paths: dict[str, str],
    data_yaml: Path,
    data_lineage: dict,
    device: str,
) -> Path:
    resolved_payload = {
        "args": {
            key: (str(value) if isinstance(value, Path) else value)
            for key, value in vars(args).items()
        },
        "paths": paths,
        "data_yaml": to_project_relative(data_yaml),
        "device_resolved": device,
        "data_lineage": data_lineage,
        "source_config": str(args.config) if args.config else None,
        "source_config_payload": config,
    }
    output_path = train_save_dir / "resolved_config.yaml"
    output_path.write_text(
        yaml.safe_dump(resolved_payload, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    return output_path


def build_mlflow_context(args: argparse.Namespace) -> tuple[object | None, bool, str | None]:
    tracking_uri = args.mlflow_tracking_uri or os.getenv("MLFLOW_TRACKING_URI")
    if not args.enable_mlflow and not tracking_uri:
        return None, False, tracking_uri

    try:
        import mlflow
    except ImportError as exc:
        raise RuntimeError(
            "MLflow logging is enabled but `mlflow` is not installed. "
            "Install it from requirements-experiment.txt or run with `--disable-mlflow`."
        ) from exc

    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(args.mlflow_experiment)
    return mlflow, True, tracking_uri


def safe_log_artifact(mlflow_module: object | None, path: Path, artifact_path: str | None = None) -> None:
    if mlflow_module is None or not path.exists():
        return
    if artifact_path:
        mlflow_module.log_artifact(str(path), artifact_path=artifact_path)
        return
    mlflow_module.log_artifact(str(path))


def prepare_mlflow_artifact_alias(source_path: Path, alias_path: Path) -> Path:
    alias_path.parent.mkdir(parents=True, exist_ok=True)
    alias_path.write_bytes(source_path.read_bytes())
    return alias_path


def safe_get_registry_alias_version(client: object, model_name: str, alias_name: str) -> str | None:
    try:
        model_version = client.get_model_version_by_alias(model_name, alias_name)
    except Exception:
        return None
    version = getattr(model_version, "version", None)
    if version is None:
        return None
    return str(version)


def register_and_update_model_aliases(
    mlflow_module: object,
    run_id: str,
    best_ckpt_path: str,
    registered_model_name: str,
    staging_alias: str,
    production_alias: str,
    set_staging_alias: bool,
    set_production_alias: bool,
) -> dict[str, str | bool | None]:
    if not registered_model_name:
        raise ValueError("registered model name is empty")

    model_artifact_name = Path(best_ckpt_path).name
    model_uri = f"runs:/{run_id}/model/{model_artifact_name}"
    registration = mlflow_module.register_model(model_uri=model_uri, name=registered_model_name)
    registered_version = str(getattr(registration, "version", ""))
    if not registered_version:
        raise RuntimeError("registered model version is missing")

    from mlflow.tracking import MlflowClient

    client = MlflowClient()
    previous_production_version = None
    if set_production_alias:
        previous_production_version = safe_get_registry_alias_version(
            client=client,
            model_name=registered_model_name,
            alias_name=production_alias,
        )

    if set_staging_alias:
        client.set_registered_model_alias(registered_model_name, staging_alias, registered_version)

    if set_production_alias:
        client.set_registered_model_alias(registered_model_name, production_alias, registered_version)
        if previous_production_version and previous_production_version != registered_version:
            client.set_model_version_tag(registered_model_name, previous_production_version, "status", "archived")
            client.set_model_version_tag(
                registered_model_name,
                previous_production_version,
                "archived_at_utc",
                datetime.datetime.now(datetime.timezone.utc).isoformat(),
            )
        for key in ("status", "archived_at_utc"):
            try:
                client.delete_model_version_tag(registered_model_name, registered_version, key)
            except Exception:
                pass

    return {
        "model_name": registered_model_name,
        "model_version": registered_version,
        "staging_alias": staging_alias if set_staging_alias else None,
        "production_alias": production_alias if set_production_alias else None,
        "previous_production_version": previous_production_version,
        "set_staging_alias": set_staging_alias,
        "set_production_alias": set_production_alias,
    }


def disable_ultralytics_mlflow_callbacks(model: YOLO) -> None:
    """Remove Ultralytics built-in MLflow callbacks so only project logging remains."""
    callbacks = getattr(model, "callbacks", None)
    if not isinstance(callbacks, dict):
        return

    removed = 0
    for event, callback_list in callbacks.items():
        if not isinstance(callback_list, list):
            continue
        filtered = []
        for callback in callback_list:
            module_name = getattr(callback, "__module__", "")
            if module_name.endswith("ultralytics.utils.callbacks.mlflow"):
                removed += 1
                continue
            filtered.append(callback)
        callbacks[event] = filtered

    if removed:
        print(f"Ultralytics MLflow callbacks disabled: {removed}")


def disable_ultralytics_mlflow_integration_in_process() -> None:
    """Disable Ultralytics MLflow integration for this process only (no settings.json write)."""
    try:
        from ultralytics.utils import SETTINGS as ULTRALYTICS_SETTINGS
    except Exception:
        return

    # SETTINGS is persisted; bypass its overridden __setitem__ to avoid writing settings.json.
    try:
        dict.__setitem__(ULTRALYTICS_SETTINGS, "mlflow", False)
    except Exception:
        return


def save_auto_inference_config(
    train_name: str,
    best_ckpt_path: str | None,
    data_yaml: Path,
    imgsz: int,
) -> Path:
    """
    학습 종료 후 기준 추론 설정 YAML을 자동 생성/갱신한다.
    파일명 규칙: expXX_train_* -> expXX_inference_*.yaml
    """
    DEFAULT_INFER_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    inference_stem = ensure_inference_token(train_name)
    infer_config_path = DEFAULT_INFER_CONFIG_DIR / f"{inference_stem}.yaml"

    infer_yaml = build_inference_config_payload(
        train_name=train_name,
        best_ckpt_path=best_ckpt_path,
        data_yaml=data_yaml,
        imgsz=imgsz,
    )
    with infer_config_path.open("w", encoding="utf-8") as f:
        yaml.dump(infer_yaml, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
    return infer_config_path


def build_inference_config_payload(
    train_name: str,
    best_ckpt_path: str | None,
    data_yaml: Path,
    imgsz: int,
) -> dict:
    inference_stem = ensure_inference_token(train_name)
    model_path = best_ckpt_path or str(RUNS_DIR / train_name / "weights" / "best.pt")
    return {
        "paths": default_paths_config(),
        "model": to_project_relative(model_path),
        "imgsz": int(imgsz),
        "conf": 0.25,
        "iou": 0.70,
        "output": f"submission/{infer_default_submission_name(inference_stem)}.csv",
        "data": to_project_relative(data_yaml),
        "save_config": True,
    }


def read_training_time_stats(run_dir: Path) -> dict:
    """
    Ultralytics results.csv의 마지막 누적 time 값을 읽어 학습 시간 통계를 반환한다.
    """
    results_csv_path = run_dir / "results.csv"
    stats = {
        "train_results_csv": to_project_relative(results_csv_path),
        "train_completed_epochs": 0,
        "train_time_seconds": None,
        "train_time_minutes": None,
        "avg_epoch_time_seconds": None,
    }
    if not results_csv_path.exists():
        return stats

    try:
        with results_csv_path.open("r", encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))
    except Exception:
        return stats

    if not rows:
        return stats

    last_row = rows[-1]
    stats["train_completed_epochs"] = len(rows)

    raw_time = (last_row.get("time") or "").strip()
    if not raw_time:
        return stats

    try:
        elapsed_seconds = float(raw_time)
    except ValueError:
        return stats

    stats["train_time_seconds"] = elapsed_seconds
    stats["train_time_minutes"] = elapsed_seconds / 60.0
    if stats["train_completed_epochs"] > 0:
        stats["avg_epoch_time_seconds"] = elapsed_seconds / float(stats["train_completed_epochs"])

    return stats


def main() -> None:
    args = parse_args()
    training_config = load_config(args.config) if args.config else {"paths": default_paths_config()}
    resolved_paths = training_config.get("paths") or default_paths_config()
    log_session = start_run_logging(
        project_root=PROJECT_ROOT,
        category="train",
        run_name=args.name,
        enabled=args.save_log,
    )
    device = args.device or get_device()
    print(f"Using device: {device}")

    if args.config:
        print(f"Config: {args.config}")

    print(f"Project root: {PROJECT_ROOT}")
    print(f"Runs dir: {RUNS_DIR}")

    data_yaml = args.data or find_default_dataset_yaml()
    if not data_yaml.exists():
        raise FileNotFoundError(f"dataset.yaml not found: {data_yaml}")

    processed_root = resolved_paths.get("processed_root", default_paths_config()["processed_root"])
    data_lineage = load_data_lineage(
        project_root=PROJECT_ROOT,
        processed_root=processed_root,
        data_yaml=data_yaml,
    )
    if data_lineage.get("manifest_path"):
        print(f"Data manifest: {data_lineage['manifest_path']}")
        print(f"Data version : {data_lineage.get('data_version')}")
    else:
        print("Data manifest: not found (training continues without data_version linkage)")

    mlflow_module, mlflow_enabled, tracking_uri = build_mlflow_context(args)
    mlflow_run = None
    mlflow_run_id = None
    registry_result: dict[str, str | bool | None] | None = None
    mlflow_params = build_resolved_mlflow_params(
        args=args,
        resolved_paths=resolved_paths,
        data_yaml=data_yaml,
        device=device,
    )
    if mlflow_enabled:
        # Ultralytics adds its own MLflow autolog callbacks by default; disable to avoid a second experiment/run.
        disable_ultralytics_mlflow_integration_in_process()
        mlflow_run = mlflow_module.start_run(run_name=args.name)
        mlflow_run_id = mlflow_run.info.run_id

        # MLFLOW_USER 환경변수 또는 시스템 유저명 사용 (getpass가 더 견고함)
        user_id = os.getenv("MLFLOW_USER") or getpass.getuser()

        mlflow_module.log_params(mlflow_params)
        mlflow_module.set_tags(
            {
                "mlflow.user": user_id,
                "project": "pill_detection_v2",
                "run_type": "train",
                "device": device,
                "tracking_uri": tracking_uri or "local_default",
                "dataset_family": str(data_lineage.get("dataset_family") or "unknown"),
                "variant_id": str(data_lineage.get("variant_id") or "unknown"),
                "data_version": str(data_lineage.get("data_version") or "unknown"),
                "dataset_hash": str(data_lineage.get("dataset_hash") or "unknown"),
            }
        )
        print(f"MLflow run id: {mlflow_run_id}")
        print(f"MLflow params : {len(mlflow_params)} keys")

    # 👉 YOLO 모델 로드
    model = YOLO(args.model)
    if mlflow_enabled:
        disable_ultralytics_mlflow_callbacks(model)

    # 👉 학습
    try:
        results = model.train(
            data=str(data_yaml),
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            device=device,
            workers=args.workers,
            patience=args.patience,
            amp=args.amp,

            # ⭐ 핵심: 절대경로 사용
            project=str(RUNS_DIR),

            # 실험 이름
            name=args.name,

            save=True,
            plots=True,
            pretrained=args.pretrained,
            resume=args.resume,
            verbose=True,
            optimizer=args.optimizer,
            close_mosaic=args.close_mosaic,
            box=args.box,
            cls=args.cls,
            dfl=args.dfl,


            # # === 실험 1: Baseline 증강 유지 ===
            # fliplr=0.5,             # YOLOv8 기본값
            # flipud=0.0,
            # degrees=0.0,
            # mosaic=1.0,

            # === 증강 설정 (명령어 인자 기반 제어) ===
            hsv_h=args.hsv_h,
            hsv_s=args.hsv_s,
            hsv_v=args.hsv_v,
            translate=args.translate,
            scale=args.scale,
            shear=args.shear,
            perspective=args.perspective,
            erasing=args.erasing,
            auto_augment=args.auto_augment,
            fliplr=args.fliplr,
            flipud=args.flipud,
            degrees=args.degrees,
            mosaic=args.mosaic,
            copy_paste=args.copy_paste,
            mixup=args.mixup,
            
            # === 최적화 설정 (기본값 위주) ===
            lr0=args.lr0,
            lrf=args.lrf,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
            warmup_epochs=args.warmup_epochs,
            warmup_momentum=args.warmup_momentum,
            warmup_bias_lr=args.warmup_bias_lr,
            cos_lr=args.cos_lr,
            seed=args.seed,
            deterministic=args.deterministic,
        )
    except Exception:
        if mlflow_enabled and mlflow_module is not None and mlflow_module.active_run() is not None:
            mlflow_module.set_tag("run_status", "FAILED")
            mlflow_module.set_tag("failed_stage", "train")
            mlflow_module.end_run(status="FAILED")
        raise

    # Capture the actual optimizer selected by Ultralytics, especially when optimizer='auto'.
    resolved_optimizer = "unknown"
    resolved_lr = None
    resolved_weight_decay = None
    trainer = getattr(model, "trainer", None)
    train_save_dir = Path(getattr(trainer, "save_dir", RUNS_DIR / args.name))
    actual_run_name = train_save_dir.name
    best_ckpt_path = str(train_save_dir / "weights" / "best.pt")
    if trainer is not None:
        opt = getattr(trainer, "optimizer", None)
        if opt is not None:
            resolved_optimizer = opt.__class__.__name__
            param_groups = getattr(opt, "param_groups", None)
            if param_groups:
                first_group = param_groups[0]
                resolved_lr = first_group.get("lr")
                resolved_weight_decay = first_group.get("weight_decay")
    train_time_stats = read_training_time_stats(train_save_dir)
    resolved_config_path = save_resolved_train_config(
        train_save_dir=train_save_dir,
        args=args,
        config=training_config,
        paths=resolved_paths,
        data_yaml=data_yaml,
        data_lineage=data_lineage,
        device=device,
    )

    print("Training finished. Evaluating best model metrics...")
    
    # [Final Validation for detailed metrics]
    # This evaluates the 'best.pt' model found in the results directory.
    val_results = model.val(
        data=str(data_yaml),
        split='val',
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        plots=True
    )
    
    # Extract metrics
    map50 = val_results.results_dict.get('metrics/mAP50(B)', 0.0)
    map75 = val_results.maps[5] if hasattr(val_results, 'maps') and len(val_results.maps) > 5 else 0.0
    map50_95 = val_results.results_dict.get('metrics/mAP50-95(B)', 0.0)
    
    precision = val_results.results_dict.get('metrics/precision(B)', 0.0)
    recall = val_results.results_dict.get('metrics/recall(B)', 0.0)
    runtime_env = collect_runtime_env(args.device)

    # Model footprint metadata (for deployment/reporting)
    model_size_mb = None
    best_ckpt_file = Path(best_ckpt_path)
    if best_ckpt_file.exists():
        model_size_mb = best_ckpt_file.stat().st_size / (1024.0 * 1024.0)

    total_params = None
    trainable_params = None
    # Ultralytics may swap to EMA/eval weights for validation where requires_grad is False.
    # Prefer the trainer's training model when available to avoid reporting trainable=0.
    params_source = None
    if trainer is not None:
        params_source = getattr(trainer, "model", None)
    if params_source is None:
        params_source = getattr(model, "model", None)
    if params_source is not None:
        try:
            total_params = int(sum(p.numel() for p in params_source.parameters()))
            trainable_params = int(sum(p.numel() for p in params_source.parameters() if p.requires_grad))
        except Exception:
            total_params = None
            trainable_params = None

    # Per-class AP50-95 from validation (optional; useful for class-wise error analysis)
    per_class_ap50_95 = {}
    val_maps = getattr(val_results, "maps", None)
    val_names = getattr(val_results, "names", None)
    if val_maps is not None and len(val_maps) > 0:
        idx_to_name = {}
        if isinstance(val_names, dict):
            for k, v in val_names.items():
                try:
                    idx_to_name[int(k)] = str(v)
                except Exception:
                    continue
        elif isinstance(val_names, (list, tuple)):
            idx_to_name = {i: str(v) for i, v in enumerate(val_names)}
        for i, ap in enumerate(val_maps):
            cls_name = idx_to_name.get(i, f"class_{i}")
            per_class_ap50_95[cls_name] = float(ap)
    
    f1_score = 0.0
    if precision + recall > 0:
        f1_score = 2 * (precision * recall) / (precision + recall)

    metrics_payload = {
        "experiment": args.name,
        "timestamp": datetime.datetime.now().isoformat(),
        "mlflow_enabled": mlflow_enabled,
        "mlflow_experiment": args.mlflow_experiment,
        "mlflow_tracking_uri": tracking_uri,
        "mlflow_param_count": len(mlflow_params),
        "optimizer_requested": str(args.optimizer),
        "optimizer_resolved": resolved_optimizer,
        "optimizer_lr": float(resolved_lr) if resolved_lr is not None else None,
        "optimizer_weight_decay": float(resolved_weight_decay) if resolved_weight_decay is not None else None,
        "loss_weight_box": float(args.box),
        "loss_weight_cls": float(args.cls),
        "loss_weight_dfl": float(args.dfl),
        "os_platform": runtime_env.get("os_platform"),
        "python_version": runtime_env.get("python_version"),
        "torch_version": runtime_env.get("torch_version"),
        "cuda_version": runtime_env.get("cuda_version"),
        "gpu_name": runtime_env.get("gpu_name"),
        "gpu_driver": runtime_env.get("gpu_driver"),
        "seed": args.seed,
        "deterministic": args.deterministic,
        "dataset_split": "val",
        "data": str(data_yaml),
        "manifest_path": data_lineage.get("manifest_path"),
        "data_version": data_lineage.get("data_version"),
        "dataset_hash": data_lineage.get("dataset_hash"),
        "annotation_source_dir": data_lineage.get("annotation_source_dir"),
        "model_path": args.model,
        "model_name": infer_model_name(args.model),
        "source_experiment": infer_source_experiment(args.model),
        "epoch": int(args.epochs),
        "imgsz": int(args.imgsz),
        "batch": int(args.batch),
        "model_size_mb": float(model_size_mb) if model_size_mb is not None else None,
        "model_params_total": total_params,
        "model_params_trainable": trainable_params,
        "best_epoch": int(getattr(val_results, "epoch", args.epochs)),
        "train_results_csv": train_time_stats["train_results_csv"],
        "train_completed_epochs": int(train_time_stats["train_completed_epochs"]),
        "train_time_seconds": float(train_time_stats["train_time_seconds"]) if train_time_stats["train_time_seconds"] is not None else None,
        "train_time_minutes": float(train_time_stats["train_time_minutes"]) if train_time_stats["train_time_minutes"] is not None else None,
        "avg_epoch_time_seconds": float(train_time_stats["avg_epoch_time_seconds"]) if train_time_stats["avg_epoch_time_seconds"] is not None else None,
        "Precision": float(precision),
        "Recall": float(recall),
        "F1-Score": float(f1_score),
        "mAP50": float(map50),
        "mAP75": float(map75),
        "mAP50-95": float(map50_95),
        "per_class_ap50_95": per_class_ap50_95,
    }
    if mlflow_run_id is not None:
        metrics_payload["mlflow_run_id"] = mlflow_run_id
    val_metrics_dir = METRICS_DIR / "val"
    val_metrics_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = val_metrics_dir / f"{args.name}_val_metrics.json"
    infer_config_path = save_auto_inference_config(
        train_name=actual_run_name,
        best_ckpt_path=best_ckpt_path,
        data_yaml=data_yaml,
        imgsz=args.imgsz,
    )
    infer_config_payload = build_inference_config_payload(
        train_name=actual_run_name,
        best_ckpt_path=best_ckpt_path,
        data_yaml=data_yaml,
        imgsz=args.imgsz,
    )
    metrics_payload["resolved_config_path"] = to_project_relative(resolved_config_path)
    metrics_payload["infer_config_path"] = to_project_relative(infer_config_path)
    metrics_path.write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")
    if mlflow_enabled:
        mlflow_artifact_dir = train_save_dir / ".mlflow_artifacts"
        resolved_train_artifact = prepare_mlflow_artifact_alias(
            resolved_config_path,
            mlflow_artifact_dir / "train_config.yaml",
        )
        resolved_inference_artifact = prepare_mlflow_artifact_alias(
            infer_config_path,
            mlflow_artifact_dir / "inference_config.yaml",
        )
        metrics_artifact = prepare_mlflow_artifact_alias(
            metrics_path,
            mlflow_artifact_dir / "metrics.json",
        )
        mlflow_module.log_metrics(
            {
                "mAP50": float(map50),
                "F1": float(f1_score),
                "mAP75": float(map75),
                "mAP50_95": float(map50_95),
                "precision": float(precision),
                "recall": float(recall),
            }
        )
        mlflow_module.log_params(flatten_mlflow_params({"infer": infer_config_payload}))
        safe_log_artifact(mlflow_module, Path(best_ckpt_path), artifact_path="model")
        safe_log_artifact(mlflow_module, resolved_train_artifact, artifact_path="config")
        safe_log_artifact(mlflow_module, metrics_artifact, artifact_path="metrics")
        safe_log_artifact(mlflow_module, resolved_inference_artifact, artifact_path="config")
        
        # YOLO 시각화 결과물(results.png, confusion_matrix 등) 전수 업로드
        if results is not None and hasattr(results, 'save_dir') and os.path.exists(results.save_dir):
            mlflow_module.log_artifacts(results.save_dir, artifact_path="plots")

        if mlflow_run_id is not None:
            try:
                registry_result = register_and_update_model_aliases(
                    mlflow_module=mlflow_module,
                    run_id=mlflow_run_id,
                    best_ckpt_path=best_ckpt_path,
                    registered_model_name=args.mlflow_registered_model_name,
                    staging_alias=args.mlflow_staging_alias,
                    production_alias=args.mlflow_production_alias,
                    set_staging_alias=args.mlflow_set_staging_alias,
                    set_production_alias=args.mlflow_set_production_alias,
                )
                mlflow_module.set_tags(
                    {
                        "registry_update_status": "SUCCESS",
                        "registered_model_name": str(registry_result["model_name"]),
                        "registered_model_version": str(registry_result["model_version"]),
                        "registry_staging_alias": str(registry_result["staging_alias"] or "none"),
                        "registry_production_alias": str(registry_result["production_alias"] or "none"),
                    }
                )
                metrics_payload["registered_model_name"] = registry_result["model_name"]
                metrics_payload["registered_model_version"] = registry_result["model_version"]
                metrics_payload["registry_staging_alias"] = registry_result["staging_alias"]
                metrics_payload["registry_production_alias"] = registry_result["production_alias"]
                metrics_path.write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")
            except Exception as exc:
                error_text = str(exc)
                mlflow_module.set_tag("registry_update_status", "FAILED")
                mlflow_module.set_tag("registry_update_error", error_text[:250])
                if args.mlflow_registry_strict:
                    mlflow_module.end_run(status="FAILED")
                    raise
                print(f"[WARN] MLflow registry update skipped: {exc}")

        mlflow_module.set_tag("best_ckpt_path", to_project_relative(best_ckpt_path))
        mlflow_module.set_tag("resolved_config_path", to_project_relative(resolved_config_path))
        mlflow_module.set_tag("metrics_path", to_project_relative(metrics_path))
        mlflow_module.set_tag("infer_config_path", to_project_relative(infer_config_path))
        mlflow_module.end_run(status="FINISHED")

    print("\n" + "=" * 60)
    print(f"      EXPERIMENT REPORT: {args.name}")
    print("=" * 60)
    print(f" ➡️  Optimizer(req): {args.optimizer}")
    print(f" ➡️  Optimizer(use): {resolved_optimizer}")
    print(f" ➡️  Precision: {precision:.4f}")
    print(f" ➡️  Recall:    {recall:.4f}")
    print(f" ➡️  F1-Score:  {f1_score:.4f}")
    print(f" ➡️  mAP@50:    {map50:.4f}")
    print(f" ➡️  mAP@75:    {map75:.4f}")
    print(f" ➡️  mAP@50-95: {map50_95:.4f}")
    if model_size_mb is not None:
        print(f" ➡️  model size: {model_size_mb:.2f} MB")
    if total_params is not None:
        print(f" ➡️  params:     {total_params:,} (trainable {trainable_params:,})")
    if train_time_stats["train_time_seconds"] is not None:
        print(
            " ➡️  train time:"
            f" {train_time_stats['train_time_seconds']:.1f}s"
            f" ({train_time_stats['train_time_minutes']:.2f}m)"
            f" / avg {train_time_stats['avg_epoch_time_seconds']:.2f}s/epoch"
        )
    print(f" ➡️  saved:     {metrics_path}")
    print(f" ➡️  config:    {resolved_config_path}")
    if mlflow_run_id is not None:
        print(f" ➡️  mlflow id: {mlflow_run_id}")
    if registry_result is not None:
        print(
            " ➡️  registry: "
            f"{registry_result['model_name']} v{registry_result['model_version']}"
        )
        print(
            " ➡️  aliases:  "
            f"staging={registry_result['staging_alias'] or 'none'}, "
            f"production={registry_result['production_alias'] or 'none'}"
        )
    print(f" ➡️  run dir:   {train_save_dir}")
    print(f" ➡️  infer cfg: {infer_config_path}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
