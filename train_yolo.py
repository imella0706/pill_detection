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
import json
import re
import sys

import torch
import yaml
from ultralytics import YOLO

from src.utils.logging_utils import start_run_logging
from src.utils.metrics_utils import collect_runtime_env


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_YAML_PRIMARY = PROJECT_ROOT / "data" / "yolo_dataset" / "dataset.yaml"
METRICS_DIR = PROJECT_ROOT / "metrics"
DEFAULT_INFER_CONFIG_DIR = PROJECT_ROOT / "configs" / "inference"
DEFAULT_TEST_IMAGES_DIR = PROJECT_ROOT / "data" / "raw" / "sprint_ai_project1_data" / "test_images"
DEFAULT_JSON_DIR = PROJECT_ROOT / "data" / "raw" / "sprint_ai_project1_data" / "train_annotations"

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

    return data


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

    return parser


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
        }
        unknown_keys = sorted(set(defaults.keys()) - allowed_keys)
        if unknown_keys:
            print(
                f"Warning: ignoring unknown config keys: {unknown_keys}",
                file=sys.stderr,
            )

    parser = build_parser(defaults)
    return parser.parse_args()


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

    model_path = best_ckpt_path or str(RUNS_DIR / train_name / "weights" / "best.pt")
    infer_yaml = {
        "model": to_project_relative(model_path),
        "imgsz": int(imgsz),
        "conf": 0.25,
        "iou": 0.70,
        "output": f"submission/{infer_default_submission_name(inference_stem)}.csv",
        "test_images": to_project_relative(DEFAULT_TEST_IMAGES_DIR),
        "data": to_project_relative(data_yaml),
        "json_dir": to_project_relative(DEFAULT_JSON_DIR),
        "save_config": True,
    }
    with infer_config_path.open("w", encoding="utf-8") as f:
        yaml.dump(infer_yaml, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
    return infer_config_path


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

    # 👉 YOLO 모델 로드
    model = YOLO(args.model)

    # 👉 학습
    model.train(
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
    model_module = getattr(model, "model", None)
    if model_module is not None:
        try:
            total_params = int(sum(p.numel() for p in model_module.parameters()))
            trainable_params = int(sum(p.numel() for p in model_module.parameters() if p.requires_grad))
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
    val_metrics_dir = METRICS_DIR / "val"
    val_metrics_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = val_metrics_dir / f"{args.name}_val_metrics.json"
    metrics_path.write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")
    infer_config_path = save_auto_inference_config(
        train_name=actual_run_name,
        best_ckpt_path=best_ckpt_path,
        data_yaml=data_yaml,
        imgsz=args.imgsz,
    )
    
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
    print(f" ➡️  run dir:   {train_save_dir}")
    print(f" ➡️  infer cfg: {infer_config_path}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
