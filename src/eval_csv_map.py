"""
예측 CSV를 로컬 validation 라벨과 비교해 mAP를 계산하고 결과를 저장하는 평가 스크립트.

YOLO 형식의 validation 정답 라벨을 기준으로 prediction CSV를 평가하며,
mAP@50, mAP@75, mAP@50-95 같은 탐지 성능 지표를 출력한다.
필요 시 metrics JSON 파일로 저장하며, experiment/model/source_experiment 같은
메타정보도 함께 기록할 수 있다.
WBF 같은 앙상블의 로컬 검증용으로 사용하며, Kaggle 제출 파일 생성용 스크립트는 아니다.
"""
import argparse
import json
import os
import re
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import yaml
from ultralytics.utils.metrics import ap_per_class, box_iou


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.utils.logging_utils import start_run_logging

DEFAULT_VAL_IMG_DIR = PROJECT_ROOT / "data" / "yolo_dataset" / "images" / "val"
DEFAULT_VAL_LABEL_DIR = PROJECT_ROOT / "data" / "yolo_dataset" / "labels" / "val"
DEFAULT_YAML_PATH = PROJECT_ROOT / "data" / "yolo_dataset" / "dataset.yaml"
DEFAULT_JSON_DIR = PROJECT_ROOT / "data" / "raw" / "sprint_ai_project1_data" / "train_annotations"
DEFAULT_METRICS_DIR = PROJECT_ROOT / "metrics"


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate prediction CSV against local YOLO val labels.")
    parser.add_argument("--pred_csv", required=True, help="Prediction CSV path")
    parser.add_argument("--image_dir", default=str(DEFAULT_VAL_IMG_DIR), help="Validation image directory")
    parser.add_argument("--label_dir", default=str(DEFAULT_VAL_LABEL_DIR), help="Validation label directory")
    parser.add_argument("--data", default=str(DEFAULT_YAML_PATH), help="Path to dataset.yaml")
    parser.add_argument("--json_dir", default=str(DEFAULT_JSON_DIR), help="Raw JSON annotation directory")
    parser.add_argument("--experiment", default=None, help="Experiment id/name (e.g. exp9)")
    parser.add_argument("--model_name", default=None, help="Model family/name (e.g. yolo11s)")
    parser.add_argument("--model_path", default=None, help="Model checkpoint path used for evaluation")
    parser.add_argument("--source_experiment", default=None, help="Optional source experiment id override")
    parser.add_argument("--dataset_split", default="val", help="Dataset split name recorded in metrics json")
    parser.add_argument("--epoch", type=int, default=None, help="Training epoch count if applicable")
    parser.add_argument("--best_epoch", type=int, default=None, help="Best epoch if applicable")
    parser.add_argument("--conf", type=float, default=None, help="Confidence threshold used for inference")
    parser.add_argument("--iou", type=float, default=None, help="IoU threshold used for inference/WBF")
    parser.add_argument("--scales", default=None, help="Optional comma-separated scales (e.g. 640,960,1024)")
    parser.add_argument("--save_json", default=None, help="Optional path to save summary json")
    parser.add_argument(
        "--save-log",
        dest="save_log",
        action="store_true",
        help="Save runtime log to logs/inference",
    )
    parser.add_argument(
        "--no-save-log",
        dest="save_log",
        action="store_false",
        help="Disable runtime log file saving",
    )
    parser.set_defaults(save_log=True)
    return parser.parse_args()


def default_metrics_path(pred_csv: str) -> Path:
    stem = Path(pred_csv).stem
    suffix = "_metrics" if stem.endswith("_val") else "_val_metrics"
    return DEFAULT_METRICS_DIR / f"{stem}{suffix}.json"


def infer_source_experiment(model_path: str | None) -> str | None:
    if not model_path:
        return None
    match = re.search(r"(?:^|[/_])exp[_-]?(\d+)(?:[^0-9]|$)", model_path, flags=re.IGNORECASE)
    if not match:
        match = re.search(r"pill_exp(\d+)", model_path, flags=re.IGNORECASE)
    return f"exp{match.group(1)}" if match else None


def build_inverse_class_map(yaml_path: str, json_dir: str) -> tuple[dict[int, int], dict[int, str]]:
    with open(yaml_path, "r", encoding="utf-8") as f:
        yolo_names = yaml.safe_load(f)["names"]

    json_category_map = {}
    original_json_names = {}

    for root, _, files in os.walk(json_dir):
        for file in files:
            if not file.endswith(".json"):
                continue
            try:
                with open(os.path.join(root, file), "r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception:
                continue

            for cat in data.get("categories", []):
                clean_name = str(cat["name"]).replace(" ", "")
                json_category_map[clean_name] = int(cat["id"])
                original_json_names[clean_name] = cat["name"]

    inverse_class_map = {}
    category_names = {}

    for yolo_id, yaml_name in yolo_names.items():
        clean_yaml = str(yaml_name).replace(" ", "")
        if clean_yaml in json_category_map:
            category_id = json_category_map[clean_yaml]
        else:
            raise KeyError(f"Could not map YOLO class '{yaml_name}' to JSON category id")
        inverse_class_map[int(yolo_id)] = category_id
        category_names[category_id] = original_json_names[clean_yaml]

    return inverse_class_map, category_names


def extract_image_id(name: str) -> str:
    image_id = "".join(re.findall(r"\d+", name))
    return normalize_image_id(image_id if image_id else "0")


def normalize_image_id(value) -> str:
    digits = re.sub(r"\D", "", str(value))
    if not digits:
        return "0"
    stripped = digits.lstrip("0")
    return stripped if stripped else "0"


def load_ground_truth(image_dir: str, label_dir: str, inverse_class_map: dict[int, int]):
    gt_by_image = {}
    target_classes = []

    image_paths = sorted(
        Path(image_dir).glob("*"),
        key=lambda p: extract_image_id(p.name),
    )

    for image_path in image_paths:
        if image_path.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
            continue

        image = cv2.imread(str(image_path))
        if image is None:
            raise FileNotFoundError(f"Failed to read image: {image_path}")
        height, width = image.shape[:2]

        image_id = extract_image_id(image_path.name)
        label_path = Path(label_dir) / f"{image_path.stem}.txt"

        boxes = []
        classes = []
        if label_path.exists():
            for line in label_path.read_text(encoding="utf-8").splitlines():
                parts = line.strip().split()
                if len(parts) != 5:
                    continue
                yolo_cls = int(parts[0])
                x_center, y_center, box_w, box_h = map(float, parts[1:])
                x1 = (x_center - box_w / 2) * width
                y1 = (y_center - box_h / 2) * height
                x2 = (x_center + box_w / 2) * width
                y2 = (y_center + box_h / 2) * height
                boxes.append([x1, y1, x2, y2])
                category_id = inverse_class_map[yolo_cls]
                classes.append(category_id)
                target_classes.append(category_id)

        gt_by_image[image_id] = {
            "boxes": torch.tensor(boxes, dtype=torch.float32),
            "classes": torch.tensor(classes, dtype=torch.int64),
        }

    return gt_by_image, np.array(target_classes, dtype=np.int64)


def match_predictions(pred_classes: torch.Tensor, true_classes: torch.Tensor, iou_matrix: torch.Tensor) -> torch.Tensor:
    iouv = torch.linspace(0.5, 0.95, 10)
    correct = np.zeros((pred_classes.shape[0], iouv.shape[0]), dtype=bool)
    if pred_classes.numel() == 0 or true_classes.numel() == 0:
        return torch.tensor(correct, dtype=torch.bool)

    correct_class = true_classes[:, None] == pred_classes
    iou = (iou_matrix * correct_class).cpu().numpy()

    for i, threshold in enumerate(iouv.tolist()):
        matches = np.nonzero(iou >= threshold)
        matches = np.array(matches).T
        if matches.shape[0]:
            if matches.shape[0] > 1:
                matches = matches[iou[matches[:, 0], matches[:, 1]].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
            correct[matches[:, 1].astype(int), i] = True

    return torch.tensor(correct, dtype=torch.bool)


def evaluate_predictions(pred_csv: str, gt_by_image: dict[str, dict], target_classes: np.ndarray):
    df = pd.read_csv(pred_csv)
    if df.empty:
        raise ValueError(f"Prediction CSV is empty: {pred_csv}")

    df["image_id"] = df["image_id"].map(normalize_image_id)

    stats = []
    pred_image_ids = set(df["image_id"].tolist())
    all_image_ids = sorted(set(gt_by_image.keys()) | pred_image_ids)

    for image_id in all_image_ids:
        gt = gt_by_image.get(
            image_id,
            {"boxes": torch.zeros((0, 4), dtype=torch.float32), "classes": torch.zeros((0,), dtype=torch.int64)},
        )
        gt_boxes = gt["boxes"]
        gt_classes = gt["classes"]

        df_img = df[df["image_id"] == image_id]
        pred_boxes = torch.tensor(
            df_img[["bbox_x", "bbox_y", "bbox_w", "bbox_h"]].assign(
                x2=df_img["bbox_x"] + df_img["bbox_w"],
                y2=df_img["bbox_y"] + df_img["bbox_h"],
            )[["bbox_x", "bbox_y", "x2", "y2"]].to_numpy(),
            dtype=torch.float32,
        )
        pred_scores = torch.tensor(df_img["score"].to_numpy(), dtype=torch.float32)
        pred_classes = torch.tensor(df_img["category_id"].astype(int).to_numpy(), dtype=torch.int64)

        if pred_boxes.numel() == 0:
            continue

        if gt_boxes.numel() == 0:
            correct = torch.zeros((pred_boxes.shape[0], 10), dtype=torch.bool)
        else:
            iou_matrix = box_iou(gt_boxes, pred_boxes)
            correct = match_predictions(pred_classes, gt_classes, iou_matrix)

        stats.append(
            (
                correct.cpu().numpy(),
                pred_scores.cpu().numpy(),
                pred_classes.cpu().numpy(),
            )
        )

    if not stats:
        raise ValueError("No predictions found to evaluate.")

    tp = np.concatenate([x[0] for x in stats], axis=0)
    conf = np.concatenate([x[1] for x in stats], axis=0)
    pred_cls = np.concatenate([x[2] for x in stats], axis=0)

    _, _, _, _, _, ap, unique_classes, _, _, _, _, _ = ap_per_class(
        tp=tp,
        conf=conf,
        pred_cls=pred_cls,
        target_cls=target_classes,
        plot=False,
        names={},
    )

    if ap.size == 0:
        raise ValueError("AP computation returned no class metrics.")

    return {
        "mAP50": float(ap[:, 0].mean()),
        "mAP75": float(ap[:, 5].mean()),
        "mAP50-95": float(ap.mean()),
        "evaluated_classes": int(len(unique_classes)),
        "num_predictions": int(len(df)),
        "num_targets": int(len(target_classes)),
    }


def main():
    args = parse_args()
    start_run_logging(
        project_root=PROJECT_ROOT,
        category="inference",
        run_name=f"{Path(args.pred_csv).stem}_eval",
        enabled=args.save_log,
    )
    inverse_class_map, _ = build_inverse_class_map(args.data, args.json_dir)
    gt_by_image, target_classes = load_ground_truth(args.image_dir, args.label_dir, inverse_class_map)
    metrics = evaluate_predictions(args.pred_csv, gt_by_image, target_classes)
    metrics.update(
        {
            "experiment": args.experiment,
            "dataset_split": args.dataset_split,
            "model_name": args.model_name,
            "model_path": args.model_path,
            "source_experiment": args.source_experiment or infer_source_experiment(args.model_path),
            "epoch": args.epoch,
            "best_epoch": args.best_epoch,
            "conf": args.conf,
            "iou": args.iou,
            "scales": [int(x.strip()) for x in args.scales.split(",")] if args.scales else None,
            "pred_csv": args.pred_csv,
        }
    )

    print("=" * 60)
    print("CSV Evaluation Report")
    print("=" * 60)
    print(f"pred_csv:      {args.pred_csv}")
    print(f"image_dir:     {args.image_dir}")
    print(f"label_dir:     {args.label_dir}")
    print(f"mAP@50:        {metrics['mAP50']:.4f}")
    print(f"mAP@75:        {metrics['mAP75']:.4f}")
    print(f"mAP@50-95:     {metrics['mAP50-95']:.4f}")
    print(f"classes:       {metrics['evaluated_classes']}")
    print(f"predictions:   {metrics['num_predictions']}")
    print(f"targets:       {metrics['num_targets']}")
    print("=" * 60)

    save_path = Path(args.save_json) if args.save_json else default_metrics_path(args.pred_csv)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    save_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"Saved summary JSON to: {save_path}")


if __name__ == "__main__":
    main()
