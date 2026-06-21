from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml
from sklearn.model_selection import train_test_split

from src.utils.config_paths import apply_train_config_paths, merge_runtime_overrides
from src.utils.project_paths import PROJECT_ROOT

RARE_CLASS_THRESHOLD = 5  # 객체 개수가 5개 이하인 클래스를 rare class로 간주


# =========================
# Utils
# =========================
def read_json(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, data: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def as_project_path(path_str: str) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def compute_annotation_dataset_hash(annotation_files: list[Path], annotation_root: Path) -> str:
    md5 = hashlib.md5()
    for annotation_file in sorted(annotation_files):
        relative_path = annotation_file.relative_to(annotation_root).as_posix()
        file_md5 = hashlib.md5(annotation_file.read_bytes()).hexdigest()
        md5.update(relative_path.encode("utf-8"))
        md5.update(b"\n")
        md5.update(file_md5.encode("ascii"))
        md5.update(b"\n")
    return md5.hexdigest()


def compute_file_collection_hash(files: list[Path], root: Path) -> str:
    md5 = hashlib.md5()
    for file_path in sorted(files):
        relative_path = file_path.relative_to(root).as_posix()
        file_md5 = hashlib.md5(file_path.read_bytes()).hexdigest()
        md5.update(relative_path.encode("utf-8"))
        md5.update(b"\n")
        md5.update(str(file_path.stat().st_size).encode("ascii"))
        md5.update(b"\n")
        md5.update(file_md5.encode("ascii"))
        md5.update(b"\n")
    return md5.hexdigest()


def build_data_version(annotation_files: list[Path], annotation_root: Path, variant_id: str) -> tuple[str, str]:
    dataset_hash = compute_annotation_dataset_hash(annotation_files, annotation_root)
    data_version = f"{dataset_hash[:8]}_{variant_id}"
    return dataset_hash, data_version


def build_raw_image_version(image_files: list[Path], image_root: Path) -> tuple[str, str]:
    raw_image_hash = compute_file_collection_hash(image_files, image_root)
    raw_image_version = f"raw_{raw_image_hash[:8]}"
    return raw_image_hash, raw_image_version


def get_image_size_from_annotation(image_info: dict[str, Any]) -> tuple[int | None, int | None]:
    width = image_info.get("width")
    height = image_info.get("height")
    return width, height


def validate_bbox_xywh(
    bbox: list[float] | tuple[float, float, float, float],
    width: int | None,
    height: int | None,
) -> tuple[bool, str]:
    """
    bbox format: [x, y, w, h]
    """
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return False, "bbox_format_error"

    x, y, w, h = bbox

    if w <= 0 or h <= 0:
        return False, "non_positive_size"

    if x < 0 or y < 0:
        return False, "negative_coordinate"

    if width is not None and x + w > width:
        return False, "x_out_of_bounds"

    if height is not None and y + h > height:
        return False, "y_out_of_bounds"

    return True, "ok"


def bbox_area_ratio(
    bbox: list[float],
    width: int | None,
    height: int | None,
) -> float | None:
    if width is None or height is None or width <= 0 or height <= 0:
        return None

    _, _, w, h = bbox
    return float((w * h) / (width * height))


def load_config(config_path: Path) -> dict[str, Any]:
    if not config_path.exists():
        raise FileNotFoundError(f"config not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise TypeError("config must be a dict")
    return apply_train_config_paths(data)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preprocessing with config paths")
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "train" / "exp15_train_baseline_yolo11s_2.0.yaml",
        help="train config yaml path (defaults to exp15 baseline)",
    )
    parser.add_argument("--data-seed", type=int, default=None)
    parser.add_argument("--val-ratio", type=float, default=None)
    parser.add_argument("--copy-paste-target-count", type=int, default=None)
    parser.add_argument("--use-stratified", type=int, choices=[0, 1], default=None)
    parser.add_argument("--use-copy-paste", type=int, choices=[0, 1], default=None)
    return parser.parse_args()


def resolve_runtime_settings(args: argparse.Namespace, config: dict[str, Any]) -> dict[str, Any]:
    paths = config.get("paths", {})
    if not isinstance(paths, dict):
        raise TypeError("config.paths must be a dict")

    overrides = merge_runtime_overrides(
        config=config,
        cli=vars(args),
        schema={
            "data_seed": {"aliases": ("data_seed", "seed"), "default": 42, "cast": int},
            "val_ratio": {"aliases": ("val_ratio",), "default": 0.2, "cast": float},
            "copy_paste_target_count": {
                "aliases": ("copy_paste_target_count",),
                "default": 20,
                "cast": int,
            },
            "use_stratified": {"aliases": ("use_stratified",), "default": True, "cast": bool},
            "use_copy_paste": {"aliases": ("use_copy_paste",), "default": True, "cast": bool},
        },
    )
    data_seed = overrides["data_seed"]
    val_ratio = overrides["val_ratio"]
    copy_paste_target_count = overrides["copy_paste_target_count"]
    use_stratified = overrides["use_stratified"]
    use_copy_paste = overrides["use_copy_paste"]

    processed_dir = as_project_path(str(paths["processed_root"]))
    train_img_dir = as_project_path(str(paths["train_images_dir"]))
    train_ann_dir = as_project_path(str(paths["annotation_source_dir"]))
    copy_paste_variant_id = f"cp_t{copy_paste_target_count}_data_seed{data_seed}"

    processed_reports_dir = processed_dir / "reports"
    processed_splits_dir = processed_dir / "splits"
    processed_merged_dir = processed_dir / "merged"
    processed_offline_dir = processed_dir / "offline"
    augmented_img_dir = processed_offline_dir / "copy_paste" / copy_paste_variant_id

    for d in [
        processed_reports_dir,
        processed_splits_dir,
        processed_merged_dir,
        processed_offline_dir,
        augmented_img_dir,
    ]:
        d.mkdir(parents=True, exist_ok=True)

    return {
        "data_seed": data_seed,
        "val_ratio": val_ratio,
        "copy_paste_target_count": copy_paste_target_count,
        "use_stratified": use_stratified,
        "use_copy_paste": use_copy_paste,
        "copy_paste_variant_id": copy_paste_variant_id,
        "train_img_dir": train_img_dir,
        "train_ann_dir": train_ann_dir,
        "processed_dir": processed_dir,
        "augmented_img_dir": augmented_img_dir,
        "merged_annotations_path": processed_merged_dir / "merged_annotations.json",
        "label_map_path": processed_reports_dir / "label_map.json",
        "data_report_path": processed_reports_dir / "data_report.json",
        "image_meta_path": processed_reports_dir / "image_meta.json",
        "train_split_path": processed_splits_dir / "train_split.json",
        "val_split_path": processed_splits_dir / "val_split.json",
        "rare_classes_path": processed_reports_dir / "rare_classes.json",
        "copy_paste_manifest_path": augmented_img_dir / "manifest.json",
    }


# =========================
# Step 1. annotation 파일 목록 읽기
# =========================
def load_raw_annotation_files(ann_dir: Path) -> list[Path]:
    return sorted(ann_dir.glob("**/*.json"))


def load_raw_image_files(image_dir: Path) -> list[Path]:
    return sorted(path for path in image_dir.glob("**/*") if path.is_file())


# =========================
# Step 2. 객체 단위 json → 이미지 단위로 병합
# =========================
def merge_annotations_by_image(ann_files: list[Path]) -> dict[str, dict[str, Any]]:
    """
    결과 구조 예시:
    {
      "image1.png": {
        "image_name": "image1.png",
        "width": 976,
        "height": 1280,
        "objects": [
          {
            "bbox": [x, y, w, h],
            "label": "pill_x",
            "category_id": 12,
            "source_json": "xxx.json"
          }
        ]
      }
    }
    """
    merged: dict[str, dict[str, Any]] = {}
    inconsistent_category_names: list[dict[str, Any]] = []

    for ann_file in ann_files:
        ann = read_json(ann_file)

        images = ann.get("images", [])
        categories = ann.get("categories", [])
        annotations = ann.get("annotations", [])

        if not images:
            continue

        image_info = images[0]
        image_name = image_info.get("file_name")
        width, height = get_image_size_from_annotation(image_info)

        if image_name is None:
            continue

        category_map = {cat["id"]: cat["name"] for cat in categories}

        if image_name not in merged:
            merged[image_name] = {
                "image_name": image_name,
                "width": width,
                "height": height,
                "objects": [],
            }
        else:
            prev_w = merged[image_name]["width"]
            prev_h = merged[image_name]["height"]

            if prev_w is None and width is not None:
                merged[image_name]["width"] = width
            if prev_h is None and height is not None:
                merged[image_name]["height"] = height

        for obj in annotations:
            category_id = obj.get("category_id")
            bbox = obj.get("bbox")

            if category_id not in category_map:
                inconsistent_category_names.append(
                    {
                        "image_name": image_name,
                        "source_json": ann_file.name,
                        "category_id": category_id,
                        "reason": "category_id_not_found_in_categories",
                    }
                )
                label = "UNKNOWN"
            else:
                label = category_map[category_id]

            merged[image_name]["objects"].append(
                {
                    "bbox": bbox,
                    "label": label,
                    "category_id": category_id,
                    "source_json": ann_file.name,
                }
            )

    if inconsistent_category_names:
        print(f"[WARN] category mapping issues: {len(inconsistent_category_names)}")

    return merged


# =========================
# Step 3. label map 생성
# =========================
def build_label_map(merged: dict[str, dict[str, Any]]) -> dict[str, int]:
    labels = sorted({obj["label"] for item in merged.values() for obj in item["objects"]})
    return {label: idx for idx, label in enumerate(labels)}


# =========================
# Step 4. 데이터 리포트 생성
# =========================
def build_data_report(
    merged: dict[str, dict[str, Any]],
    train_img_dir: Path,
    train_ann_files: list[Path],
) -> dict[str, Any]:
    report: dict[str, Any] = {}

    train_images = sorted(train_img_dir.glob("**/*.*"))
    train_image_names = {p.name for p in train_images}
    merged_image_names = set(merged.keys())

    missing_annotation_images = sorted(list(train_image_names - merged_image_names))
    missing_image_files = sorted(list(merged_image_names - train_image_names))

    class_counter: Counter[str] = Counter()
    objects_per_image_counter: Counter[int] = Counter()
    invalid_bboxes: list[dict[str, Any]] = []
    images_with_invalid_bbox: set[str] = set()
    images_with_zero_objects: list[str] = []
    bbox_area_ratios: list[float] = []

    for image_name, item in merged.items():
        width = item.get("width")
        height = item.get("height")
        objects = item.get("objects", [])

        objects_per_image_counter[len(objects)] += 1

        if len(objects) == 0:
            images_with_zero_objects.append(image_name)

        for obj in objects:
            label = obj["label"]
            bbox = obj["bbox"]

            class_counter[label] += 1

            ok, reason = validate_bbox_xywh(bbox, width, height)
            if not ok:
                invalid_bboxes.append(
                    {
                        "image_name": image_name,
                        "label": label,
                        "bbox": bbox,
                        "reason": reason,
                        "source_json": obj.get("source_json"),
                    }
                )
                images_with_invalid_bbox.add(image_name)

            ratio = bbox_area_ratio(bbox, width, height)
            if ratio is not None:
                bbox_area_ratios.append(ratio)

    rare_classes = sorted([cls_name for cls_name, count in class_counter.items() if count <= RARE_CLASS_THRESHOLD])

    report["summary"] = {
        "num_train_images": len(train_images),
        "num_train_annotation_jsons": len(train_ann_files),
        "num_unique_images_in_merged_annotations": len(merged),
        "num_classes": len(class_counter),
        "num_missing_annotation_images": len(missing_annotation_images),
        "num_missing_image_files": len(missing_image_files),
        "num_invalid_bboxes": len(invalid_bboxes),
        "num_images_with_invalid_bbox": len(images_with_invalid_bbox),
        "num_images_with_zero_objects": len(images_with_zero_objects),
    }

    report["class_distribution"] = dict(class_counter)
    report["objects_per_image_distribution"] = dict(sorted(objects_per_image_counter.items()))
    report["rare_classes"] = rare_classes
    report["missing_annotation_images"] = missing_annotation_images
    report["missing_image_files"] = missing_image_files
    report["images_with_zero_objects"] = sorted(images_with_zero_objects)
    report["invalid_bboxes"] = invalid_bboxes[:200]

    if bbox_area_ratios:
        report["bbox_area_ratio_stats"] = {
            "count": len(bbox_area_ratios),
            "min": min(bbox_area_ratios),
            "max": max(bbox_area_ratios),
            "mean": sum(bbox_area_ratios) / len(bbox_area_ratios),
        }
    else:
        report["bbox_area_ratio_stats"] = None

    return report


# =========================
# Step 5. 이미지 단위 메타데이터 생성
# =========================
def build_image_meta(
    merged: dict[str, dict[str, Any]],
    rare_classes: set[str],
) -> list[dict[str, Any]]:
    image_meta: list[dict[str, Any]] = []

    for image_name, item in merged.items():
        objects = item.get("objects", [])
        labels = [obj["label"] for obj in objects]
        label_counter = Counter(labels)

        image_meta.append(
            {
                "image_name": image_name,
                "width": item.get("width"),
                "height": item.get("height"),
                "num_objects": len(objects),
                "labels": sorted(list(set(labels))),
                "label_counts": dict(label_counter),
                "has_rare_class": any(label in rare_classes for label in labels),
            }
        )

    image_meta = sorted(image_meta, key=lambda x: x["image_name"])
    return image_meta


# =========================
# Step 6. train/val 분할
# =========================
def split_train_val_random(
    image_names: list[str],
    val_ratio: float = 0.2,
    seed: int = 42,
) -> tuple[list[str], list[str]]:
    rng = random.Random(seed)
    image_names = list(image_names)
    rng.shuffle(image_names)

    n_total = len(image_names)
    n_val = max(1, int(n_total * val_ratio))

    val_images = sorted(image_names[:n_val])
    train_images = sorted(image_names[n_val:])
    return train_images, val_images


def split_train_val_stratified(
    image_meta: list[dict[str, Any]],
    val_ratio: float = 0.2,
    seed: int = 42,
) -> tuple[list[str], list[str]]:
    """
    [개선된 데이터 분할 로직 - 팀원 한별님 버전]
    최소 클래스 개수가 3개 이상인 환경에 최적화되어,
    모든 클래스를 비율(8:2)에 맞춰 엄격하게 분할합니다.
    """
    global_label_counts = Counter()
    for meta in image_meta:
        for label in meta["labels"]:
            global_label_counts[label] += 1

    image_primary_labels = []
    image_names = []

    for meta in image_meta:
        image_names.append(meta["image_name"])
        if not meta["labels"]:
            image_primary_labels.append("empty")
            continue
        primary_label = min(meta["labels"], key=lambda l: global_label_counts[l])
        image_primary_labels.append(primary_label)

    stratify_labels = list(image_primary_labels)
    counts = Counter(stratify_labels)

    singletons = {lbl for lbl, cnt in counts.items() if cnt < 2}
    if singletons:
        for i, lbl in enumerate(stratify_labels):
            if lbl in singletons:
                stratify_labels[i] = "__other__"
        counts = Counter(stratify_labels)

    n_val = max(1, int(len(image_names) * val_ratio))
    if len(counts) > n_val:
        labels_by_rarity = sorted(
            [lbl for lbl in counts.keys() if lbl != "__other__"],
            key=lambda l: counts[l],
        )
        for lbl in labels_by_rarity:
            if len(counts) <= n_val:
                break
            for i, cur in enumerate(stratify_labels):
                if cur == lbl:
                    stratify_labels[i] = "__other__"
            counts = Counter(stratify_labels)

    still_invalid = any(cnt < 2 for cnt in counts.values()) or (len(counts) > n_val)
    if still_invalid:
        print(
            "[WARN] Stratified split not possible with current label distribution. "
            "Falling back to random split."
        )
        return split_train_val_random(image_names, val_ratio=val_ratio, seed=seed)

    try:
        train_images, val_images = train_test_split(
            image_names,
            test_size=val_ratio,
            random_state=seed,
            stratify=stratify_labels,
        )
    except ValueError as e:
        print(f"[WARN] Stratified split failed ({e}). Falling back to random split.")
        return split_train_val_random(image_names, val_ratio=val_ratio, seed=seed)

    return sorted(train_images), sorted(val_images)


# =========================
# Step 7. Copy-Paste 증강 (Exp2)
# =========================
def apply_copy_paste_augmentation(
    train_image_names: list[str],
    merged: dict[str, dict[str, Any]],
    rare_classes: set[str],
    train_img_dir: Path,
    augmented_img_dir: Path,
    target_count: int,
) -> dict[str, dict[str, Any]]:
    """
    희귀 클래스에 대해 Copy-Paste 증강을 수행합니다.
    - 알약 크기를 랜덤하게 조절(Random Scale) 기능 포함.
    """
    print(f"[7] Starting Copy-Paste augmentation (Target: {target_count} instances per rare class)")

    def sanitize_filename(text: str) -> str:
        cleaned = re.sub(r"[^0-9A-Za-z가-힣._-]+", "_", text)
        return cleaned.strip("_") or "unknown"

    pill_bank: dict[str, list[dict[str, Any]]] = {cls: [] for cls in rare_classes}

    for img_name in train_image_names:
        img_info = merged[img_name]
        img_path = train_img_dir / img_name
        if not img_path.exists():
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            continue

        for obj in img_info["objects"]:
            if obj["label"] in rare_classes:
                x, y, w, h = map(int, obj["bbox"])
                crop = img[y : y + h, x : x + w].copy()
                pill_bank[obj["label"]].append(
                    {
                        "image": crop,
                        "label": obj["label"],
                        "category_id": obj["category_id"],
                        "original_size": (w, h),
                    }
                )

    augmented_annots: dict[str, dict[str, Any]] = {}
    bg_color = (211, 211, 211)  # 연회색 배경 (원본 데이터 특성 반영)

    for cls_name, crops in pill_bank.items():
        if not crops:
            continue

        current_count = len(crops)
        needed = target_count - current_count
        if needed <= 0:
            continue

        print(f"    - Augmenting {cls_name}: {current_count} -> {target_count}")

        for i in range(needed):
            aug_img = np.full((1280, 976, 3), bg_color, dtype=np.uint8)
            aug_objects = []

            num_pills = random.randint(2, 4)
            for p_idx in range(num_pills):
                selected_crop_info = (
                    random.choice(crops) if p_idx == 0 else random.choice(random.choice(list(pill_bank.values())))
                )

                pill_img = selected_crop_info["image"]
                scale = random.uniform(0.8, 1.2)
                new_w = int(pill_img.shape[1] * scale)
                new_h = int(pill_img.shape[0] * scale)
                pill_img = cv2.resize(pill_img, (new_w, new_h))

                center = (new_w // 2, new_h // 2)
                matrix = cv2.getRotationMatrix2D(center, random.uniform(0, 360), 1.0)
                pill_img = cv2.warpAffine(pill_img, matrix, (new_w, new_h), borderValue=bg_color)

                max_y, max_x = aug_img.shape[0] - new_h, aug_img.shape[1] - new_w
                start_x = random.randint(50, max_x - 50)
                start_y = random.randint(50, max_y - 50)

                aug_img[start_y : start_y + new_h, start_x : start_x + new_w] = pill_img

                aug_objects.append(
                    {
                        "bbox": [float(start_x), float(start_y), float(new_w), float(new_h)],
                        "label": selected_crop_info["label"],
                        "category_id": selected_crop_info["category_id"],
                        "source_json": "augmented",
                    }
                )

            safe_cls = sanitize_filename(cls_name)
            aug_name = f"aug_{safe_cls}_{i}.png"
            out_path = augmented_img_dir / aug_name
            ok = cv2.imwrite(str(out_path), aug_img)
            if not ok:
                raise RuntimeError(f"Failed to write augmented image: {out_path}")

            augmented_annots[aug_name] = {
                "image_name": aug_name,
                "width": 976,
                "height": 1280,
                "objects": aug_objects,
                "is_augmented": True,
            }

    return augmented_annots


def summarize_split(
    split_image_names: list[str],
    merged: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    class_counter: Counter[str] = Counter()
    num_objects_per_image: Counter[int] = Counter()

    for image_name in split_image_names:
        item = merged[image_name]
        objects = item["objects"]

        num_objects_per_image[len(objects)] += 1

        for obj in objects:
            class_counter[obj["label"]] += 1

    return {
        "num_images": len(split_image_names),
        "num_objects": sum(class_counter.values()),
        "num_classes_present": len(class_counter),
        "class_distribution": dict(class_counter),
        "objects_per_image_distribution": dict(sorted(num_objects_per_image.items())),
    }


# =========================
# Main
# =========================
def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    runtime = resolve_runtime_settings(args, config)

    random.seed(runtime["data_seed"])

    print("=" * 60)
    print("Start preprocessing")
    print("=" * 60)
    print(f"PROJECT_ROOT     : {PROJECT_ROOT}")
    print(f"CONFIG           : {args.config}")
    print(f"TRAIN_IMG_DIR    : {runtime['train_img_dir']}")
    print(f"TRAIN_ANN_DIR    : {runtime['train_ann_dir']}")
    print(f"PROCESSED_DIR    : {runtime['processed_dir']}")
    print(f"DATA_SEED        : {runtime['data_seed']}")
    print(f"VAL_RATIO        : {runtime['val_ratio']}")
    print(f"COPY_PASTE_DIR   : {runtime['augmented_img_dir']}")

    if not runtime["train_img_dir"].exists():
        raise FileNotFoundError(f"Train image directory not found: {runtime['train_img_dir']}")

    if not runtime["train_ann_dir"].exists():
        raise FileNotFoundError(f"Train annotation directory not found: {runtime['train_ann_dir']}")

    ann_files = load_raw_annotation_files(runtime["train_ann_dir"])
    dataset_hash, data_version = build_data_version(
        annotation_files=ann_files,
        annotation_root=runtime["train_ann_dir"],
        variant_id=runtime["copy_paste_variant_id"],
    )
    raw_train_image_files = load_raw_image_files(runtime["train_img_dir"])
    raw_train_image_hash, raw_train_image_version = build_raw_image_version(
        image_files=raw_train_image_files,
        image_root=runtime["train_img_dir"],
    )
    print(f"[1] Loaded annotation json files: {len(ann_files)}")
    print(f"    - dataset_hash             : {dataset_hash}")
    print(f"    - data_version             : {data_version}")
    print(f"    - raw_train_image_hash     : {raw_train_image_hash}")
    print(f"    - raw_train_image_version  : {raw_train_image_version}")
    print(f"    - raw_train_image_count    : {len(raw_train_image_files)}")

    merged = merge_annotations_by_image(ann_files)
    print(f"[2] Merged into image-level annotations: {len(merged)} images")

    label_map = build_label_map(merged)
    print(f"[3] Built label map: {len(label_map)} classes")

    report = build_data_report(
        merged=merged,
        train_img_dir=runtime["train_img_dir"],
        train_ann_files=ann_files,
    )

    rare_classes = set(report["rare_classes"])
    print("[4] Built report")
    print(f"    - num classes              : {report['summary']['num_classes']}")
    print(f"    - missing annotation imgs  : {report['summary']['num_missing_annotation_images']}")
    print(f"    - missing image files      : {report['summary']['num_missing_image_files']}")
    print(f"    - invalid bboxes           : {report['summary']['num_invalid_bboxes']}")
    print(f"    - rare classes             : {len(rare_classes)}")

    image_meta = build_image_meta(merged, rare_classes)
    print(f"[5] Built image metadata: {len(image_meta)} items")

    if runtime["use_stratified"]:
        train_images, val_images = split_train_val_stratified(
            image_meta=image_meta,
            val_ratio=runtime["val_ratio"],
            seed=runtime["data_seed"],
        )
        print(f"[6] Performed Stratified Split (ratio: {runtime['val_ratio']})")
    else:
        train_images, val_images = train_test_split(
            list(merged.keys()),
            test_size=runtime["val_ratio"],
            random_state=runtime["data_seed"],
        )
        train_images = sorted(train_images)
        val_images = sorted(val_images)
        print(f"[6] Performed Random Split (ratio: {runtime['val_ratio']})")

    print(f"    - train images: {len(train_images)}")
    print(f"    - val images  : {len(val_images)}")

    augmented_data: dict[str, dict[str, Any]] = {}
    if runtime["use_copy_paste"]:
        augmented_data = apply_copy_paste_augmentation(
            train_image_names=train_images,
            merged=merged,
            rare_classes=rare_classes,
            train_img_dir=runtime["train_img_dir"],
            augmented_img_dir=runtime["augmented_img_dir"],
            target_count=runtime["copy_paste_target_count"],
        )

        for aug_name, aug_info in augmented_data.items():
            merged[aug_name] = aug_info
            train_images.append(aug_name)

        print(f"[7] Added {len(augmented_data)} augmented images to train set")
    else:
        print("[7] Copy-Paste augmentation disabled")

    train_summary = summarize_split(train_images, merged)
    val_summary = summarize_split(val_images, merged)

    split_info = {
        "data_seed": runtime["data_seed"],
        "seed": runtime["data_seed"],
        "val_ratio": runtime["val_ratio"],
        "copy_paste_variant_id": runtime["copy_paste_variant_id"],
        "copy_paste_target_count": runtime["copy_paste_target_count"],
        "train_summary": train_summary,
        "val_summary": val_summary,
    }

    copy_paste_manifest = {
        "variant_id": runtime["copy_paste_variant_id"],
        "data_version": data_version,
        "dataset_hash": dataset_hash,
        "raw_train_image_hash": raw_train_image_hash,
        "raw_train_image_version": raw_train_image_version,
        "augmentation": "copy_paste",
        "data_seed": runtime["data_seed"],
        "target_count_per_rare_class": runtime["copy_paste_target_count"],
        "enabled": runtime["use_copy_paste"],
        "num_generated_images": len(augmented_data) if runtime["use_copy_paste"] else 0,
        "input_images_dir": str(runtime["train_img_dir"]),
        "input_annotations_dir": str(runtime["train_ann_dir"]),
        "input_paths": {
            "train_images_dir": str(runtime["train_img_dir"]),
            "annotation_source_dir": str(runtime["train_ann_dir"]),
        },
        "annotation_file_count": len(ann_files),
        "raw_train_image_file_count": len(raw_train_image_files),
        "raw_train_image_total_bytes": sum(path.stat().st_size for path in raw_train_image_files),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }

    merged_output = {
        "metadata": {
            "description": "Merged image-level annotations generated from object-level json files",
            "num_images": len(merged),
            "num_classes": len(label_map),
            "label_map_path": str(runtime["label_map_path"].name),
        },
        "images": merged,
    }

    write_json(runtime["merged_annotations_path"], merged_output)
    write_json(runtime["label_map_path"], label_map)
    write_json(runtime["image_meta_path"], image_meta)
    write_json(runtime["train_split_path"], train_images)
    write_json(runtime["val_split_path"], val_images)
    write_json(runtime["rare_classes_path"], sorted(list(rare_classes)))
    write_json(runtime["copy_paste_manifest_path"], copy_paste_manifest)

    report_with_split = report.copy()
    report_with_split["split_info"] = split_info
    write_json(runtime["data_report_path"], report_with_split)

    print("[8] Saved files")
    print(f"    - {runtime['merged_annotations_path']}")
    print(f"    - {runtime['label_map_path']}")
    print(f"    - {runtime['data_report_path']}")
    print(f"    - {runtime['image_meta_path']}")
    print(f"    - {runtime['train_split_path']}")
    print(f"    - {runtime['val_split_path']}")
    print(f"    - {runtime['rare_classes_path']}")
    print(f"    - {runtime['copy_paste_manifest_path']}")

    print("=" * 60)
    print("Preprocessing done.")
    print("=" * 60)


if __name__ == "__main__":
    main()
