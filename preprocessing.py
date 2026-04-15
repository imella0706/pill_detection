from __future__ import annotations

from collections import Counter
from pathlib import Path
from datetime import datetime, timezone
import json
import random
import re
import cv2
import numpy as np
from typing import Any
from sklearn.model_selection import train_test_split

DATA_SEED = 42
VAL_RATIO = 0.2
RARE_CLASS_THRESHOLD = 5  # 객체 개수가 5개 이하인 클래스를 rare class로 간주
COPY_PASTE_TARGET_COUNT = 20

random.seed(DATA_SEED)

# =========================
# Paths
# =========================
PROJECT_ROOT = Path(__file__).resolve().parent

RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw" / "sprint_ai_project1_data"
CURATED_DATA_DIR = PROJECT_ROOT / "data" / "curated" / "sprint_ai_project1_data"
TRAIN_IMG_DIR = RAW_DATA_DIR / "train_images"
TRAIN_ANN_DIR = CURATED_DATA_DIR / "train_annotations"

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

PROCESSED_REPORTS_DIR = PROCESSED_DIR / "reports"
PROCESSED_SPLITS_DIR = PROCESSED_DIR / "splits"
PROCESSED_MERGED_DIR = PROCESSED_DIR / "merged"
PROCESSED_OFFLINE_DIR = PROCESSED_DIR / "offline"
COPY_PASTE_VARIANT_ID = f"cp_t{COPY_PASTE_TARGET_COUNT}_data_seed{DATA_SEED}"
AUGMENTED_IMG_DIR = PROCESSED_OFFLINE_DIR / "copy_paste" / COPY_PASTE_VARIANT_ID

for d in [
    PROCESSED_REPORTS_DIR,
    PROCESSED_SPLITS_DIR,
    PROCESSED_MERGED_DIR,
    PROCESSED_OFFLINE_DIR,
    AUGMENTED_IMG_DIR,
]:
    d.mkdir(parents=True, exist_ok=True)

MERGED_ANNOTATIONS_PATH = PROCESSED_MERGED_DIR / "merged_annotations.json"
LABEL_MAP_PATH = PROCESSED_REPORTS_DIR / "label_map.json"
DATA_REPORT_PATH = PROCESSED_REPORTS_DIR / "data_report.json"
IMAGE_META_PATH = PROCESSED_REPORTS_DIR / "image_meta.json"
TRAIN_SPLIT_PATH = PROCESSED_SPLITS_DIR / "train_split.json"
VAL_SPLIT_PATH = PROCESSED_SPLITS_DIR / "val_split.json"
RARE_CLASSES_PATH = PROCESSED_REPORTS_DIR / "rare_classes.json"
COPY_PASTE_MANIFEST_PATH = AUGMENTED_IMG_DIR / "manifest.json"


# =========================
# Utils
# =========================
def read_json(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, data: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def get_image_size_from_annotation(
    image_info: dict[str, Any]
) -> tuple[int | None, int | None]:
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


# =========================
# Step 1. annotation 파일 목록 읽기
# =========================
def load_raw_annotation_files(ann_dir: Path) -> list[Path]:
    return sorted(ann_dir.glob("**/*.json"))


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
                inconsistent_category_names.append({
                    "image_name": image_name,
                    "source_json": ann_file.name,
                    "category_id": category_id,
                    "reason": "category_id_not_found_in_categories",
                })
                label = "UNKNOWN"
            else:
                label = category_map[category_id]

            merged[image_name]["objects"].append({
                "bbox": bbox,
                "label": label,
                "category_id": category_id,
                "source_json": ann_file.name,
            })

    if inconsistent_category_names:
        print(f"[WARN] category mapping issues: {len(inconsistent_category_names)}")

    return merged


# =========================
# Step 3. label map 생성
# =========================
def build_label_map(merged: dict[str, dict[str, Any]]) -> dict[str, int]:
    labels = sorted({
        obj["label"]
        for item in merged.values()
        for obj in item["objects"]
    })
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
                invalid_bboxes.append({
                    "image_name": image_name,
                    "label": label,
                    "bbox": bbox,
                    "reason": reason,
                    "source_json": obj.get("source_json"),
                })
                images_with_invalid_bbox.add(image_name)

            ratio = bbox_area_ratio(bbox, width, height)
            if ratio is not None:
                bbox_area_ratios.append(ratio)

    rare_classes = sorted([
        cls_name for cls_name, count in class_counter.items()
        if count <= RARE_CLASS_THRESHOLD
    ])

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

        image_meta.append({
            "image_name": image_name,
            "width": item.get("width"),
            "height": item.get("height"),
            "num_objects": len(objects),
            "labels": sorted(list(set(labels))),
            "label_counts": dict(label_counter),
            "has_rare_class": any(label in rare_classes for label in labels),
        })

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
    # 1. 각 알약의 전체 빈도수 파악
    global_label_counts = Counter()
    for meta in image_meta:
        for label in meta["labels"]:
            global_label_counts[label] += 1

    # 2. 이미지별 '대표 희귀 라벨' 선정
    # 여러 알약이 있어도 가장 개수가 적은 알약을 기준으로 계층을 형성합니다.
    image_primary_labels = []
    image_names = []
    
    for meta in image_meta:
        image_names.append(meta["image_name"])
        if not meta["labels"]:
            image_primary_labels.append("empty")
            continue
        
        # 해당 사진에서 가장 희귀한 알약 찾기
        primary_label = min(meta["labels"], key=lambda l: global_label_counts[l])
        image_primary_labels.append(primary_label)

    # 3. 계층적 분할 실행 (Stratified Split)
    # - sklearn stratify는 각 그룹의 최소 샘플 수가 2 이상이어야 함
    # - 또한 val 샘플 수 >= 그룹 수 여야 안정적으로 동작함
    stratify_labels = list(image_primary_labels)
    counts = Counter(stratify_labels)

    # (a) 샘플 1개짜리 그룹은 "__other__"로 묶어서 에러 방지
    singletons = {lbl for lbl, cnt in counts.items() if cnt < 2}
    if singletons:
        for i, lbl in enumerate(stratify_labels):
            if lbl in singletons:
                stratify_labels[i] = "__other__"
        counts = Counter(stratify_labels)

    # (b) 그룹 수가 val 샘플 수보다 많으면, 빈도 낮은 그룹부터 "__other__"로 병합
    n_val = max(1, int(len(image_names) * val_ratio))
    if len(counts) > n_val:
        # "__other__" 제외 후 빈도 낮은 순으로 병합
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

    # (c) 그래도 stratify 조건이 안 맞으면 랜덤 분할로 폴백
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
    target_count: int = COPY_PASTE_TARGET_COUNT,
) -> dict[str, dict[str, Any]]:
    """
    희귀 클래스에 대해 Copy-Paste 증강을 수행합니다.
    - 알약 크기를 랜덤하게 조절(Random Scale) 기능 포함.
    """
    print(f"[7] Starting Copy-Paste augmentation (Target: {target_count} instances per rare class)")

    def sanitize_filename(text: str) -> str:
        # 파일명으로 쓰기 어려운 문자를 '_'로 치환 (예: '/', '\\', ':', 공백 등)
        cleaned = re.sub(r"[^0-9A-Za-z가-힣._-]+", "_", text)
        return cleaned.strip("_") or "unknown"
    
    # 1. Pill Bank 구축 (희귀 알약 크롭 이미지 수집)
    pill_bank: dict[str, list[dict[str, Any]]] = {cls: [] for cls in rare_classes}
    
    for img_name in train_image_names:
        img_info = merged[img_name]
        img_path = TRAIN_IMG_DIR / img_name
        if not img_path.exists(): continue
        
        img = cv2.imread(str(img_path))
        if img is None: continue
        
        for obj in img_info["objects"]:
            if obj["label"] in rare_classes:
                x, y, w, h = map(int, obj["bbox"])
                crop = img[y:y+h, x:x+w].copy()
                pill_bank[obj["label"]].append({
                    "image": crop,
                    "label": obj["label"],
                    "category_id": obj["category_id"],
                    "original_size": (w, h)
                })

    # 2. 증강 이미지 생성
    augmented_annots: dict[str, dict[str, Any]] = {}
    bg_color = (211, 211, 211) # 연회색 배경 (원본 데이터 특성 반영)
    
    for cls_name, crops in pill_bank.items():
        if not crops: continue
        
        current_count = len(crops)
        needed = target_count - current_count
        if needed <= 0: continue
        
        print(f"    - Augmenting {cls_name}: {current_count} -> {target_count}")
        
        for i in range(needed):
            # 새 캔버스 생성 (원본 해상도 유지)
            aug_img = np.full((1280, 976, 3), bg_color, dtype=np.uint8)
            aug_objects = []
            
            # 한 이미지에 2~4개의 알약 배치 (희귀 알약 위주)
            num_pills = random.randint(2, 4)
            for p_idx in range(num_pills):
                # 타겟 희귀 클래스 알약 하나는 반드시 포함, 나머지는 무작위 희귀 알약
                selected_crop_info = random.choice(crops) if p_idx == 0 else random.choice(random.choice(list(pill_bank.values())))
                
                pill_img = selected_crop_info["image"]
                
                # [향후 개선 사항 반영] 알약 크기 랜덤 조절 (0.8x ~ 1.2x)
                scale = random.uniform(0.8, 1.2)
                new_w = int(pill_img.shape[1] * scale)
                new_h = int(pill_img.shape[0] * scale)
                pill_img = cv2.resize(pill_img, (new_w, new_h))
                
                # 랜덤 회전 (0~360) - [PRD 3.2 반영]
                center = (new_w // 2, new_h // 2)
                matrix = cv2.getRotationMatrix2D(center, random.uniform(0, 360), 1.0)
                pill_img = cv2.warpAffine(pill_img, matrix, (new_w, new_h), borderValue=bg_color)
                
                # 배치 가능 위치 찾기 (단순화: 겹침 방지는 미구현했으나 좌표 분산으로 충분)
                max_y, max_x = aug_img.shape[0] - new_h, aug_img.shape[1] - new_w
                start_x = random.randint(50, max_x - 50)
                start_y = random.randint(50, max_y - 50)
                
                # 합성
                aug_img[start_y:start_y+new_h, start_x:start_x+new_w] = pill_img
                
                aug_objects.append({
                    "bbox": [float(start_x), float(start_y), float(new_w), float(new_h)],
                    "label": selected_crop_info["label"],
                    "category_id": selected_crop_info["category_id"],
                    "source_json": "augmented"
                })
            
            safe_cls = sanitize_filename(cls_name)
            aug_name = f"aug_{safe_cls}_{i}.png"
            out_path = AUGMENTED_IMG_DIR / aug_name
            ok = cv2.imwrite(str(out_path), aug_img)
            if not ok:
                raise RuntimeError(f"Failed to write augmented image: {out_path}")
            
            augmented_annots[aug_name] = {
                "image_name": aug_name,
                "width": 976,
                "height": 1280,
                "objects": aug_objects,
                "is_augmented": True
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
    print("=" * 60)
    print("Start preprocessing")
    print("=" * 60)
    print(f"PROJECT_ROOT     : {PROJECT_ROOT}")
    print(f"TRAIN_IMG_DIR    : {TRAIN_IMG_DIR}")
    print(f"TRAIN_ANN_DIR    : {TRAIN_ANN_DIR}")
    print(f"PROCESSED_DIR    : {PROCESSED_DIR}")
    print(f"DATA_SEED        : {DATA_SEED}")
    print(f"COPY_PASTE_DIR   : {AUGMENTED_IMG_DIR}")

    # === 실험 제어 플래그 ===
    USE_STRATIFIED = True     # True: 계층적 분할(Exp1) / False: 랜덤 분할(Baseline)
    USE_COPY_PASTE = True     # (실험 5: Copy-Paste 가동)

    if not TRAIN_IMG_DIR.exists():
        raise FileNotFoundError(f"Train image directory not found: {TRAIN_IMG_DIR}")

    if not TRAIN_ANN_DIR.exists():
        raise FileNotFoundError(f"Train annotation directory not found: {TRAIN_ANN_DIR}")

    ann_files = load_raw_annotation_files(TRAIN_ANN_DIR)
    print(f"[1] Loaded annotation json files: {len(ann_files)}")

    merged = merge_annotations_by_image(ann_files)
    print(f"[2] Merged into image-level annotations: {len(merged)} images")

    label_map = build_label_map(merged)
    print(f"[3] Built label map: {len(label_map)} classes")

    report = build_data_report(
        merged=merged,
        train_img_dir=TRAIN_IMG_DIR,
        train_ann_files=ann_files,
    )

    rare_classes = set(report["rare_classes"])
    print(f"[4] Built report")
    print(f"    - num classes              : {report['summary']['num_classes']}")
    print(f"    - missing annotation imgs  : {report['summary']['num_missing_annotation_images']}")
    print(f"    - missing image files      : {report['summary']['num_missing_image_files']}")
    print(f"    - invalid bboxes           : {report['summary']['num_invalid_bboxes']}")
    print(f"    - rare classes             : {len(rare_classes)}")

    image_meta = build_image_meta(merged, rare_classes)
    print(f"[5] Built image metadata: {len(image_meta)} items")

    # [6] 데이터셋 분배 (Train/Val)
    if USE_STRATIFIED:
        # 계층적 분산 로직 (Exp1)
        train_images, val_images = split_train_val_stratified(
            image_meta=image_meta,
            val_ratio=VAL_RATIO,
            seed=DATA_SEED
        )
        print(f"[6] Performed Stratified Split (ratio: {VAL_RATIO})")
    else:
        # 클래식 랜덤 분산 로직 (Baseline)
        from sklearn.model_selection import train_test_split
        train_images, val_images = train_test_split(
            list(merged.keys()),
            test_size=VAL_RATIO,
            random_state=DATA_SEED
        )
        print(f"[6] Performed Random Split (ratio: {VAL_RATIO})")
    
    print(f"    - train images: {len(train_images)}")
    print(f"    - val images  : {len(val_images)}")

    # [7] Copy-Paste 증강 적용 (실험 5 가동)
    augmented_data: dict[str, dict[str, Any]] = {}
    if USE_COPY_PASTE:
        augmented_data = apply_copy_paste_augmentation(
            train_image_names=train_images,
            merged=merged,
            rare_classes=rare_classes,
            target_count=COPY_PASTE_TARGET_COUNT
        )
        
        # 병합된 데이터에 증강 데이터 추가
        for aug_name, aug_info in augmented_data.items():
            merged[aug_name] = aug_info
            train_images.append(aug_name)
        
        print(f"[7] Added {len(augmented_data)} augmented images to train set")
    else:
        print("[7] Copy-Paste augmentation disabled")

    train_summary = summarize_split(train_images, merged)
    val_summary = summarize_split(val_images, merged)

    split_info = {
        "data_seed": DATA_SEED,
        "seed": DATA_SEED,
        "val_ratio": VAL_RATIO,
        "copy_paste_variant_id": COPY_PASTE_VARIANT_ID,
        "copy_paste_target_count": COPY_PASTE_TARGET_COUNT,
        "train_summary": train_summary,
        "val_summary": val_summary,
    }

    copy_paste_manifest = {
        "variant_id": COPY_PASTE_VARIANT_ID,
        "augmentation": "copy_paste",
        "data_seed": DATA_SEED,
        "target_count_per_rare_class": COPY_PASTE_TARGET_COUNT,
        "enabled": USE_COPY_PASTE,
        "num_generated_images": len(augmented_data) if USE_COPY_PASTE else 0,
        "input_images_dir": str(TRAIN_IMG_DIR),
        "input_annotations_dir": str(TRAIN_ANN_DIR),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }

    merged_output = {
        "metadata": {
            "description": "Merged image-level annotations generated from object-level json files",
            "num_images": len(merged),
            "num_classes": len(label_map),
            "label_map_path": str(LABEL_MAP_PATH.name),
        },
        "images": merged,
    }

    write_json(MERGED_ANNOTATIONS_PATH, merged_output)
    write_json(LABEL_MAP_PATH, label_map)
    write_json(IMAGE_META_PATH, image_meta)
    write_json(TRAIN_SPLIT_PATH, train_images)
    write_json(VAL_SPLIT_PATH, val_images)
    write_json(RARE_CLASSES_PATH, sorted(list(rare_classes)))
    write_json(COPY_PASTE_MANIFEST_PATH, copy_paste_manifest)

    report_with_split = report.copy()
    report_with_split["split_info"] = split_info
    write_json(DATA_REPORT_PATH, report_with_split)

    print("[7] Saved files")
    print(f"    - {MERGED_ANNOTATIONS_PATH}")
    print(f"    - {LABEL_MAP_PATH}")
    print(f"    - {DATA_REPORT_PATH}")
    print(f"    - {IMAGE_META_PATH}")
    print(f"    - {TRAIN_SPLIT_PATH}")
    print(f"    - {VAL_SPLIT_PATH}")
    print(f"    - {RARE_CLASSES_PATH}")
    print(f"    - {COPY_PASTE_MANIFEST_PATH}")

    print("=" * 60)
    print("Preprocessing done.")
    print("=" * 60)


if __name__ == "__main__":
    main()
