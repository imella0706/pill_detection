from __future__ import annotations

from pathlib import Path
import json
import shutil
import os
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw" / "sprint_ai_project1_data"
TRAIN_IMG_DIR = RAW_DATA_DIR / "train_images"
DATA_SEED = int(os.getenv("DATA_SEED", "42"))
COPY_PASTE_TARGET_COUNT = int(os.getenv("COPY_PASTE_TARGET_COUNT", "20"))
COPY_PASTE_VARIANT_ID = os.getenv(
    "COPY_PASTE_VARIANT_ID",
    f"cp_t{COPY_PASTE_TARGET_COUNT}_data_seed{DATA_SEED}",
)
AUGMENTED_IMG_DIR = PROCESSED_DIR / "offline" / "copy_paste" / COPY_PASTE_VARIANT_ID

MERGED_ANNOTATIONS_PATH = PROCESSED_DIR / "merged" / "merged_annotations.json"
LABEL_MAP_PATH = PROCESSED_DIR / "reports" / "label_map.json"
TRAIN_SPLIT_PATH = PROCESSED_DIR / "splits" / "train_split.json"
VAL_SPLIT_PATH = PROCESSED_DIR / "splits" / "val_split.json"

DATASET_FAMILY = os.getenv("DATASET_FAMILY", "copy_paste")
DATASET_VARIANT_ID = os.getenv("DATASET_VARIANT_ID", COPY_PASTE_VARIANT_ID)
CLEAN_OUTPUT = os.getenv("CLEAN_OUTPUT", "1").lower() not in {"0", "false", "no"}
YOLO_ROOT = PROJECT_ROOT / "data" / "datasets" / "yolo" / DATASET_FAMILY / DATASET_VARIANT_ID
YOLO_IMAGES_TRAIN = YOLO_ROOT / "images" / "train"
YOLO_IMAGES_VAL = YOLO_ROOT / "images" / "val"
YOLO_LABELS_TRAIN = YOLO_ROOT / "labels" / "train"
YOLO_LABELS_VAL = YOLO_ROOT / "labels" / "val"
YOLO_DATA_YAML = YOLO_ROOT / "dataset.yaml"


def read_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def resolve_existing_path(primary: Path, legacy: Path) -> Path:
    if primary.exists():
        return primary
    if legacy.exists():
        return legacy
    return primary


def prepare_output_dirs(clean: bool = CLEAN_OUTPUT) -> None:
    for d in [
        YOLO_IMAGES_TRAIN,
        YOLO_IMAGES_VAL,
        YOLO_LABELS_TRAIN,
        YOLO_LABELS_VAL,
    ]:
        if clean and d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True, exist_ok=True)


def xywh_to_yolo(
    bbox: list[float],
    img_w: int,
    img_h: int,
) -> tuple[float, float, float, float]:
    """
    입력: [x, y, w, h] (absolute)
    출력: [x_center, y_center, w, h] (normalized for YOLO)
    """
    x, y, w, h = bbox

    x_center = (x + w / 2) / img_w
    y_center = (y + h / 2) / img_h
    w_norm = w / img_w
    h_norm = h / img_h

    return x_center, y_center, w_norm, h_norm


def clip01(v: float) -> float:
    return max(0.0, min(1.0, v))


def write_label_file(
    image_info: dict[str, Any],
    label_map: dict[str, int],
    out_path: Path,
) -> None:
    img_w = image_info["width"]
    img_h = image_info["height"]
    objects = image_info["objects"]

    lines: list[str] = []

    for obj in objects:
        label = obj["label"]
        bbox = obj["bbox"]
        class_id = label_map[label]

        x_center, y_center, w_norm, h_norm = xywh_to_yolo(bbox, img_w, img_h)

        x_center = clip01(x_center)
        y_center = clip01(y_center)
        w_norm = clip01(w_norm)
        h_norm = clip01(h_norm)

        lines.append(
            f"{class_id} "
            f"{x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}"
        )

    out_path.write_text("\n".join(lines), encoding="utf-8")


def copy_image(src: Path, dst: Path) -> None:
    if not src.exists():
        raise FileNotFoundError(f"Image not found: {src}")
    shutil.copy2(src, dst)

def find_source_image_path(image_name: str) -> Path:
    candidates = [
        TRAIN_IMG_DIR / image_name,
        AUGMENTED_IMG_DIR / image_name,
        PROCESSED_DIR / "offline" / "copy_paste" / image_name,
    ]
    for p in candidates:
        if p.exists():
            return p
    searched = "\n".join(f"- {p}" for p in candidates)
    raise FileNotFoundError(
        f"Image not found for '{image_name}'. Looked in:\n{searched}\n\n"
        "If this is an augmented image, ensure `preprocessing.py` wrote it under "
        "`data/processed/offline/copy_paste/<variant_id>/`."
    )


def save_dataset_yaml(label_map: dict[str, int]) -> None:
    id_to_label = {idx: label for label, idx in label_map.items()}
    names = [id_to_label[i] for i in range(len(id_to_label))]

    yaml_text = "\n".join([
        f"path: {YOLO_ROOT.resolve()}",
        "train: images/train",
        "val: images/val",
        f"nc: {len(names)}",
        "names:",
        *[f"  {i}: '{name}'" for i, name in enumerate(names)],
    ])

    YOLO_DATA_YAML.write_text(yaml_text, encoding="utf-8")


def main() -> None:
    prepare_output_dirs()

    merged_annotations_path = resolve_existing_path(
        MERGED_ANNOTATIONS_PATH,
        PROCESSED_DIR / "merged_annotations.json",
    )
    label_map_path = resolve_existing_path(
        LABEL_MAP_PATH,
        PROCESSED_DIR / "label_map.json",
    )
    train_split_path = resolve_existing_path(
        TRAIN_SPLIT_PATH,
        PROCESSED_DIR / "train_split.json",
    )
    val_split_path = resolve_existing_path(
        VAL_SPLIT_PATH,
        PROCESSED_DIR / "val_split.json",
    )

    missing = [
        p
        for p in [
            merged_annotations_path,
            label_map_path,
            train_split_path,
            val_split_path,
        ]
        if not p.exists()
    ]
    if missing:
        missing_list = "\n".join(f"- {p}" for p in missing)
        raise FileNotFoundError(
            "Missing processed artifacts required to build `data/datasets/yolo/<variant_id>/`.\n"
            f"{missing_list}\n\n"
            "Next steps:\n"
            "- If you have curated COCO jsons under `data/curated/sprint_ai_project1_data/train_annotations`, run: `python preprocessing.py` first.\n"
            "- If you already have a ready-to-train YOLO dataset under `data/datasets/yolo/<dataset_family>/<variant_id>/`,\n"
            "  you can skip this script and train directly with its `dataset.yaml`.\n"
        )

    merged = read_json(merged_annotations_path)
    label_map = read_json(label_map_path)
    train_images = read_json(train_split_path)
    val_images = read_json(val_split_path)

    images_dict: dict[str, Any] = merged["images"]

    for image_name in train_images:
        image_info = images_dict[image_name]

        src_img = find_source_image_path(image_name)
            
        dst_img = YOLO_IMAGES_TRAIN / image_name
        dst_label = YOLO_LABELS_TRAIN / f"{Path(image_name).stem}.txt"

        copy_image(src_img, dst_img)
        write_label_file(image_info, label_map, dst_label)

    for image_name in val_images:
        image_info = images_dict[image_name]

        src_img = find_source_image_path(image_name)
        dst_img = YOLO_IMAGES_VAL / image_name
        dst_label = YOLO_LABELS_VAL / f"{Path(image_name).stem}.txt"

        copy_image(src_img, dst_img)
        write_label_file(image_info, label_map, dst_label)

    save_dataset_yaml(label_map)

    print("YOLO dataset prepared.")
    print(f"dataset_family: {DATASET_FAMILY}")
    print(f"variant_id: {DATASET_VARIANT_ID}")
    print(f"clean_output: {CLEAN_OUTPUT}")
    print(f"dataset.yaml: {YOLO_DATA_YAML}")


if __name__ == "__main__":
    main()
