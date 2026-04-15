from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import yaml

from src.utils.config_paths import apply_train_config_paths, merge_runtime_overrides

PROJECT_ROOT = Path(__file__).resolve().parent


def read_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_config(config_path: Path) -> dict[str, Any]:
    if not config_path.exists():
        raise FileNotFoundError(f"config not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise TypeError("config must be a dict")
    return apply_train_config_paths(data)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare YOLO dataset with unified config paths")
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "train" / "exp15_train_baseline_yolo11s_2.0.yaml",
        help="train config yaml path (defaults to exp15 baseline)",
    )
    parser.add_argument("--dataset-family", type=str, default=None)
    parser.add_argument("--dataset-variant-id", type=str, default=None)
    parser.add_argument("--clean-output", type=int, choices=[0, 1], default=None)
    parser.add_argument("--data-seed", type=int, default=None)
    parser.add_argument("--copy-paste-target-count", type=int, default=None)
    return parser.parse_args()


def as_project_path(path_str: str) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def resolve_runtime_settings(args: argparse.Namespace, config: dict[str, Any]) -> dict[str, Any]:
    paths = config.get("paths", {})
    if not isinstance(paths, dict):
        raise TypeError("config.paths must be a dict")

    overrides = merge_runtime_overrides(
        config=config,
        cli=vars(args),
        schema={
            "data_seed": {"aliases": ("data_seed", "seed"), "default": 42, "cast": int},
            "copy_paste_target_count": {
                "aliases": ("copy_paste_target_count",),
                "default": 20,
                "cast": int,
            },
            "dataset_family": {
                "aliases": ("dataset_family",),
                "default": "copy_paste",
                "cast": str,
            },
            "dataset_variant_id": {
                "aliases": ("dataset_variant_id",),
                "default": None,
                "cast": str,
            },
            "clean_output": {"aliases": ("clean_output",), "default": 1, "cast": bool},
        },
    )

    data_seed = overrides["data_seed"]
    copy_paste_target_count = overrides["copy_paste_target_count"]
    copy_paste_variant_id = f"cp_t{copy_paste_target_count}_data_seed{data_seed}"
    dataset_variant_id = overrides["dataset_variant_id"] or copy_paste_variant_id

    train_img_dir = as_project_path(str(paths["train_images_dir"]))
    processed_dir = as_project_path(str(paths["processed_root"]))
    datasets_root = as_project_path(str(paths["datasets_root"]))
    augmented_img_dir = processed_dir / "offline" / "copy_paste" / dataset_variant_id

    merged_annotations_path = resolve_existing_path(
        processed_dir / "merged" / "merged_annotations.json",
        processed_dir / "merged_annotations.json",
    )
    label_map_path = resolve_existing_path(
        processed_dir / "reports" / "label_map.json",
        processed_dir / "label_map.json",
    )
    train_split_path = resolve_existing_path(
        processed_dir / "splits" / "train_split.json",
        processed_dir / "train_split.json",
    )
    val_split_path = resolve_existing_path(
        processed_dir / "splits" / "val_split.json",
        processed_dir / "val_split.json",
    )

    yolo_root = datasets_root / "yolo" / overrides["dataset_family"] / dataset_variant_id
    yolo_images_train = yolo_root / "images" / "train"
    yolo_images_val = yolo_root / "images" / "val"
    yolo_labels_train = yolo_root / "labels" / "train"
    yolo_labels_val = yolo_root / "labels" / "val"
    yolo_data_yaml = yolo_root / "dataset.yaml"

    return {
        "data_seed": data_seed,
        "copy_paste_target_count": copy_paste_target_count,
        "dataset_family": overrides["dataset_family"],
        "dataset_variant_id": dataset_variant_id,
        "clean_output": overrides["clean_output"],
        "train_img_dir": train_img_dir,
        "processed_dir": processed_dir,
        "augmented_img_dir": augmented_img_dir,
        "merged_annotations_path": merged_annotations_path,
        "label_map_path": label_map_path,
        "train_split_path": train_split_path,
        "val_split_path": val_split_path,
        "yolo_root": yolo_root,
        "yolo_images_train": yolo_images_train,
        "yolo_images_val": yolo_images_val,
        "yolo_labels_train": yolo_labels_train,
        "yolo_labels_val": yolo_labels_val,
        "yolo_data_yaml": yolo_data_yaml,
    }


def resolve_existing_path(primary: Path, legacy: Path) -> Path:
    if primary.exists():
        return primary
    if legacy.exists():
        return legacy
    return primary


def prepare_output_dirs(
    yolo_images_train: Path,
    yolo_images_val: Path,
    yolo_labels_train: Path,
    yolo_labels_val: Path,
    clean: bool,
) -> None:
    for d in [
        yolo_images_train,
        yolo_images_val,
        yolo_labels_train,
        yolo_labels_val,
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


def find_source_image_path(
    image_name: str,
    train_img_dir: Path,
    augmented_img_dir: Path,
    processed_dir: Path,
) -> Path:
    candidates = [
        train_img_dir / image_name,
        augmented_img_dir / image_name,
        processed_dir / "offline" / "copy_paste" / image_name,
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


def save_dataset_yaml(label_map: dict[str, int], yolo_root: Path, yolo_data_yaml: Path) -> None:
    id_to_label = {idx: label for label, idx in label_map.items()}
    names = [id_to_label[i] for i in range(len(id_to_label))]

    yaml_text = "\n".join(
        [
            f"path: {yolo_root.resolve()}",
            "train: images/train",
            "val: images/val",
            f"nc: {len(names)}",
            "names:",
            *[f"  {i}: '{name}'" for i, name in enumerate(names)],
        ]
    )

    yolo_data_yaml.write_text(yaml_text, encoding="utf-8")


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    runtime = resolve_runtime_settings(args, config)
    prepare_output_dirs(
        yolo_images_train=runtime["yolo_images_train"],
        yolo_images_val=runtime["yolo_images_val"],
        yolo_labels_train=runtime["yolo_labels_train"],
        yolo_labels_val=runtime["yolo_labels_val"],
        clean=runtime["clean_output"],
    )

    missing = [
        p
        for p in [
            runtime["merged_annotations_path"],
            runtime["label_map_path"],
            runtime["train_split_path"],
            runtime["val_split_path"],
        ]
        if not p.exists()
    ]
    if missing:
        missing_list = "\n".join(f"- {p}" for p in missing)
        raise FileNotFoundError(
            "Missing processed artifacts required to build `data/datasets/yolo/<variant_id>/`.\n"
            f"{missing_list}\n\n"
            "Next steps:\n"
            "- If you have curated COCO jsons under `data/curated/train_annotations`, run: `python preprocessing.py` first.\n"
            "- If you already have a ready-to-train YOLO dataset under `data/datasets/yolo/<dataset_family>/<variant_id>/`,\n"
            "  you can skip this script and train directly with its `dataset.yaml`.\n"
        )

    merged = read_json(runtime["merged_annotations_path"])
    label_map = read_json(runtime["label_map_path"])
    train_images = read_json(runtime["train_split_path"])
    val_images = read_json(runtime["val_split_path"])

    images_dict: dict[str, Any] = merged["images"]

    for image_name in train_images:
        image_info = images_dict[image_name]

        src_img = find_source_image_path(
            image_name=image_name,
            train_img_dir=runtime["train_img_dir"],
            augmented_img_dir=runtime["augmented_img_dir"],
            processed_dir=runtime["processed_dir"],
        )

        dst_img = runtime["yolo_images_train"] / image_name
        dst_label = runtime["yolo_labels_train"] / f"{Path(image_name).stem}.txt"

        copy_image(src_img, dst_img)
        write_label_file(image_info, label_map, dst_label)

    for image_name in val_images:
        image_info = images_dict[image_name]

        src_img = find_source_image_path(
            image_name=image_name,
            train_img_dir=runtime["train_img_dir"],
            augmented_img_dir=runtime["augmented_img_dir"],
            processed_dir=runtime["processed_dir"],
        )
        dst_img = runtime["yolo_images_val"] / image_name
        dst_label = runtime["yolo_labels_val"] / f"{Path(image_name).stem}.txt"

        copy_image(src_img, dst_img)
        write_label_file(image_info, label_map, dst_label)

    save_dataset_yaml(
        label_map=label_map,
        yolo_root=runtime["yolo_root"],
        yolo_data_yaml=runtime["yolo_data_yaml"],
    )

    print("YOLO dataset prepared.")
    print(f"config: {args.config}")
    print(f"dataset_family: {runtime['dataset_family']}")
    print(f"variant_id: {runtime['dataset_variant_id']}")
    print(f"clean_output: {runtime['clean_output']}")
    print(f"dataset.yaml: {runtime['yolo_data_yaml']}")


if __name__ == "__main__":
    main()
