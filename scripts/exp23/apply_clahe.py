from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import cv2
import yaml
from tqdm import tqdm


IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


def parse_tile_grid_size(value: str) -> tuple[int, int]:
    parts = [p.strip() for p in value.split(",")]
    if len(parts) != 2:
        raise ValueError("--tile-grid-size must be like '8,8'")
    w, h = int(parts[0]), int(parts[1])
    if w <= 0 or h <= 0:
        raise ValueError("tile grid size values must be positive")
    return w, h


def load_base_dataset_yaml(base_data_dir: Path) -> dict:
    dataset_yaml = base_data_dir / "dataset.yaml"
    if not dataset_yaml.exists():
        raise FileNotFoundError(f"dataset.yaml not found: {dataset_yaml}")
    with dataset_yaml.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def infer_dataset_identity(dataset_dir: Path) -> tuple[str | None, str | None]:
    variant_id = dataset_dir.name if dataset_dir.name else None
    dataset_family = dataset_dir.parent.name if dataset_dir.parent != dataset_dir else None
    return dataset_family, variant_id


def load_base_manifest(base_data_dir: Path, processed_root: Path) -> dict:
    dataset_family, variant_id = infer_dataset_identity(base_data_dir)
    if dataset_family is None or variant_id is None:
        return {}

    manifest_path = processed_root / "offline" / dataset_family / variant_id / "manifest.json"
    if not manifest_path.exists():
        return {}

    with manifest_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        return {}
    return payload


def collect_images(images_dir: Path) -> list[Path]:
    return sorted([p for p in images_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS])


def apply_clahe_to_split(
    input_images_dir: Path,
    output_images_dir: Path,
    clip_limit: float,
    tile_grid_size: tuple[int, int],
) -> int:
    output_images_dir.mkdir(parents=True, exist_ok=True)
    image_paths = collect_images(input_images_dir)
    if not image_paths:
        raise FileNotFoundError(f"No images found in: {input_images_dir}")

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)

    for image_path in tqdm(image_paths, desc=f"CLAHE {input_images_dir.name}", unit="img"):
        img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if img is None:
            continue
        yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
        yuv[:, :, 0] = clahe.apply(yuv[:, :, 0])
        out = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)
        cv2.imwrite(str(output_images_dir / image_path.name), out)

    return len(image_paths)


def copy_labels_split(input_labels_dir: Path, output_labels_dir: Path) -> int:
    output_labels_dir.mkdir(parents=True, exist_ok=True)
    label_paths = sorted([p for p in input_labels_dir.iterdir() if p.is_file() and p.suffix.lower() == ".txt"])
    for label_path in label_paths:
        shutil.copy2(label_path, output_labels_dir / label_path.name)
    return len(label_paths)


def write_dataset_yaml(base_meta: dict, output_root: Path) -> Path:
    out_yaml = {
        "path": str(output_root.resolve()),
        "train": "images/train",
        "val": "images/val",
        "nc": int(base_meta.get("nc", 0)),
        "names": base_meta.get("names", {}),
    }

    output_root.mkdir(parents=True, exist_ok=True)
    output_yaml_path = output_root / "dataset.yaml"
    with output_yaml_path.open("w", encoding="utf-8") as f:
        yaml.dump(out_yaml, f, allow_unicode=True, sort_keys=False)
    return output_yaml_path


def write_clahe_manifest(
    *,
    output_root: Path,
    processed_root: Path,
    base_data_dir: Path,
    base_manifest: dict,
    clip_limit: float,
    tile_grid_size: tuple[int, int],
    total_images: int,
    total_labels: int,
    dataset_yaml_path: Path,
) -> Path:
    dataset_family, base_variant_id = infer_dataset_identity(base_data_dir)
    variant_id = output_root.name
    base_data_version = base_manifest.get("data_version")
    data_version = f"{base_data_version}_{variant_id}" if base_data_version else variant_id

    manifest = {
        "variant_id": variant_id,
        "data_version": data_version,
        "dataset_hash": base_manifest.get("dataset_hash"),
        "base_dataset_family": dataset_family,
        "base_variant_id": base_variant_id,
        "base_data_version": base_data_version,
        "base_dataset_dir": str(base_data_dir),
        "input_images_dir": base_manifest.get("input_images_dir"),
        "input_annotations_dir": base_manifest.get("input_annotations_dir"),
        "input_paths": base_manifest.get("input_paths") or {},
        "augmentation": "clahe",
        "clip_limit": clip_limit,
        "tile_grid_size": list(tile_grid_size),
        "materialized_yolo_dataset": str(dataset_yaml_path),
        "raw_train_image_hash": base_manifest.get("raw_train_image_hash"),
        "raw_train_image_version": base_manifest.get("raw_train_image_version"),
        "raw_train_image_file_count": base_manifest.get("raw_train_image_file_count"),
        "raw_train_image_total_bytes": base_manifest.get("raw_train_image_total_bytes"),
        "annotation_file_count": base_manifest.get("annotation_file_count"),
        "images_processed": total_images,
        "labels_copied": total_labels,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }

    manifest_path = processed_root / "offline" / "clahe" / variant_id / "manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    return manifest_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Build CLAHE-processed YOLO dataset clone")
    parser.add_argument(
        "--base-data-dir",
        type=Path,
        default=Path("data/datasets/yolo/copy_paste/cp_t20_data_seed42"),
        help="Base YOLO dataset directory containing images/, labels/, dataset.yaml",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/datasets/yolo/clahe/cl20_data_seed42"),
        help="Output root for CLAHE dataset clone",
    )
    parser.add_argument(
        "--processed-root",
        type=Path,
        default=Path("data/processed"),
        help="Processed data root containing offline manifests",
    )
    parser.add_argument(
        "--clip-limit",
        type=float,
        default=2.0,
        help="OpenCV CLAHE clipLimit",
    )
    parser.add_argument(
        "--tile-grid-size",
        type=str,
        default="8,8",
        help="OpenCV CLAHE tileGridSize as 'w,h' (e.g., 8,8)",
    )
    parser.add_argument(
        "--overwrite",
        dest="overwrite",
        action="store_true",
        help="Delete output-root before creating the new dataset (default: enabled)",
    )
    parser.add_argument(
        "--no-overwrite",
        dest="overwrite",
        action="store_false",
        help="Keep existing output-root and overwrite only matching filenames",
    )
    parser.set_defaults(overwrite=True)
    args = parser.parse_args()

    base_data_dir = args.base_data_dir.resolve()
    output_root = args.output_root.resolve()
    processed_root = args.processed_root.resolve()
    tile_grid_size = parse_tile_grid_size(args.tile_grid_size)

    if args.overwrite and output_root.exists():
        shutil.rmtree(output_root)
    elif output_root.exists():
        print(
            f"[WARN] output_root already exists and --no-overwrite was used: {output_root}\n"
            "       stale files may remain if the base dataset split changed."
        )

    base_meta = load_base_dataset_yaml(base_data_dir)
    base_manifest = load_base_manifest(base_data_dir=base_data_dir, processed_root=processed_root)

    total_images = 0
    total_labels = 0
    for split in ("train", "val"):
        input_images_dir = base_data_dir / "images" / split
        input_labels_dir = base_data_dir / "labels" / split
        output_images_dir = output_root / "images" / split
        output_labels_dir = output_root / "labels" / split

        if not input_images_dir.exists():
            raise FileNotFoundError(f"Missing images split: {input_images_dir}")
        if not input_labels_dir.exists():
            raise FileNotFoundError(f"Missing labels split: {input_labels_dir}")

        total_images += apply_clahe_to_split(
            input_images_dir=input_images_dir,
            output_images_dir=output_images_dir,
            clip_limit=args.clip_limit,
            tile_grid_size=tile_grid_size,
        )
        total_labels += copy_labels_split(
            input_labels_dir=input_labels_dir,
            output_labels_dir=output_labels_dir,
        )

    dataset_yaml_path = write_dataset_yaml(base_meta=base_meta, output_root=output_root)
    manifest_path = write_clahe_manifest(
        output_root=output_root,
        processed_root=processed_root,
        base_data_dir=base_data_dir,
        base_manifest=base_manifest,
        clip_limit=args.clip_limit,
        tile_grid_size=tile_grid_size,
        total_images=total_images,
        total_labels=total_labels,
        dataset_yaml_path=dataset_yaml_path,
    )
    print("CLAHE dataset created")
    print(f"- output_root: {output_root}")
    print(f"- dataset_yaml: {dataset_yaml_path}")
    print(f"- manifest: {manifest_path}")
    print(f"- overwrite: {args.overwrite}")
    print(f"- clip_limit: {args.clip_limit}")
    print(f"- tile_grid_size: {tile_grid_size}")
    print(f"- images_processed: {total_images}")
    print(f"- labels_copied: {total_labels}")


if __name__ == "__main__":
    main()
