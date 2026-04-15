from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Raw is immutable storage for original Kaggle assets.
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
CURATED_DATA_DIR = DATA_DIR / "curated"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
DATASETS_DIR = DATA_DIR / "datasets"

TRAIN_IMAGES_DIR = RAW_DATA_DIR / "train_images"
TEST_IMAGES_DIR = RAW_DATA_DIR / "test_images"
RAW_TRAIN_ANNOTATIONS_DIR = RAW_DATA_DIR / "train_annotations"
CURATED_TRAIN_ANNOTATIONS_DIR = CURATED_DATA_DIR / "train_annotations"


def get_annotation_source_dir() -> Path:
    return CURATED_TRAIN_ANNOTATIONS_DIR
