from __future__ import annotations

from pathlib import Path
import hashlib
import json
import platform
import subprocess

import numpy as np


def sha256_file(file_path: str | Path | None) -> str | None:
    """
    Return SHA256 hex digest for a file path.
    Returns None when path is missing or file does not exist.
    """
    if file_path is None:
        return None
    path = Path(file_path)
    if not path.exists():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()


def percentile_ms(latency_sec: list[float], q: float) -> float | None:
    """
    Convert second-based latency list to millisecond percentile.
    """
    if not latency_sec:
        return None
    return float(np.percentile(np.array(latency_sec) * 1000.0, q))


def parse_cuda_device_index(device: str) -> int:
    """
    Parse CUDA index from strings like '0', '0,1', 'cpu'.
    Returns 0 for cpu/non-parsable values for compatibility.
    """
    text = str(device).strip().lower()
    if text == "cpu":
        return 0
    first = text.split(",")[0]
    try:
        return int(first)
    except ValueError:
        return 0


def get_gpu_name(device: str = "0") -> str | None:
    """
    Return GPU name for selected CUDA device; None when unavailable.
    """
    try:
        import torch  # local import to keep this module import-safe without torch
    except Exception:
        return None
    if not torch.cuda.is_available() or str(device).strip().lower() == "cpu":
        return None
    try:
        return torch.cuda.get_device_name(parse_cuda_device_index(device))
    except Exception:
        return None


def get_nvidia_driver_version() -> str | None:
    """
    Read NVIDIA driver version via nvidia-smi.
    Returns None on macOS/CPU-only/driver-missing environments.
    """
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except Exception:
        return None
    line = out.strip().splitlines()[0].strip() if out.strip() else ""
    return line or None


def collect_runtime_env(device: str = "0") -> dict:
    """
    Collect runtime environment metadata for reproducibility logging.
    """
    try:
        import torch  # local import to keep this module import-safe without torch
        torch_version = str(torch.__version__)
        cuda_version = str(torch.version.cuda) if torch.version.cuda is not None else None
    except Exception:
        torch_version = None
        cuda_version = None

    return {
        "os_platform": platform.platform(),
        "python_version": platform.python_version(),
        "torch_version": torch_version,
        "cuda_version": cuda_version,
        "gpu_name": get_gpu_name(device),
        "gpu_driver": get_nvidia_driver_version(),
    }


def safe_len_dataset(loader: object) -> int | None:
    """
    Safely read len(loader.dataset), returns None on failure.
    """
    if loader is None:
        return None
    dataset = getattr(loader, "dataset", None)
    if dataset is None:
        return None
    try:
        return int(len(dataset))
    except Exception:
        return None


def get_value(obj: object, key: str) -> object:
    """
    Access key from dict-like or attribute from object.
    """
    if obj is None:
        return None
    if isinstance(obj, dict):
        return obj.get(key)
    return getattr(obj, key, None)


def resolve_train_metrics_path(project_root: Path, experiment: str) -> Path:
    metrics_dir = project_root / "metrics" / "train"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    return metrics_dir / f"{experiment}_metrics.json"


def resolve_infer_metrics_path(project_root: Path, run_name: str, metrics_output: str | Path | None = None) -> Path:
    if metrics_output:
        output = Path(metrics_output)
        output.parent.mkdir(parents=True, exist_ok=True)
        return output
    metrics_dir = project_root / "metrics" / "infer"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    return metrics_dir / f"{run_name}_infer_metrics.json"


def write_metrics_json(payload: dict, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return output_path
