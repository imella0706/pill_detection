"""
Validation 셋 기준으로 NMS 임계값(conf, iou)을 탐색하는 스크립트.

학습이 끝난 가중치에 대해 여러 conf/iou 조합으로 `validation` 성능을 반복 평가하고,
가장 좋은 조합과 mAP 결과를 확인한다. 최종 탐색 결과는 metrics JSON으로 저장해
`experiments.md` 업데이트나 추론 설정 재현에 활용할 수 있다.
"""

import argparse
import gc
import json
import time
import sys
import re
from pathlib import Path
import torch
import pandas as pd
from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.utils.logging_utils import start_run_logging

METRICS_DIR = PROJECT_ROOT / "metrics"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Exp8: validation-based conf/iou grid search (Ultralytics YOLO)")
    p.add_argument(
        "--model",
        default="runs/pill_exp5_yolo11s_copypaste/weights/best.pt",
        help="Path to trained weights (e.g. runs/.../weights/best.pt)",
    )
    p.add_argument(
        "--data",
        default="data/datasets/yolo/copy_paste/cp_t20_data_seed42/dataset.yaml",
        help="Path to dataset.yaml",
    )
    p.add_argument("--confs", default="0.15,0.20,0.25", help="Comma-separated conf values")
    p.add_argument("--ious", default="0.50,0.60,0.70", help="Comma-separated iou values (NMS iou)")
    p.add_argument("--imgsz", type=int, default=None, help="Image size for validation (default: Ultralytics default)")
    p.add_argument("--batch", type=int, default=None, help="Batch size for validation (default: Ultralytics default)")
    p.add_argument("--device", default=None, help="Device (e.g. 0, cpu). Default: Ultralytics auto")
    p.add_argument(
        "--verbose",
        action="store_true",
        help="Show Ultralytics validation progress/logs (recommended if you can't see progress).",
    )
    p.add_argument(
        "--save-log",
        dest="save_log",
        action="store_true",
        help="Save runtime log to logs/inference",
    )
    p.add_argument(
        "--no-save-log",
        dest="save_log",
        action="store_false",
        help="Disable runtime log file saving",
    )
    p.set_defaults(save_log=True)
    return p.parse_args()


def parse_float_list(csv: str) -> list[float]:
    values: list[float] = []
    for part in csv.split(","):
        part = part.strip()
        if not part:
            continue
        values.append(float(part))
    return values


def validate_thresholds(values: list[float], name: str) -> None:
    if not values:
        raise ValueError(f"`{name}` is empty. Provide at least one value (e.g. --{name} 0.20).")
    for value in values:
        if not (0.0 <= value <= 1.0):
            raise ValueError(f"`{name}` must be in [0, 1]. Invalid value: {value}")


def default_device() -> str:
    return "0" if torch.cuda.is_available() else "cpu"


def infer_source_experiment(model_path: str) -> str | None:
    match = re.search(r"(?:^|[/_])exp[_-]?(\d+)(?:[^0-9]|$)", model_path, flags=re.IGNORECASE)
    if not match:
        match = re.search(r"pill_exp(\d+)", model_path, flags=re.IGNORECASE)
    return f"exp{match.group(1)}" if match else None


def infer_model_name(model_path: str) -> str | None:
    match = re.search(r"(yolo\d+[a-z]+|yolov\d+[a-z]+|rtdetr[\w-]*)", model_path, flags=re.IGNORECASE)
    return match.group(1) if match else None


def main() -> None:
    args = parse_args()
    start_run_logging(
        project_root=PROJECT_ROOT,
        category="inference",
        run_name=f"exp8_search_{Path(args.model).stem}",
        enabled=args.save_log,
    )
    model_path = args.model
    data_path = args.data

    device = args.device if args.device is not None else default_device()

    print(f"🚀 [Exp 8] 파라미터 동적 튜닝 (Threshold Grid Search)", flush=True)
    print(f" - 모델: {model_path}", flush=True)
    print(f" - 검증 데이터셋: {data_path}", flush=True)
    print(f" - device: {device} (cuda_available={torch.cuda.is_available()})", flush=True)
    if args.imgsz is not None:
        print(f" - imgsz: {args.imgsz}", flush=True)
    if args.batch is not None:
        print(f" - batch: {args.batch}", flush=True)
    print(
        " - 경고: 이 과정은 '모델 학습(Train)'이 아니며, 학습된 모델의 '추론 필터(Val)'를 조율하는 과정입니다.\n",
        flush=True,
    )
    
    model = YOLO(model_path)
    
    confs = parse_float_list(args.confs)
    ious = parse_float_list(args.ious)
    validate_thresholds(confs, "confs")
    validate_thresholds(ious, "ious")
    total = len(confs) * len(ious)
    
    results_list = []
    
    idx = 0
    t0 = time.perf_counter()
    for conf in confs:
        for iou in ious:
            idx += 1
            step_t0 = time.perf_counter()
            print(f"▶️ [{idx}/{total}] 테스트 진행 중 -> conf: {conf:.2f}, iou: {iou:.2f}", flush=True)
            # Validation 셋에 대해 추론 성능 평가 (학습 X)
            val_kwargs = dict(
                data=data_path,
                split="val",
                conf=conf,
                iou=iou,
                plots=False,
                verbose=args.verbose,
            )
            val_kwargs["device"] = device
            if args.imgsz is not None:
                val_kwargs["imgsz"] = args.imgsz
            if args.batch is not None:
                val_kwargs["batch"] = args.batch
            metrics = model.val(**val_kwargs)
            
            map50 = metrics.box.map50
            map95 = metrics.box.map
            
            step_s = time.perf_counter() - step_t0
            total_s = time.perf_counter() - t0
            print(
                f"   [결과] mAP@50: {map50:.4f} | mAP@50-95: {map95:.4f} (step {step_s:.1f}s, total {total_s/60:.1f}m)",
                flush=True,
            )
            results_list.append({
                "conf": conf,
                "iou": iou,
                "mAP@50": round(map50, 4),
                "mAP@50-95": round(map95, 4)
            })
            
            # VRAM 메모리 누수 방지
            del metrics
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            sys.stdout.flush()
            
    # DataFrame으로 랭킹 산출
    df = pd.DataFrame(results_list)
    df = df.sort_values(by="mAP@50-95", ascending=False).reset_index(drop=True)
    
    print("\n" + "=" * 50, flush=True)
    print("🏆 Grid Search 최종 결과 랭킹 🏆", flush=True)
    print(df.to_string(index=True), flush=True)
    print("=" * 50, flush=True)
    
    best_row = df.iloc[0]
    metrics_payload = {
        "experiment": "exp8",
        "dataset_split": "val",
        "model": model_path,
        "model_name": infer_model_name(model_path),
        "source_experiment": infer_source_experiment(model_path),
        "data": data_path,
        "epoch": None,
        "best_epoch": None,
        "best_conf": float(best_row["conf"]),
        "best_iou": float(best_row["iou"]),
        "mAP50": float(best_row["mAP@50"]),
        "mAP50-95": float(best_row["mAP@50-95"]),
        "rankings": results_list,
    }
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    metrics_path = METRICS_DIR / "exp8_val_metrics.json"
    metrics_path.write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")
    print(f"\n✅ [최종 결론] 앞으로 추론(test_custom_v12.py) 시 추천하는 설정값:", flush=True)
    print(f"👉 python src/test_custom_v12.py --model {model_path} --conf {best_row['conf']:.2f}", flush=True)
    print(
        f"   (주의: test_custom_v12.py 내에 iou={best_row['iou']:.2f} 로직이 없다면 코드에 iou값도 반영할 것)",
        flush=True,
    )
    print(f"💾 metrics saved to: {metrics_path}", flush=True)

if __name__ == "__main__":
    main()
