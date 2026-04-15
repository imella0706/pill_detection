"""
모델 1개로 이미지들을 추론해서 결과를 1개의 CSV로 저장하는 범용 inference 스크립트.
예원 test.py를 수정한 버전임. 

기본적으로는 `test_images`를 대상으로 Kaggle 제출용 CSV를 생성하는 데 사용함. 
(다만 입력 이미지 경로를 바꾸면 `validation` 이미지에 대해서도 동일한 형식의
예측 CSV를 만들 수 있음.) 
"""

import argparse
import datetime
import os
import pandas as pd
import re
import yaml
import json
import difflib 
import sys
import time
from ultralytics import YOLO
from pathlib import Path

# --- DEFAULT PATHS --- #
# 현재 파일 위치(/src/test_custom_v12.py) 기준으로 프로젝트 루트 설정
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.utils.logging_utils import start_run_logging
from src.utils.metrics_utils import collect_runtime_env, percentile_ms, write_metrics_json

DEFAULT_TEST_IMG_DIR = PROJECT_ROOT / "data" / "raw" / "sprint_ai_project1_data" / "test_images"
DEFAULT_YAML_PATH = PROJECT_ROOT / "data" / "yolo_dataset" / "dataset.yaml"
DEFAULT_JSON_DIR = PROJECT_ROOT / "data" / "raw" / "sprint_ai_project1_data" / "train_annotations"
DEFAULT_INFER_METRICS_DIR = PROJECT_ROOT / "metrics" / "infer"


def save_infer_runtime_metadata(
    run_name: str,
    model_path: str,
    output_csv: str,
    device: str = "0",
    imgsz: int | None = None,
    conf: float | None = None,
    iou: float | None = None,
    num_images: int | None = None,
    num_predictions: int | None = None,
    infer_latency_sec: list[float] | None = None,
    infer_loop_seconds: float | None = None,
    total_wall_seconds: float | None = None,
) -> Path:
    """
    추론 실행 환경 메타데이터를 저장한다.
    실무 재현성 추적용으로 train_yolo.py와 동일한 환경 필드를 기록한다.
    """
    runtime_env = collect_runtime_env(device)
    latency_sec = infer_latency_sec or []
    total_infer_sec = float(sum(latency_sec)) if latency_sec else None
    avg_latency_ms = (total_infer_sec * 1000.0 / len(latency_sec)) if latency_sec else None
    fps = (len(latency_sec) / total_infer_sec) if latency_sec and total_infer_sec and total_infer_sec > 0 else None

    payload = {
        "run_name": run_name,
        "timestamp": datetime.datetime.now().isoformat(),
        "model_path": str(model_path),
        "output_csv": str(Path(output_csv).resolve()),
        "imgsz": int(imgsz) if imgsz is not None else None,
        "conf": float(conf) if conf is not None else None,
        "iou": float(iou) if iou is not None else None,
        "num_images": int(num_images) if num_images is not None else None,
        "num_predictions": int(num_predictions) if num_predictions is not None else None,
        # Standardized inference runtime naming
        "model_inference_time_seconds": total_infer_sec,
        "end_to_end_inference_time_seconds": float(infer_loop_seconds) if infer_loop_seconds is not None else None,
        "wall_clock_time_seconds": float(total_wall_seconds) if total_wall_seconds is not None else None,
        "latency_mean_ms": float(avg_latency_ms) if avg_latency_ms is not None else None,
        "latency_p50_ms": percentile_ms(latency_sec, 50),
        "latency_p90_ms": percentile_ms(latency_sec, 90),
        "latency_p95_ms": percentile_ms(latency_sec, 95),
        "latency_p99_ms": percentile_ms(latency_sec, 99),
        "throughput_fps": float(fps) if fps is not None else None,
        "os_platform": runtime_env.get("os_platform"),
        "python_version": runtime_env.get("python_version"),
        "torch_version": runtime_env.get("torch_version"),
        "cuda_version": runtime_env.get("cuda_version"),
        "gpu_name": runtime_env.get("gpu_name"),
        "gpu_driver": runtime_env.get("gpu_driver"),
    }
    DEFAULT_INFER_METRICS_DIR.mkdir(parents=True, exist_ok=True)
    save_path = DEFAULT_INFER_METRICS_DIR / f"{run_name}_infer_runtime.json"
    write_metrics_json(payload, save_path)
    return save_path

def parse_args():
    """
    터미널 인자(CLI)를 파싱하는 함수.
    [우선순위] 1. 사용자가 직접 입력한 인자 > 2. Config YAML 파일 > 3. 아래 설정된 기본값(Default)
    """
    parser = argparse.ArgumentParser(description="Professional Inference Script for Kaggle Submission (Custom Version)")
    
    # 설정 파일 관련
    parser.add_argument("--config", type=str, default=None, help="실행 설정을 담은 yaml 파일 경로 (사용 시 아래 기본값들을 무시하고 덮어씌움)")
    
    # 모델 및 성능 관련
    parser.add_argument("--model", type=str, default=None, help="학습된 YOLO 모델 가중치 파일(*.pt) 경로 (필수)")
    parser.add_argument("--imgsz", type=int, default=1024, help="추론 시 사용할 이미지 해상도 (기본값: 1024)")
    
    # NOTE:
    # 기본값은 Ultralytics YOLO 기본 추론값(conf=0.25, iou=0.70)을 따른다.
    # 실험별 값 변경은 코드 수정이 아니라 configs/inference/*.yaml에서 조절한다.
    parser.add_argument("--conf", type=float, default=0.25, help="검출 신뢰도 임계값 (YOLO 기본값: 0.25)")
    parser.add_argument("--iou", type=float, default=0.70, help="NMS 수행 시 사용할 IoU 임계값 (YOLO 기본값: 0.70)")
    
    # 입출력 경로 관련
    parser.add_argument("--output", type=str, default="submission.csv", help="결과를 저장할 CSV 파일명")
    parser.add_argument("--data", type=str, default=str(DEFAULT_YAML_PATH), help="학습 시 사용한 dataset.yaml 경로 (클래스 이름 매핑용)")
    parser.add_argument("--test_images", type=str, default=str(DEFAULT_TEST_IMG_DIR), help="추론을 수행할 이미지들이 담긴 폴더 경로")
    parser.add_argument("--json_dir", type=str, default=str(DEFAULT_JSON_DIR), help="원본 COCO 어노테이션 폴더 (클래스 ID 역매핑을 위한 원본 정보 추출용)")
    
    # 실험 관리 (Reproducibility) 관련
    parser.add_argument(
        "--save-config",
        dest="save_config",
        action="store_true",
        help="추론 설정을 configs/ 폴더에 yaml 파일로 자동 저장할지 여부",
    )
    parser.add_argument(
        "--no-save-config",
        dest="save_config",
        action="store_false",
        help="설정 파일 자동 저장을 비활성화",
    )
    
    
    # 추론 로그는 기본 비활성화 (필요할 때만 코드에서 True로 전환)
    parser.set_defaults(save_config=True, save_log=False)
    return parser.parse_args()

def get_next_exp_id(config_dir):
    """
    configs/ 폴더를 탐색하여 중복되지 않는 다음 실험 번호(ID)를 자동으로 찾는 함수.
    예: exp0, exp1이 있으면 2를 반환함.
    """
    if not os.path.exists(config_dir):
        os.makedirs(config_dir)
        return 0
    
    existing_nums = []
    for f in os.listdir(config_dir):
        nums = re.findall(r'exp(\d+)', f)
        if nums:
            existing_nums.append(int(nums[0]))
    
    return max(existing_nums) + 1 if existing_nums else 0

def save_inference_config(args, next_id):
    """
    현재 실행에 사용된 모든 하이퍼파라미터와 경로 설정을 configs/ 폴더에 yaml로 저장하는 함수.
    나중에 이 파일을 --config 인자로 넘기면 동일한 설정을 완벽하게 재현 가능함.
    """
    config_dir = PROJECT_ROOT / "configs"
    filename = f"exp{next_id}_inference.yaml"
    save_path = config_dir / filename
    
    config_data = vars(args).copy()
    # Remove config itself from the saved file to avoid recursion
    if 'config' in config_data:
        del config_data['config']
        
    with open(save_path, 'w', encoding='utf-8') as f:
        yaml.dump(config_data, f, default_flow_style=False, allow_unicode=True)
    
    print(f"📝 [Auto-Guide] Inference config saved to: {save_path}")
    return save_path

def run_test_and_save_csv():
    """
    전체 추론 프로세스를 관리하는 메인 실행 함수.
    1. 인자 파싱 및 설정 로드
    2. 클래스 ID 역매핑 (YOLO -> 오리지널 COCO ID)
    3. 모델 로드 및 배치 추론
    4. 결과를 Kaggle 양식의 CSV로 저장
    """
    cli_args = parse_args()
    total_start = time.perf_counter()
    
    # --- Config Loading Logic --- #
    if cli_args.config:
        print(f"Loading config from: {cli_args.config}")
        with open(cli_args.config, 'r', encoding='utf-8') as f:
            config_data = yaml.safe_load(f)
        
        # Override CLI args with config file values (unless explicitly provided in CLI)
        # For simplicity, we prioritize the config file if --config is passed
        for key, value in config_data.items():
            setattr(cli_args, key, value)

    run_name = Path(cli_args.output).stem if getattr(cli_args, "output", None) else "inference"
    start_run_logging(
        project_root=PROJECT_ROOT,
        category="inference",
        run_name=run_name,
        enabled=getattr(cli_args, "save_log", True),
    )
    
    if cli_args.model is None:
        print("Error: --model is required (either via CLI or --config file)")
        return

    # Set derived variables
    MODEL_PATH = cli_args.model
    TEST_IMG_DIR = cli_args.test_images
    OUTPUT_CSV = cli_args.output
    YAML_PATH = cli_args.data
    JSON_DIR = cli_args.json_dir

    # ---------------------------------------------------------
    # 2. 스마트 역매핑 딕셔너리 만들기 (difflib 유사도 검사)
    # ---------------------------------------------------------
    # A. YAML 파일 읽기
    with open(YAML_PATH, 'r', encoding='utf-8') as f:
        yolo_names = yaml.safe_load(f)['names']
        
    json_category_map = {} 
    original_json_names = {} 

    print(f"🔍 '{JSON_DIR}' 폴더 탐색을 시작합니다.")
    
    for root, dirs, files in os.walk(JSON_DIR):
        for file in files:
            if file.endswith('.json'):
                try:
                    with open(os.path.join(root, file), 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        if 'categories' in data:
                            for cat in data['categories']:
                                clean_name = str(cat['name']).replace(" ", "")
                                json_category_map[clean_name] = int(cat['id'])
                                original_json_names[clean_name] = cat['name']
                except:
                    pass
    
    print(f" JSON에서 총 {len(json_category_map)}개의 고유 알약 카테고리를 찾아냈습니다.\n")

    # C. 스마트 매핑 시작
    inverse_class_map = {}
    unmapped_classes = []

    for yolo_id, yaml_name in yolo_names.items():
        clean_yaml = str(yaml_name).replace(" ", "")
        
        if clean_yaml in json_category_map:
            inverse_class_map[int(yolo_id)] = json_category_map[clean_yaml]
        else:
            possible_matches = difflib.get_close_matches(clean_yaml, json_category_map.keys(), n=1, cutoff=0.5)
            if possible_matches:
                best_match = possible_matches[0]
                matched_id = json_category_map[best_match]
                inverse_class_map[int(yolo_id)] = matched_id
                print(f" 자동 보정 성공: '{yaml_name}' -> JSON의 '{original_json_names[best_match]}' (ID: {matched_id})")
            else:
                unmapped_classes.append(yaml_name)

    if not unmapped_classes:
        print("\n 모든 알약의 클래스 매핑이 완벽하게 완료되었습니다!\n")

    # ---------------------------------------------------------
    # 3. 모델 로드 및 추론
    # ---------------------------------------------------------
    model = YOLO(MODEL_PATH)
    
    if not os.path.exists(TEST_IMG_DIR):
        raise FileNotFoundError(f"Test image directory not found: {TEST_IMG_DIR}")

    image_files = [f for f in os.listdir(TEST_IMG_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    image_files.sort(key=lambda x: int("".join(re.findall(r'\d+', x))))
    
    results_list = []
    ann_id_counter = 1 

    print(f" 총 {len(image_files)}장의 이미지에 대해 검출을 시작합니다.")
    infer_loop_start = time.perf_counter()
    infer_latency_sec: list[float] = []
    for idx, img_name in enumerate(image_files):
        if idx % 100 == 0:
            print(f"   - Processing image {idx}/{len(image_files)}...")
            
        img_path = os.path.join(TEST_IMG_DIR, img_name)
        image_id_str = "".join(re.findall(r'\d+', img_name))
        image_id = int(image_id_str) if image_id_str else 0
            
        t0 = time.perf_counter()
        outputs = model.predict(source=img_path, conf=cli_args.conf, iou=cli_args.iou, imgsz=cli_args.imgsz, verbose=False)
        infer_latency_sec.append(time.perf_counter() - t0)
        
        for r in outputs:
            boxes = r.boxes
            for box in boxes:
                yolo_cls_id = int(box.cls[0]) 
                score = float(box.conf[0])
                real_category_id = inverse_class_map.get(yolo_cls_id, -1)
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                
                results_list.append({
                    'annotation_id': ann_id_counter,
                    'image_id': image_id,
                    'category_id': real_category_id,
                    'bbox_x': int(x1),
                    'bbox_y': int(y1),
                    'bbox_w': int(x2 - x1),
                    'bbox_h': int(y2 - y1),
                    'score': round(score, 3)
                })
                ann_id_counter += 1

    df = pd.DataFrame(results_list)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\n✅ 분석 완료! 파일 저장됨: {os.path.abspath(OUTPUT_CSV)}")
    infer_loop_seconds = time.perf_counter() - infer_loop_start
    total_wall_seconds = time.perf_counter() - total_start

    runtime_meta_path = save_infer_runtime_metadata(
        run_name=run_name,
        model_path=MODEL_PATH,
        output_csv=OUTPUT_CSV,
        device=str(getattr(cli_args, "device", "0")),
        imgsz=cli_args.imgsz,
        conf=cli_args.conf,
        iou=cli_args.iou,
        num_images=len(image_files),
        num_predictions=len(results_list),
        infer_latency_sec=infer_latency_sec,
        infer_loop_seconds=infer_loop_seconds,
        total_wall_seconds=total_wall_seconds,
    )
    print(f"📊 추론 런타임 메타 저장됨: {runtime_meta_path}")
    
    # --- Auto Logging Logic --- #
    if cli_args.save_config and not cli_args.config:
        next_id = get_next_exp_id(PROJECT_ROOT / "configs")
        save_inference_config(cli_args, next_id)

if __name__ == '__main__':
    run_test_and_save_csv()
