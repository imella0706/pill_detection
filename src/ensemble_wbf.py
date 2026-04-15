"""
test_custom_v12.py를 통해 만든 여러 개의 예측 CSV 파일을 Weighted Box Fusion(WBF)으로 하나의 최종 제출 파일로 병합하는 스크립트.

이 스크립트는 입력 CSV들이 동일한 submission 스키마를 따르는 경우,
다음과 같은 다양한 앙상블 실험에서 재사용할 수 있습니다:
- 하나의 모델로 생성한 multi-scale 예측
- 동일 모델의 서로 다른 seed 예측
- 서로 다른 모델 계열의 예측
"""

import os
import pandas as pd
import cv2
import re
from ensemble_boxes import weighted_boxes_fusion
import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.utils.logging_utils import start_run_logging
from src.utils.project_paths import TEST_IMAGES_DIR

def get_image_sizes(image_dir):
    sizes = {}
    for img_name in os.listdir(image_dir):
        if not img_name.lower().endswith(('.png', '.jpg', '.jpeg')): continue
        image_id_str = "".join(re.findall(r'\d+', img_name))
        if not image_id_str: continue
        image_id = int(image_id_str)
        img_path = os.path.join(image_dir, img_name)
        img = cv2.imread(img_path)
        if img is not None:
            h, w = img.shape[:2]
            sizes[image_id] = (w, h)
    return sizes


def resolve_image_dir(csv_paths, image_dir):
    if image_dir:
        return image_dir
    # Auto-detect based on csv filename hints.
    if 'val' in csv_paths[0]:
        return 'data/datasets/yolo/copy_paste/cp_t20_data_seed42/images/val'
    return str(TEST_IMAGES_DIR)

def parse_args():
    parser = argparse.ArgumentParser(
        description="Apply Weighted Box Fusion (WBF) to multiple prediction CSV files."
    )
    parser.add_argument(
        '--csvs',
        nargs='+',
        required=True,
        help='Prediction CSV files to fuse (same schema required)',
    )
    parser.add_argument(
        '--weights',
        nargs='+',
        type=float,
        help='Optional per-CSV weights; defaults to equal weighting',
    )
    parser.add_argument('--iou', type=float, default=0.7, help='IoU threshold for WBF')
    parser.add_argument(
        '--output',
        type=str,
        default='submission/wbf_submission.csv',
        help='Output path for the fused submission CSV',
    )
    parser.add_argument(
        '--image_dir',
        type=str,
        default=None,
        help='Image directory used for box normalization (test or val).',
    )
    parser.add_argument(
        '--save-log',
        dest='save_log',
        action='store_true',
        help='Save runtime log to logs/inference',
    )
    parser.add_argument(
        '--no-save-log',
        dest='save_log',
        action='store_false',
        help='Disable runtime log file saving',
    )
    parser.set_defaults(save_log=True)
    return parser.parse_args()

def main():
    args = parse_args()
    start_run_logging(
        project_root=PROJECT_ROOT,
        category="inference",
        run_name=Path(args.output).stem,
        enabled=args.save_log,
    )
    csv_paths = args.csvs
    weights = args.weights
    if weights is None:
        weights = [1.0] * len(csv_paths)
    
    print(f" Ensembling {len(csv_paths)} CSVs: {csv_paths}")
    df_list = [pd.read_csv(f) for f in csv_paths]
    
    # 앙상블 전 총 박스 수 확인
    total_boxes = sum(len(df) for df in df_list)
    print(f" 병합 전 총 Bounding Box 수: {total_boxes}")

    args.image_dir = resolve_image_dir(csv_paths, args.image_dir)
    print(f" Reading image sizes from {args.image_dir} (WBF normalization requires W/H)...")
    img_sizes = get_image_sizes(args.image_dir)
    
    if not img_sizes:
        raise ValueError(f"No images found in {args.image_dir}. WBF cannot normalize coordinates without valid width/height.")
    
    all_image_ids = set()
    for df in df_list:
        all_image_ids.update(df['image_id'].unique())
        
    final_results = []
    ann_id_counter = 1
    
    print(" Applying Weighted Boxes Fusion...")
    for img_id in sorted(list(all_image_ids)):
        if img_id not in img_sizes:
            continue
        w, h = img_sizes[img_id]
        
        boxes_list = []
        scores_list = []
        labels_list = []
        
        # 모델별 예측값 모으기
        for df in df_list:
            df_img = df[df['image_id'] == img_id]
            boxes = []
            scores = []
            labels = []
            for _, row in df_img.iterrows():
                bx, by, bw, bh = row['bbox_x'], row['bbox_y'], row['bbox_w'], row['bbox_h']
                
                # YOLO->WBF 좌표 정규화 (0.0 ~ 1.0)
                x1 = bx / w
                y1 = by / h
                x2 = (bx + bw) / w
                y2 = (by + bh) / h
                
                # Bounding Box가 화면을 벗어나지 않도록 클리핑 보호 (Robustness)
                x1 = max(0.0, min(1.0, x1))
                y1 = max(0.0, min(1.0, y1))
                x2 = max(0.0, min(1.0, x2))
                y2 = max(0.0, min(1.0, y2))
                
                boxes.append([x1, y1, x2, y2])
                scores.append(float(row['score']))
                labels.append(int(row['category_id']))
                
            boxes_list.append(boxes)
            scores_list.append(scores)
            labels_list.append(labels)
            
        # WBF 수행 (빈 박스 무시 기준 0.0)
        f_boxes, f_scores, f_labels = weighted_boxes_fusion(
            boxes_list, scores_list, labels_list, 
            weights=weights, iou_thr=args.iou, skip_box_thr=0.0
        )
        
        # WBF->YOLO 좌표 역정규화 (원상복구)
        for i in range(len(f_boxes)):
            x1, y1, x2, y2 = f_boxes[i]
            x1 = x1 * w
            y1 = y1 * h
            x2 = x2 * w
            y2 = y2 * h
            
            box_w = x2 - x1
            box_h = y2 - y1
            
            final_results.append({
                'annotation_id': ann_id_counter,
                'image_id': img_id,
                'category_id': int(f_labels[i]),
                'bbox_x': int(round(x1)),
                'bbox_y': int(round(y1)),
                'bbox_w': int(round(box_w)),
                'bbox_h': int(round(box_h)),
                'score': round(float(f_scores[i]), 3)
            })
            ann_id_counter += 1
            
    final_df = pd.DataFrame(final_results)
    final_df.to_csv(args.output, index=False)
    print(f"\n WBF ensemble completed! Saved to {args.output}")
    print(f" 최종 앙상블 완료된 Bounding Box 수: {len(final_df)} (노이즈 박스 감소 및 통합 성공!)")

if __name__ == "__main__":
    main()
