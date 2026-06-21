# Pill Detection Experiment Log

## 📑 지표 정의 및 측정 이유 (Metric Definitions & Rationale)

1.  **mAP@50 (Detection 성공률)**: **"일단 찾기는 했는가?"** 
    *   가장 기본적이고 너그러운 기준임. 알약의 존재를 놓치지는 않았는지를 판단함 
2.  **mAP@75 (Localization 정밀도)**: **"칼같이 찾았는가?"** 
    *   알약 테두리를 얼마나 정확하게 감쌌는지를 봄. 초급 프로젝트처럼 **알약의 각인(문자/로고)을 분석해야 하는 작업**이 뒤따른다면, 박스 오차가 작아야 글자가 잘리지 않고 정확한 OCR이 가능해짐
3.  **mAP@50-95 (종합 실력)**: **"전체적인 품질이 어떤가?"** 
    *   너그러운 기준(@50)부터 아주 깐깐한 기준(@95)까지의 평균임. 모델의 전반적인 완성도를 나타내는 가장 신뢰할 만한 점수임.
4.  **Best Epoch (최적화 지점)**: **"언제 더 이상 학습할 필요가 없었는가?"** 
    *   최대 성능을 낸 시점을 알아야 불필요한 과적합(Overfitting) 여부를 판단하고 학습 속도를 조절할 수 있음. 


## 📌 Pill Detection Experiments Log

> [!IMPORTANT]
> **중대 기술 부채 발견 및 상환 (2026-03-26)**: 
> Exp 1~10까지의 실험 과정에서 아마도 **9건의 학습 데이터 오염**이 존재했을 것으로 추정됨을 10단계까지 한 후 알게 됨.  
> - **현상**: 8장의 라벨 누락(결측) 및 1건의 Invalid BBox(x=6567).
> - **영향**: 파이프라인의 `clip01()` 함수가 쓰레기 좌표를 화면 끝(`x_center=1.0`)에 강제 고정하여 모델에게 왜곡된 피처를 학습시킴.
> - **조치**: 
  1. 예원 브랜치(`pred/yewon`)로부터 `git restore` 명령어를 통해 원본 JSON 데이터 복구.
```
git switch model/exp-sujin
git restore --source=pred/yewon train_annotations
```
  2. `python preprocessing.py`를 실행하여 중간 산출물(`merged_annotations.json`)에 수정 사항 반영 (Report에서 `invalid bboxes: 0` 확인).
  3. `python prepare_yolo_dataset.py`를 실행하여 최종 YOLO TXT 레이블셋 최신화 및 `dataset.yaml` 업데이트.
> - **결과**: Exp 11부터는 **'Cleaned Baseline'**으로 명명하며 모든 지표의 신뢰성을 재확보함.

> [!NOTE]
> **데이터 경계 분리 반영 (2026-04-14)**:
> - 재현성과 계보 추적 강화를 위해 데이터 레이어를 분리함.
> - `data/raw/`: 원본 이미지(`train_images`, `test_images`) 보존
> - `data/curated/`: 정제 완료 어노테이션(`train_annotations`) 관리
> - `preprocessing.py`는 curated 어노테이션을 입력으로 사용하도록 경로를 갱신함.
> - `data/raw/`는 원본 데이터 보관 영역이다.
> - 학습/전처리/데이터셋 생성에서 사용하는 annotation source of truth는 `data/curated/train_annotations`다.
> - `data_version`은 `data/curated/train_annotations` 기준으로 계산한다.

> [!NOTE]
> **raw/curated 분리 근거 (2026-04-14)**:
> - 정제로 실제 변경되는 대상은 어노테이션(`train_annotations`)이므로, 변경된 데이터만 `curated`에서 관리함.
> - 이미지는 원본 기준점(`data/raw/train_images`)으로 고정해, 원본-정제 경계를 명확히 유지함.
> - 이미지까지 `curated`에서 읽으면 원본/정제 경계가 흐려지고, 추후 이미지 변경 감지가 늦어질 수 있음.
> - 대용량 이미지 중복 저장을 줄여 저장소/복사 비용을 절감함.

## 🧪 실험 목록 (Experiment Table)
| ID | 실험명 | 변경 사항 (Strategy) | Backbone | Size | Epoch | Seed | Opt | C | I | P | R | F1 | mAP@.5 | mAP@.75 | mAP@.5:.95 | Best | Train(min) | Eff(K/min) | 인사이트 및 결과 |
|:---:|:---:|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---|
| **Exp 1~10**| **Polluted-Sets** | **데이터 오염 상태에서의 실험들 (잠정 지표)** | - | - | - | 0 중심 / 42·123·777 포함 | - | - | - | - | - | - | - | - | - | - | - | - | 데이터 무결성 결함 잔존했을 것으로 추정됨 |
| **캐글** | **캐글용** | Stratified Split/욜로기본증강설정 | v8n | 640 | 50 | 50 | - | - | - | - | - | - | 0.79 | - | - | - | - | - | Kaggle: 0.70166 |
| **Ref 1**| **예원** | Random Split / 욜로기본증강설정 | v8n | 640 | 50 | - | - | - | - | **0.790** | 0.760 | - | - | - | - | - | - | - | 예원님 결과 |
| **Exp 0** | **YOLOv8n** | Random Split / 욜로기본증강설정 | v8n | 640 | 50 | 0 | auto | - | - | 0.959 | 0.935 | - | **0.950** | - | - | 50 | 2.00 | - | • 극단적 데이터 누수 발생 / 캐글 베이스라인 로컬 재현 |
| **Exp 1-1** | **Stratified** | YOLO 기본증강 + Stratified | v8n | 640 | 20 | 0(추정) | - | - | - | 0.402 | 0.384 | - | **0.395** | - | - | 20 | 2.14 | - | 계층적 분할 시작 |
| **Exp 1-2**| **Stratified** | YOLO 기본증강 + Stratified | v8n | 640 | 50 | 0 | auto | - | - | **0.960** | 0.943 | - | 0.945 | - | - | **36** | - | - | 누수 잔존 확인 |
| **Exp 2** | **v11s Pivot** | YOLO 기본증강 + Stratified | **v11s** | 640 | 50 | 0 | auto | - | - | **0.994** | **0.990** | - | **0.995** | - | - | **50** | 3.08 | - | 모델 교체 |
| **Exp 3** | **Hi-Res** | YOLO 기본증강 + Stratified | **v11s** | 960 | 50 | 0 | auto | - | - | **0.995** | **0.990** | - | **0.995** | - | - | **50** | 6.30 | - | 해상도 증량 |
| **Exp 4** | **Flip-Off** | YOLO 기본증강 (Flip Off) | **v11s** | 960 | 50 | 0 | auto | - | - | **0.995** | **0.992** | - | **0.995** | - | - | **50** | 6.06 | - | 각인 보호 효과 |
| **Exp 5** | **Custom CP** | YOLO 기본증강 (Flip Off) + 커스텀 CP | **v11s** | 960 | 50 | 0 | auto | .25 | .70 | **0.995** | **0.993** | - | **0.995** | - | - | **50** | 13.01 | 0.074 | **Kaggle 0.968** (YOLO 내장 copy_paste옵션 아님) |
| **Exp 6** | **Rotation** | YOLO 기본 (Flip Off) + CP + Rotation | **v11s** | 960 | 50 | 0 | auto | - | - | 0.990 | 0.940 | - | 0.995 | - | - | 50 | 13.17 | 0.065 | **Kaggle 0.854** |
| **Exp 7** | **Final-Res** | YOLO 기본 (Flip Off) + CP (No Rot) | **v11s** | 1024| 50 | 0 | auto | - | - | **0.995** | **0.993** | - | **0.995** | - | - | **50** | 14.48 | - | 성능 정체 |
| **Exp 8** | **Dynamic NMS** | Exp5(best) 기준 NMS 튜닝 | v11s | 1024 | - | 0(Exp5 기반) | - | .20 | .60 | 0.9941 | 0.9926 | - | 0.9941 | - | - | - | - | - | Kaggle: **0.96670**, iou=0.60 튜닝 결과 |
| **Exp 9** | **Multi-scale WBF** | Exp5 `best.pt` 기반 640/960/1024 추론 + WBF | v11s | Mix | - | 0(Exp5 기반) | - | .20 | .60 | 0.9932 | 0.9896 | - | 0.9932 | - | - | - | - | - | Kaggle: **0.97243** (최고점) |
| **Exp 10-1**| **Seed 42** | Exp 5 (Seed 42) | v11s | 960 | 50 | **42** | auto | .25 | .70 | 0.9949 | 0.9942 | - | 0.9949 | 0.9950 | 0.9942 | 50 | 13.00 | - | 개별 시드 훈련 1 |
| **Exp 10-2**| **Seed 123** | Exp 5 (Seed 123) | v11s | 960 | 50 | **123** | auto | .25 | .70 | 0.9950 | 0.9904 | - | 0.9950 | 0.9950 | 0.9904 | 50 | 12.96 | - | 개별 시드 훈련 2 |
| **Exp 10-3**| **Seed 777** | Exp 5 (Seed 777) | v11s | 960 | 50 | **777** | auto | .25 | .70 | 0.9944 | 0.9932 | - | 0.9944 | 0.9950 | 0.9932 | 50 | 13.03 | - | 개별 시드 훈련 3 |
| **Exp 10(F)**| **3-Seed Ens**| Exp 10-1~3 WBF 앙상블 | v11s | 960 | - | 42/123/777 | - | .25 | .60 | **0.9942** | **0.9927** | - | **0.9942** | 0.9942 | **0.9927** | - | - | - | **Kaggle: 0.98073 (현재 최고점!!)** |
| **Exp 11** | **Dirty-Aug** | Exp 5 Clean (fliplr: 0.5 오설정) | v11s | 960 | 50 | 0 | auto | .25 | .70 | 0.9696 | 0.9696 | 0.9696 | 0.9931 | 0.9950 | 0.9899 | 50 | 13.70 | 0.070 | **[기각]** 변인 통제 실패 (Kaggle: 0.95910) |
| **Exp 12** | **Cleaned-Base** | **Exp 5 Clean (fliplr: 0.0 복구)** | v11s | 960 | 50 | 0 | auto | .25 | .70 | **0.9740** | **0.9649** | **0.9694** | **0.9946** | **0.9950** | **0.9933** | 50 | 13.40 | 0.072 |  (Kaggle: 0.96528) |
| **Exp 13** | **Baseline-1.0 Rebuild** | 베이스라인 1.0재현 실패(fliplr: 0.0 오설정) | v8n | 640 | 50 | 42 | auto | .25 | .70 | 0.9362 | 0.9887 | 0.9617 | 0.9890 | 0.9950 | 0.9800 | 50 | 4.36 | 0.216 | Kaggle: **0.94032** |
| **Exp 14** | **Baseline-1.0 Rebuild** | 베이스라인 2.0 재현(`fliplr: 0.5` 복구) | v8n | 640 | 50 | 42 | auto(→AdamW) | .25 | .70 | 0.9485 | 0.9918 | 0.9697 | 0.9878 | 0.9950 | 0.9717 | 50 | 4.29 | 0.219 | Kaggle: **0.93808** |
| **Exp 15** | **Baseline-2.0 Canonical** | Exp12 applied 값(AdamW) 고정 + seed42 기준 팀원 공유용 베이스라인 정리 | v11s | 960 | 50 | 42 | AdamW | .25 | .70 | 0.9704 | 0.9761 | 0.9733 | 0.9918 | 0.9950 | 0.9897 | 50 | 14.11 | 0.068 | Kaggle: **0.96455** |
| **Exp 16** | **Res-640 Sweep** | Exp15 고정값 유지 + 해상도만 `960→640` 변경 (증강 파라미터 동결) | v11s | 640 | 50 | 42 | AdamW | .25 | .70 | 0.9690 | 0.9861 | 0.9774 | 0.9916 | 0.9950 | 0.9892 | 50 | 6.83 | 0.142 | Kaggle: **0.96723** |
| **Exp 17** | **Res-1024 Sweep** | Exp15 고정값 유지 + 해상도만 `960→1024` 변경 (증강 파라미터 동결) | v11s | 1024 | 50 | 42 | AdamW | .25 | .70 | 0.9692 | 0.9863 | 0.9777 | 0.9929 | 0.9950 | 0.9900 | 50 | 14.90 | 0.065 | Kaggle: **0.97292** |
| **Exp 18** | **Res-1280 Benchmark** | Exp15 고정값 유지 + 해상도만 `960→1280` 변경 (증강 파라미터 동결) | v11s | 1280 | 50 | 42 | AdamW | .25 | .70 | 0.9706* | 0.9634* | 0.9670* | 0.9909* | - | 0.9890* | 50 | 112.21 | - | final val OOM / inference 완료 / Kaggle: **TBD** |
| **Exp 19-A** | **No-HSVH** | Exp15 파생 실험: `hsv_h 0.015→0.0` 단일 변수 검증 | v11s | 960 | 50 | 42 | AdamW | .25 | .70 | 0.9631 | 0.9642 | 0.9636 | 0.9929 | 0.9950 | 0.9920 | 50 | 13.44 | 0.074 | inference 완료 / Kaggle: **0.96678** |
| **Exp 19-B** | **Mosaic-0.5** | Exp15 파생 실험: `mosaic 1.0→0.5` 단일 변수 검증 | v11s | 960 | 50 | 42 | AdamW | .25 | .70 | 0.9765 | 0.9617 | 0.9690 | 0.9940 | 0.9950 | 0.9920 | 50 | 13.30 | 0.075 | inference 완료 / Kaggle: **TBD** |
| **Exp 19-C** | **No-Translate** | Exp15 파생 실험: `translate 0.1→0.0` 단일 변수 검증 | v11s | 960 | 50 | 42 | AdamW | .25 | .70 | 0.9543 | 0.9712 | 0.9626 | 0.9939 | 0.9950 | 0.9909 | 50 | 13.35 | 0.074 | inference 완료 / Kaggle: **TBD** |
| **Exp 19-D** | **Scale-0.3** | Exp15 파생 실험: `scale 0.5→0.3` 단일 변수 검증 | v11s | 960 | 50 | 42 | AdamW | .25 | .70 | 0.9710 | 0.9689 | 0.9699 | 0.9909 | 0.9950 | 0.9893 | 50 | 13.35 | 0.074 | inference 완료 / Kaggle: **TBD** |
| **Exp 20-H1** | **HSVH-0.005** | `hsv_h` 국소 탐색(`0.015→0.005`) | v11s | 960 | 50 | 42 | AdamW | .25 | .70 | 0.9509 | 0.9877 | 0.9690 | 0.9915 | 0.9950 | 0.9895 | 50 | 13.50 | 0.073 | train/inference 완료 / Kaggle: **TBD** |
| **Exp 20-H2** | **HSVH-0.010** | `hsv_h` 국소 탐색(`0.015→0.010`) | v11s | 960 | 50 | 42 | AdamW | .25 | .70 | 0.9655 | 0.9649 | 0.9652 | 0.9934 | 0.9950 | 0.9915 | 50 | 13.28 | 0.075 | train/inference 완료 / Kaggle: **보류(로컬 기준 후보)** |
| **Exp 20-H3** | **HSVH-0.020** | `hsv_h` 상한 가드레일 검증(`0.015→0.020`) | v11s | 960 | 50 | 42 | AdamW | .25 | .70 | 0.9709 | 0.9667 | 0.9688 | 0.9935 | 0.9950 | 0.9919 | 50 | 13.35 | 0.074 | train/inference 완료 / Kaggle: **0.97154** |
| **Exp 20-H4** | **HSVH-0.030** | `hsv_h` 상한 가드레일 검증(`0.015→0.030`) | v11s | 960 | 50 | 42 | AdamW | .25 | .70 | 0.9656 | 0.9625 | 0.9641 | 0.9936 | 0.9950 | 0.9886 | 50 | 13.38 | 0.074 | train/inference 완료 / Kaggle: **TBD** |
| **Exp 20-H5** | **HSVH-0.018** | `hsv_h` 피크 구간 좌측 미세탐색(`0.015→0.018`) | v11s | 960 | 50 | 42 | AdamW | .25 | .70 | 0.9722 | 0.9606 | 0.9664 | 0.9927 | 0.9950 | 0.9911 | 50 | 14.02 | 0.071 | train 완료 / inference 대기 / Kaggle: **TBD** |
| **Exp 20-H6** | **HSVH-0.022** | `hsv_h` 피크 구간 우측 미세탐색(`0.015→0.022`) | v11s | 960 | 50 | 42 | AdamW | .25 | .70 | 0.9694 | 0.9669 | 0.9682 | 0.9919 | 0.9950 | 0.9901 | 50 | 14.02 | 0.071 | train 완료 / inference 대기 / Kaggle: **TBD** |
| **Exp 20-H7** | **HSVH-0.025** | `hsv_h` 상한 구간 미세탐색(`0.015→0.025`) | v11s | 960 | 50 | 42 | AdamW | .25 | .70 | 0.9726 | 0.9749 | 0.9738 | 0.9934 | 0.9950 | 0.9911 | 50 | 13.89 | 0.072 | train 완료 / inference 대기 / Kaggle: **TBD** |
| **Exp 21-M0** | **Mosaic-0.0** | `mosaic` 경계값 검증(`1.0→0.0`) | v11s | 960 | 50 | 42 | AdamW | .25 | .70 | 0.9659 | 0.9707 | 0.9683 | 0.9945 | 0.9950 | 0.9916 | 50 | 13.85 | 0.072 | train/inference 완료 / Kaggle: **0.96894** |
| **Exp 21-M1** | **Mosaic-0.4** | `mosaic` 국소 탐색(`0.5→0.4`) | v11s | 960 | 50 | 42 | AdamW | .25 | .70 | 0.9727 | 0.9622 | 0.9674 | 0.9918 | 0.9950 | 0.9897 | 50 | 14.11 | 0.071 | train 완료 / inference 대기 / Kaggle: **TBD** |
| **Exp 21-M2** | **Mosaic-0.6** | `mosaic` 국소 탐색(`0.5→0.6`) | v11s | 960 | 50 | 42 | AdamW | .25 | .70 | 0.9747 | 0.9648 | 0.9697 | 0.9939 | 0.9950 | 0.9920 | 50 | 13.69 | 0.073 | train/inference 완료 / Kaggle: **0.96154** |
| **Exp 22-H3M0** | **H3+M0 Combo** | `hsv_h=0.020(H3)` + `mosaic=0.0(M0)` 조합 검증 | v11s | 960 | 50 | 42 | AdamW | .25 | .70 | 0.9753 | 0.9630 | 0.9691 | 0.9935 | 0.9950 | 0.9915 | 50 | 13.48 | 0.072 | train/inference 완료 / Kaggle: **0.97199** |
| **Exp 23-C1** | **CLAHE-15** | `Exp22-H3M0` 고정 + CLAHE 데이터셋(`seed_42_cl15`) | v11s | 960 | 50 | 42 | AdamW | .25 | .70 | 0.9683 | 0.9643 | 0.9663 | 0.9948 | 0.9950 | **0.9941** | 50 | 13.48 | - | train/inference 완료 / Kaggle: **0.96435** |
| **Exp 23-C2** | **CLAHE-20** | `Exp22-H3M0` 고정 + CLAHE 데이터셋(`seed_42_cl20`) | v11s | 960 | 50 | 42 | AdamW | .25 | .70 | 0.9548 | 0.9801 | **0.9673** | 0.9931 | 0.9950 | 0.9914 | 50 | 13.25 | - | train/inference 완료 / Kaggle: **0.94796** |
| **Exp 23-C3** | **CLAHE-25** | `Exp22-H3M0` 고정 + CLAHE 데이터셋(`seed_42_cl25`) | v11s | 960 | 50 | 42 | AdamW | .25 | .70 | 0.9347 | **0.9926** | 0.9627 | 0.9946 | 0.9950 | 0.9928 | 50 | 13.23 | - | train 완료 / inference 대기 / Kaggle: **TBD** |
| **Exp 24-V1** | **HSVV-0.0** | `Exp22-H3M0` 고정 + `hsv_v 0.4→0.0` | v11s | 960 | 50 | 42 | AdamW | .25 | .70 | 0.9749 | 0.9641 | 0.9695 | 0.9935 | 0.9950 | 0.9906 | 50 | 13.52 | - | train 완료 / inference 대기 / Kaggle: **TBD** |
| **Exp 24-V2** | **HSVV-0.2** | `Exp22-H3M0` 고정 + `hsv_v 0.4→0.2` | v11s | 960 | 50 | 42 | AdamW | .25 | .70 | 0.9559 | 0.9854 | 0.9704 | 0.9943 | 0.9950 | **0.9928** | 50 | 13.33 | - | train/inference 완료 / Kaggle: **0.96308** |
| **Exp 24-V3** | **HSVV-0.5** | `Exp22-H3M0` 고정 + `hsv_v 0.4→0.5` | v11s | 960 | 50 | 42 | AdamW | .25 | .70 | 0.9600 | 0.9818 | **0.9708** | 0.9945 | 0.9950 | 0.9922 | 50 | 13.42 | - | train 완료 / inference 대기 / Kaggle: **TBD** |
| **Exp 25-LB1** | **LossGain-Box** | `Exp22-H3M0` 고정 + `box 7.5→8.5` 단일 변수 | v11s | 960 | 50 | 42 | AdamW | .25 | .70 | 0.9730 | 0.9632 | 0.9681 | 0.9932 | 0.9950 | 0.9919 | 50 | 14.04 | - | train/inference 완료 / Kaggle: **TBD** |
| **Exp 25-LD1** | **LossGain-DFL** | `Exp22-H3M0` 고정 + `dfl 1.5→2.0` 단일 변수 | v11s | 960 | 50 | 42 | AdamW | .25 | .70 | 0.9787 | 0.9599 | 0.9692 | 0.9944 | 0.9950 | **0.9920** | 50 | 13.89 | - | train/inference 완료 / Kaggle: **TBD** |
| **Exp 25-LBD1** | **LossGain-Box+DFL** | `Exp22-H3M0` 고정 + `(box,dfl)=(8.5,2.0)` 조합 | v11s | 960 | 50 | 42 | AdamW | .25 | .70 | 0.9741 | **0.9856** | **0.9798** | 0.9941 | 0.9950 | **0.9923** | 50 | 13.84 | - | train/inference 완료 / Kaggle: **제출 완료(Exp22 대비 하락, 수치 입력 대기)** |
| **Exp 26** | **Exp22 Multi-scale WBF** | `Exp22-H3M0 best.pt` 단일 가중치로 `640/960/1024` 추론 후 WBF(`WBF IoU=0.70`) | v11s | Mix | - | 42(Exp22 기반) | - | .25 | .70 | - | - | - | - | - | - | - | - | - | inference 전용(submission-only) / Kaggle: **0.97345** (**vs Exp22 +0.00146**) |
| **Exp 27-S123** | **Exp22 Seed-123 재학습** | `Exp22-H3M0` 완전 동일 + `seed=123` | v11s | 960 | 50 | 123 | AdamW | .25 | .70 | 0.9369 | 0.9753 | 0.9557 | 0.9875 | 0.9950 | 0.9851 | 50 | 13.93 | - | train 완료 / inference 완료 / Kaggle: **TBD** |
| **Exp 27-S777** | **Exp22 Seed-777 재학습** | `Exp22-H3M0` 완전 동일 + `seed=777` | v11s | 960 | 50 | 777 | AdamW | .25 | .70 | 0.9801 | 0.9653 | 0.9727 | 0.9934 | 0.9950 | 0.9923 | 50 | 13.90 | - | train 완료 / inference 완료 / Kaggle: **TBD** |
| **Exp 27(F)** | **Exp22 3-Seed WBF 재현(Exp10 방식)** | `seed 42/123/777` 3개 예측(960) WBF | v11s | 960 | - | 42/123/777 | - | .25 | .70 | - | - | - | - | - | - | - | - | - | 완료 / Kaggle: **0.98077** (**vs Exp22 +0.00878, vs Exp26 +0.00732**) |

> `*` Exp18의 P/R/F1/mAP는 `results.csv`의 50epoch 행 값 기준(학습 중 val 로그)이며, `train_yolo.py`의 final `model.val()` 결과값은 OOM 종료로 미기록.


### 평가 기준 정리
*   **용어 고정**:
    *   **Local mAP** = 각 실험에서 지정한 `dataset.yaml`의 `val` split에서 계산한 mAP 지표
    *   **Kaggle score** = `test_images` 제출 CSV 기준 Kaggle 리더보드 점수
    *   **Seed 규칙** = `data_seed`는 데이터셋 생성 seed, `seed`(단독 표기)는 학습 seed(`train_seed`)로 간주
*   **기록 규칙**: 모든 실험은 `Local mAP`와 `Kaggle score`를 분리해서 기록하고, 두 수치를 직접 동일 지표로 비교하지 않음.
*   Exp 1~9의 mAP 지표는 로컬 `validation` 셋(`data/datasets/yolo/copy_paste/cp_t20_data_seed42/images/val`) 기준으로 산출했고, 이후 파생 실험은 각 실험 YAML의 `data` 경로 기준으로 산출함.
*   캐글 제출용 CSV는 별도 추론 스크립트 `src/test_custom.py`를 사용해 `test_images` 기준으로 생성함.
*   따라서 로컬 mAP와 캐글 점수는 서로 다른 데이터셋에서 측정된 값이며, 직접적으로 동일 지표가 아님.
*   Exp 8은 `validation` 기준으로 Exp5의 `best.pt`를 사용해 NMS 탐색을 수행한 뒤, 동일 가중치로 `test_images` 제출 CSV를 생성한 실험임.
*   Exp 0~14 학습 실험의 optimizer 기록은 대부분 `auto`로 운영됨. 다만 `auto`는 내부 선택 결과(예: SGD/AdamW)가 별도 로그로 남지 않아, 옵티마이저 관점의 정밀 통제/재현성 해석에는 한계가 있어서 로그기록이 되도록 변경함. 

### 일반 실험과 앙상블 실험의 차이
*   일반 단일 모델 실험(Exp 1~8)은 `validation` 셋에 대해 직접 평가하여 mAP를 산출하며, 별도의 `val` 예측 CSV 저장은 필수가 아님.
*   앙상블 실험(예: Exp 9)은 여러 예측 결과를 먼저 병합한 뒤 최종 예측으로 mAP를 계산해야 하므로, 검증 과정에서 해상도별 또는 모델별 중간 예측이 일시적으로 필요할 수 있음.
*   다만 `val` 기준 중간 예측은 로컬 검증용 임시 산출물일 뿐이며, 캐글 제출용 파일은 항상 `test_images` 기준 최종 CSV만 사용해야 0점처리가 안됨(주의!)
*   향후 앙상블 실험에서도 원칙은 동일하며, `val` 중간 산출물은 필요 시에만 생성하고 최종 기록에는 mAP 수치만 남길 것임


---
## 🔬 정성적 분석 및 이슈 사항 (Insights & Issues)
### [Exp 0] Baseline (YOLOv8n / 640px)
*   **분석**: mAP가 0.959라는 비정상적인 결과 도출. 이는 유사도가 높은 사진들이 학습/검증 셋에 골고루 섞여 검증 데이터가 유출된 **Data Leakage(데이터 누수)** 현상임.
*   **통찰**: 초기 단계에서 높은 Local 점수는 성능 향상 신호가 아니라 분할 전략 오류 신호일 수 있으며, 본 프로젝트에서는 이 구간 점수를 의사결정 기준에서 제외해야 함.
*   **교훈**: 랜덤 분할은 지표를 지나치게 낙관적으로 만들며, 실제 대회(Kaggle: 0.701) 성적과 극심한 괴리를 발생시킴을 확인.

### [Exp 1] Stratified Split (YOLOv8n / 640px / 20ep)
*   **분석**: 640px 해상도에서 랜덤 분할 대비 지표가 0.402로 급감함. 이는 희귀 클래스들이 검증 데이터셋에 정직하게 포함되면서 나타나는 "정상적인 지표 하락" 현상임.
*   **통찰**: 지표 하락 자체가 실패가 아니라, 데이터 분포를 현실에 가깝게 만든 정상화 과정이며 이후 실험의 기준선 신뢰도를 회복하는 단계였음.
*   **결정**: 0.402를 기준점으로 잡고 다음 실험 진행.

### [Exp 1-2] Stratified Split (YOLOv8n / 640px / 50ep)
*   **현상**: 640px 환경에서 50 Epoch까지 학습 시 mAP50이 다시 0.96까지 상승함.
*   **분석**: 계층적 분할을 하더라도 **동일 세션에서 촬영된 이미지들 간의 높은 유사성(Session Similarity)** 때문에 모델이 검증 셋을 '쉽게' 맞추는 경향이 여전히 존재함.
*   **통찰**: 분할 전략 개선만으로는 세션 유사도 문제를 해결할 수 없고, 일반화 향상을 위해서는 데이터 다양성 증강(합성/도메인 변주) 개입이 필수임이 확인됨.
*   **한계**: 단순히 데이터를 나누는 것(Split Strategy)만으로는 데이터 불균형과 리큐지 문제를 완벽히 해결할 수 없음을 시사함.
*   **결정**: 모델의 일반화 성능(Generalization) 및 강건성(Robustness) 향상을 위한 Copy-Paste 증강 기법 도입이 절대적으로 필요함.

### [Exp 2] YOLO11 Transition (YOLO11s / 640px)
*   **분석**: 모델을 YOLOv8n에서 최신형인 YOLOv11s로 교체(640px 유지)한 결과, **mAP75 기준 0.995**라는 압도적인 수치를 기록함.
*   **통찰**: YOLOv11의 개선된 아키텍처가 특징 추출 및 정밀한 Localization에서 뛰어난 피지컬을 보여줌을 입증함.
*   **주의**: 0.99라는 수치는 현재의 검증 환경(Data Leakage) 내에서 이미 '학습의 정점'에 도달했음을 의미하며, 이 상태로 캐글에 제출할 경우 'Overfitting to Session' 이슈로 인해 점수 하락이 예상됨.
*   **결정**: 모델 성능은 확인되었으므로, 이제 인위적인 데이터 변주(Copy-Paste)가 주어진 상황에서 모델 실험을 해야 함  

### [Exp 3] High-Res Pivot (YOLO11s / 960px)
*   **분석**: 이미지 해상도를 640px에서 960px로 대폭 키워 학습했음에도, mAP 지표는 기존 640px(Exp 2)와 거의 동일한 **한계점(Ceiling)**에 부딪힘.
*   **통찰**: 알약의 각인 등 세부 특징을 더 명확하게 제공 하더라도, '기본적인 배경이 동일한 쌍둥이 이미지'라는 데이터셋 자체의 치명적 결함(데이터 누수)을 넘어서지는 못함.

### [Exp 4] No-Flip Fix (YOLO11s / 960px)
*   **분석**: 각인 보호를 위해 욜로 기본 설정인 fliplr=0.0(좌우 반전 금지) 조치 후 학습한 결과, mAP@50-95 지표가 0.989에서 0.992로 미세하게 상승함. 
*   **통찰**: 엉망이 된 오염 데이터(뒤집힌 문자)를 배제하는 것이 정밀한 위치 파악(Localization)에 실질적인 도움이 됨을 수치로 증명함.
*   **결정**: **"다양성(Exp 5) 이전에 무결성(Exp 4) 적용이 먼저임."** 
    - 만약 뒤집힌 각인이 방치된 채 Copy-Paste를 진행했다면, 모델은 '오염된 알약'을 더 많이 학습하며 혼란에 빠졌을 것임.
    - 똑바른 각인 정보가 복구되었으므로, 이 데이터를 기반으로 배경 누수를 제거하는 실험 5(Copy-Paste)를 하기로 결정

### [Exp 5] Custom Copy-Paste Synthesis (YOLO11s / 960px)
*   **분석**: 317장의 합성 데이터를 추가하여 데이터셋을 약 3배로 증량한 결과, **Kaggle Public Score 0.96804**라는 성적이 나옴. 
*   **통찰**: 로컬 mAP(0.99)는 누수 때문에 정체되었으나, 커스텀 합성 데이터를 통한 '데이터 다양성 확보'가 실전 테스트 데이터(캐글)에서의 일반화 성능을 폭발적으로 향상시켰음을 입증함. (주의: YOLO 내부 하이퍼파라미터 `copy_paste`를 켠 것이 아님)
*   **결정**: 이제는 공간적 변형을 극한으로 주는 **실험 6(Rotation)**을 해보기로 결정.

### [Exp 6] Rotation Aug (YOLO11s / 960px)
*   **분석**: 전각도 회전(Degrees=180.0) 적용 결과, **Kaggle Public Score 0.85483**으로 대폭 하락. 
*   **통찰**: 캐글 테스트 데이터가 이미 정방향 위주로 정렬되어 있을 가능성이 높음. 과도한 회전 변형은 오히려 정방향 데이터에 대한 모델의 정밀도를 떨어뜨리는 독이 됨(Over-fitting to rotation-invariance). 
*   **결정**: 실험 7에서는 **회전을 다시 제거하고**, 성공했던 Exp 5 설정에 **1024px 해상도**만 추가

### [Exp 7] Final High-Res (YOLO11s / 1024px)
*   **분석**: 실험 5의 성공 베이스라인에 1024px 초고해상도를 투입한 결과, 로컬 mAP@50-95 **0.9934** 달성. 
*   **통찰**: 실험 6에서 배운 "테스트 셋 정방향 정렬" 특성을 적극 활용하여 회전을 제거하고, 해상도 증량(960px -> 1024px)에 집중함. 이는 알약의 미세 각인(세부 특징 추출) 및 테두리 정밀도를 극한으로 살리기 위한 전략임. 
*   **결정**: 해상도 단순 리사이즈는 임계점에 도달한 것으로 판단됨. YOLO11m으로 높이는 것은 보류하고, 기본기가 탄탄한 모델(Exp 5)을 기반으로 **1순위 전략(NMS 튜닝 및 Multi-scale 앙상블)** 을 하는 것이 맞다는 생각이 듦
*   **재현**: `python train_yolo.py --config configs/train/exp7_train.yaml`

### [Exp 8] Dynamic NMS Tuning (Validation Threshold Search)
*   **목적**: conf, iou 값을 바꿔가며 어떤 조합이 validation 성능(mAP@50-95)이 가장 좋은지 찾는 것
*   **방식**: 학습이 아니라 Exp5의 `best.pt`를 사용해 `validation` 셋에서 반복 평가한 뒤, 최종 설정으로 `test_images` 제출 CSV를 생성함.
*   **분석**: Exp5 best 가중치 기준으로 `conf=0.20` 고정 후 `iou=0.50/0.60/0.70` 비교 시 mAP@50-95가 각각 **0.9926 / 0.9926 / 0.9925**로 거의 동일함.
*   **통찰**: 로컬 검증셋에서는 NMS 임계값 변화 영향이 매우 작고, `iou=0.70`은 미세하게 불리함.
*   **결정**: 실전 추론 기본값을 **`conf=0.20`, `iou=0.60`**으로 고정하고 다음 단계(Exp9, Exp10)로 진행.

*   **실험 결과 데이터**:
    | 구분 | IoU=0.50 | **IoU=0.60 (선택)** | IoU=0.70 |
    | :--- | :--- | :--- | :--- |
    | **mAP@50-95** | 0.9926 | **0.9926** | 0.9925 |
    | **mAP@50** | 0.9941 | **0.9941** | 0.9941 |

*   **재현(임계값 탐색)**: `python -u src/exp8_search.py --device 0 --verbose --confs 0.20 --ious 0.50,0.60,0.70`
*   **재현(추론)**: `python src/test_custom.py --model runs/pill_exp5_yolo11s_copypaste/weights/best.pt --imgsz 1024 --conf 0.20 --iou 0.60 --output submission/exp8_submission_iou0.6.csv`



### [Exp 9] Multi-Scale WBF (다중 해상도 앙상블)
*   **배경**: 원래 계획했던 Snapshot 앙상블은 과거 epoch 가중치가 남아 있지 않아 진행 불가. 따라서 추가 학습 없이도 앙상블 효과를 노릴 수 있는 `Multi-scale WBF`로 전략을 전환함.
*   **방식**: Exp5의 최종 가중치 `runs/pill_exp5_yolo11s_copypaste/weights/best.pt` 하나를 사용해 `640 / 960 / 1024` 세 해상도로 각각 추론한 뒤, 생성된 CSV를 WBF로 병합함.
*   **실험 통제 원칙(Exp22 재현 실험 적용)**: `Exp22-H3M0` 기반 WBF 효과 비교 시, 기준 추론 설정과 동일하게 **NMS IoU=0.7**, **WBF IoU=0.7**로 고정해 변수 통제를 유지함.
*   **실행 결과**: 개별 추론 결과 3개(`exp9_submission_640.csv`, `exp9_submission_960.csv`, `exp9_submission_1024.csv`)를 생성했고, WBF 적용 후 최종 제출 파일 `submission/exp9_final_wbf.csv`를 생성함. 이후 `validation` 기준 재평가를 별도로 수행하여 로컬 mAP 지표를 기록함.
*   **로컬 검증 결과**: `metrics/exp9_val_metrics.json` 기준 `mAP50=0.9932`, `mAP@50-95=0.9896`, `mAP@75=0.9932`.
*   **통찰**: 재학습 없이 단일 가중치의 다중 해상도 예측을 WBF로 병합하는 방식만으로도, 로컬 `validation` 기준 높은 성능을 유지하면서 Kaggle Public Score **0.97243**까지 도달함.
*   **시행착오**: Exp 9는 처음에는 `test_images` 기반 제출용 WBF 실험으로만 수행되었고, 이후 표 기록 및 근거 정리를 위해 `validation` 기준 WBF 평가를 별도로 추가 수행함.
*   **재현**: `bash scripts/exp9/run_exp9_val.sh` (로컬 validation 평가), `bash scripts/exp9/run_exp9_test.sh` (캐글 제출용 test 추론)

### [심층 리포트] NMS vs WBF의 IoU 파라미터 역할 차이 (Exp 8 vs Exp 9)
Exp 8에서 로컬 검증 기준 최적 밸런스(`mAP@50-95=0.9926`)를 보인 `iou=0.60` 설정을 Exp 9의 멀티스케일 앙상블 파이프라인에 적용함. 다만 Exp 9의 Kaggle 0.97243은 `iou=0.60` 하나의 효과라기보다, **멀티스케일 추론 + 해상도별 NMS + WBF 병합이 함께 작동한 결과**로 해석하는 것이 더 정확함. 

**1. 해상도별 추론 단계의 NMS (`test_custom.py`)**
*   단일 모델 NMS에서 `iou=0.60`은 기본값 `0.70`보다 박스 억제 조건이 더 공격적임. 즉, 조금만 겹쳐도 중복 박스로 판단해 제거할 가능성이 더 높음.
*   Exp 8에서는 이 설정이 단일 모델 추론 환경에서 Kaggle 점수를 소폭 하락시킴. 따라서 단일 모델 기준으로는 `0.60`이 항상 유리하다고 볼 수는 없음.
*   하지만 Exp 9처럼 640/960/1024 세 해상도에서 예측이 동시에 생성되는 환경에서는, 각 해상도 단계에서 중복 박스를 1차로 정리하는 역할을 했다고 볼 수 있음. 

**2. 앙상블 단계의 WBF (`ensemble_wbf.py`)**
*   WBF의 `iou=0.60`은 병합 기준으로 작동함. 따라서 `0.70`보다 더 쉽게 같은 객체로 묶이게 만드는 조건임.
*   해상도별로 조금씩 어긋난 박스들이 너무 엄격한 기준 때문에 따로 남지 않도록, `0.60` 기준으로 더 유연하게 병합할 수 있었음. 

**[결론]**
Exp 8에서 단일 모델 기준으로는 다소 불리했던 `iou=0.60` 설정이, Exp 9의 멀티스케일 WBF 파이프라인에서는 충분히 잘 작동했음. 다만 Exp 9의 성능 향상은 `0.60` 하나 때문이라기보다, **다중 해상도 예측과 WBF 병합 구조 전체의 효과**로 보는 것이 적절함.

### [Exp 10] 3-Seed Ensemble Training (The Masterpiece)
*   **배경**: 단일 모델(Exp 5)의 한계를 극복하기 위해 3-Seed(42, 123, 777) 기반의 모델 분산(Variance) 제어 전략 도입.
*   **실험 결과**:
    | ID | Seed | mAP50 | mAP75 | mAP50-95 |
    | :--- | :--- | :--- | :--- | :--- |
    | Exp 10-1 | 42 | 0.9949 | 0.9950 | **0.9942** |
    | Exp 10-2 | 123 | 0.9950 | 0.9950 | **0.9904** |
    | Exp 10-3 | 777 | 0.9944 | 0.9950 | **0.9932** |
*   **결과**: **Kaggle Public Score 0.98073**
*   **통찰**: 
    1.  **앙상블의 압도적 위력**: 로컬 mAP(0.9927)는 누수 때문에 단일 모델과 비슷해 보였으나, 캐글 리더보드에서는 **+0.0083**이라는 점수 상승을 보여줌.
    2.  **데이터 오염의 역설**: 비록 데이터에 일부 결함이 있었음에도 불구하고, 시드 다양성이 그 결함으로 인한 왜곡을 '상쇄'하며 강력한 강건성(Robustness)을 보여줬음.
    3. 3개 시드의 평균 mAP50-95는 **0.9926 (±0.0019)** 로 매우 높은 안정성을 보였고, 시드 42가 개별 최고 성능(0.9942)을 기록함. 실무에서는 3개 시드를 통해 검증하는 작업이 중요하지만 한정된 시간 안에 캐글 점수 올리는 데는 불필요한 작업일 수도 있었으나 성적이 올라서 원인파악을 해야 함. 

*   **최종 앙상블 결과**: 3개 시드의 validation 예측을 WBF로 병합한 결과, `metrics/exp10_final_ensemble_metrics.json` 기준 **mAP50=0.9942 / mAP75=0.9942 / mAP@50-95=0.9927** 을 기록함.
*   **조치**: `submission/exp10_final_submission.csv`를 현재까지의 마스터피스로 선언. 향후 모든 고도화 모델은 이 점수를 넘어설 것을 목표로 함.

### [Exp 11] Dirty-Aug Fail (Lesson Learned)
*   **원인**: 데이터는 정제했으나 `fliplr: 0.5` 옵션이 켜져 Exp 5와 1:1 비교 불가 상태로 진행됨.
*   **교훈**: 정제 후 성능 하락 시, 데이터 자체가 아닌 '실험 환경 매칭' 유무를 먼저 파악해야 함. 
*   **현상**: Exp 11 제출 결과 Kaggle 점수가 `0.9680` -> `0.9591`로 하락. 
*   **원인**: 데이터뿐만 아니라 `fliplr`(수평 뒤집기) 옵션이 Exp 5(`0.0`)와 달리 **`0.5`**(기본값)로 잘못 적용됨. 
*   **통찰**: 수평 뒤집기가 활성화되면서 알약 내 텍스트 정보(각인) 특징이 오염되어, 데이터 정제 효과가 가려짐.
*   **조치**: No-Flip(`fliplr: 0.0`) 환경을 완벽히 복원한 **Exp 12**를 통해 데이터 정제 ROI를 재측정함. 
*   **교훈**: A/B 테스트 시 오직 단 하나의 변수(데이터)만 변경해야 하며, 다른 모델 하이퍼파라미터는 100% 일치시켜야 함.
*   **재현**: `python train_yolo.py --config configs/train/exp11_train_yolo11s_flip.yaml`

### [Exp 12] Cleaned Baseline (The True Outcome)
*   **현상**: Exp 11의 변인 통제 오류(fliplr: 0.5)를 바로잡고, No-Flip(`0.0`) 환경에서 재학습 완료.
*   **결과**: **mAP@50 0.9946** (+0.0015), **mAP@50-95 0.9933** (+0.0034) 달성 (Exp 11 대비).
*   **통찰**: 
    1.  **BBox 정밀도 대폭 향상**: mAP@50-95 지표가 0.99대로 진입하며 우리가 정제한 '깨끗한 좌표'가 물리적 정밀도를 극대화했음을 증명함.
    2.  **노이즈 제거**: 뒤집힌 텍스트 특징을 배제함으로써 Precision(`0.9740`)이 개선됨.
*   **조치**: `runs/exp12_train_yolo11s_noflip/weights/best.pt`를 최종 Cleaned Baseline으로 확정.
*   **추론**: `python src/test_custom.py --config configs/inference/exp12_inference_yolo11s_noflip.yaml`
*   **재현**: `python train_yolo.py --config configs/train/exp12_train_yolo11s_noflip.yaml`

### [Exp 13] Baseline-1.0 Rebuild (YOLOv8n, Partial-Default Aug)
*   **현상**: YOLOv8n + 640 + 50epoch + seed42 + YOLO 기본증강 일부 배제(플립 OFF) 조건으로 Baseline 1.0을 재현했을 때, Kaggle Public Score **0.94032**를 기록함.
*   **결과**: 로컬 `validation` 기준 `mAP50=0.9890`, `mAP75=0.9950`, `mAP@50-95=0.9800`.
*   **해석**: 초기 Baseline 점수(0.70166) 대비 큰 상승은 yolo 기본 증강인 flip을 꺼서 그런 거 같아서 exp14에서는 default값인 0.5로 켜서 실험하기로 결정함 
*   **통찰**: Baseline 재현 단계에서 설정 차이(특히 flip 계열) 하나만으로도 결과가 크게 흔들릴 수 있어, 이후 실험은 YAML 기준 설정 고정/추적이 필수임.
*   **근거 파일**: `metrics/exp_baseline_yolov8n_1.0_val_metrics.json`, `submission/exp13_baseline_yolov8n_1.0.csv`

### [Exp 14] Baseline-1.0 Rebuild (YOLOv8n, Flip Restore)
*   **현상**: YOLOv8n + 640 + 50epoch + seed42 조건에서 `fliplr: 0.5`를 복구해 Baseline 2.0 재현 실험 수행.
*   **결과**: 로컬 `validation` 기준 `Precision=0.9485`, `Recall=0.9918`, `F1=0.9697`, `mAP50=0.9878`, `mAP75=0.9950`, `mAP@50-95=0.9717`.
*   **옵티마이저 기록**: `optimizer_requested=auto`, `optimizer_resolved=AdamW` 확인.
*   **해석**: Exp13 대비 Recall은 상승했지만 `mAP@50-95`가 하락해, flip 복구가 정밀 localization에는 불리하게 작용했을 가능성이 있음.
*   **통찰**: 본 도메인(각인/방향 민감)에서는 일반적인 좌우반전 증강이 항상 유효하지 않으며, 클래스 특성 기반으로 증강 선택이 필요함.
*   **Kaggle 결과**: Public Score **0.93808**
*   **근거 파일**: `metrics/exp14_train_baseline_yolov8n_1.0_val_metrics.json`, `submission/exp14_baseline_yolov8n_1.0.csv`
*   **후속 액션**: 초기 0.7점대 제출을 기록한 팀원에게 당시 학습/추론 설정 YAML 원본 공유 요청 필요하나 당시 상황에서는 이런 부분을 생각하지 못하고 기록을 안했기 때문에, 모든 팀원이 동일한 환경으로 베이스라인 1.0 재현 불가

### [Exp 15] Baseline-2.0 Canonical (YOLO11s, Applied-Fix)
*   **목적**: Exp12에서 `optimizer=auto`로 요청했을 때 실제 적용된 AdamW 계열 하이퍼파라미터를 명시 고정해, 팀 배포용 베이스라인을 하나로 통일.
*   **실행 조건**: `optimizer=AdamW`, `lr0=0.000167`, `momentum=0.9`, `warmup_bias_lr=0.0`, `seed=42`, `imgsz=960`, `batch=16`.
*   **결과**: `Precision=0.9704`, `Recall=0.9761`, `F1=0.9733`, `mAP50=0.9918`, `mAP75=0.9950`, `mAP@50-95=0.9897`.
*   **학습 시간**: `results.csv` 누적 시간 기준 `846.882s` (`14.11m`).
*   **Kaggle 재현성 비교**: `exp12=0.96528`, `exp15=0.96455`, 차이 `-0.00073`.
*   **해석**: `0.00073` 차이는 매우 작은 편으로 실질적으로는 거의 동일 성능대이며, `seed`를 `0 -> 42`로 변경한 만큼 완전 동일 점수가 나오지 않는 것은 정상 범주로 판단.
*   **통찰**: Exp15는 최고점 실험이 아니라 팀 공통 기준선(재현 가능한 기준점) 역할을 수행하며, 이후 단일변수 실험의 기준 anchor로 적합함.
*   **명명 정리**: 학습/추론 기준명을 모두 `2.0`으로 통일함.
    *   train: `runs/exp15_train_baseline_yolo11s_2.0`
    *   infer config: `configs/inference/exp15_inference_baseline_yolo11s_2.0.yaml`
*   **근거 파일**: `metrics/exp15_train_baseline_yolo11s_2.0_val_metrics.json`

### [Exp 16] Resolution Sweep (YOLO11s, 640px)
*   **목적**: 증강 변수 개입 없이 해상도 변화 효과만 분리 측정하기 위해, Exp15 설정을 그대로 유지한 채 입력 해상도만 `640`으로 축소.
*   **실행 조건**: `optimizer=AdamW`, `seed=42`, `imgsz=640`, `batch=16`, 증강 파라미터(`hsv/mosaic/erasing/auto_augment`)는 Exp15와 동일.
*   **결과**: `Precision=0.9690`, `Recall=0.9861`, `F1=0.9774`, `mAP50=0.9916`, `mAP75=0.9950`, `mAP@50-95=0.9892`.
*   **학습 시간**: `results.csv` 누적 시간 기준 `409.986s` (`6.83m`).
*   **Kaggle 결과**: Public Score **0.96723** (`vs Exp15 +0.00268`).
*   **해석**: Local `mAP@50-95`는 Exp15 대비 소폭 하락(`0.9897 → 0.9892`)했지만 Kaggle은 상승해, 실전 일반화 관점에서는 640 설정이 유리하게 작동함.
*   **통찰**: 해상도 증대가 항상 일반화 개선으로 이어지지 않으며, 본 데이터셋에서는 계산비용 대비 640도 경쟁력 있는 운영 옵션임이 확인됨.
*   **산출물**: `runs/exp16_train_yolo11s_res640/weights/best.pt`, `submission/exp16_yolo11s_res640.csv`
*   **근거 파일**: `metrics/exp16_train_yolo11s_res640_val_metrics.json`, `configs/inference/exp16_inference_yolo11s_res640.yaml`

### [Exp 17] Resolution Sweep (YOLO11s, 1024px)
*   **목적**: Exp15 고정 하이퍼파라미터에서 고해상도(`1024`) 입력이 성능에 미치는 영향을 단일 변수로 검증.
*   **실행 조건**: `optimizer=AdamW`, `seed=42`, `imgsz=1024`, `batch=16`, 나머지 조건 Exp15와 동일.
*   **결과**: `Precision=0.9692`, `Recall=0.9863`, `F1=0.9777`, `mAP50=0.9929`, `mAP75=0.9950`, `mAP@50-95=0.9900`.
*   **학습 시간**: `results.csv` 누적 시간 기준 `894.107s` (`14.90m`).
*   **Kaggle 결과**: Public Score **0.97292** (`vs Exp15 +0.00837`, `vs Exp16 +0.00569`).
*   **해석**: Local/Kaggle 모두 Exp16 대비 우세하며, 현재 단일 모델 해상도 비교군(640/960/1024)에서는 `1024`가 가장 강한 선택지.
*   **통찰**: 본 프로젝트의 단일 모델 상위권 성능은 1024 해상도에서 안정적으로 확보되며, 960은 탐색 비용 절감을 위한 실험 트랙으로 운용하는 것이 타당함.
*   **산출물**: `runs/exp17_train_yolo11s_res1024/weights/best.pt`, `submission/exp17_yolo11s_res1024.csv`
*   **근거 파일**: `metrics/exp17_train_yolo11s_res1024_val_metrics.json`, `configs/inference/exp17_inference_yolo11s_res1024.yaml`

### [Exp 18] Resolution Benchmark (YOLO11s, 1280px)
*   **목적**: 해상도 상한선(1280)에서 성능/시간 효율을 벤치마크하고, 이후 실험 해상도 전략(탐색 vs 최종검증)의 근거를 확보.
*   **실행 조건**: `optimizer=AdamW`, `seed=42`, `imgsz=1280`, `batch=16`, `workers=8`, 증강 파라미터는 Exp15와 동일.
*   **학습 상태**: `50 epochs completed in 1.870 hours`로 학습 완료. `best.pt`, `last.pt` 저장 확인.
*   **학습 시간**: `results.csv` 누적 시간 기준 `6732.89s` (`112.21m`).
*   **지표 기록 주의**: 학습 종료 직후 실행된 final `model.val()` 단계에서 OOM으로 프로세스가 `Killed`되어 `metrics/exp18_..._val_metrics.json` 자동 저장은 실패.
*   **임시 지표(`results.csv` 50epoch 행)**: `Precision=0.97062`, `Recall=0.96341`, `mAP50=0.99085`, `mAP@50-95=0.98902`.
*   **산출물**: `runs/exp18_train_yolo11s_res1280/weights/best.pt`, `configs/inference/exp18_inference_yolo11s_res1280.yaml`, `submission/exp18_yolo11s_res1280.csv`
*   **추론 실행 상태**: inference 완료(`metrics/infer/exp18_yolo11s_res1280_infer_runtime.json` 생성 확인).
*   **Kaggle 결과**: **TBD** (추후 업데이트 예정)
*   **통찰**: 1280은 잠재 성능보다 운영 리스크(OOM/시간비용)가 커서, 현 로컬 자원에서는 상시 실험 트랙이 아니라 최종 검증용 제한 트랙으로 분리하는 것이 합리적임.

### [Exp 19-A] No-HSVH Ablation (YOLO11s, 960px)
*   **목적**: Exp15/기본증강 기반에서 `hsv_h` 민감도를 단일 변수로 확인 (`0.015 → 0.0`).
*   **실행 조건**: `optimizer=AdamW`, `seed=42`, `imgsz=960`, `batch=16`, `hsv_h=0.0`, 그 외 Exp15/기본값 유지.
*   **결과**: `Precision=0.9631`, `Recall=0.9642`, `F1=0.9636`, `mAP50=0.9929`, `mAP75=0.9950`, `mAP@50-95=0.9920`.
*   **학습 시간**: `806.449s` (`13.44m`).
*   **산출물**: `runs/exp19A_train_yolo11s_res960_no_hsvh/weights/best.pt`, `submission/exp19A_yolo11s_res960_no_hsvh.csv`
*   **근거 파일**: `metrics/exp19A_train_yolo11s_res960_no_hsvh_val_metrics.json`, `metrics/infer/exp19A_yolo11s_res960_no_hsvh_infer_runtime.json`
*   **Kaggle 결과**: **0.96678** (`vs Exp15 +0.00223`)
*   **통찰**: hue 변형을 완전히 제거해도 성능이 유지/개선되어, 기본 `hsv_h=0.015`가 본 데이터 분포에는 과한 변동일 가능성을 열어줌.

### [Exp 19-B] Mosaic 0.5 Ablation (YOLO11s, 960px)
*   **목적**: `mosaic` 강도를 절반으로 낮췄을 때(`1.0 → 0.5`) 정밀도/재현율 트레이드오프 확인.
*   **실행 조건**: `optimizer=AdamW`, `seed=42`, `imgsz=960`, `batch=16`, `mosaic=0.5`, 나머지 조건 유지.
*   **결과**: `Precision=0.9765`, `Recall=0.9617`, `F1=0.9690`, `mAP50=0.9940`, `mAP75=0.9950`, `mAP@50-95=0.9920`.
*   **학습 시간**: `797.952s` (`13.30m`).
*   **산출물**: `runs/exp19B_train_yolo11s_res960_mosaic05/weights/best.pt`, `submission/exp19B_yolo11s_res960_mosaic05.csv`
*   **근거 파일**: `metrics/exp19B_train_yolo11s_res960_mosaic05_val_metrics.json`, `metrics/infer/exp19B_yolo11s_res960_mosaic05_infer_runtime.json`
*   **Kaggle 결과**: **TBD**
*   **통찰**: Local 기준 개선 신호가 분명해 Mosaic 강도는 고정값(1.0)보다 하향 탐색이 유망하며, 후속 M-sweep 실험의 근거로 채택됨.

### [Exp 19-C] No-Translate Ablation (YOLO11s, 960px)
*   **목적**: `translate` 제거 효과를 단일 변수로 검증 (`0.1 → 0.0`).
*   **실행 조건**: `optimizer=AdamW`, `seed=42`, `imgsz=960`, `batch=16`, `translate=0.0`, 그 외 동일.
*   **결과**: `Precision=0.9543`, `Recall=0.9712`, `F1=0.9626`, `mAP50=0.9939`, `mAP75=0.9950`, `mAP@50-95=0.9909`.
*   **학습 시간**: `800.844s` (`13.35m`).
*   **산출물**: `runs/exp19C_train_yolo11s_res960_no_translate/weights/best.pt`, `submission/exp19C_yolo11s_res960_no_translate.csv`
*   **근거 파일**: `metrics/exp19C_train_yolo11s_res960_no_translate_val_metrics.json`, `metrics/infer/exp19C_yolo11s_res960_no_translate_infer_runtime.json`
*   **Kaggle 결과**: **TBD**
*   **통찰**: `translate` 제거 시 mAP는 소폭 개선되지만 F1 하락이 동반되어, 단독 채택보다 후순위 후보(조합 검증 전 대기)로 두는 것이 리스크 관리에 유리함.

### [Exp 19-D] Scale 0.3 Ablation (YOLO11s, 960px)
*   **목적**: `scale` 범위를 축소했을 때(`0.5 → 0.3`) 박스 정밀도 변화 확인.
*   **실행 조건**: `optimizer=AdamW`, `seed=42`, `imgsz=960`, `batch=16`, `scale=0.3`, 나머지 조건 동일.
*   **결과**: `Precision=0.9710`, `Recall=0.9689`, `F1=0.9699`, `mAP50=0.9909`, `mAP75=0.9950`, `mAP@50-95=0.9893`.
*   **학습 시간**: `800.743s` (`13.35m`).
*   **산출물**: `runs/exp19D_train_yolo11s_res960_scale03/weights/best.pt`, `submission/exp19D_yolo11s_res960_scale03.csv`
*   **근거 파일**: `metrics/exp19D_train_yolo11s_res960_scale03_val_metrics.json`, `metrics/infer/exp19D_yolo11s_res960_scale03_infer_runtime.json`
*   **Kaggle 결과**: **TBD**
*   **통찰**: `scale` 축소는 Local 기준 성능 저하가 확인되어, 현 파이프라인에서는 독성 변수로 분류하고 추가 탐색 우선순위에서 제외함.

### [Exp 20-H Sweep] HSVH Fine-Grained Search (YOLO11s, 960px)
*   **목적**: Exp19-A(`hsv_h=0.0`) 결과를 바탕으로 `hsv_h` 민감도를 미세 탐색하고, `0.015` 초과 구간의 상한 가드레일까지 확인.
*   **실행 조건**: `optimizer=AdamW`, `seed=42`, `imgsz=960`, `batch=16`, 고정값(`mosaic=1.0`, `translate=0.1`, `scale=0.5`) 유지 후 `hsv_h`만 변경.
*   **실험값**: `H1=0.005`, `H2=0.010`, `H3=0.020`, `H4=0.030`, `H5=0.018`, `H6=0.022`, `H7=0.025`.
*   **결과 요약**:
    *   `H3 (hsv_h=0.020)`가 Local `mAP@50-95=0.9919`로 최고.
    *   `H2 (0.010)`도 `mAP@50-95=0.9915`로 강한 후보.
    *   `H5 (0.018)`와 `H7 (0.025)`는 각각 `0.9911`로 근접했지만 `H3/H2`를 넘지 못함.
    *   `H6 (0.022)`는 `0.9901`로 피크 구간 우측에서 하락 신호를 보임.
    *   `H4 (0.030)`는 `mAP@50-95=0.9886`로 하락해 과도한 hue 변형 리스크 신호 확인.
*   **학습 시간**: `13.28~14.02m` 범위.
*   **산출물**: `runs/exp20H*_train_yolo11s_res960_hsvh0*/weights/best.pt` (H1~H7)
*   **근거 파일**:
    *   `metrics/exp20H1_train_yolo11s_res960_hsvh005_val_metrics.json`
    *   `metrics/exp20H2_train_yolo11s_res960_hsvh010_val_metrics.json`
    *   `metrics/exp20H3_train_yolo11s_res960_hsvh020_val_metrics.json`
    *   `metrics/exp20H4_train_yolo11s_res960_hsvh030_val_metrics.json`
    *   `metrics/exp20H5_train_yolo11s_res960_hsvh018_val_metrics.json`
    *   `metrics/exp20H6_train_yolo11s_res960_hsvh022_val_metrics.json`
    *   `metrics/exp20H7_train_yolo11s_res960_hsvh025_val_metrics.json`
    *   `metrics/infer/exp20H1_yolo11s_res960_hsvh005_infer_runtime.json`
    *   `metrics/infer/exp20H2_yolo11s_res960_hsvh010_infer_runtime.json`
    *   `metrics/infer/exp20H3_yolo11s_res960_hsvh020_infer_runtime.json`
    *   `metrics/infer/exp20H4_yolo11s_res960_hsvh030_infer_runtime.json`
*   **추론/캐글 상태**: `H1~H4` inference 완료, `H5~H7`은 train 완료/inference 대기.
*   **Kaggle 업데이트(2026-03-31)**:
    *   `H3 (hsv_h=0.020)`: **0.97154** (`vs Exp15 +0.00699`, `vs 19-A +0.00476`)
    *   `H2 (hsv_h=0.010)`: 제출 보류(로컬 기준 후보 유지, 제출 슬롯 여유 시 검증)
*   **통찰**:
    *   `hsv_h` 효과는 단조 증가가 아니며, `0.020` 부근에서 국소 최적점이 형성됨.
    *   `0.018~0.025` 구간은 모두 상위권이지만, 추가 미세탐색의 이득 폭은 `~0.001` 내외로 작음.
    *   즉, Exp20 추가 실험의 의미는 "최고점 갱신"보다 "`hsv_h` 민감도 곡선과 안전 구간 확인"에 있음.
*   **결론**: Kaggle 기준으로도 `H3(0.020)`가 `H-best`로 확정되었다. `H2`는 로컬 후보로만 유지하고, 다음 단계는 `Exp21(mosaic)`에서 `M-best`를 선별한 뒤 `Exp22(H-best+M-best)` 조합 검증으로 전환한다.

### [Exp 21-M Sweep] Mosaic Fine-Grained Search (YOLO11s, 960px)
*   **실험 가설**: `mosaic=0.5`에서 개선 신호가 있었으므로, `0.5` 근방(`0.4/0.6`)과 경계값(`0.0`)에서 추가 개선 또는 안정 구간이 확인될 수 있다.
*   **가설 설정 이유**: `mosaic`은 소형 객체의 경계 절단/배경 혼합 강도에 직접 영향하여 과소/과대 설정 시 성능 저하 위험이 있다. 따라서 `0.5` 근처 국소 탐색이 필요했다.
*   **실행 조건**: `optimizer=AdamW`, `seed=42`, `imgsz=960`, `batch=16`, 고정값(`hsv_h=0.015`, `translate=0.1`, `scale=0.5`) 유지 후 `mosaic`만 변경.
*   **실험값**: `M0=0.0`, `M1=0.4`, `M2=0.6`.
*   **결과 요약**:
    *   Local에서는 `M2 (mosaic=0.6)`가 `mAP@50-95=0.9920`로 최고.
    *   Kaggle에서는 `M0 (mosaic=0.0)=0.96894`가 `M2 (0.96154)`보다 크게 우세.
    *   즉 `mosaic`은 Local 우열과 Kaggle 우열이 역전되었고, 최종 선택은 Kaggle 기준으로 해야 한다.
*   **통찰**:
    *   본 실험군에서는 강한 합성(`mosaic` 상향)이 Local 지표를 상승시켜 실전 일반화를 악화시킬 수 있어, 최종 채택 기준은 Kaggle 성능으로 고정해야 한다.
*   **점수 변화 원인 분석(가설 기반)**:
    *   `0.4`는 합성 강도가 낮아져 데이터 다양성 이득이 제한적이었을 가능성.
    *   `0.0`은 인위적 합성 왜곡을 제거해 박스 정밀도에 유리할 수 있으나, 일부 다양성 이득은 상실했을 가능성.
    *   `0.6`은 과도한 절단을 피하면서도 다양성을 충분히 확보해 일반화에 가장 유리했을 가능성.
*   **학습 시간**: `M0=13.85m`, `M1=14.11m`, `M2=13.69m`.
*   **근거 파일**:
    *   `metrics/exp21M0_train_yolo11s_res960_mosaic00_val_metrics.json`
    *   `metrics/exp21M1_train_yolo11s_res960_mosaic04_val_metrics.json`
    *   `metrics/exp21M2_train_yolo11s_res960_mosaic06_val_metrics.json`
*   **추론/캐글 상태**:
    *   `M0`: inference 완료 / Kaggle **0.96894**
    *   `M2`: inference 완료 / Kaggle **0.96154**
    *   `M1`: train 완료 / Kaggle 미제출
*   **결론(업데이트)**:
    *   `Exp15 baseline(0.96455)` 대비 `M0`는 `+0.00439`, `M2`는 `-0.00301`.
    *   현재 mosaic 단일변수 기준 `M-best`는 `M0 (mosaic=0.0)`로 재지정한다.
    *   실험 방향은 `M1` 1회 검증 여부만 결정한 뒤, 조합 단계는 `H3+M0` 우선으로 진행한다.

### [Exp 22-H3M0] H3 + M0 Combination (YOLO11s, 960px)
*   **실험 가설**: `H-best(0.020)`와 `M-best(0.0)`를 결합하면 단일변수 대비 추가 일반화 이득이 발생할 수 있다.
*   **가설 설정 이유**: `hsv_h`와 `mosaic`은 각각 Kaggle 개선 신호가 확인된 변수이며, 변인 통제 원칙에 따라 `H3` 기준에 `M0` 한 변수만 추가해 시너지 여부를 점검한다.
*   **실행 조건**: `optimizer=AdamW`, `seed=42`, `imgsz=960`, `batch=16`, 기준 `Exp20-H3`에서 `mosaic: 1.0 → 0.0`만 변경.
*   **결과**: `Precision=0.9753`, `Recall=0.9630`, `F1=0.9691`, `mAP50=0.9935`, `mAP75=0.9950`, `mAP50-95=0.9915`.
*   **운영 메타**: `model size=18.37MB`, `params=9,434,472`, `train time=808.7s (13.48m)`.
*   **근거 파일**:
    *   `metrics/exp22H3M0_train_yolo11s_res960_hsvh020_mosaic00_val_metrics.json`
    *   `configs/inference/exp22H3M0_inference_yolo11s_res960_hsvh020_mosaic00.yaml`
*   **추론/캐글 상태**: inference 완료 / Kaggle **0.97199**.
*   **통찰**:
    *   Local 기준으로는 `H3` 단독(`0.9919`) 대비 소폭 하락(`0.9915`)했지만, Kaggle은 `H3(0.97154)` 대비 **+0.00045** 상승했다.
    *   개선 폭은 크지 않지만 방향은 양수이므로, 현재 제출/추론 고도화 기준 모델은 `H3M0`를 우선 채택하고 `H3`를 폴백으로 유지한다.

### [Exp 23-C Sweep] CLAHE 강도 탐색 (YOLO11s, 960px)
*   **실험 목적**: `Exp22-H3M0` 설정을 고정한 상태에서 CLAHE 강도(`cl15/cl20/cl25`)만 바꿔, 로컬 validation 기준 실효성이 있는지 확인.
*   **실행 조건**: `optimizer=AdamW`, `seed=42`, `imgsz=960`, `batch=16`, `hsv_h=0.020`, `mosaic=0.0`, `fliplr=0.0` 고정. 변경 변수는 데이터셋(`data/datasets/yolo/clahe/clXX_data_seed42/dataset.yaml`)만 적용.
*   **결과 요약 (Local + Kaggle)**:

    | 실험 | Precision | Recall | F1 | mAP50 | mAP75 | mAP50-95 | vs Exp22-H3M0 (Local) | Kaggle |
    | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
    | **Exp23-C1 (cl15)** | 0.9683 | 0.9643 | 0.9663 | 0.9948 | 0.9950 | **0.9941** | **+0.0026** | **0.96435** |
    | Exp23-C2 (cl20) | 0.9548 | 0.9801 | **0.9673** | 0.9931 | 0.9950 | 0.9914 | -0.0001 | **0.94796** |
    | Exp23-C3 (cl25) | 0.9347 | **0.9926** | 0.9627 | 0.9946 | 0.9950 | 0.9928 | +0.0013 | TBD |

*   **해석**:
    *   로컬 기준으로는 CLAHE가 **조건부로 유효**했다. `cl15`가 최고 성능(`0.9941`)이며 기준 실험 `Exp22-H3M0(0.9915)` 대비 **+0.0026** 개선.
    *   `cl20`은 실질적 개선이 없고(`-0.0001`), `cl25`는 개선은 있으나 `cl15`를 넘지 못함.
    *   강도를 올릴수록(`cl20→cl25`) Recall은 올라가지만 Precision이 크게 하락하는 패턴이 보여, 과도한 대비 강화가 오탐을 늘릴 가능성이 있음.
    *   **중요 업데이트(실전 일반화 실패)**: `Exp23-C1`의 Kaggle 점수는 **0.96435**로, 기준 `Exp22-H3M0(0.97199)` 대비 **-0.00764** 하락했다.
    *   `Exp23-C2`의 Kaggle 점수는 **0.94796**로, 기준 대비 **-0.02403** 하락해 CLAHE 강도 증가가 실전 분포에서 더 큰 성능 붕괴를 유발했다.
    *   즉, `CLAHE val`에서 얻은 Local 이득이 `raw test` 일반화로 이어지지 않았고, 분포 불일치(domain mismatch) 리스크가 실제 점수 하락으로 확인됐다.
*   **추론/캐글 상태**:
    *   `C1`: inference/제출 완료, Kaggle **0.96435** (하락)
    *   `C2`: inference/제출 완료, Kaggle **0.94796** (대폭 하락)
    *   `C3`: train/val 완료, Kaggle 미제출(`TBD`)
*   **결론**: 현 세팅(`960 + H3M0`)에서 CLAHE는 실전 기준 독성 가능성이 높다. `Exp23` 라인은 우선 **보류/기각**하고, 제출 기준은 `Exp22-H3M0`를 유지한다.
*   **근거 파일**:
    *   `metrics/exp23C1_train_yolo11s_res960_hsvh020_mosaic00_clahe15_val_metrics.json`
    *   `metrics/exp23C2_train_yolo11s_res960_hsvh020_mosaic00_clahe20_val_metrics.json`
    *   `metrics/exp23C3_train_yolo11s_res960_hsvh020_mosaic00_clahe25_val_metrics.json`
    *   `configs/train/exp23C1_train_yolo11s_res960_hsvh020_mosaic00_clahe.yaml`
    *   `configs/train/exp23C2_train_yolo11s_res960_hsvh020_mosaic00_clahe.yaml`
    *   `configs/train/exp23C3_train_yolo11s_res960_hsvh020_mosaic00_clahe.yaml`
    *   `configs/inference/exp23C1_inference_yolo11s_res960_hsvh020_mosaic00_clahe15.yaml`
    *   `configs/inference/exp23C2_inference_yolo11s_res960_hsvh020_mosaic00_clahe20.yaml`
    *   `configs/inference/exp23C3_inference_yolo11s_res960_hsvh020_mosaic00_clahe25.yaml`

### [Exp 24-V Sweep] HSV_V 강도 재탐색 (YOLO11s, 960px)
*   **실험 목적**: `Exp22-H3M0` 기준 설정에서 `hsv_v`만 단일 변수로 조정(`0.0/0.2/0.5`)해, CLAHE 라인 기각 이후 밝기 변형 강도의 실효성을 재검증.
*   **실행 조건**: `optimizer=AdamW`, `seed=42`, `imgsz=960`, `batch=16`, `hsv_h=0.020`, `mosaic=0.0`, `fliplr=0.0` 고정. 변경 변수는 `hsv_v`만 적용.
*   **결과 요약 (Local + Kaggle)**:

    | 실험 | hsv_v | Precision | Recall | F1 | mAP50 | mAP50-95 | vs Exp22-H3M0 (Local) | Kaggle |
    | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
    | **Exp24-V2** | **0.2** | 0.9559 | 0.9854 | 0.9704 | 0.9943 | **0.9928** | **+0.0013** | **0.96308** |
    | Exp24-V3 | 0.5 | 0.9600 | 0.9818 | **0.9708** | 0.9945 | 0.9922 | +0.0007 | TBD |
    | Exp24-V1 | 0.0 | **0.9749** | 0.9641 | 0.9695 | 0.9935 | 0.9906 | -0.0009 | TBD |
    | Exp22-H3M0 (기준) | 0.4 | 0.9753 | 0.9630 | 0.9691 | 0.9935 | 0.9915 | - | **0.97199** |

*   **해석**:
    *   `hsv_v` 하향(0.2)과 상향(0.5) 모두 Local 기준으로는 `Exp22-H3M0` 대비 개선 신호가 확인됨.
    *   `hsv_v=0.0`(완전 OFF)은 Precision은 높지만 mAP50-95가 기준보다 낮아, 과도한 축소(밝기 변형 제거)는 불리할 가능성이 큼.
    *   현재 우열은 `V2(0.2) > V3(0.5) > Base(0.4) > V1(0.0)` 순.
    *   **중요 업데이트(실전 일반화 실패)**: `Exp24-V2`는 Kaggle **0.96308**로, 기준 `Exp22-H3M0(0.97199)` 대비 **-0.00891** 하락했다.
    *   즉, `hsv_v=0.2`의 Local 개선은 실전 일반화로 이어지지 않았고, 현재 기준으로는 `V2`를 채택할 수 없다.
*   **제출 우선순위 (업데이트, 실전 기준)**:

    | 제출순위(실전) | 실험명 | hsv_v | Precision | Recall | mAP50 | mAP50-95 | Kaggle |
    | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
    | **1** | **Exp24-V3** | **0.5** | 0.9600 | 0.9818 | 0.9945 | 0.9922 | 미제출 |
    | 2 | Exp24-V1 | 0.0 | **0.9749** | 0.9641 | 0.9935 | 0.9906 | 미제출 |
    | 3 | Exp24-V2 | 0.2 | 0.9559 | 0.9854 | 0.9943 | **0.9928** | **0.96308 (기각)** |
    | - | Exp22-H3M0 (기준) | 0.4 | 0.9753 | 0.9630 | 0.9935 | 0.9915 | **0.97199** |

*   **결론(업데이트)**:
    *   `V2`는 실전 하락이 명확하므로 즉시 기각한다.
    *   남은 검증 가치는 `V3` 1회 확인 제출에 한정한다(`V1`은 슬롯 여유 있을 때만).
    *   `V3`도 기준(`0.97199`)을 넘지 못하면 `hsv_v` 라인은 보류/종료하고 기준 모델 `Exp22-H3M0` 유지가 합리적이다.
*   **근거 파일**:
    *   `metrics/exp24V1_train_yolo11s_res960_hsvh020_mosaic00_hsvv000_val_metrics.json`
    *   `metrics/exp24V2_train_yolo11s_res960_hsvh020_mosaic00_hsvv020_val_metrics.json`
    *   `metrics/exp24V3_train_yolo11s_res960_hsvh020_mosaic00_hsvv050_val_metrics.json`
    *   `configs/train/exp24V1_train_yolo11s_res960_hsvh020_mosaic00_hsvv000.yaml`
    *   `configs/train/exp24V2_train_yolo11s_res960_hsvh020_mosaic00_hsvv020.yaml`
    *   `configs/train/exp24V3_train_yolo11s_res960_hsvh020_mosaic00_hsvv050.yaml`
    *   `configs/inference/exp24V1_inference_yolo11s_res960_hsvh020_mosaic00_hsvv000.yaml`
    *   `configs/inference/exp24V2_inference_yolo11s_res960_hsvh020_mosaic00_hsvv020.yaml`
    *   `configs/inference/exp24V3_inference_yolo11s_res960_hsvh020_mosaic00_hsvv050.yaml`

### [Exp 25-L Sweep] Loss Gain(box/cls/dfl) 전이 검증 (YOLO11s, 960px)
*   **실험 목적**: 팀 내 640 해상도 선행 탐색에서 얻은 loss gain 후보를 `Exp22-H3M0(960)` 기준으로 재실험해, 성능이 높게 재현되는지 검증.
*   **가설 설정 이유**:
    *   640에서의 선행 탐색은 저비용 후보 발굴 단계로 유효하며, 960은 최종 채택 검증 단계로 분리 운영하는 것이 실무적으로 합리적이다.
    *   단, 640에서 유효했던 수치를 960의 확정값으로 그대로 쓰는 것은 금지하고, 960에서 재검증/재탐색을 거쳐야 한다.
*   **실행 조건**: `optimizer=AdamW`, `seed=42`, `imgsz=960`, `batch=16`, `hsv_h=0.020`, `mosaic=0.0`, `fliplr=0.0` 고정. 변경 변수는 `box/cls/dfl`만 적용(`cls=0.5` 고정).
*   **결과 요약 (Local + Kaggle)**:

    | 제출순위(로컬) | 실험명 | box, cls, dfl | Precision | Recall | mAP50 | mAP50-95 | vs Exp22-H3M0 (Local) | Kaggle |
    | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
    | **1** | **Exp25-LBD1** | **(8.5, 0.5, 2.0)** | 0.9741 | **0.9856** | 0.9941 | **0.9923** | **+0.0008** | 제출 완료 (Exp22 대비 하락, 수치 입력 대기) |
    | 2 | Exp25-LD1 | (7.5, 0.5, 2.0) | **0.9787** | 0.9599 | **0.9944** | 0.9920 | +0.0005 | TBD |
    | 3 | Exp25-LB1 | (8.5, 0.5, 1.5) | 0.9730 | 0.9632 | 0.9932 | 0.9919 | +0.0004 | TBD |
    | - | Exp22-H3M0 (기준) | (7.5, 0.5, 1.5) | 0.9753 | 0.9630 | 0.9935 | 0.9915 | - | **0.97199** |
    | - | Exp15-Baseline | (7.5, 0.5, 1.5) | 0.9704 | 0.9761 | 0.9918 | 0.9897 | -0.0018 | **0.96455** |

*   **해석**:
    *   Local 기준으로는 `LB1/LD1/LBD1` 모두 `Exp22-H3M0` 대비 소폭 개선되었지만, 개선폭이 `+0.001` 게이트를 넘지 못해 강한 신호로 보기는 어렵다.
    *   `LBD1`은 Local 1위이지만, Kaggle에서는 기준 `Exp22-H3M0` 대비 하락이 관찰되어 최종 채택 근거가 약하다.
    *   따라서 이번 결과는 "640 탐색이 무의미했다"가 아니라, **640에서 성능상 이득이 있었던 후보가 960에서는 그대로 전이가 되지 않은 사례**로 해석하는 것이 타당하다.
*   **결론**:
    *   640 선행 탐색 자체는 유효한 전략이며(후보 발굴), 최종 채택은 960 재검증으로 확정해야 한다.
    *   현재 기준 제출 모델은 `Exp22-H3M0`를 유지하고, `Exp25` 라인은 보조 후보로 보관한다.
    *   마감 시점 운영 우선순위는 960 추가 하이퍼튜닝보다 `Phase4(WBF/TTA/SAHI)` 추론 최적화로 전환한다.
*   **근거 파일**:
    *   `metrics/exp25LB1_train_yolo11s_res960_hsvh020_mosaic00_box850_cls050_dfl150_val_metrics.json`
    *   `metrics/exp25LD1_train_yolo11s_res960_hsvh020_mosaic00_box750_cls050_dfl200_val_metrics.json`
    *   `metrics/exp25LBD1_train_yolo11s_res960_hsvh020_mosaic00_box850_cls050_dfl200_val_metrics.json`
    *   `configs/train/exp25LB1_train_yolo11s_res960_hsvh020_mosaic00_box850_cls050_dfl150.yaml`
    *   `configs/train/exp25LD1_train_yolo11s_res960_hsvh020_mosaic00_box750_cls050_dfl200.yaml`
    *   `configs/train/exp25LBD1_train_yolo11s_res960_hsvh020_mosaic00_box850_cls050_dfl200.yaml`
    *   `configs/inference/exp25LB1_inference_yolo11s_res960_hsvh020_mosaic00_box850_cls050_dfl150.yaml`
    *   `configs/inference/exp25LD1_inference_yolo11s_res960_hsvh020_mosaic00_box750_cls050_dfl200.yaml`
    *   `configs/inference/exp25LBD1_inference_yolo11s_res960_hsvh020_mosaic00_box850_cls050_dfl200.yaml`
    *   `submission/exp25LBD1_yolo11s_res960_hsvh020_mosaic00_box850_cls050_dfl200.csv`

### [Exp 26] Exp22 기반 Multi-scale WBF (Phase4 1차)
*   **실험 목적**: `Exp22-H3M0` 단일 모델에서 재학습 없이 추론 성능을 개선할 수 있는지 검증.
*   **설정**: 동일 가중치(`Exp22 best.pt`)로 `640/960/1024` 예측 CSV 생성 후 WBF 병합. 비교군 통제를 위해 `NMS IoU=0.70`, `WBF IoU=0.70` 고정.
*   **Kaggle 결과**:
    *   `Exp22-H3M0`: **0.97199**
    *   `Exp26-WBF`: **0.97345**
    *   **개선폭: +0.00146**
*   **해석**: 상단 점수 구간에서 재학습 없이 얻은 유의미한 개선으로 판단. 다음 단계는 `Exp27(다중 seed WBF)`에서 추가 개선 가능성 확인.

### [Exp 27] Exp10 방식 재현: Exp22 기반 3-Seed WBF (Phase4 2차, 완료)
*   **실험 목적**: `Exp26`에서 확인한 WBF 개선 신호를 확장해, Exp10과 동일한 전략(다중 시드 앙상블)로 추가 성능 향상 가능성을 검증.
*   **설정**:
    *   베이스라인: `Exp22-H3M0 (seed=42)`
    *   추가 학습: `seed=123`, `seed=777` (`Exp22`와 완전 동일 조건, seed만 변경)
    *   최종 병합: 3개 seed의 960 예측 CSV를 WBF(`IoU=0.70`)로 결합
*   **로컬 결과 (Seed별)**:
    *   `Exp27-S123`: Precision **0.9369**, Recall **0.9753**, mAP50 **0.9875**, mAP50-95 **0.9851**
    *   `Exp27-S777`: Precision **0.9801**, Recall **0.9653**, mAP50 **0.9934**, mAP50-95 **0.9923**
*   **최종 Kaggle 결과**:
    *   `Exp27(F) 3-seed WBF`: **0.98077**
    *   `vs Exp22-H3M0 (0.97199)`: **+0.00878**
    *   `vs Exp26 (0.97345)`: **+0.00732**
*   **통찰(최종)**:
    *   `Exp26`의 단일 가중치 멀티스케일 WBF도 유효했지만, 시드 다양성을 추가한 3-seed WBF가 훨씬 큰 개선폭을 제공했다.
    *   단일 시드 간 성능 편차가 존재하므로, 최종 제출 의사결정은 개별 시드가 아니라 앙상블 점수 기준으로 하는 것이 안정적이다.
    *   결론적으로 `Exp22` 계열의 최종 제출 후보는 `Exp27(F)`로 결정한다. 



---
## 🛠️ 명명 규칙 (Tri-Match Naming Policy)
*   **원칙**: `학습파일명` == `YAML 내 name` == `추론파일명` == `runs 폴더명`을 1:1:1:1로 일치시킨다.
*   **예시**: 
    *   학습: `configs/train/exp12_train_yolo11s_noflip.yaml`
    *   내부: `name: "exp12_train_yolo11s_noflip"` (결과물은 `runs/exp12_train_yolo11s_noflip` 자동 생성)
    *   추론: `configs/inference/exp12_inference_yolo11s_noflip.yaml`
*   **이유**: 수백 개의 실험이 쌓였을 때, 특정 가중치가 어떤 설정(해상도, 모델, 증강)으로 학습되었는지 즉각적으로 추적하기 위함.

---

## 🛠️ 실행 가이드 (How to Reproduce)

1.  **`preprocessing.py`** 설정 
    * `USE_STRATIFIED`: 계층적 분할 여부 세팅
    * `USE_COPY_PASTE`: Copy-Paste 증강 여부 세팅
2.  **`train_yolo.py`** 실행 (학습)
    * `python train_yolo.py --config configs/train/exp11_train_yolo11s_flip.yaml`
3.  **`exp8_search.py`** 실행 (NMS 최적화)
    * `python -u src/exp8_search.py --device 0 --verbose --confs 0.20 --ious 0.50,0.60,0.70`
4.  **`test_custom.py`** 실행 (추론 + Config 자동 반영)
    * `python src/test_custom.py --config configs/inference/exp8_inference.yaml`
---


