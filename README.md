# 🏥 ASAN Rehabilitation Medicine - YOLOv11 Pose Fine-tuning

이 프로젝트는 재활 의학 데이터를 활용하여 YOLOv11-Pose (Keypoint Estimation) 모델을 파인튜닝하기 위한 전체 파이프라인을 제공합니다.  
데이터 전처리부터 모델 학습, 그리고 WandB를 통한 실험 관리까지 Docker 환경 위에서 수행됩니다.  

## 📂 Project Structure
.
├── 0_jupyter.sh                # [Entry] Docker 컨테이너 실행 및 Jupyter Lab 시작  
├── 1_yolo_finetunin.sh         # [Train] YOLO Fine-tuning 실행 스크립트 (SLURM/Batch)  
├── config/                     # [Config] 실험 설정 파일 (Hyperparameters)  
│   ├── exp_v1.0_step1.yaml  
│   ├── exp_v1.0_step15.yaml  
│   └── exp_v1.0_step30.yaml  
├── docker/                     # [Env] Docker 환경 설정  
│   ├── Dockerfile  
│   └── requirements.txt  
├── funcs/                      # [Utils] 데이터 처리 유틸리티  
├── runner/                     # [Core] 핵심 실행 스크립트 (전처리 및 학습)  
│   ├── create_dataset.py  
│   ├── json2yolo.py  
│   └── yolo_finetuning.py  
├── tree.txt                    # 프로젝트 구조 트리  
└── yolo.ipynb                  # 실험 및 테스트용 노트북  


## ⚙️ Prerequisites

### Environment Setup (.env)

프로젝트 루트 경로에 .env 파일을 생성하여 WandB API Key를 설정합니다.  
이 파일은 보안을 위해 Git에 업로드되지 않아야 합니다.  

*.env 파일 생성*  
WANDB_API_KEY=your_wandb_api_key_here


## 🚀 Workflow & Scripts Description

이 프로젝트의 핵심 워크플로우는 runner/ 디렉토리 내의 파이썬 스크립트로 구성되며, 모든 데이터 처리는 metadata.csv를 기준으로 동작합니다.

---
### 1. 라벨 포맷 변환 (runner/json2yolo.py)

Role: Raw 데이터(JSON)를 YOLO 학습용 포맷(.txt)으로 변환합니다.

Logic:

metadata.csv 파일을 로드합니다.

is_train 또는 is_val 컬럼이 True인 행(Row)들을 필터링합니다.

해당 행의 JSON 파일을 읽어 YOLO Pose 포맷(Normalized xywh + keypoints)으로 변환하여 저장합니다.

---
### 2. 데이터셋 구축 (runner/create_dataset.py)

Role: YOLO 학습을 위한 디렉토리 구조(images/train, labels/val 등)를 생성합니다.

Optimization (Symlink):

metadata.csv를 참조하여 학습/검증 데이터셋 폴더를 구성합니다.

원본 이미지 파일을 복제하지 않고 **심볼릭 링크(Symlink)**를 사용하여 디스크 용량을 최소화하고 데이터셋 생성 속도를 비약적으로 높입니다.

---
### 3. 모델 학습 (runner/yolo_finetuning.py)

Role: 설정된 Config 파일을 기반으로 모델 학습을 수행하고 로그를 기록합니다.

Logic:

config/*.yaml 파일의 설정을 로드합니다.

YOLOv11-Pose 모델을 Fine-tuning 합니다.

학습 과정(Loss, mAP 등)을 WandB에 실시간으로 기록합니다.

중단된 학습을 감지하면 자동으로 last.pt를 로드하여 Resume 기능을 수행합니다.

---
### 📝 Configuration Guide

실험 설정은 config/ 디렉토리 내의 YAML 파일로 관리합니다. 새로운 실험을 시작할 때 다음 항목을 수정하세요.

config/exp_v1.0_stepXX.yaml 주요 항목

run_name: 실험 식별자입니다. WandB 프로젝트 내의 Run 이름 및 체크포인트 저장 폴더명으로 사용됩니다.

config_path: 학습에 사용할 데이터 정보 파일(data.yaml)의 경로입니다. Step 별로 다른 데이터셋을 연결할 때 이 경로를 변경합니다.

## 📊 Experiment Monitoring

모든 학습 로그와 시각화 결과는 Weights & Biases (WandB) 대시보드에서 확인할 수 있습니다.

WandB Dashboard: [ASAN_YOLO_FINETUNING Project Overview](https://wandb.ai/KJH1358/ASAN_YOLO_FINETUNING?nw=nwuserjihu6033)

👨‍💻 Maintainer

Developer: ojihoo (jihu6033@gmail.com)