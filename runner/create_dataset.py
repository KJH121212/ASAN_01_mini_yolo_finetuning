import os
import sys
import shutil
import yaml
import pandas as pd
from pathlib import Path
from tqdm import tqdm
# 경로 설정 (사용자 환경에 맞게 유지)
BASE_DIR = Path("/workspace/nas203/ds_RehabilitationMedicineData/IDs/tojihoo/ASAN_01_mini_yolo_finetuning/")
sys.path.append(str(BASE_DIR))
from funcs.data_utils import create_yolo_dataset_structure

if __name__ == "__main__":
    # 경로 설정
    DATA_DIR = Path("/workspace/nas203/ds_RehabilitationMedicineData/IDs/tojihoo/data")
    CSV_PATH = DATA_DIR / "metadata.csv" 
    TEST_DATASET_DIR = DATA_DIR / "6_YOLO_TRAINING_DATA/v1.0_step1"
    SAMPLING_STEP = 1

    # 데이터 로드
    print(f"📖 메타데이터 로드 중... ({CSV_PATH})")
    df = pd.read_csv(CSV_PATH)
    
    # Train 또는 Val로 마킹된 데이터만 필터링 (불필요한 루프 방지)
    target_df = df[(df['is_train'] == True) | (df['is_val'] == True)]
    print(f"🎯 처리 대상 폴더: {len(target_df)}개 (Train + Val)")

    # 함수 실행
    generated_yaml = create_yolo_dataset_structure(
        df=target_df, 
        dataset_dir=TEST_DATASET_DIR, 
        data_dir=DATA_DIR, 
        step=SAMPLING_STEP
    )
    
    print(f"\n✅ 모든 작업이 끝났습니다. 학습을 시작할 준비가 되었습니다!")
