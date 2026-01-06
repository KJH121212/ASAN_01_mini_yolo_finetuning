import json
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import sys

# 경로 설정 (사용자 환경에 맞게 유지)
BASE_DIR = Path("/workspace/nas203/ds_RehabilitationMedicineData/IDs/tojihoo/ASAN_01_mini_yolo_finetuning/")
sys.path.append(str(BASE_DIR))
from funcs.data_utils import convert_json_to_yolo_kpt_fixed, create_yolo_dataset_structure

# ==========================================
# 1. 경로 및 데이터 로드
# ==========================================
DATA_DIR = Path("/workspace/nas203/ds_RehabilitationMedicineData/IDs/tojihoo/data")
CSV_PATH = DATA_DIR / "metadata.csv" 

# 메타데이터 로드
df = pd.read_csv(CSV_PATH)

# Train 및 Val 데이터만 필터링
target_df = df[(df['is_train'] == True) | (df['is_val'] == True)]

print(f"📊 총 처리 대상 폴더 수: {len(target_df)}개 (Train + Val)")

# ==========================================
# 2. 전체 데이터 순회 및 변환 실행
# ==========================================

total_success_files = 0  # 전체 변환 성공 파일 수 카운트
error_folders = []       # 문제가 발생한 폴더 목록

# target_df를 순회하도록 변경했습니다.
for idx, row in tqdm(target_df.iterrows(), total=len(target_df), desc="Processing Folders"):
    
    try:
        common_path = row['common_path']
        
        # 각 폴더별 경로 설정
        FRAME_DIR = DATA_DIR / "1_FRAME" / common_path
        INTERP_DIR = DATA_DIR / "4_INTERP_DATA" / common_path
        YOLO_DIR = DATA_DIR / "5_YOLO_TXT" / common_path
        
        # 저장할 폴더가 없으면 생성합니다.
        YOLO_DIR.mkdir(parents=True, exist_ok=True)
        
        # 이미지 파일 확인 (너비/높이 정보를 얻기 위해 하나만 읽음)
        img_files = list(FRAME_DIR.glob("*.jpg")) + list(FRAME_DIR.glob("*.png"))
        
        if not img_files:
            # 이미지가 없으면 해당 폴더는 건너뜁니다.
            continue

        # 첫 번째 이미지를 읽어 해상도(H, W) 정보를 가져옵니다.
        sample_img = cv2.imread(str(img_files[0]))
        if sample_img is None:
             # 이미지가 깨져있거나 읽을 수 없는 경우
            error_folders.append(f"{common_path} (Image Read Error)")
            continue
            
        H, W = sample_img.shape[:2]

        # JSON 파일 목록 가져오기
        json_files = list(INTERP_DIR.glob("*.json"))
        
        if not json_files:
             continue

        # 폴더 내 파일 변환 루프
        folder_success_count = 0
        for json_file in json_files:
            txt_file = YOLO_DIR / f"{json_file.stem}.txt"
            
            # 함수 호출
            if convert_json_to_yolo_kpt_fixed(json_file, txt_file, W, H):
                folder_success_count += 1
        
        total_success_files += folder_success_count

    except Exception as e:
        print(f"\n❌ 오류 발생 ({common_path}): {e}")
        error_folders.append(f"{common_path} ({str(e)})")
        continue

# ==========================================
# 3. 결과 요약
# ==========================================
print("\n" + "="*40)
print(f"✅ 총 변환된 파일 수: {total_success_files}개")

if error_folders:
    print(f"⚠️ 오류가 발생한 폴더 ({len(error_folders)}개):")
    for err in error_folders:
        print(f" - {err}")
else:
    print("✨ 모든 폴더가 오류 없이 처리되었습니다.")