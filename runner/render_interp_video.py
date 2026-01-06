import shutil
from pathlib import Path
import pandas as pd

# Path 설정 (기존과 동일)
DATA_DIR = Path("/workspace/nas203/ds_RehabilitationMedicineData/IDs/tojihoo/data")
CSV_PATH = DATA_DIR / "metadata.csv"

# 데이터 로드
df = pd.read_csv(CSV_PATH)
train_df = df[df['is_train'] == True]

print(f"총 {len(train_df)}개의 데이터 처리를 시작합니다. (고속 Resume 모드)")

for idx, row in train_df.iterrows():
    c_path = row['common_path']
    
    src_dir = DATA_DIR / "2_KEYPOINTS" / c_path
    dst_dir = DATA_DIR / "4_INTERP_DATA" / c_path
    
    # ✅ [핵심] 마커 파일 경로 정의
    # 이 파일이 있다는 것은 지난번에 복사가 '성공적으로' 끝났음을 의미합니다.
    marker_file = dst_dir / "copy_done.txt"

    # 1. 마커 파일이 존재하면 즉시 건너뜀 (가장 빠른 체크)
    if marker_file.exists():
        # print(f"⏭️  Skip: {c_path}") # 속도를 위해 로그도 생략 가능
        continue

    # 2. 원본 확인 및 복사 시작
    if src_dir.exists():
        dst_dir.mkdir(parents=True, exist_ok=True)
        
        # 실제 복사 로직
        src_files = list(src_dir.glob("*.json"))
        if src_files:
            for f in src_files:
                shutil.copy2(f, dst_dir)
            
            # ✅ [중요] 복사가 에러 없이 다 끝나면 마커 파일 생성 (빈 파일)
            marker_file.touch()
            print(f"✅ 완료: {c_path} ({len(src_files)} files)")
        else:
            print(f"⚠️ 빈 폴더: {c_path}")
            # 빈 폴더도 '처리 완료'로 볼 것인지에 따라 여기서 touch()를 할지 결정합니다.
            # 보통은 빈 폴더면 넘어가므로 touch() 안함.
            
    else:
        print(f"❌ 원본 없음: {c_path}")

print("\n모든 작업이 종료되었습니다.")