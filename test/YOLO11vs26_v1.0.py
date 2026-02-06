import cv2
import time
import torch
import os
import pandas as pd
from pathlib import Path
from ultralytics import YOLO
from tqdm import tqdm

# 1. 경로 및 기본 설정
DATA_DIR = Path("/workspace/nas203/ds_RehabilitationMedicineData/IDs/tojihoo/data")
YOLO11_PATH = DATA_DIR / "checkpoints/YOLO/" / "yolo11m.pt"
YOLO26_PATH = DATA_DIR / "checkpoints/YOLO/" / "yolo26m.pt"
YOLO11_POSE_PATH = DATA_DIR / "checkpoints/YOLO/" / "yolo11m-pose.pt"
YOLO26_POSE_PATH = DATA_DIR / "checkpoints/YOLO/" / "yolo26m-pose.pt"

# 메타데이터 로드
metadata_path = DATA_DIR / "metadata.csv"
if not metadata_path.exists():
    raise FileNotFoundError("metadata.csv 파일을 찾을 수 없습니다.")
df = pd.read_csv(metadata_path)

# 2. GPU 장치 설정
device = 0 if torch.cuda.is_available() else 'cpu'
if device == 0:
    torch.cuda.set_device(device)
    print(f"✅ GPU 가속 활성화: {torch.cuda.get_device_name(0)}")

# 3. 모델 로드 (루프 외부에서 단 한 번만 수행하여 메모리를 절약합니다)
print("📦 모델 로딩 중... (한 번만 실행됩니다)")
model_info = [
    {"name": "YOLO11m", "path": str(YOLO11_PATH)},
    {"name": "YOLO26m", "path": str(YOLO26_PATH)},
    {"name": "YOLO11m-Pose", "path": str(YOLO11_POSE_PATH)},
    {"name": "YOLO26m-Pose", "path": str(YOLO26_POSE_PATH)}
]
# 모델을 GPU로 이동시킵니다.
models = [YOLO(m["path"]).to(device) for m in model_info]
print("🚀 모델 로드 완료!")

# 4. Target 루프 실행 (예: 0부터 32까지)
# 원하시는 범위로 수정 가능합니다: range(0, 32) -> 0부터 31까지
TARGET_START = 35
TARGET_END = 40

for target_idx in range(TARGET_START, TARGET_END):
    try:
        # 데이터프레임 인덱스 확인
        if target_idx not in df.index:
            print(f"⚠️ Target {target_idx}: 메타데이터에 존재하지 않아 건너뜁니다.")
            continue

        COMMON_PATH = df.loc[target_idx, "common_path"]
        
        # 입력 및 출력 경로 설정
        FRAME_DIR = DATA_DIR / "1_FRAME" / COMMON_PATH
        OUTPUT_DIR = DATA_DIR / "test" / COMMON_PATH
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        
        # 프레임 파일 확보
        frame_files = sorted([f for f in FRAME_DIR.glob("*.jpg")])
        if not frame_files:
            print(f"⚠️ Target {target_idx} ({COMMON_PATH}): 프레임이 없어 건너뜁니다.")
            continue

        # 비디오 저장 설정
        sample_img = cv2.imread(str(frame_files[0]))
        h, w = sample_img.shape[:2]
        
        video_filename = f"Comparison_v1.0.mp4"
        output_video_path = str(OUTPUT_DIR / video_filename)
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_video_path, fourcc, 30.0, (w * 2, h * 2))
        
        print(f"\n▶️ [Target {target_idx}] 분석 시작: {len(frame_files)} Frames")
        print(f"   📂 저장 경로: {output_video_path}")

        # --- 프레임 처리 루프 (tqdm 적용) ---
        # desc에 현재 target 번호를 표시하여 진행 상황을 명확히 합니다.
        for frame_path in tqdm(frame_files, desc=f"Target {target_idx}", unit="frame"):
            input_img = cv2.imread(str(frame_path))
            processed_results = []

            for i, model in enumerate(models):
                # 정밀 시간 측정을 위한 GPU 동기화
                if device != 'cpu': torch.cuda.synchronize()
                
                start_t = time.perf_counter()
                
                # Tracking 수행 (persist=True로 ID 유지)
                # imgsz=640으로 고정하여 추론 속도를 최적화합니다.
                result = model.track(input_img, imgsz=640, classes=[0], device=device, 
                                     persist=True, verbose=False)[0]
                
                if device != 'cpu': torch.cuda.synchronize()
                end_t = time.perf_counter()
                
                fps = 1.0 / (end_t - start_t)
                
                # 시각화 (Bounding Box + ID + Skeleton)
                res_frame = result.plot()
                
                # 정보 텍스트 오버레이
                display_text = f"{model_info[i]['name']} | FPS: {fps:.1f}"
                font_scale, thickness = 1.0, 2
                text_size = cv2.getTextSize(display_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
                
                tx, ty = w - text_size[0] - 20, 45 
                # 가독성을 위한 검은색 배경 박스
                cv2.rectangle(res_frame, (tx - 5, ty - text_size[1] - 5), (tx + text_size[0] + 5, ty + 5), (0, 0, 0), -1)
                # 녹색 텍스트
                cv2.putText(res_frame, display_text, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), thickness, cv2.LINE_AA)
                
                processed_results.append(res_frame)

            # 4분할 화면 병합 (2x2 Grid)
            top_row = cv2.hconcat([processed_results[0], processed_results[1]])
            bottom_row = cv2.hconcat([processed_results[2], processed_results[3]])
            final_frame = cv2.vconcat([top_row, bottom_row])
            
            out.write(final_frame)

        # 현재 Target 작업 종료 및 리소스 해제
        out.release()
        
    except Exception as e:
        print(f"\n❌ [Error] Target {target_idx} 처리 중 오류 발생: {e}")
        # 오류가 나도 다음 Target으로 넘어가도록 continue 처리할 수 있습니다.
        if 'out' in locals(): out.release()
        continue

print("\n🎉 모든 Target에 대한 분석 작업이 완료되었습니다!")