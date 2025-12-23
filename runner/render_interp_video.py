import pandas as pd
from pathlib import Path
import sys
import os
import cv2
import json
import numpy as np

# -----------------------------------------------------------
# 1. 핵심 렌더링 함수 (수정됨)
# -----------------------------------------------------------
def render_from_interp_json(frame_dir, interp_dir, output_path, fps=30):
    """
    JSON 데이터와 이미지 프레임을 읽어 비디오를 생성합니다.
    - 선: 검정색
    - 점: 왼쪽(파랑), 오른쪽(빨강)
    - KPT 0~4 (얼굴) 제외
    - Confidence Score 무시하고 강제 렌더링
    """
    frame_dir = Path(frame_dir)
    interp_dir = Path(interp_dir)
    output_path = Path(output_path)

    # 1. 파일 목록 정렬
    json_files = sorted(list(interp_dir.glob("*.json")))
    if not json_files:
        print(f"   ❌ [Error] JSON 파일을 찾을 수 없습니다: {interp_dir}")
        return False

    # 2. 첫 번째 이미지를 로드하여 비디오 크기 확인
    jpg_files = sorted(list(frame_dir.glob("*.jpg")))
    if not jpg_files:
        print(f"   ❌ [Error] JPG 이미지 파일을 찾을 수 없습니다: {frame_dir}")
        return False

    sample_frame = cv2.imread(str(jpg_files[0]))
    if sample_frame is None:
        print("   ❌ [Error] 첫 번째 프레임을 읽을 수 없습니다.")
        return False
        
    h, w, _ = sample_frame.shape
    img_center = np.array([w / 2, h / 2])

    # 3. 비디오 라이터 설정
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (w, h))

    print(f"   🎬 렌더링 시작: {output_path.name} (Total frames: {len(json_files)})")

    # 내부 함수: 중심 인물 찾기
    # (렌더링은 score 무시하지만, '누가 환자인지' 고를 때는 유효한 점들만 쓰는 것이 안전하여 유지)
    def _get_center_person(instances, center_point):
        best_instance = None
        min_dist = float('inf')

        for instance in instances:
            kpts = np.array(instance['keypoints'])
            scores = np.array(instance['keypoint_scores'])
            
            # 중심점 계산 시에는 너무 튀는 값 제외를 위해 최소한의 threshold 유지 (선택 로직용)
            valid_kpts = kpts[scores > 0.05]
            if len(valid_kpts) == 0:
                # 만약 모든 점수가 낮다면 그냥 전체 평균 사용
                valid_kpts = kpts
            
            person_center = np.mean(valid_kpts, axis=0)
            dist = np.linalg.norm(person_center - center_point)

            if dist < min_dist:
                min_dist = dist
                best_instance = instance
        
        return best_instance

    # 4. 프레임 루프
    for json_file in json_files:
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        img_path = frame_dir / f"{json_file.stem}.jpg"
        if not img_path.exists():
            continue

        frame = cv2.imread(str(img_path))
        if frame is None:
            continue

        # 메타 정보
        meta = data.get('meta_info', {})
        skeleton_links = meta.get('skeleton_links', [])
        
        # 인스턴스 정보
        instances = data.get('instance_info', [])
        target_instance = _get_center_person(instances, img_center)

        # -------------------------------------------------------
        # [수정] 렌더링 로직
        # -------------------------------------------------------
        if target_instance:
            kpts = np.array(target_instance['keypoints'])
            # scores는 요청에 의해 그리는 조건에서 제외됨

            # 제외할 관절 인덱스 (코, 눈, 귀)
            excluded_indices = {0, 1, 2, 3, 4}

            # (1) 스켈레톤 연결선 그리기 (검정색)
            if skeleton_links:
                for link in skeleton_links:
                    idx1, idx2 = link
                    
                    # 인덱스 범위 체크
                    if idx1 >= len(kpts) or idx2 >= len(kpts): continue

                    # 0~4번과 연결된 선은 그리지 않음
                    if idx1 in excluded_indices or idx2 in excluded_indices:
                        continue

                    # 점수 체크 없이 무조건 그리기
                    p1 = tuple(kpts[idx1].astype(int))
                    p2 = tuple(kpts[idx2].astype(int))
                    
                    # 색상: 검정색 (BGR: 0, 0, 0)
                    cv2.line(frame, p1, p2, (0, 0, 0), 2)

            # (2) 관절 포인트 그리기 (왼쪽:파랑, 오른쪽:빨강)
            for i, pt in enumerate(kpts):
                # 0~4번 포인트 그리지 않음
                if i in excluded_indices:
                    continue
                
                # 색상 결정 (COCO 포맷 기준)
                # 5, 7, 9... (홀수) : 왼쪽 -> 파란색 (255, 0, 0)
                # 6, 8, 10... (짝수) : 오른쪽 -> 빨간색 (0, 0, 255)
                if i % 2 != 0:
                    color = (255, 0, 0) # Blue
                else:
                    color = (0, 0, 255) # Red

                cv2.circle(frame, tuple(pt.astype(int)), 4, color, -1)

        out.write(frame)

    out.release()
    return True

# -----------------------------------------------------------
# 2. 메인 실행부 (기존과 동일)
# -----------------------------------------------------------

new_meta = Path("/workspace/nas203/ds_RehabilitationMedicineData/IDs/tojihoo/data/metadata.csv")
DATA_DIR = Path("/workspace/nas203/ds_RehabilitationMedicineData/IDs/tojihoo/data")
BASE_DIR = Path("/workspace/nas203/ds_RehabilitationMedicineData/IDs/tojihoo/ASAN_01_mini_lebeling_postprocessing")

sys.path.append(str(BASE_DIR))

# 데이터 로드
if not new_meta.exists():
    print(f"❌ 메타데이터 파일을 찾을 수 없습니다: {new_meta}")
    sys.exit()

df = pd.read_csv(new_meta)

# ⭐ [사용자 설정] 처리하고 싶은 행 인덱스
targets = df.index.tolist()

print(f"🚀 총 {len(targets)}개의 데이터를 처리를 시작합니다.")
print("-" * 60)

for current_count, target_idx in enumerate(targets):
    try:
        if target_idx not in df.index:
            print(f"⚠️ [Skip] Index {target_idx} is not in metadata.")
            continue

        common_path = df.loc[target_idx, 'common_path']
        
        frame_dir = DATA_DIR / f"1_FRAME/{common_path}"
        interp_dir = DATA_DIR / f"4_INTERP_DATA/{common_path}"
        output_mp4_path = DATA_DIR / f"7_INTERP_MP4/{common_path}.mp4"

        print(f"[{current_count + 1}/{len(targets)}] Processing Index {target_idx}: {common_path}")

        output_mp4_path.parent.mkdir(parents=True, exist_ok=True)

        success = render_from_interp_json(
            frame_dir=frame_dir,
            interp_dir=interp_dir,
            output_path=output_mp4_path,
            fps=30
        )
        
        if success:
            print(f"   ✅ Render Completed: {output_mp4_path.name}")
        else:
            print(f"   ⚠️ Render Failed or Skipped")

    except Exception as e:
        print(f"   ❌ Error at Index {target_idx}: {e}")
        import traceback
        traceback.print_exc()
        continue

print("-" * 60)
print("🎉 지정된 모든 행의 처리가 완료되었습니다.")