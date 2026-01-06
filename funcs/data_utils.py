import json
import yaml
import os
import shutil
import pandas as pd
from pathlib import Path
from tqdm import tqdm

# ==========================================
# 1. JSON -> YOLO TXT 변환 함수 (Head Padding 포함)
# ==========================================
def convert_json_to_yolo_kpt_fixed(json_path, txt_path, img_w, img_h, head_ratio=0.20, padding=20):
    """
    JSON 파일의 키포인트(5~16번)를 추출하여 YOLO Pose 포맷(.txt)으로 변환합니다.
    헤드 패딩(Head Padding)을 적용하여 머리 부분을 포함한 BBox를 자동 생성합니다.
    """
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        # 데이터 유효성 검사
        if 'instance_info' not in data or not data['instance_info']:
            return False
            
        person = data['instance_info'][0]
        raw_kpts = person.get('keypoints', []) # [[x,y], [x,y]...]

        if not raw_kpts: return False

        # --- [Logic] 5~16번 키포인트 추출 ---
        selected_kpts = []
        valid_x = []
        valid_y = [] 

        start_idx, end_idx = 5, 16
        
        if len(raw_kpts) <= end_idx: return False

        for i in range(start_idx, end_idx + 1):
            x, y = raw_kpts[i]
            
            # Visibility: 좌표가 있으면 2 (Visible)
            v = 2 if (x > 0 and y > 0) else 0

            # 정규화 (Normalization)
            nx = x / img_w
            ny = y / img_h
            selected_kpts.extend([nx, ny, v])

            # BBox 계산용 좌표 수집
            if x > 0 and y > 0:
                valid_x.append(x)
                valid_y.append(y)

        # 유효한 키포인트가 너무 적으면 변환 실패
        if len(valid_x) < 2: return False

        # --- [Logic] Bounding Box 자동 계산 (헤드 패딩 적용) ---
        
        # 1. 몸통 범위 계산
        min_x_body, max_x_body = min(valid_x), max(valid_x)
        min_y_body, max_y_body = min(valid_y), max(valid_y)
        
        # 2. 헤드 패딩 (Head Padding) 적용
        body_h = max_y_body - min_y_body
        head_extension = body_h * head_ratio 
        
        final_min_y = min_y_body - head_extension
        
        # 3. 기본 패딩 및 클리핑
        min_x = max(0, min_x_body - padding)
        min_y = max(0, final_min_y - padding)
        max_x = min(img_w, max_x_body + padding)
        max_y = min(img_h, max_y_body + padding)

        # 4. XYXY -> XYWH (Normalized Center)
        box_w = max_x - min_x
        box_h = max_y - min_y
        box_cx = min_x + (box_w / 2)
        box_cy = min_y + (box_h / 2)

        yolo_bbox = [
            box_cx / img_w,
            box_cy / img_h,
            box_w / img_w,
            box_h / img_h
        ]

        # --- [File Write] ---
        # Class 0 + BBox + Keypoints
        line = f"0 {' '.join(f'{v:.6f}' for v in yolo_bbox)} {' '.join(f'{v:.6f}' for v in selected_kpts)}\n"
        
        with open(txt_path, 'w') as f:
            f.write(line)
            
        return True

    except Exception as e:
        print(f"❌ Error converting {Path(json_path).name}: {e}")
        return False


# ==========================================
# 2. 데이터셋 구조화 및 샘플링 함수 (Symlink + Step)
# ==========================================
def create_yolo_dataset_structure(df, dataset_dir, data_dir, step=30):
    """
    DataFrame을 기반으로 YOLO 학습용 폴더 구조를 생성하고,
    지정된 프레임 간격(step)으로 데이터를 샘플링하여 연결합니다.
    (YAML 파일에 step 정보를 포함하여 저장합니다.)
    """
    print(f"🚀 [Sampling Mode] 데이터셋 구조화 시작 (간격: {step})")
    print(f"📂 저장 경로: {dataset_dir}")

    # 폴더 생성
    for split in ['train', 'val']:
        (dataset_dir / 'images' / split).mkdir(parents=True, exist_ok=True)
        (dataset_dir / 'labels' / split).mkdir(parents=True, exist_ok=True)

    counts = {'train': 0, 'val': 0, 'skip': 0, 'fixed': 0}
    
    # tqdm 진행률 표시
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Linking Files"):
        if row.get('is_train') == True: split = 'train'
        elif row.get('is_val') == True: split = 'val'
        else: continue 

        common_path = row['common_path']
        src_label_dir = data_dir / "5_YOLO_TXT" / common_path
        src_image_dir = data_dir / "1_FRAME" / common_path

        if not src_label_dir.exists() or not src_image_dir.exists():
            continue

        label_files = sorted(list(src_label_dir.glob("*.txt")))
        if not label_files: continue

        # Step 간격 샘플링
        sampled_files = label_files[::step]

        for label_file in sampled_files:
            file_stem = label_file.stem
            
            image_file = src_image_dir / f"{file_stem}.jpg"
            if not image_file.exists():
                image_file = src_image_dir / f"{file_stem}.png"
            if not image_file.exists(): continue

            safe_common_path = common_path.replace("/", "_").replace("\\", "_")
            unique_name = f"{safe_common_path}_{file_stem}"

            dst_image = dataset_dir / 'images' / split / f"{unique_name}{image_file.suffix}"
            dst_label = dataset_dir / 'labels' / split / f"{unique_name}.txt"

            if dst_image.is_symlink() and not dst_image.exists():
                dst_image.unlink()

            if dst_image.exists() and dst_label.exists():
                counts['skip'] += 1
                continue

            try:
                if not dst_image.exists():
                    os.symlink(image_file, dst_image)
                    counts['fixed'] += 1 # 심볼릭 링크 생성 시 카운트
                
                if not dst_label.exists():
                    shutil.copy2(label_file, dst_label)
                
                # 카운트 로직: 새로 링크를 걸었거나(fixed), 이미 존재해서 건너뛰지 않았을 때
                # 여기서는 루프를 돌 때마다 해당 split 카운트를 올리는 것이 직관적이므로 수정
                counts[split] += 1
                
            except OSError as e:
                print(f"❌ 에러: {e}")

    # ---------------------------------------------------------
    # ✅ [수정됨] data.yaml 생성 (sampling_step 정보 추가)
    # ---------------------------------------------------------
    yaml_content = {
        'path': str(dataset_dir.absolute()),
        'sampling_step': step,               
        'train': 'images/train',
        'val': 'images/val',
        'names': {0: 'person'},
        'kpt_shape': [12, 3],
        'flip_idx': [1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10]
    }

    yaml_path = dataset_dir / "data.yaml"
    with open(yaml_path, 'w') as f:
        yaml.dump(yaml_content, f, sort_keys=False)

    print("\n📊 [완료] 데이터셋 구축 결과:")
    print(f"   - 적용 Step: {step}")
    print(f"   - Train Images: {counts['train']:,} 장")
    print(f"   - Val Images:   {counts['val']:,} 장")
    print(f"   - YAML Path:    {yaml_path}")
    
    return yaml_path