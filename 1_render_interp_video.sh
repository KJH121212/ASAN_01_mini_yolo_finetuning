#!/bin/bash
#SBATCH -J tojihoo_yolo_finetune
#SBATCH -t 7-00:00:00
#SBATCH -o /home/tojihoo/logs/%A.out
#SBATCH --mail-type END,TIME_LIMIT_90,REQUEUE,INVALID_DEPEND
#SBATCH --mail-user jihu6033@gmail.com
#SBATCH -p RTX3090
#SBATCH --gpus 1

# ------------------------------------------------------------
# 환경 설정
# ------------------------------------------------------------
export HTTP_PROXY=http://192.168.45.108:3128
export HTTPS_PROXY=http://192.168.45.108:3128
export http_proxy=http://192.168.45.108:3128
export https_proxy=http://192.168.45.108:3128

DOCKER_IMAGE_NAME="tojihoo/yolo"
DOCKER_CONTAINER_NAME="tojihoo_yolo_step30"
DOCKERFILE_PATH="/mnt/nas203/ds_RehabilitationMedicineData/IDs/tojihoo/ASAN_01_mini_yolo_finetuning/docker/Dockerfile"
RANDOM_PORT=$(shuf -i 8000-9000 -n 1) 

# ------------------------------------------------------------
# Docker 이미지 빌드
# ------------------------------------------------------------
# 'docker images -q'는 이미지 ID만 반환합니다. 결과가 비어있으면("") 이미지가 없는 것입니다.
if [[ "$(docker images -q ${DOCKER_IMAGE_NAME} 2> /dev/null)" == "" ]]; then
    echo "[INFO] Image not found. Building Docker image: ${DOCKER_IMAGE_NAME}"
    
    # 주의: docker build 명령어 끝에 빌드 컨텍스트(경로)가 필요합니다. 
    # 보통 Dockerfile이 있는 폴더나 현재 폴더(.)를 지정합니다. 아래에 ${BUILD_CONTEXT}를 추가했습니다.
    docker build -t ${DOCKER_IMAGE_NAME} -f ${DOCKERFILE_PATH} ${BUILD_CONTEXT}
    
    if [ $? -ne 0 ]; then
        echo "[❌ ERROR] Docker build failed."
        exit 1
    fi
else
    echo "[INFO] Image ${DOCKER_IMAGE_NAME} already exists. Skipping build."
fi

# ------------------------------------------------------------
# Docker 컨테이너 실행
# ------------------------------------------------------------
echo "[INFO] Running container: ${DOCKER_CONTAINER_NAME}"
docker run -it --device=nvidia.com/gpu=all --shm-size 1TB \
    --name "${DOCKER_CONTAINER_NAME}" \
    -e JUPYTER_ENABLE_LAB=yes \
    -p ${RANDOM_PORT}:${RANDOM_PORT} \
    -v /mnt:/workspace \
    -e HTTP_PROXY=${HTTP_PROXY} \
    -e HTTPS_PROXY=${HTTPS_PROXY} \
    -e http_proxy=${http_proxy} \
    -e https_proxy=${https_proxy} \
    --rm \
    ${DOCKER_IMAGE_NAME} \
    bash -c "
        cd /workspace/nas203/ds_RehabilitationMedicineData/IDs/tojihoo/ASAN_01_mini_yolo_finetuning/runner && \
        python3 yolo_finetuning.py
    "