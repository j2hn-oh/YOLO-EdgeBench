#!/bin/bash

ROOT="/home/orin/yolo"
IMAGE="ultralytics/ultralytics:latest-jetson-jetpack6"
PIPE="/tmp/nvidia-mps"
MPSLOG="/tmp/nvidia-log"
DELAY=5
LOG_DIR="$ROOT/all_logs"

mkdir -p "$LOG_DIR" "$PIPE" "$MPSLOG"

# 1. 환경변수 설정 (pidstat 시간을 ISO 8601 형식으로 기록하여 tegrastats와 매칭 용이)
export S_TIME_FORMAT=ISO

if ! pgrep -f nvidia-cuda-mps-control >/dev/null; then
    echo "MPS 설정되지 않음"
    exit 1
fi

# 2. tegrastats 시작
sudo tegrastats --stop >/dev/null 2>&1 || true
sudo tegrastats --interval 1000 --logfile "$LOG_DIR/all_tegrastat.log" &

TARGET_NS=$(python3 -c "import time; print(time.time_ns() + $DELAY * 1000000000)")

run_task() {
    NAME=$1
    FILE=$2
    PIDSTAT_LOG="$LOG_DIR/pidstat_$NAME.log"

    sudo docker rm -f "$NAME" >/dev/null 2>&1 || true

    # 컨테이너 실행
    sudo docker run --rm \
        --name "$NAME" \
        --runtime=nvidia \
        --gpus=all \
        --cap-add=SYS_ADMIN \
        -v "$ROOT:/home" \
        -v "$PIPE:$PIPE" \
        -v "$MPSLOG:$MPSLOG" \
        -v /opt/nvidia/nsight-systems/2024.5.4:/opt/nsys:ro \
        -e CUDA_MPS_PIPE_DIRECTORY="$PIPE" \
        -e CUDA_MPS_LOG_DIRECTORY="$MPSLOG" \
        -e CUDA_MPS_ENABLE_PER_CTX_DEVICE_MULTIPROCESSOR_PARTITIONING=1 \
        -e CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=20 \
        -e TARGET_NS="$TARGET_NS" \
        -e FILE_NAME="$FILE" \
        "$IMAGE" \
        /bin/bash -lc "cd /home && python3 - << 'PY'
import os
import time
import shlex

target_ns = int(os.environ['TARGET_NS'])
file_name = os.environ['FILE_NAME']

# 모든 workload가 동일한 TARGET_NS까지 대기
while time.time_ns() < target_ns:
    time.sleep(0.0005)

# Host의 Nsight Systems를 container에 mount하여 사용
nsys = '/opt/nsys/target-linux-tegra-armv8/nsys'
task_name = os.path.splitext(file_name)[0]
output = f'/home/all_logs/nsys_{task_name}'
task_log = f'/home/all_logs/{task_name}.log'

# 실제 workload의 stdout/stderr는 task별 log에 직접 기록
command = (
    f'exec python3 {shlex.quote(file_name)} '
    f'> {shlex.quote(task_log)} 2>&1'
)

os.execvp(
    nsys,
    [
        nsys,
        'profile',
        '--trace=cuda,nvtx,osrt',
        '--force-overwrite=true',
        '-o',
        output,
        '/bin/bash',
        '-lc',
        command
    ]
)
PY" > "$LOG_DIR/nsys_$NAME.log" 2>&1 &

    DOCKER_RUN_PID=$! # docker run 명령의 PID

    # 컨테이너가 생성되고 실제 Python workload가 뜰 때까지 최대 10초간 반복 확인
    ACTUAL_PID=""
    for i in {1..20}; do
        # 1단계: 컨테이너 내부에서 실행되는 실제 Python workload의 호스트 PID 찾기
        ACTUAL_PID=$(sudo docker top "$NAME" -eo pid,comm,args 2>/dev/null | \
            awk -v file="$FILE" '$2 ~ /^python/ && $0 ~ file {print $1; exit}')

        # 2단계: Python workload PID를 찾았으면 반복문 종료
        if [ ! -z "$ACTUAL_PID" ]; then
            break
        fi

        sleep 0.5 # 못 찾았으면 0.5초 대기 후 다시 시도
    done

    if [ ! -z "$ACTUAL_PID" ]; then
        # 3) pidstat 시작 (CPU + MEM)
        pidstat -p "${ACTUAL_PID}" -u -r -h 1 > "${PIDSTAT_LOG}" &
        PIDSTAT_MON_PID=$!
    else
        echo "python PID 찾을 수 없어 pidstat을 실행X"
        PIDSTAT_MON_PID=""
    fi

    # Docker 작업이 끝날 때까지 대기
    wait $DOCKER_RUN_PID

    # 작업 종료 후 pidstat 프로세스 종료
    [ ! -z "$PIDSTAT_MON_PID" ] && kill $PIDSTAT_MON_PID 2>/dev/null
}

# 각 테스크 병렬 실행
run_task detection detection.py &
P1=$!
run_task classification classification.py &
P2=$!
run_task estimation estimation.py &
P3=$!
run_task segmentation segmentation.py &
P4=$!
run_task obb obb.py &
P5=$!

wait $P1 $P2 $P3 $P4 $P5

sudo tegrastats --stop >/dev/null 2>&1 || true
echo "All tasks done. Logs are in $LOG_DIR"


