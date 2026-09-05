#!/bin/bash

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMAGE="ultralytics/ultralytics:latest-jetson-jetpack6"
PIPE="/tmp/nvidia-mps"
MPSLOG="/tmp/nvidia-log"
DELAY=5
LOG_DIR="$ROOT/all_logs"
NSYS_DIR="$LOG_DIR/nsys"

mkdir -p "$LOG_DIR" "$NSYS_DIR" "$PIPE" "$MPSLOG"

# 1. 환경변수 설정 (pidstat 시간을 ISO 8601 형식으로 기록하여 tegrastats와 매칭 용이)
export S_TIME_FORMAT=ISO

if ! pgrep -f nvidia-cuda-mps-control >/dev/null; then
    echo "MPS 설정되지 않음"
    exit 1
fi

# 2. tegrastats 시작
sudo tegrastats --stop >/dev/null 2>&1 || true
sudo tegrastats --interval 100 --logfile "$LOG_DIR/all_tegrastat.log" &

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
output = f'/home/all_logs/nsys/nsys_{task_name}'
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
PY" > "$NSYS_DIR/nsys_$NAME.log" 2>&1 &

    DOCKER_RUN_PID=$! # docker run 명령의 PID

    # 컨테이너가 생성되고 실제 Python workload가 뜰 때까지 최대 10초간 반복 확인
    ACTUAL_PID=""
    for i in $(seq 1 240); do
        # 1단계: 컨테이너 내부에서 실행되는 실제 Python workload의 호스트 PID 찾기
        ACTUAL_PID=$(sudo docker top "$NAME" -eo pid,comm,args 2>/dev/null | \
            awk -v file="$FILE" '$2 ~ /^python/ && $0 ~ file {print $1; exit}')

        # 2단계: Python workload PID를 찾았으면 반복문 종료
        if [ ! -z "$ACTUAL_PID" ]; then
            break
        fi

        sleep 0.05 # 못 찾았으면 0.05초 대기 후 다시 시도
    done

    if [ ! -z "$ACTUAL_PID" ]; then

        # workload별 tegrastats 구간 파일
        INTERVAL_FILE="$LOG_DIR/tegrastat_${NAME}_interval.csv"

        # workload 시작 시점의 tegrastats line 위치 기록
        START_LINE=$(wc -l < "$LOG_DIR/all_tegrastat.log")

        # CPU + Memory 측정 시작
        pidstat \
            -p "$ACTUAL_PID" \
            -u \
            -r \
            -h \
            1 \
            > "$PIDSTAT_LOG" &

        PIDSTAT_MON_PID=$!

        # 실제 Python workload가 종료될 때까지 대기
        while ps -p "$ACTUAL_PID" >/dev/null 2>&1; do
            sleep 0.05
        done

        # workload 종료 시점의 tegrastats line 위치 기록
        END_LINE=$(wc -l < "$LOG_DIR/all_tegrastat.log")

        # CSV 저장
        echo "start_line,end_line" > "$INTERVAL_FILE"
        echo "$START_LINE,$END_LINE" >> "$INTERVAL_FILE"

        # pidstat 종료
        if [ ! -z "${PIDSTAT_MON_PID:-}" ]; then
            kill "$PIDSTAT_MON_PID" >/dev/null 2>&1 || true
            wait "$PIDSTAT_MON_PID" 2>/dev/null || true
        fi

    else

        echo "[WARNING] $NAME Python PID 찾을 수 없음"

        PIDSTAT_MON_PID=""

        echo "start_line,end_line" \
            > "$LOG_DIR/tegrastat_${NAME}_interval.csv"

    fi

    # Nsight report까지 완전히 생성될 때까지 Docker 종료 대기
    wait "$DOCKER_RUN_PID"
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

# Nsight Systems 결과를 SQLite 및 CSV로 저장
for task in detection classification estimation segmentation obb; do
    REPORT="$NSYS_DIR/nsys_${task}.nsys-rep"
    SQLITE="$NSYS_DIR/nsys_${task}.sqlite"

    if [ -f "$REPORT" ]; then
        # 1) 전체 trace 데이터를 SQLite로 export
        /usr/local/bin/nsys export \
            --type sqlite \
            --force-overwrite=true \
            --output "$SQLITE" \
            "$REPORT"

        # 2) CUDA kernel 실행 정보
        sqlite3 -header -csv "$SQLITE" "
SELECT
    k.start,
    k.end,
    (k.end - k.start) AS duration_ns,
    (k.end - k.start) / 1000000.0 AS duration_ms,
    s.value AS kernel_name,
    k.deviceId,
    k.contextId,
    k.streamId,
    k.correlationId,
    k.registersPerThread,
    k.gridX,
    k.gridY,
    k.gridZ,
    k.blockX,
    k.blockY,
    k.blockZ,
    k.staticSharedMemory,
    k.dynamicSharedMemory
FROM CUPTI_ACTIVITY_KIND_KERNEL AS k
LEFT JOIN StringIds AS s
    ON k.demangledName = s.id;
" > "$NSYS_DIR/nsys_${task}_kernel.csv"

        # 3) CUDA Runtime API 호출 정보
        sqlite3 -header -csv "$SQLITE" \
            "SELECT * FROM CUPTI_ACTIVITY_KIND_RUNTIME;" \
            > "$NSYS_DIR/nsys_${task}_runtime.csv"

        # 4) CUDA memory copy 정보
        sqlite3 -header -csv "$SQLITE" \
            "SELECT * FROM CUPTI_ACTIVITY_KIND_MEMCPY;" \
            > "$NSYS_DIR/nsys_${task}_memcpy.csv"

        # 5) CUDA synchronization 정보
        sqlite3 -header -csv "$SQLITE" \
            "SELECT * FROM CUPTI_ACTIVITY_KIND_SYNCHRONIZATION;" \
            > "$NSYS_DIR/nsys_${task}_synchronization.csv"

        # 6) NVTX event 정보
        sqlite3 -header -csv "$SQLITE" \
            "SELECT * FROM NVTX_EVENTS;" \
            > "$NSYS_DIR/nsys_${task}_nvtx.csv"

        # 7) OS runtime event 정보
        sqlite3 -header -csv "$SQLITE" \
            "SELECT * FROM OSRT_API;" \
            > "$NSYS_DIR/nsys_${task}_osrt.csv"
    fi
done

echo "All tasks done. Logs are in $LOG_DIR"