#!/bin/bash

set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

IMAGE="ultralytics/ultralytics:latest-jetson-jetpack6"

PIPE="/tmp/nvidia-mps"

NSYS_ROOT="/opt/nvidia/nsight-systems/2024.5.4"
NSYS_BIN="/opt/nvidia/nsight-systems/2024.5.4/target-linux-tegra-armv8/nsys"

DELAY="${DELAY:-5}"
COOLDOWN="${COOLDOWN:-5}"
REPEATS="${REPEATS:-1}"

LOG_DIR="$ROOT/isolated_logs"

mkdir -p "$LOG_DIR"
mkdir -p "$LOG_DIR/nsys"

export S_TIME_FORMAT=ISO

declare -A FILES

FILES[detection]="detection.py"
FILES[classification]="classification.py"
FILES[estimation]="estimation.py"
FILES[segmentation]="segmentation.py"
FILES[obb]="obb.py"

WORKLOADS=(
    detection
    classification
    estimation
    segmentation
    obb
)

check_mps_off()
{
    echo "============================================================"
    echo "[CHECK] NVIDIA MPS"
    echo "============================================================"

    if pgrep -f '[n]vidia-cuda-mps-control|[n]vidia-cuda-mps-server' \
        >/dev/null 2>&1; then

        echo "[ERROR] MPS가 실행 중입니다."
        echo "MPS를 종료한 후 다시 실행하세요."
        pgrep -af '[n]vidia-cuda-mps-control|[n]vidia-cuda-mps-server'
        exit 1
    fi

    echo "[OK] MPS OFF"
    echo

    unset CUDA_MPS_PIPE_DIRECTORY 2>/dev/null || true
    unset CUDA_MPS_LOG_DIRECTORY 2>/dev/null || true
}

check_nsys()
{
    echo "============================================================"
    echo "[CHECK] Nsight Systems"
    echo "============================================================"

    if [ ! -x "$NSYS_BIN" ]; then
        echo "[ERROR] Host Nsight Systems를 찾을 수 없습니다."
        echo "$NSYS_BIN"
        exit 1
    fi

    echo "[INFO] Host Nsight:"
    "$NSYS_BIN" --version
    echo

    if sudo docker run --rm \
        --runtime=nvidia \
        --gpus=all \
        --cap-add=SYS_ADMIN \
        -v "$NSYS_ROOT:$NSYS_ROOT:ro" \
        "$IMAGE" \
        /bin/bash -lc \
        "$NSYS_BIN --version"
    then
        echo "[OK] Docker에서 Host Nsight 사용 가능"
        echo
    else
        echo "[ERROR] Docker 내부에서 mounted Nsight 실행 실패"
        exit 1
    fi
}

check_mps_off
check_nsys

echo "[INFO] Cleaning previous logs..."

for NAME in "${WORKLOADS[@]}"; do
    rm -f "$LOG_DIR/${NAME}.log"
    rm -f "$LOG_DIR/pidstat_${NAME}.log"
done

rm -f "$LOG_DIR/all_tegrastat.log"

rm -rf "$LOG_DIR/nsys"
mkdir -p "$LOG_DIR/nsys"

CURRENT_CONTAINER=""
PIDSTAT_PID=""

cleanup()
{
    if [ -n "${PIDSTAT_PID:-}" ]; then
        kill "$PIDSTAT_PID" \
            >/dev/null 2>&1 || true
    fi

    sudo tegrastats --stop \
        >/dev/null 2>&1 || true

    if [ -n "${CURRENT_CONTAINER:-}" ]; then
        sudo docker rm -f "$CURRENT_CONTAINER" \
            >/dev/null 2>&1 || true
    fi

    stty sane 2>/dev/null || true
}

trap cleanup INT TERM EXIT

sudo tegrastats --stop \
    >/dev/null 2>&1 || true

sudo tegrastats \
    --interval 1000 \
    --logfile "$LOG_DIR/all_tegrastat.log" \
    >/dev/null 2>&1 &

TEGRA_PID=$!

run_workload()
{
    local NAME="$1"
    local FILE="$2"
    local REPEAT="$3"

    # 이전 workload 실행 후 터미널 상태 복구
    stty sane 2>/dev/null || true

    echo "[RUN] $NAME"

    CURRENT_CONTAINER="nsys_${NAME}_r${REPEAT}"

    sudo docker rm -f \
        "$CURRENT_CONTAINER" \
        >/dev/null 2>&1 || true

    TARGET_NS=$(python3 -c \
        "import time; print(time.time_ns() + $DELAY * 1000000000)"
    )

    NSYS_BASE="/logs/nsys/${NAME}"

    sudo docker run --rm \
        --name "$CURRENT_CONTAINER" \
        --runtime=nvidia \
        --gpus=all \
        --cap-add=SYS_ADMIN \
        -v "$ROOT:/home" \
        -v "$LOG_DIR:/logs" \
        -v "$NSYS_ROOT:$NSYS_ROOT:ro" \
        -e TARGET_NS="$TARGET_NS" \
        -e FILE_NAME="$FILE" \
        -e NSYS_BASE="$NSYS_BASE" \
        -e NSYS_BIN="$NSYS_BIN" \
        "$IMAGE" \
        /bin/bash -lc "cd /home && python3 - << 'PY'

import os
import time
import shlex

target_ns = int(os.environ['TARGET_NS'])
file_name = os.environ['FILE_NAME']
nsys_base = os.environ['NSYS_BASE']
nsys_bin = os.environ['NSYS_BIN']

while time.time_ns() < target_ns:
    time.sleep(0.0005)

task_name = os.path.splitext(file_name)[0]
task_log = f'/logs/{task_name}.log'

# 실제 workload의 stdout/stderr는 task별 log에 직접 기록
command = (
    f'exec python3 {shlex.quote(file_name)} '
    f'> {shlex.quote(task_log)} 2>&1'
)

os.execv(
    nsys_bin,
    [
        nsys_bin,
        'profile',
        '--trace=cuda,nvtx,osrt',
        '--sample=none',
        '--force-overwrite=true',
        '-o',
        nsys_base,
        '/bin/bash',
        '-lc',
        command,
    ]
)

PY" > "$LOG_DIR/nsys/${NAME}.log" 2>&1 &

    DOCKER_PID=$!

    ACTUAL_PID=""

    for i in $(seq 1 60); do

        ACTUAL_PID=$(
            sudo docker top \
                "$CURRENT_CONTAINER" \
                -eo pid,comm,args \
                2>/dev/null |
            awk -v f="$FILE" \
                '$2 ~ /^python/ && index($0,f) {
                    print $1
                    exit
                }'
        )

        if [ -n "${ACTUAL_PID:-}" ]; then
            break
        fi

        sleep 0.2
    done

    PIDSTAT_PID=""

    if [ -n "${ACTUAL_PID:-}" ]; then

        pidstat \
            -p "$ACTUAL_PID" \
            -u \
            -r \
            -h \
            1 \
            > "$LOG_DIR/pidstat_${NAME}.log" &

        PIDSTAT_PID=$!

    else

        echo "[WARNING] Python PID not found"

        : > "$LOG_DIR/pidstat_${NAME}.log"
    fi

    wait "$DOCKER_PID"

    EXIT_STATUS=$?

    if [ -n "${PIDSTAT_PID:-}" ]; then

        kill "$PIDSTAT_PID" \
            >/dev/null 2>&1 || true

        wait "$PIDSTAT_PID" \
            2>/dev/null || true

        PIDSTAT_PID=""
    fi

    # workload 실패 시 Nsight 후처리를 수행하지 않음
    if [ "$EXIT_STATUS" -ne 0 ]; then
        stty sane 2>/dev/null || true
        echo "[ERROR] $NAME failed (exit status=$EXIT_STATUS)"
        CURRENT_CONTAINER=""
        sleep "$COOLDOWN"
        return
    fi

    NSYS_REP="$LOG_DIR/nsys/${NAME}.nsys-rep"

    if [ -f "$NSYS_REP" ]; then

        SQLITE="$LOG_DIR/nsys/${NAME}.sqlite"

        # Nsight Systems 결과를 SQLite 및 CSV로 저장
        if "$NSYS_BIN" export \
            --type sqlite \
            --force-overwrite=true \
            --output "$SQLITE" \
            "$NSYS_REP" \
            > "$LOG_DIR/nsys/${NAME}_export.log" 2>&1
        then
            :
        else
            echo "[ERROR] SQLite export failed: $NAME"
        fi

        # 1) CUDA kernel 실행 정보
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
    " > "$LOG_DIR/nsys/${NAME}_kernel.csv"

        # 2) CUDA Runtime API 호출 정보
        sqlite3 -header -csv "$SQLITE" \
            "SELECT * FROM CUPTI_ACTIVITY_KIND_RUNTIME;" \
            > "$LOG_DIR/nsys/${NAME}_runtime.csv"

        # 3) CUDA memory copy 정보
        sqlite3 -header -csv "$SQLITE" \
            "SELECT * FROM CUPTI_ACTIVITY_KIND_MEMCPY;" \
            > "$LOG_DIR/nsys/${NAME}_memcpy.csv"

        # 4) CUDA synchronization 정보
        sqlite3 -header -csv "$SQLITE" \
            "SELECT * FROM CUPTI_ACTIVITY_KIND_SYNCHRONIZATION;" \
            > "$LOG_DIR/nsys/${NAME}_synchronization.csv"

        # 5) NVTX event 정보
        sqlite3 -header -csv "$SQLITE" \
            "SELECT * FROM NVTX_EVENTS;" \
            > "$LOG_DIR/nsys/${NAME}_nvtx.csv"

        # 6) OS runtime event 정보
        sqlite3 -header -csv "$SQLITE" \
            "SELECT * FROM OSRT_API;" \
            > "$LOG_DIR/nsys/${NAME}_osrt.csv"

    else

        echo "[WARNING] Nsight report not found:"
        echo "$NSYS_REP"

    fi

    # Nsight 실행 후 터미널 상태 복구
    stty sane 2>/dev/null || true

    if [ "$EXIT_STATUS" -eq 0 ]; then
        echo "[DONE] $NAME"
    else
        echo "[ERROR] $NAME failed (exit status=$EXIT_STATUS)"
    fi

    CURRENT_CONTAINER=""

    sleep "$COOLDOWN"
}

for REPEAT in $(seq 1 "$REPEATS"); do
    for NAME in "${WORKLOADS[@]}"; do

        run_workload \
            "$NAME" \
            "${FILES[$NAME]}" \
            "$REPEAT"

    done
done

sudo tegrastats --stop \
    >/dev/null 2>&1 || true

trap - INT TERM EXIT

echo
echo "============================================================"
echo "ALL WORKLOADS DONE"
echo
echo "MPS    : OFF"
echo "Nsight : ON"
echo
echo "Logs:"
echo "$LOG_DIR"
echo
echo "Run parser:"
echo "python3 parse_logs.py"
echo
echo "Nsight:"
echo "$LOG_DIR/nsys/"
echo "============================================================"
