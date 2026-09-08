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
    classification
    detection
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
    rm -f "$LOG_DIR/tegrastat_${NAME}"_r*.log
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

# Idle baseline power 측정
rm -f "$LOG_DIR/baseline_tegrastat.log"

sudo tegrastats \
    --interval 1000 \
    --logfile "$LOG_DIR/baseline_tegrastat.log" \
    >/dev/null 2>&1 &

sleep 5

sudo tegrastats --stop \
    >/dev/null 2>&1 || true

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

    TEGRA_LOG="$LOG_DIR/tegrastat_${NAME}_r${REPEAT}.log"

    TARGET_NS=$(python3 -c \
        "import time; print(time.time_ns() + $DELAY * 1000000000)"
    )

    NSYS_BASE="/logs/nsys/${NAME}"

    # NVTX predictor 사용 시 아래 mount 추가 
    # -v "$ROOT/predictor.py:/ultralytics/ultralytics/engine/predictor.py:ro"
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

    for i in $(seq 1 240); do

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

        sleep 0.05
    done

    PIDSTAT_PID=""
    TEGRA_PID=""

    if [ -n "${ACTUAL_PID:-}" ]; then

        echo "[INFO] Workload PID: $ACTUAL_PID"

        # 실제 workload Python process가 확인된 시점부터 GPU 측정
        sudo tegrastats --stop \
            >/dev/null 2>&1 || true

        sudo tegrastats \
            --interval 1000 \
            --logfile "$TEGRA_LOG" \
            >/dev/null 2>&1 &

        TEGRA_PID=$!

        # workload Python process의 CPU/Memory 측정
        pidstat \
            -p "$ACTUAL_PID" \
            -u \
            -r \
            -h \
            1 \
            > "$LOG_DIR/pidstat_${NAME}.log" &

        PIDSTAT_PID=$!

        # 실제 workload Python process가 종료될 때까지 대기
        while ps -p "$ACTUAL_PID" >/dev/null 2>&1; do
            sleep 0.05
        done

        # workload 종료 직후 tegrastats 종료
        sudo tegrastats --stop \
            >/dev/null 2>&1 || true

        if [ -n "${TEGRA_PID:-}" ]; then
            wait "$TEGRA_PID" \
                2>/dev/null || true
            TEGRA_PID=""
        fi

        # workload 종료 직후 pidstat도 종료
        if [ -n "${PIDSTAT_PID:-}" ]; then
            kill "$PIDSTAT_PID" \
                >/dev/null 2>&1 || true

            wait "$PIDSTAT_PID" \
                2>/dev/null || true

            PIDSTAT_PID=""
        fi

    else

        echo "[WARNING] Python PID not found"

        : > "$LOG_DIR/pidstat_${NAME}.log"
        : > "$TEGRA_LOG"
    fi

    # Nsight/Docker가 완전히 종료되고 report가 생성될 때까지 대기
    wait "$DOCKER_PID"

    EXIT_STATUS=$?

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

        # Nsight Systems 결과를 SQLite로 export한 뒤 workload별 XLSX 생성
        if "$NSYS_BIN" export \
            --type sqlite \
            --force-overwrite=true \
            --output "$SQLITE" \
            "$NSYS_REP" \
            > "$LOG_DIR/nsys/${NAME}_export.log" 2>&1
        then
            XLSX="$LOG_DIR/nsys/${NAME}.xlsx"

            # kernel/runtime/memcpy/synchronization/nvtx/osrt를 한 XLSX의 개별 sheet로 저장
            SQLITE_PATH="$SQLITE" XLSX_PATH="$XLSX" python3 - << 'PYXLSX'
import os
import sqlite3
from openpyxl import Workbook

sqlite_path = os.environ["SQLITE_PATH"]
xlsx_path = os.environ["XLSX_PATH"]

queries = {
    "kernel": """
        SELECT
            k.start, k.end,
            (k.end - k.start) AS duration_ns,
            (k.end - k.start) / 1000000.0 AS duration_ms,
            s.value AS kernel_name,
            k.deviceId, k.contextId, k.streamId, k.correlationId,
            k.registersPerThread,
            k.gridX, k.gridY, k.gridZ,
            k.blockX, k.blockY, k.blockZ,
            k.staticSharedMemory, k.dynamicSharedMemory
        FROM CUPTI_ACTIVITY_KIND_KERNEL AS k
        LEFT JOIN StringIds AS s ON k.demangledName = s.id;
    """,
    "runtime": "SELECT * FROM CUPTI_ACTIVITY_KIND_RUNTIME;",
    "memcpy": "SELECT * FROM CUPTI_ACTIVITY_KIND_MEMCPY;",
    "synchronization": "SELECT * FROM CUPTI_ACTIVITY_KIND_SYNCHRONIZATION;",
    "nvtx": "SELECT * FROM NVTX_EVENTS;",
    "osrt": "SELECT * FROM OSRT_API;",
}

conn = sqlite3.connect(sqlite_path)
wb = Workbook(write_only=True)
for sheet_name, query in queries.items():
    ws = wb.create_sheet(title=sheet_name)
    try:
        cur = conn.execute(query)
        ws.append([col[0] for col in cur.description])
        for row in cur:
            ws.append(list(row))
    except sqlite3.Error as e:
        ws.append(["ERROR"])
        ws.append([str(e)])
conn.close()
wb.save(xlsx_path)
PYXLSX
        else
            echo "[ERROR] SQLite export failed: $NAME"
        fi

    else

        echo "[WARNING] Nsight report not found:"
        echo "$NSYS_REP"

    fi

    # Nsight 실행 후 터미널 상태 복구
    stty sane 2>/dev/null || true
    printf '\n[DONE] %s\n' "$NAME"

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