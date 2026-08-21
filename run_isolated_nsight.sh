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

stop_mps()
{
    echo "============================================================"
    echo "[CHECK] NVIDIA MPS"
    echo "============================================================"

    if pgrep -f 'nvidia-cuda-mps-control -d|nvidia-cuda-mps-server' \
        >/dev/null 2>&1; then

        echo "[INFO] MPS 실행 중 -> 종료"

        echo quit | \
            env CUDA_MPS_PIPE_DIRECTORY="$PIPE" \
            nvidia-cuda-mps-control \
            >/dev/null 2>&1 || true

        sleep 1

        if pgrep -f \
            'nvidia-cuda-mps-control -d|nvidia-cuda-mps-server' \
            >/dev/null 2>&1; then

            echo quit | \
                sudo env CUDA_MPS_PIPE_DIRECTORY="$PIPE" \
                nvidia-cuda-mps-control \
                >/dev/null 2>&1 || true
        fi

        sleep 1

        if pgrep -f \
            'nvidia-cuda-mps-control -d|nvidia-cuda-mps-server' \
            >/dev/null 2>&1; then

            echo quit | \
                nvidia-cuda-mps-control \
                >/dev/null 2>&1 || true
        fi

        sleep 1
    fi

    if pgrep -f \
        'nvidia-cuda-mps-control -d|nvidia-cuda-mps-server' \
        >/dev/null 2>&1; then

        echo "[ERROR] MPS process remains."

        pgrep -af \
            'nvidia-cuda-mps-control|nvidia-cuda-mps-server'

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

stop_mps
check_nsys

echo "[INFO] Cleaning previous logs..."

for NAME in "${WORKLOADS[@]}"; do
    rm -f "$LOG_DIR/${NAME}.log"
    rm -f "$LOG_DIR/pidstat_${NAME}.log"
    rm -f "$LOG_DIR/${NAME}_per_image_timing.csv"
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
}

trap cleanup INT TERM EXIT

sudo tegrastats --stop \
    >/dev/null 2>&1 || true

sudo tegrastats \
    --interval 1000 \
    --logfile "$LOG_DIR/all_tegrastat.log" &

TEGRA_PID=$!

extract_per_image_timing()
{
    local NAME="$1"
    local LOG_FILE="$LOG_DIR/${NAME}.log"
    local CSV_FILE="$LOG_DIR/${NAME}_per_image_timing.csv"

    python3 - \
        "$LOG_FILE" \
        "$CSV_FILE" \
        "$NAME" <<'PY'

import csv
import re
import sys
from pathlib import Path

log_file = Path(sys.argv[1])
csv_file = Path(sys.argv[2])
task = sys.argv[3]

ansi = re.compile(r"\x1b\[[0-9;]*m")

pattern = re.compile(
    r"Image\s+(\d+)"
    r"(?:\s+\((.*?)\))?"
    r"\s*:\s*"
    r"preprocess=([\d.]+)\s*ms,\s*"
    r"inference=([\d.]+)\s*ms,\s*"
    r"postprocess=([\d.]+)\s*ms"
    r"(?:,\s*"
    r"(?:processing_total|total)="
    r"([\d.]+)\s*ms"
    r")?",
    re.IGNORECASE
)

if not log_file.exists():
    print(f"[WARNING] {task}: log file not found")
    sys.exit(0)

text = log_file.read_text(errors="ignore")
text = ansi.sub("", text)

matches = pattern.findall(text)

with csv_file.open("w", newline="") as f:
    writer = csv.writer(f)

    writer.writerow([
        "image_index",
        "image_name",
        "preprocess_ms",
        "inference_ms",
        "postprocess_ms",
        "processing_total_ms",
    ])

    for (
        image_index,
        image_name,
        preprocess,
        inference,
        postprocess,
        total
    ) in matches:

        preprocess = float(preprocess)
        inference = float(inference)
        postprocess = float(postprocess)

        if total:
            total = float(total)
        else:
            total = preprocess + inference + postprocess

        writer.writerow([
            int(image_index),
            image_name,
            f"{preprocess:.6f}",
            f"{inference:.6f}",
            f"{postprocess:.6f}",
            f"{total:.6f}",
        ])

print(
    f"[PER-IMAGE] {task}: "
    f"{len(matches)} images -> "
    f"{csv_file}"
)

PY
}

run_workload()
{
    local NAME="$1"
    local FILE="$2"
    local REPEAT="$3"

    echo
    echo "============================================================"
    echo "[RUN]"
    echo " workload : $NAME"
    echo " repeat   : $REPEAT"
    echo " MPS      : OFF"
    echo " Nsight   : ON"
    echo "============================================================"

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

target_ns = int(os.environ['TARGET_NS'])
file_name = os.environ['FILE_NAME']
nsys_base = os.environ['NSYS_BASE']
nsys_bin = os.environ['NSYS_BIN']

while time.time_ns() < target_ns:
    time.sleep(0.0005)

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
        'python3',
        file_name,
    ]
)

PY" > "$LOG_DIR/${NAME}.log" 2>&1 &

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

        echo "[INFO] Python PID = $ACTUAL_PID"

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

    extract_per_image_timing "$NAME"

    NSYS_REP="$LOG_DIR/nsys/${NAME}.nsys-rep"

    if [ -f "$NSYS_REP" ]; then

        echo "[INFO] Nsight report:"
        echo "$NSYS_REP"

        "$NSYS_BIN" stats \
            --report cuda_gpu_kern_sum \
            --format csv \
            "$NSYS_REP" \
            > "$LOG_DIR/nsys/${NAME}_cuda_gpu_kern_sum.csv" \
            2> "$LOG_DIR/nsys/${NAME}_cuda_gpu_kern_sum.err" \
            || true

        "$NSYS_BIN" stats \
            --report cuda_api_sum \
            --format csv \
            "$NSYS_REP" \
            > "$LOG_DIR/nsys/${NAME}_cuda_api_sum.csv" \
            2> "$LOG_DIR/nsys/${NAME}_cuda_api_sum.err" \
            || true

    else

        echo "[WARNING] Nsight report not found:"
        echo "$NSYS_REP"
    fi

    echo
    echo "[DONE] $NAME"
    echo "exit status = $EXIT_STATUS"
    echo

    CURRENT_CONTAINER=""

    sleep "$COOLDOWN"
}

for REPEAT in $(seq 1 "$REPEATS"); do

    echo
    echo "############################################################"
    echo " REPEAT $REPEAT / $REPEATS"
    echo "############################################################"

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
echo "Per-image timing CSV:"
for NAME in "${WORKLOADS[@]}"; do
    echo "$LOG_DIR/${NAME}_per_image_timing.csv"
done
echo
echo "Nsight:"
echo "$LOG_DIR/nsys/"
echo "============================================================"
