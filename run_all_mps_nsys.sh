#!/bin/bash

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMAGE="ultralytics/ultralytics:latest-jetson-jetpack6"
PIPE="/tmp/nvidia-mps"
MPSLOG="/tmp/nvidia-log"
DELAY=5
LOG_DIR="$ROOT/all_logs"
NSYS_DIR="$LOG_DIR/nsys"
INTERVAL_TMP="$LOG_DIR/tegrastat_intervals.tmp"
INTERVAL_XLSX="$LOG_DIR/tegrastat_intervals.xlsx"

# 이전 Nsight 결과와 interval 중간/결과 파일 정리
mkdir -p "$LOG_DIR" "$PIPE" "$MPSLOG"
rm -rf "$NSYS_DIR"
mkdir -p "$NSYS_DIR"
rm -f "$INTERVAL_TMP" "$INTERVAL_XLSX"
rm -f "$LOG_DIR"/tegrastat_*_interval.csv

# 1. 환경변수 설정 (pidstat 시간을 ISO 8601 형식으로 기록하여 tegrastats와 매칭 용이)
export S_TIME_FORMAT=ISO

save_experiment_config()
{
    local CONFIG_LOG="$LOG_DIR/experiment_config.log"

    rm -f "$CONFIG_LOG"

    {
        echo "============================================================"
        echo "Experiment Configuration"
        echo "============================================================"

        echo
        echo "[Timestamp]"
        date '+%Y-%m-%d %H:%M:%S %Z'

        echo
        echo "[Hostname]"
        hostname

        echo
        echo "[nvpmodel -q]"
        sudo nvpmodel -q 2>&1 || true

        echo
        echo "[jetson_clocks --show]"
        sudo jetson_clocks --show 2>&1 || true

        echo
        echo "[Online CPUs]"
        cat /sys/devices/system/cpu/online 2>&1 || true

        echo
        echo "[Present CPUs]"
        cat /sys/devices/system/cpu/present 2>&1 || true

        echo
        echo "[MPS Status]"

        if pgrep -f '[n]vidia-cuda-mps-control|[n]vidia-cuda-mps-server' \
            >/dev/null 2>&1; then
            echo "ON"
        else
            echo "OFF"
        fi

        echo
        echo "============================================================"
    } > "$CONFIG_LOG"

    echo "[INFO] Experiment configuration saved:"
    echo "$CONFIG_LOG"
}


if ! pgrep -f nvidia-cuda-mps-control >/dev/null; then
    echo "MPS 설정되지 않음"
    exit 1
fi

# 실험 시작 직전 power mode / clock / CPU / MPS 상태 저장
save_experiment_config

# 2. Baseline power 측정
sudo tegrastats --stop >/dev/null 2>&1 || true
rm -f "$LOG_DIR/baseline_tegrastat.log"

sudo tegrastats \
    --interval 1000 \
    --logfile "$LOG_DIR/baseline_tegrastat.log" &

sleep 5

sudo tegrastats --stop >/dev/null 2>&1 || true

# 3. Workload용 tegrastats 시작
rm -f "$LOG_DIR/all_tegrastat.log"

sudo tegrastats \
    --interval 1000 \
    --logfile "$LOG_DIR/all_tegrastat.log" &

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
output = f'/home/all_logs/nsys/{task_name}'
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
        '--sample=none',
        '--force-overwrite=true',
        '-o',
        output,
        '/bin/bash',
        '-lc',
        command
    ]
)
PY" > "$NSYS_DIR/$NAME.log" 2>&1 &

    DOCKER_RUN_PID=$!

    # 컨테이너가 생성되고 실제 Python workload가 뜰 때까지 최대 12초간 반복 확인
    ACTUAL_PID=""
    for i in $(seq 1 240); do
        ACTUAL_PID=$(sudo docker top "$NAME" -eo pid,comm,args 2>/dev/null | \
            awk -v file="$FILE" '$2 ~ /^python/ && $0 ~ file {print $1; exit}')

        if [ -n "$ACTUAL_PID" ]; then
            break
        fi

        sleep 0.05
    done

    if [ -n "$ACTUAL_PID" ]; then

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

        # 병렬 workload들이 하나의 임시 파일에 안전하게 interval 정보 추가
        (
            flock -x 200
            echo "$NAME,$((START_LINE + 1)),$END_LINE" >> "$INTERVAL_TMP"
        ) 200>"$LOG_DIR/.tegrastat_interval.lock"

        # pidstat 종료
        if [ -n "${PIDSTAT_MON_PID:-}" ]; then
            kill "$PIDSTAT_MON_PID" >/dev/null 2>&1 || true
            wait "$PIDSTAT_MON_PID" 2>/dev/null || true
        fi

    else
        echo "[WARNING] $NAME Python PID 찾을 수 없음"
        PIDSTAT_MON_PID=""

        # PID를 찾지 못한 workload도 기록
        (
            flock -x 200
            echo "$NAME,," >> "$INTERVAL_TMP"
        ) 200>"$LOG_DIR/.tegrastat_interval.lock"
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

# workload별 tegrastats interval을 하나의 XLSX에 sheet별로 저장
INTERVAL_TMP_PATH="$INTERVAL_TMP" INTERVAL_XLSX_PATH="$INTERVAL_XLSX" python3 - << 'PY_INTERVAL'
import os
from pathlib import Path
from openpyxl import Workbook

src = Path(os.environ["INTERVAL_TMP_PATH"])
dst = Path(os.environ["INTERVAL_XLSX_PATH"])

workloads = ["classification", "detection", "estimation", "segmentation", "obb"]
records = {}

if src.exists():
    for line in src.read_text().splitlines():
        parts = line.strip().split(",")
        if len(parts) != 3:
            continue
        name, start, end = parts
        records[name] = (start, end)

wb = Workbook()
wb.remove(wb.active)

for name in workloads:
    ws = wb.create_sheet(title=name)
    ws.append(["start_line", "end_line"])

    if name in records:
        start, end = records[name]
        if start and end:
            ws.append([int(start), int(end)])

wb.save(dst)
PY_INTERVAL

rm -f "$INTERVAL_TMP" "$LOG_DIR/.tegrastat_interval.lock"

# Nsight Systems 결과를 workload별 SQLite 및 XLSX로 저장
for task in detection classification estimation segmentation obb; do
    REPORT="$NSYS_DIR/${task}.nsys-rep"
    SQLITE="$NSYS_DIR/${task}.sqlite"
    XLSX="$NSYS_DIR/${task}.xlsx"

    if [ -f "$REPORT" ]; then
        /usr/local/bin/nsys export \
            --type sqlite \
            --force-overwrite=true \
            --output "$SQLITE" \
            "$REPORT"

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
        echo "[WARNING] Nsight report not found: $REPORT"
    fi
done

echo "All tasks done. Logs are in $LOG_DIR"
