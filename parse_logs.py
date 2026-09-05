import os
import re
import pandas as pd
import io
from pathlib import Path

ROOT = Path(__file__).resolve().parent

TASKS = ["detection", "classification", "estimation", "segmentation", "obb"]

LOG_CONFIGS = [
    {
        "log_dir": ROOT / "all_logs",
        "excel_name": "parsed_logs_all.xlsx"
    },
    {
        "log_dir": ROOT / "isolated_logs",
        "excel_name": "parsed_logs_isolated.xlsx"
    }
]


def parse_inference_logs(writer, log_dir):
    """ 이미지별 preprocess/inference/postprocess/total 시간 추출 및 엑셀 시트 추가 """

    pattern = re.compile(
        r"Image\s+(\d+):\s+"
        r"preprocess=([\d.]+)\s+ms,\s+"
        r"inference=([\d.]+)\s+ms,\s+"
        r"postprocess=([\d.]+)\s+ms,\s+"
        r"processing_total=([\d.]+)\s+ms"
    )

    processing_data = {}
    inference_data = {}

    for task in TASKS:
        filepath = log_dir / f"{task}.log"

        if not filepath.exists():
            continue

        preprocess_times = []
        inference_times = []
        postprocess_times = []
        total_times = []

        with open(filepath, "r") as f:
            for line in f:
                match = pattern.search(line.strip())

                if match:
                    preprocess_times.append(float(match.group(2)))
                    inference_times.append(float(match.group(3)))
                    postprocess_times.append(float(match.group(4)))
                    total_times.append(float(match.group(5)))

        if inference_times:
            # 기존 inference-only 시트용
            inference_data[task] = inference_times

            # 전체 processing time 시트용
            processing_data[f"{task}_preprocess"] = preprocess_times
            processing_data[f"{task}_inference"] = inference_times
            processing_data[f"{task}_postprocess"] = postprocess_times
            processing_data[f"{task}_total"] = total_times

    # 기존 Inference_Time 시트 유지
    if inference_data:
        df_inference = pd.DataFrame(
            {k: pd.Series(v) for k, v in inference_data.items()}
        )
        df_inference.to_excel(
            writer,
            sheet_name="Inference_Time",
            index=False
        )
        print("추론 시간 시트 저장 완료")

    # preprocess/inference/postprocess/total 전체 저장
    if processing_data:
        df_processing = pd.DataFrame(
            {k: pd.Series(v) for k, v in processing_data.items()}
        )
        df_processing.to_excel(
            writer,
            sheet_name="Processing_Time",
            index=False
        )
        print("Preprocess/Inference/Postprocess/Total 시간 시트 저장 완료")


def parse_tegrastats_logs(writer, log_dir):
    """ tegrastats 로그 파싱 및 엑셀 시트 추가 """

    time_pattern = re.compile(
        r"^(\d{2}-\d{2}-\d{4} \d{2}:\d{2}:\d{2})"
    )

    cpu_pattern = re.compile(r"CPU \[(.*?)\]")
    ram_pattern = re.compile(r"RAM (\d+)/(\d+)MB")
    gpu_pattern = re.compile(r"GR3D(?:_FREQ)?\s+(\d+)%")
    power_pattern = re.compile(r"VDD_IN\s+(\d+)mW/\d+mW")

    def parse_lines(lines):
        """ tegrastats line 목록을 DataFrame으로 변환 """

        data = []

        for line in lines:

            time_match = time_pattern.search(line)
            ram_match = ram_pattern.search(line)
            gpu_match = gpu_pattern.search(line)
            cpu_match = cpu_pattern.search(line)
            power_match = power_pattern.search(line)

            if not (ram_match and gpu_match and cpu_match and power_match):
                continue

            cores = cpu_match.group(1).split(",")

            cpu_values = []

            for core in cores:

                if "%" in core:

                    try:
                        cpu_values.append(
                            int(core.split("%")[0])
                        )
                    except ValueError:
                        pass

            avg_cpu = (
                round(
                    sum(cpu_values) / len(cpu_values),
                    1
                )
                if cpu_values
                else 0.0
            )

            data.append({
                "Time": (
                    time_match.group(1)
                    if time_match
                    else "Unknown"
                ),
                "Avg_CPU_Usage%": avg_cpu,
                "RAM_Used_MB": int(ram_match.group(1)),
                "RAM_Total_MB": int(ram_match.group(2)),
                "GPU_Util%": int(gpu_match.group(1)),
                "Power_mW": int(power_match.group(1)),
            })

        return pd.DataFrame(data)

    def parse_one_file(filepath):
        """ tegrastats 전체 파일 파싱 """

        with open(filepath, "r") as f:
            lines = f.readlines()

        return parse_lines(lines)

    def parse_file_interval(filepath, start_line, end_line):
        """
        tegrastats 원본 파일에서 지정된 line 범위만 파싱.
        start_line/end_line은 1-based line 번호.
        """

        with open(filepath, "r") as f:
            lines = f.readlines()

        if not lines:
            return pd.DataFrame()

        # 범위 보정
        start_line = max(1, start_line)
        end_line = min(end_line, len(lines))

        if start_line > end_line:
            return pd.DataFrame()

        # Python list는 0-based이고 끝 index는 exclusive
        selected_lines = lines[
            start_line - 1:end_line
        ]

        return parse_lines(selected_lines)

    # =========================================================
    # isolated_logs
    # workload마다 별도의 tegrastats 파일 존재
    # =========================================================

    found_isolated = False

    for task in TASKS:

        files = sorted(
            log_dir.glob(
                f"tegrastat_{task}_r*.log"
            )
        )

        if not files:
            continue

        found_isolated = True

        frames = []

        for filepath in files:

            df = parse_one_file(filepath)

            if not df.empty:
                frames.append(df)

        if frames:

            df_task = pd.concat(
                frames,
                ignore_index=True
            )

            df_task.to_excel(
                writer,
                sheet_name=f"{task}_tegrastat",
                index=False
            )

            print(
                f"{task} tegrastats 저장 완료 "
                f"(총 {len(df_task)}건)"
            )

    if found_isolated:
        return

    # =========================================================
    # all_logs
    # 하나의 all_tegrastat.log에서 workload별 구간 추출
    # =========================================================

    filepath = log_dir / "all_tegrastat.log"

    if not filepath.exists():
        return

    # ---------------------------------------------------------
    # 전체 tegrastats 시트 저장
    # ---------------------------------------------------------

    df_all = parse_one_file(filepath)

    if not df_all.empty:

        df_all.to_excel(
            writer,
            sheet_name="Tegrastats",
            index=False
        )

        print(
            f"전체 GPU/Mem/CPU 시트 저장 완료 "
            f"(총 {len(df_all)}건)"
        )

    # ---------------------------------------------------------
    # workload별 실행 구간 추출
    # ---------------------------------------------------------

    for task in TASKS:

        interval_file = (
            log_dir /
            f"tegrastat_{task}_interval.csv"
        )

        if not interval_file.exists():

            print(
                f"[WARNING] {task} interval 파일 없음"
            )

            continue

        interval_df = pd.read_csv(
            interval_file
        )

        if interval_df.empty:

            print(
                f"[WARNING] {task} interval 정보 없음"
            )

            continue

        try:

            start_line = int(
                interval_df.iloc[0]["start_line"]
            )

            end_line = int(
                interval_df.iloc[0]["end_line"]
            )

        except (ValueError, KeyError):

            print(
                f"[WARNING] {task} interval 형식 오류"
            )

            continue

        df_task = parse_file_interval(
            filepath,
            start_line,
            end_line
        )

        if df_task.empty:

            print(
                f"[WARNING] {task} tegrastats 데이터 없음"
            )

            continue

        # workload별 time-series는 0부터 다시 시작할 것이므로
        # index 초기화
        df_task.reset_index(
            drop=True,
            inplace=True
        )

        df_task.to_excel(
            writer,
            sheet_name=f"{task}_tegrastat",
            index=False
        )

        print(
            f"{task} tegrastats 저장 완료 "
            f"(line {start_line}-{end_line}, "
            f"총 {len(df_task)}건)"
        )

def parse_pidstat_logs(writer, log_dir):
    """ 워크로드별 리소스 데이터 추출 및 개별 엑셀 시트 추가 """

    for task in TASKS:
        filepath = log_dir / f"pidstat_{task}.log"

        if not filepath.exists():
            continue

        clean_lines = []
        header_found = False

        with open(filepath, "r") as f:
            for line in f:
                if "Linux" in line or line.isspace():
                    continue

                if "Time" in line and "PID" in line:
                    if not header_found:
                        line = line.replace("# Time", "Time").replace("#Time", "Time")
                        clean_lines.append(line)
                        header_found = True
                    continue

                clean_lines.append(line)

        if len(clean_lines) > 1:
            df = pd.read_csv(
                io.StringIO("".join(clean_lines)),
                sep=r"\s+"
            )

            if "RSS" in df.columns:
                df["RSS_MB"] = df["RSS"] / 1024.0

            sheet_name = f"{task}_pidstat"
            df.to_excel(writer, sheet_name=sheet_name, index=False)

    print("워크로드별 리소스 시트 저장 완료")


def parse_log_directory(log_dir, excel_name):
    """ 지정된 로그 디렉터리를 하나의 Excel 파일로 변환 """

    if not log_dir.exists():
        print(f"로그 디렉터리 없음: {log_dir}")
        return

    excel_out = log_dir / excel_name

    print(f"\nParsing: {log_dir}")

    with pd.ExcelWriter(excel_out, engine="openpyxl") as writer:
        parse_inference_logs(writer, log_dir)
        parse_tegrastats_logs(writer, log_dir)
        parse_pidstat_logs(writer, log_dir)

    print(f"Excel 저장 완료: {excel_out}")


if __name__ == "__main__":
    for config in LOG_CONFIGS:
        parse_log_directory(
            config["log_dir"],
            config["excel_name"]
        )