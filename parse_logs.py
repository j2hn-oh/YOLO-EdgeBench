import os
import re
import pandas as pd
import io
from pathlib import Path

ROOT = Path(__file__).resolve().parent

TASKS = ["classification", "detection", "estimation", "segmentation", "obb"]

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



def parse_experiment_config(writer, log_dir):
    """
    experiment_config.log와 tegrastats를 이용해 실험 설정을
    사람이 보기 쉬운 Experiment_Config 시트로 저장한다.

    - Power mode: experiment_config.log의 nvpmodel -q에서 직접 읽음
    - MPS: experiment_config.log에 기록된 MPS process 유무로 확인
    - Clock status: workload 동안 tegrastats의 CPU/GPU/EMC 주파수 변동으로 판단
    """

    config_file = log_dir / "experiment_config.log"

    if not config_file.exists():
        print("[INFO] experiment_config.log 없음 - Experiment_Config 시트 생략")
        return

    config_text = config_file.read_text(errors="replace")

    # ---------------------------------------------------------
    # experiment_config.log 파싱
    # ---------------------------------------------------------

    def get_section(name):
        pattern = re.compile(
            rf"\[{re.escape(name)}\]\s*\n(.*?)(?=\n\[[^\]]+\]|\n=+\s*$|\Z)",
            re.S
        )
        match = pattern.search(config_text)
        return match.group(1).strip() if match else ""

    timestamp = get_section("Timestamp")
    hostname = get_section("Hostname")
    nvpmodel_text = get_section("nvpmodel -q")
    jetson_clocks_text = get_section("jetson_clocks --show")
    online_cpus = get_section("Online CPUs")
    present_cpus = get_section("Present CPUs")
    mps_status = get_section("MPS Status")

    if not mps_status:
        mps_status = "Unknown"

    power_mode = ""
    mode_id = ""

    mode_match = re.search(
        r"NV Power Mode:\s*(.+)",
        nvpmodel_text
    )

    if mode_match:
        power_mode = mode_match.group(1).strip()

        # NV Power Mode 다음 줄의 숫자를 mode ID로 사용
        after_mode = nvpmodel_text[mode_match.end():].strip().splitlines()

        for line in after_mode:
            line = line.strip()

            if re.fullmatch(r"\d+", line):
                mode_id = line
                break

    # ---------------------------------------------------------
    # workload tegrastats 파일 선택
    # ---------------------------------------------------------

    all_tegrastat = log_dir / "all_tegrastat.log"

    if all_tegrastat.exists():
        tegra_files = [all_tegrastat]
    else:
        tegra_files = sorted(
            log_dir.glob("tegrastat_*_r*.log")
        )

    # tegrastats frequency parser
    cpu_pattern = re.compile(r"CPU \[(.*?)\]")
    gpu_pattern = re.compile(
        r"GR3D(?:_FREQ)?\s+\d+%@(?:\[(\d+)(?:,\d+)?\]|(\d+))"
    )
    emc_pattern = re.compile(r"EMC_FREQ\s+\d+%@(\d+)")

    cpu_freqs = {}
    gpu_freqs = []
    emc_freqs = []
    online_core_counts = []

    for filepath in tegra_files:

        with open(filepath, "r", errors="replace") as f:

            for line in f:

                # CPU
                cpu_match = cpu_pattern.search(line)

                if cpu_match:

                    cores = [
                        core.strip()
                        for core in cpu_match.group(1).split(",")
                    ]

                    online_count = 0

                    for idx, core in enumerate(cores):

                        if core.lower() == "off":
                            continue

                        freq_match = re.search(
                            r"\d+%@(\d+)",
                            core
                        )

                        if freq_match:
                            online_count += 1

                            cpu_freqs.setdefault(
                                idx,
                                []
                            ).append(
                                int(freq_match.group(1))
                            )

                    online_core_counts.append(
                        online_count
                    )

                # GPU
                gpu_match = gpu_pattern.search(line)

                if gpu_match:

                    freq = (
                        gpu_match.group(1)
                        if gpu_match.group(1) is not None
                        else gpu_match.group(2)
                    )

                    gpu_freqs.append(int(freq))

                # EMC
                emc_match = emc_pattern.search(line)

                if emc_match:
                    emc_freqs.append(
                        int(emc_match.group(1))
                    )

    # ---------------------------------------------------------
    # Clock 변동 요약
    # ---------------------------------------------------------

    # tegrastats의 몇 MHz 수준 반올림/측정 차이는 고정 상태로 취급
    CPU_TOLERANCE_MHZ = 20
    GPU_TOLERANCE_MHZ = 20
    EMC_TOLERANCE_MHZ = 20

    cpu_ranges = {}

    for core, values in cpu_freqs.items():

        if values:
            cpu_ranges[core] = (
                min(values),
                max(values)
            )

    cpu_fixed = bool(cpu_ranges) and all(
        (max_freq - min_freq) <= CPU_TOLERANCE_MHZ
        for min_freq, max_freq in cpu_ranges.values()
    )

    gpu_min = min(gpu_freqs) if gpu_freqs else None
    gpu_max = max(gpu_freqs) if gpu_freqs else None

    gpu_fixed = (
        gpu_min is not None
        and gpu_max is not None
        and (gpu_max - gpu_min) <= GPU_TOLERANCE_MHZ
    )

    emc_min = min(emc_freqs) if emc_freqs else None
    emc_max = max(emc_freqs) if emc_freqs else None

    emc_fixed = (
        emc_min is not None
        and emc_max is not None
        and (emc_max - emc_min) <= EMC_TOLERANCE_MHZ
    )

    # CPU와 GPU를 핵심 기준으로 사용.
    # EMC 데이터가 존재하면 EMC도 함께 고정되어야 Fixed로 판정.
    if cpu_ranges and gpu_freqs:

        clock_fixed = (
            cpu_fixed
            and gpu_fixed
            and (
                emc_fixed
                if emc_freqs
                else True
            )
        )

        clock_status = (
            "Fixed"
            if clock_fixed
            else "Not Fixed"
        )

    else:
        clock_status = "Unknown"

    # ---------------------------------------------------------
    # 사람이 보기 쉬운 문자열 생성
    # ---------------------------------------------------------

    if online_core_counts:
        online_min = min(online_core_counts)
        online_max = max(online_core_counts)

        online_observed = (
            str(online_min)
            if online_min == online_max
            else f"{online_min}-{online_max}"
        )

    else:
        online_observed = ""

    cpu_min_all = (
        min(
            min_freq
            for min_freq, _ in cpu_ranges.values()
        )
        if cpu_ranges
        else None
    )

    cpu_max_all = (
        max(
            max_freq
            for _, max_freq in cpu_ranges.values()
        )
        if cpu_ranges
        else None
    )

    cpu_range_text = (
        f"{cpu_min_all}-{cpu_max_all} MHz"
        if cpu_min_all is not None
        else ""
    )

    gpu_range_text = (
        f"{gpu_min}-{gpu_max} MHz"
        if gpu_min is not None
        else ""
    )

    emc_range_text = (
        f"{emc_min}-{emc_max} MHz"
        if emc_min is not None
        else ""
    )

    cpu_core_detail = ", ".join(
        (
            f"CPU{core}: {low} MHz"
            if low == high
            else f"CPU{core}: {low}-{high} MHz"
        )
        for core, (low, high)
        in sorted(cpu_ranges.items())
    )

    tegra_file_text = ", ".join(
        path.name
        for path in tegra_files
    )

    # ---------------------------------------------------------
    # Excel 시트 저장
    # ---------------------------------------------------------

    rows = [
        ["Experiment Time", timestamp],
        ["Hostname", hostname],
        ["Power Mode", power_mode],
        ["Power Mode ID", mode_id],
        ["Online CPUs (configured)", online_cpus],
        ["Present CPUs", present_cpus],
        ["Online CPU Cores (observed)", online_observed],
        ["MPS", mps_status],
        ["CPU Clock Range", cpu_range_text],
        ["GPU Clock Range", gpu_range_text],
        ["EMC Clock Range", emc_range_text],
        ["Clock Status", clock_status],
        ["CPU Clock Detail", cpu_core_detail],
        ["Tegrastats Files", tegra_file_text],
    ]

    df_config = pd.DataFrame(
        rows,
        columns=["Item", "Value"]
    )

    df_config.to_excel(
        writer,
        sheet_name="Experiment_Config",
        index=False
    )

    print(
        f"Experiment_Config 저장 완료 "
        f"(Power Mode={power_mode or 'Unknown'}, "
        f"MPS={mps_status}, "
        f"Clock={clock_status})"
    )


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
    vdd_in_pattern = re.compile(r"VDD_IN\s+(\d+)mW/\d+mW")
    vdd_gpu_soc_pattern = re.compile(r"VDD_GPU_SOC\s+(\d+)mW/\d+mW")
    vdd_cpu_cv_pattern = re.compile(r"VDD_CPU_CV\s+(\d+)mW/\d+mW")
    vin_sys_5v0_pattern = re.compile(r"VIN_SYS_5V0\s+(\d+)mW/\d+mW")

    def parse_lines(lines):
        """ tegrastats line 목록을 DataFrame으로 변환 """

        data = []

        for line in lines:

            time_match = time_pattern.search(line)
            ram_match = ram_pattern.search(line)
            gpu_match = gpu_pattern.search(line)
            cpu_match = cpu_pattern.search(line)
            vdd_in_match = vdd_in_pattern.search(line)
            vdd_gpu_soc_match = vdd_gpu_soc_pattern.search(line)
            vdd_cpu_cv_match = vdd_cpu_cv_pattern.search(line)
            vin_sys_5v0_match = vin_sys_5v0_pattern.search(line)

            if vdd_in_match:
                # Orin Nano / Orin NX
                power_mw = int(vdd_in_match.group(1))

            elif vdd_gpu_soc_match and vdd_cpu_cv_match and vin_sys_5v0_match:
                # Jetson AGX Orin
                power_mw = (
                    int(vdd_gpu_soc_match.group(1))
                    + int(vdd_cpu_cv_match.group(1))
                    + int(vin_sys_5v0_match.group(1))
                )

            else:
                continue

            if not (ram_match and gpu_match and cpu_match):
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
                "Power_mW": power_mw,
            })

        return pd.DataFrame(data)

    def parse_one_file(filepath):
        """ tegrastats 전체 파일 파싱 """

        with open(filepath, "r") as f:
            lines = f.readlines()

        return parse_lines(lines)

    def get_baseline_power(log_dir):
        """ Idle baseline의 평균 module power 계산 """

        baseline_file = log_dir / "baseline_tegrastat.log"

        if not baseline_file.exists():
            return None

        df_baseline = parse_one_file(baseline_file)

        if df_baseline.empty:
            return None

        return df_baseline["Power_mW"].mean()

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

    baseline_power_mw = get_baseline_power(log_dir)

    if baseline_power_mw is not None:
        print(
            f"Baseline power: "
            f"{baseline_power_mw / 1000:.3f} W"
        )


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

            if baseline_power_mw is not None:
                df_task["Power_Increase_mW"] = (
                    df_task["Power_mW"] - baseline_power_mw
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
    # concurrent(all_logs): tegrastat_intervals.xlsx의
    # workload별 sheet에서 start_line/end_line을 읽음
    # ---------------------------------------------------------

    interval_file = log_dir / "tegrastat_intervals.xlsx"

    if not interval_file.exists():
        print("[WARNING] tegrastat_intervals.xlsx 파일 없음")
        return

    interval_xls = pd.ExcelFile(interval_file)

    for task in TASKS:

        if task not in interval_xls.sheet_names:
            print(
                f"[WARNING] {task} interval sheet 없음"
            )
            continue

        interval_df = pd.read_excel(
            interval_file,
            sheet_name=task
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

        except (ValueError, TypeError, KeyError):

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

        if baseline_power_mw is not None:
            df_task["Power_Increase_mW"] = (
                df_task["Power_mW"] - baseline_power_mw
            )

        # workload별 time-series는 0부터 다시 시작
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
        parse_experiment_config(writer, log_dir)
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