import os
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path(__file__).resolve().parent

PLOT_CONFIGS = [
    {
        "log_dir": ROOT / "all_logs",
        "excel_file": ROOT / "all_logs" / "parsed_logs_all.xlsx"
    },
    {
        "log_dir": ROOT / "isolated_logs",
        "excel_file": ROOT / "isolated_logs" / "parsed_logs_isolated.xlsx"
    }
]

# 고정 순서 : classification, detection, estimation, segmentation, obb
DESIRED_ORDER = ["classification", "detection", "estimation", "segmentation", "obb"]


def plot_inference_stats(log_dir, excel_file):
    """ 추론시간 평균 편차 그래프 """

    if not excel_file.exists():
        return

    try:
        df = pd.read_excel(excel_file, sheet_name="Inference_Time")
    except ValueError:
        return

    if df.empty:
        return

    ordered_cols = [col for col in DESIRED_ORDER if col in df.columns]
    df = df[ordered_cols]

    means = df.mean()
    stds = df.std()
    tasks = means.index

    plt.figure(figsize=(10, 6))

    lower_error = [0] * len(means)
    upper_error = stds.values
    asymmetric_error = [lower_error, upper_error]

    plt.bar(
        tasks,
        means.values,
        yerr=asymmetric_error,
        capsize=5,
        color="skyblue",
        edgecolor="black",
        alpha=0.8
    )

    plt.title("Inference Time per Workload")
    plt.ylabel("Inference Time (ms)")
    plt.xlabel("Workload (Task)")
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.ylim(bottom=0)

    out_path = log_dir / "inference_plot.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"추론 시간 그래프 저장 완료: {out_path}")


def plot_tegrastats_box(log_dir, excel_file):
    """ 전체 CPU/GPU/메모리 사용량 """

    if not excel_file.exists():
        return

    try:
        df = pd.read_excel(excel_file, sheet_name="Tegrastats")
    except ValueError:
        return

    if df.empty:
        return

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # 첫 번째 칸: CPU
    axes[0].boxplot(
        df["Avg_CPU_Usage%"].dropna(),
        vert=True,
        patch_artist=True,
        boxprops=dict(facecolor="pink")
    )
    axes[0].set_title("CPU Usage")
    axes[0].set_ylabel("Usage (%)")
    axes[0].set_xticks([1])
    axes[0].set_xticklabels(["CPU"])
    axes[0].grid(axis="y", linestyle="--", alpha=0.7)

    # 두 번째 칸: GPU
    axes[1].boxplot(
        df["GPU_Util%"].dropna(),
        vert=True,
        patch_artist=True,
        boxprops=dict(facecolor="salmon")
    )
    axes[1].set_title("GPU Utilization")
    axes[1].set_ylabel("Usage (%)")
    axes[1].set_xticks([1])
    axes[1].set_xticklabels(["GPU"])
    axes[1].grid(axis="y", linestyle="--", alpha=0.7)

    # 세 번째 칸: RAM
    axes[2].boxplot(
        df["RAM_Used_MB"].dropna(),
        vert=True,
        patch_artist=True,
        boxprops=dict(facecolor="lightgreen")
    )
    axes[2].set_title("Total Memory Usage")
    axes[2].set_ylabel("Memory (MB)")
    axes[2].set_xticks([1])
    axes[2].set_xticklabels(["RAM"])
    axes[2].grid(axis="y", linestyle="--", alpha=0.7)

    out_path = log_dir / "tegrastats_boxplot.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"전체 CPU/GPU/Mem Box 그래프 저장 완료: {out_path}")


def plot_workload_box(log_dir, excel_file):
    """ 워크로드별 CPU 및 메모리 사용량 """

    if not excel_file.exists():
        return

    xls = pd.ExcelFile(excel_file)

    cpu_tasks = []
    cpu_data = []

    # gpu_tasks = []
    # gpu_data = []

    mem_tasks = []
    mem_data = []

    for task in DESIRED_ORDER:

        # CPU / Memory: pidstat
        pidstat_sheet = f"{task}_pidstat"

        if pidstat_sheet in xls.sheet_names:
            df_pid = pd.read_excel(
                excel_file,
                sheet_name=pidstat_sheet
            )

            if "%CPU" in df_pid.columns:
                values = df_pid["%CPU"].dropna().values

                if len(values) > 0:
                    cpu_tasks.append(task)
                    cpu_data.append(values)

            if "RSS_MB" in df_pid.columns:
                values = df_pid["RSS_MB"].dropna().values

                if len(values) > 0:
                    mem_tasks.append(task)
                    mem_data.append(values)

    # CPU
    if cpu_data:
        plt.figure(figsize=(12, 6))

        plt.boxplot(
            cpu_data,
            labels=cpu_tasks,
            patch_artist=True,
            boxprops=dict(facecolor="lightblue")
        )

        plt.title("CPU Usage per Workload")
        plt.ylabel("CPU Usage (%)")
        plt.xlabel("Workload (Task)")
        plt.grid(axis="y", linestyle="--", alpha=0.7)

        out_path = log_dir / "cpu_workload_boxplot.png"

        plt.savefig(
            out_path,
            dpi=300,
            bbox_inches="tight"
        )
        plt.close()

        print(
            f"CPU workload boxplot 저장 완료: "
            f"{out_path}"
        )

    # Memory
    if mem_data:
        plt.figure(figsize=(12, 6))

        plt.boxplot(
            mem_data,
            labels=mem_tasks,
            patch_artist=True,
            boxprops=dict(facecolor="lightcoral")
        )

        plt.title("Memory Usage per Workload")
        plt.ylabel("Memory Usage (MB)")
        plt.xlabel("Workload (Task)")
        plt.grid(axis="y", linestyle="--", alpha=0.7)

        out_path = log_dir / "memory_workload_boxplot.png"

        plt.savefig(
            out_path,
            dpi=300,
            bbox_inches="tight"
        )
        plt.close()

        print(
            f"Memory workload boxplot 저장 완료: "
            f"{out_path}"
        )

def plot_gpu_timeseries(log_dir, excel_file, sample_interval_ms=100):
    """ 워크로드별 GPU utilization을 하나의 5x1 time-series로 표시 """

    if not excel_file.exists():
        return

    xls = pd.ExcelFile(excel_file)

    gpu_series = []

    for task in DESIRED_ORDER:
        sheet_name = f"{task}_tegrastat"

        if sheet_name not in xls.sheet_names:
            continue

        df = pd.read_excel(
            excel_file,
            sheet_name=sheet_name
        )

        if "GPU_Util%" not in df.columns:
            continue

        gpu = df["GPU_Util%"].dropna().reset_index(drop=True)

        if gpu.empty:
            continue

        time_sec = (
            gpu.index.to_numpy()
            * sample_interval_ms
            / 1000.0
        )

        gpu_series.append(
            (task, time_sec, gpu.values)
        )

    if not gpu_series:
        return

    fig, axes = plt.subplots(
        len(gpu_series),
        1,
        figsize=(12, 12),
        sharex=True
    )

    if len(gpu_series) == 1:
        axes = [axes]

    for ax, (task, time_sec, gpu) in zip(axes, gpu_series):

        ax.plot(
            time_sec,
            gpu,
            linewidth=1.2
        )

        ax.set_ylabel("GPU (%)")
        ax.set_ylim(0, 100)
        ax.set_title(task.capitalize())
        ax.grid(
            axis="both",
            linestyle="--",
            alpha=0.7
        )

    axes[-1].set_xlabel("Time (s)")

    fig.suptitle(
        "GPU Utilization over Time per Workload",
        fontsize=16
    )

    plt.tight_layout(
        rect=[0, 0, 1, 0.97]
    )

    out_path = log_dir / "gpu_workload_timeseries.png"

    plt.savefig(
        out_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(
        f"GPU workload time-series 저장 완료: "
        f"{out_path}"
    )

def plot_power_timeseries(log_dir, excel_file, sample_interval_ms=100):
    """ 워크로드별 system power time-series """

    if not excel_file.exists():
        return

    xls = pd.ExcelFile(excel_file)

    power_series = []

    for task in DESIRED_ORDER:
        sheet_name = f"{task}_tegrastat"

        if sheet_name not in xls.sheet_names:
            continue

        df = pd.read_excel(
            excel_file,
            sheet_name=sheet_name
        )

        if "Power_mW" not in df.columns:
            continue

        power = (
            df["Power_mW"]
            .dropna()
            .reset_index(drop=True)
        )

        if power.empty:
            continue

        time_sec = (
            power.index.to_numpy()
            * sample_interval_ms
            / 1000.0
        )

        power_series.append(
            (
                task,
                time_sec,
                power.values / 1000.0
            )
        )

    if not power_series:
        return

    fig, axes = plt.subplots(
        len(power_series),
        1,
        figsize=(12, 12),
        sharex=True
    )

    if len(power_series) == 1:
        axes = [axes]

    for ax, (task, time_sec, power_w) in zip(
        axes,
        power_series
    ):
        ax.plot(
            time_sec,
            power_w,
            linewidth=1.2
        )

        ax.set_ylabel("Power (W)")
        ax.set_title(task.capitalize())
        ax.grid(
            axis="both",
            linestyle="--",
            alpha=0.7
        )

    axes[-1].set_xlabel("Time (s)")

    fig.suptitle(
        "System Power over Time per Workload",
        fontsize=16
    )

    plt.tight_layout(rect=[0, 0, 1, 0.97])

    out_path = (
        log_dir /
        "power_workload_timeseries.png"
    )

    plt.savefig(
        out_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(
        f"Power workload time-series 저장 완료: "
        f"{out_path}"
    )

def plot_log_directory(log_dir, excel_file):
    """ 지정된 실험 결과에 대한 그래프 생성 """

    if not excel_file.exists():
        print(f"Excel 파일 없음: {excel_file}")
        return

    print(f"\nPlotting: {excel_file}")

    plot_inference_stats(log_dir, excel_file)
    plot_tegrastats_box(log_dir, excel_file)
    plot_workload_box(log_dir, excel_file)
    plot_gpu_timeseries(log_dir, excel_file)
    plot_power_timeseries(log_dir, excel_file)


if __name__ == "__main__":
    for config in PLOT_CONFIGS:
        plot_log_directory(
            config["log_dir"],
            config["excel_file"]
        )