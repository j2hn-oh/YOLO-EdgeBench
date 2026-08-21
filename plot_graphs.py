import os
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path(__file__).resolve().parent
LOG_DIR = ROOT / "all_logs"
EXCEL_FILE = os.path.join(LOG_DIR, "parsed_logs_all.xlsx")

# 고정 순서 : classification, detection, estimation, segmentation, obb
DESIRED_ORDER = ["classification", "detection", "estimation", "segmentation", "obb"]

def plot_inference_stats():
    """ 추론시간 평균 편차 그래프 """
    if not os.path.exists(EXCEL_FILE): return
    try:
        df = pd.read_excel(EXCEL_FILE, sheet_name="Inference_Time")
    except ValueError:
        return

    if df.empty: return

    ordered_cols = [col for col in DESIRED_ORDER if col in df.columns]
    df = df[ordered_cols]

    means = df.mean()
    stds = df.std()
    tasks = means.index
    
    plt.figure(figsize=(10, 6))
    
    lower_error = [0] * len(means)
    upper_error = stds.values
    asymmetric_error = [lower_error, upper_error]
    
    plt.bar(tasks, means.values, yerr=asymmetric_error, capsize=5, color='skyblue', edgecolor='black', alpha=0.8)
    
    plt.title('Inference Time per Workload')
    plt.ylabel('Inference Time (ms)')
    plt.xlabel('Workload (Task)')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.ylim(bottom=0)
    
    out_path = os.path.join(LOG_DIR, "inference_plot.png")
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print("추론 시간 그래프 저장 완료")

def plot_tegrastats_box():
    """ 전체 GPU 사용량 및 메모리 사용량 """
    if not os.path.exists(EXCEL_FILE): return
    try:
        df = pd.read_excel(EXCEL_FILE, sheet_name="Tegrastats")
    except ValueError:
        return
        
    if df.empty: return
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # 첫 번째 칸 (axes[0]): CPU
    axes[0].boxplot(df['Avg_CPU_Usage%'].dropna(), vert=True, patch_artist=True, boxprops=dict(facecolor='pink'))
    axes[0].set_title('CPU Usage')
    axes[0].set_ylabel('Usage (%)')
    axes[0].set_xticks([1])
    axes[0].set_xticklabels(['CPU'])
    axes[0].grid(axis='y', linestyle='--', alpha=0.7)

    # 두 번째 칸 (axes[1]): GPU  <-- 여기 인덱스를 1로 수정!
    axes[1].boxplot(df['GPU_Util%'].dropna(), vert=True, patch_artist=True, boxprops=dict(facecolor='salmon'))
    axes[1].set_title('GPU Utilization')
    axes[1].set_ylabel('Usage (%)')
    axes[1].set_xticks([1])
    axes[1].set_xticklabels(['GPU'])
    axes[1].grid(axis='y', linestyle='--', alpha=0.7)
    
    # 세 번째 칸 (axes[2]): RAM  <-- 여기 인덱스를 2로 수정!
    axes[2].boxplot(df['RAM_Used_MB'].dropna(), vert=True, patch_artist=True, boxprops=dict(facecolor='lightgreen'))
    axes[2].set_title('Total Memory Usage')
    axes[2].set_ylabel('Memory (MB)')
    axes[2].set_xticks([1])
    axes[2].set_xticklabels(['RAM'])
    axes[2].grid(axis='y', linestyle='--', alpha=0.7)
    
    out_path = os.path.join(LOG_DIR, "tegrastats_boxplot.png")
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print("전체 CPU/GPU/Mem Box 그래프 저장 완료")

def plot_workload_box():
    """ 워크로드별 CPU 및 메모리 사용량 """
    if not os.path.exists(EXCEL_FILE): return
    
    xls = pd.ExcelFile(EXCEL_FILE)
    
    tasks = []
    cpu_data = []
    mem_data = []
    
    for task in DESIRED_ORDER:
        sheet_name = f"{task}_pidstat"
        
        if sheet_name in xls.sheet_names:
            df = pd.read_excel(EXCEL_FILE, sheet_name=sheet_name)
            
            if '%CPU' in df.columns and 'RSS_MB' in df.columns:
                tasks.append(task)
                cpu_data.append(df['%CPU'].dropna().values)
                mem_data.append(df['RSS_MB'].dropna().values)
                
    if not tasks: return

    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    
    axes[0].boxplot(cpu_data, labels=tasks, patch_artist=True, boxprops=dict(facecolor='lightblue'))
    axes[0].set_title('CPU Usage per Workload')
    axes[0].set_ylabel('CPU Usage (%)')
    axes[0].grid(axis='y', linestyle='--', alpha=0.7)
    
    axes[1].boxplot(mem_data, labels=tasks, patch_artist=True, boxprops=dict(facecolor='lightcoral'))
    axes[1].set_title('Memory Usage per Workload')
    axes[1].set_ylabel('Memory Usage (MB)')
    axes[1].grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    out_path = os.path.join(LOG_DIR, "workload_boxplot.png")
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print("워크로드별 리소스 Box 그래프 저장 완료")

if __name__ == "__main__":
    plot_inference_stats()
    plot_tegrastats_box()
    plot_workload_box()

