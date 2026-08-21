# YOLO-EdgeBench

YOLO-EdgeBench is an experimental framework for evaluating YOLO inference performance and resource utilization on edge computing platforms.

The current implementation evaluates five YOLO workloads on NVIDIA Jetson platforms. It supports concurrent execution with NVIDIA Multi-Process Service (MPS), system and per-workload resource monitoring, and CUDA execution profiling with NVIDIA Nsight Systems.

## 1. Workloads

The following five YOLO workloads are evaluated:

| Workload | Script | TensorRT Engine |
|---|---|---|
| Classification | `classification.py` | `yolo26n-cls.engine` |
| Object Detection | `detection.py` | `yolo26n.engine` |
| Pose Estimation | `estimation.py` | `yolo26n-pose.engine` |
| Segmentation | `segmentation.py` | `yolo26n-seg.engine` |
| Oriented Bounding Box (OBB) | `obb.py` | `yolo26n-obb.engine` |

Each workload reports the following timing information:

- preprocessing time
- inference time
- postprocessing time
- processing time (`preprocess + inference + postprocess`)
- total wall-clock time
- average wall-clock time per image

## 2. Evaluation Platforms

The experiments are currently designed for NVIDIA Jetson platforms, including:

- NVIDIA Jetson AGX Orin
- NVIDIA Jetson Orin NX
- NVIDIA Jetson Orin Nano

The framework can be extended to additional edge computing platforms by adapting the execution and resource-monitoring scripts.

## 3. Software Environment

The current experimental environment uses:

- NVIDIA JetPack 6
- CUDA
- TensorRT
- Docker
- NVIDIA Container Runtime
- Ultralytics YOLO
- NVIDIA Multi-Process Service (MPS)
- NVIDIA Nsight Systems
- `pidstat`
- `tegrastats`
- Python 3

The Docker image used to execute the YOLO workloads is:

```bash
ultralytics/ultralytics:latest-jetson-jetpack6
```

The current scripts mount the host installation of NVIDIA Nsight Systems 2024.5.4 into the Docker containers.

## 4. Repository Structure

```text
YOLO-EdgeBench/
├── classification.py
├── detection.py
├── estimation.py
├── segmentation.py
├── obb.py
├── run_all_mps_nsys.sh
├── run_isolated_nsight.sh
├── parse_logs.py
├── plot_graphs.py
└── README.md
```

### Workload Scripts

`classification.py`, `detection.py`, `estimation.py`, `segmentation.py`, and `obb.py` execute the corresponding TensorRT YOLO models and record per-image processing times.

### `run_all_mps_nsys.sh`

Runs all five workloads concurrently in separate Docker containers.

The script:

- requires NVIDIA MPS to be running before the experiment,
- launches the five workloads concurrently,
- synchronizes their start time,
- sets `CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=20` for each workload,
- enables per-context multiprocessor partitioning,
- collects per-workload CPU and memory measurements using `pidstat`,
- collects system-level CPU, GPU, and memory measurements using `tegrastats`, and
- profiles CUDA execution using NVIDIA Nsight Systems.

Nsight Systems collects:

```bash
--trace=cuda,nvtx,osrt
```

A separate `.nsys-rep` report is generated for each workload.

### `run_isolated_nsight.sh`

Runs each workload individually with MPS disabled and Nsight Systems enabled.

This script is intended for isolated workload profiling without interference from the other YOLO workloads. It also collects `pidstat` and `tegrastats` measurements and generates Nsight Systems reports and CUDA summary files.

### `parse_logs.py`

Parses the logs produced by the concurrent experiment and creates:

```text
all_logs/parsed_logs_all.xlsx
```

The generated workbook contains:

- `Inference_Time`: per-image inference times for each workload
- `Processing_Time`: preprocessing, inference, postprocessing, and processing-total times
- `Tegrastats`: system-level CPU, GPU, and memory measurements
- workload-specific `pidstat` sheets: per-workload CPU and memory measurements

### `plot_graphs.py`

Reads `parsed_logs_all.xlsx` and generates plots for:

- inference time per workload
- system-level CPU, GPU, and memory utilization
- per-workload CPU utilization
- per-workload memory usage

## 5. Input Data

The current workload scripts expect the following image directories:

```text
imagenet_images/    # Classification
coco_images/        # Detection and Pose Estimation
dota_images/        # Segmentation and OBB
```

The corresponding TensorRT engine files must also be placed in the repository directory:

```text
yolo26n-cls.engine
yolo26n.engine
yolo26n-pose.engine
yolo26n-seg.engine
yolo26n-obb.engine
```

TensorRT engine files are platform dependent. Engines should therefore be generated for the target Jetson platform and software environment.

## 6. Concurrent Experiment with MPS

### 6.1 Start NVIDIA MPS

NVIDIA MPS must be running before executing `run_all_mps_nsys.sh`.

The experiment script checks whether the MPS control process is running and terminates if MPS is not available.

### 6.2 Run the Concurrent Experiment

Make the script executable:

```bash
chmod +x run_all_mps_nsys.sh
```

Run:

```bash
./run_all_mps_nsys.sh
```

Five Docker containers are launched concurrently:

```text
detection
classification
estimation
segmentation
obb
```

Each workload is configured with:

```bash
CUDA_MPS_ENABLE_PER_CTX_DEVICE_MULTIPROCESSOR_PARTITIONING=1
CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=20
```

The workloads wait for a common target time before execution so that their execution starts approximately simultaneously.

## 7. Parse Experimental Results

After the concurrent experiment finishes, run:

```bash
python3 parse_logs.py
```

The parser reads the workload logs, `pidstat` logs, and `tegrastats` log and generates:

```text
all_logs/parsed_logs_all.xlsx
```

The basic data-processing flow is:

```text
run_all_mps_nsys.sh
        │
        ├── workload logs
        ├── pidstat logs
        ├── tegrastats log
        └── Nsight Systems reports
                 │
                 ▼
           parse_logs.py
                 │
                 ▼
       parsed_logs_all.xlsx
```

## 8. Generate Graphs

After generating `parsed_logs_all.xlsx`, run:

```bash
python3 plot_graphs.py
```

The following figures are generated in `all_logs/`:

```text
inference_plot.png
tegrastats_boxplot.png
workload_boxplot.png
```

The complete workflow is therefore:

```text
./run_all_mps_nsys.sh
          │
          ▼
       Raw logs
          │
          ▼
 python3 parse_logs.py
          │
          ▼
 parsed_logs_all.xlsx
          │
          ▼
python3 plot_graphs.py
          │
          ▼
       PNG plots
```

## 9. Collected Metrics

### Per-Image Processing Time

Each YOLO workload records:

```text
preprocess
inference
postprocess
processing_total
```

where:

```text
processing_total = preprocess + inference + postprocess
```

The scripts additionally measure the wall-clock time surrounding the complete model execution:

```text
Total wall-clock time
Average wall-clock time per image
```

The processing total and wall-clock time represent different measurements. The processing total is calculated from the timing components reported by Ultralytics, whereas the wall-clock measurement includes the elapsed time observed around the complete model call.

### Per-Workload Resource Usage

`pidstat` is used to collect:

- CPU utilization
- resident memory usage (RSS)

for each workload process.

### System-Level Resource Usage

`tegrastats` is used to collect:

- CPU utilization
- GPU utilization
- RAM usage

for the overall Jetson system.

### CUDA Execution

NVIDIA Nsight Systems is used to collect CUDA execution traces for each workload.

The profiler records CUDA, NVTX, and OS runtime activity and produces a separate `.nsys-rep` file for each workload.

## 10. Output Files

A typical concurrent experiment produces:

```text
all_logs/
├── all_tegrastat.log
├── classification.log
├── detection.log
├── estimation.log
├── segmentation.log
├── obb.log
├── pidstat_classification.log
├── pidstat_detection.log
├── pidstat_estimation.log
├── pidstat_segmentation.log
├── pidstat_obb.log
├── nsys_classification.log
├── nsys_detection.log
├── nsys_estimation.log
├── nsys_segmentation.log
├── nsys_obb.log
├── nsys_classification.nsys-rep
├── nsys_detection.nsys-rep
├── nsys_estimation.nsys-rep
├── nsys_segmentation.nsys-rep
├── nsys_obb.nsys-rep
├── parsed_logs_all.xlsx
├── inference_plot.png
├── tegrastats_boxplot.png
└── workload_boxplot.png
```

The `.log` files contain workload and resource-monitoring results, while the `.nsys-rep` files can be opened with NVIDIA Nsight Systems for detailed timeline analysis.

## 11. Isolated Nsight Systems Profiling

To profile workloads individually rather than concurrently, use:

```bash
chmod +x run_isolated_nsight.sh
./run_isolated_nsight.sh
```

This experiment disables NVIDIA MPS and executes the workloads sequentially:

```text
Detection
    ↓
Classification
    ↓
Pose Estimation
    ↓
Segmentation
    ↓
OBB
```

The isolated results are stored separately under:

```text
isolated_logs/
```

Nsight Systems reports and CUDA summary files are stored under:

```text
isolated_logs/nsys/
```

This allows the CUDA behavior of each workload during isolated execution to be compared with its behavior during concurrent execution.

## 12. Experimental Workflow

The repository supports two main execution modes:

```text
                    YOLO-EdgeBench
                         │
            ┌────────────┴────────────┐
            │                         │
            ▼                         ▼
     Concurrent Execution       Isolated Execution
    run_all_mps_nsys.sh       run_isolated_nsight.sh
            │                         │
         MPS ON                    MPS OFF
       5 workloads              1 workload
       concurrently             at a time
            │                         │
       Nsight Systems             Nsight Systems
       pidstat                    pidstat
       tegrastats                 tegrastats
            │                         │
            ▼                         ▼
        all_logs/              isolated_logs/
```

The concurrent experiment is used to evaluate multi-workload execution and resource sharing, while the isolated experiment provides a baseline for analyzing the execution characteristics of each workload without concurrent GPU workloads.

## 13. Notes

- TensorRT engine files are hardware- and software-environment dependent and may need to be regenerated for each Jetson platform.
- The current MPS experiment assigns an active thread percentage of 20 to each of the five concurrent workloads.
- Nsight Systems reports are generated separately for each workload because each workload is launched as an independent profiling target.
- `tegrastats` measures system-level resource utilization, whereas `pidstat` measures the resource usage of individual workload processes.
- Profiling introduces measurement overhead. Nsight Systems results should therefore be interpreted primarily as execution-trace and profiling data rather than assuming that profiling has no effect on execution time.
