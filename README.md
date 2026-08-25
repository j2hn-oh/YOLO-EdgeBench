# YOLO-EdgeBench

YOLO-EdgeBench is an experimental framework for evaluating YOLO inference performance and resource utilization on edge computing platforms.

The current implementation evaluates five YOLO workloads on NVIDIA Jetson platforms. It supports concurrent workload execution using NVIDIA Multi-Process Service (MPS), system and per-workload resource monitoring, and CUDA execution profiling using NVIDIA Nsight Systems.

## 1. Workloads

The following five YOLO workloads are evaluated:

| Workload | Script | TensorRT Engine |
|---|---|---|
| Classification | `classification.py` | `yolo26n-cls.engine` |
| Object Detection | `detection.py` | `yolo26n.engine` |
| Pose Estimation | `estimation.py` | `yolo26n-pose.engine` |
| Segmentation | `segmentation.py` | `yolo26n-seg.engine` |
| Oriented Bounding Box (OBB) | `obb.py` | `yolo26n-obb.engine` |

The `.pt` models are converted to TensorRT engines on the target Jetson platform before running the experiments.

Each workload records per-image:

- preprocessing time
- inference time
- postprocessing time
- total processing time (`preprocess + inference + postprocess`)

The workload scripts also report the total wall-clock time and average wall-clock time per image.

## 2. Evaluation Platforms

The experiments are currently conducted on the following NVIDIA Jetson platforms:

- NVIDIA Jetson AGX Orin
- NVIDIA Jetson Orin NX
- NVIDIA Jetson Orin Nano

## 3. Requirements

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
- `sqlite3`
- Python 3

The Docker image used for the YOLO workloads is:

```bash
ultralytics/ultralytics:latest-jetson-jetpack6
```

The experiment scripts expect NVIDIA Nsight Systems 2024.5.4 at `/opt/nvidia/nsight-systems/2024.5.4`. If a different version is installed, update the Nsight Systems path in the experiment scripts accordingly.

## 4. Repository Structure

```text
YOLO-EdgeBench/
├── coco_images/               # Input images for detection, pose, and segmentation
├── dota_images/               # Input images for OBB
├── imagenet_images/           # Input images for classification
├── models/                    # YOLO .pt models and generated TensorRT engines
│   ├── yolo26n.pt
│   ├── yolo26n-cls.pt
│   ├── yolo26n-pose.pt
│   ├── yolo26n-seg.pt
│   └── yolo26n-obb.pt
├── classification.py          # YOLO classification workload
├── detection.py               # YOLO object detection workload
├── estimation.py              # YOLO pose estimation workload
├── segmentation.py            # YOLO segmentation workload
├── obb.py                     # YOLO oriented bounding box workload
├── pt_to_trt.py               # Convert YOLO .pt models to TensorRT engines
├── run_all_mps_nsys.sh        # Concurrent execution with MPS and Nsight profiling
├── run_isolated_nsys.sh       # Isolated execution with Nsight profiling
├── parse_logs.py              # Parse experiment logs into Excel
├── plot_graphs.py             # Generate plots from parsed results
└── README.md
```

## 5. Setup

### 5.1 Clone the Repository

Clone the repository and move to the repository directory:

```bash
git clone <repository-url>
cd YOLO-EdgeBench
```

### 5.2 Input Images

The repository contains the input image directories used by the five workloads:

```text
coco_images/
dota_images/
imagenet_images/
```

The workloads use these image sets as follows:

| Workload | Input Directory |
|---|---|
| Classification | `imagenet_images/` |
| Object Detection | `coco_images/` |
| Pose Estimation | `coco_images/` |
| Segmentation | `coco_images/` |
| OBB | `dota_images/` |

### 5.3 Generate TensorRT Engines

The `models/` directory contains the YOLO `.pt` models used to generate TensorRT engines.

Because TensorRT engines depend on the target hardware and TensorRT environment, generate the engines directly on each target Jetson platform.

Run `pt_to_trt.py` using the same Docker image used for the workloads:

```bash
docker run --rm \
    --runtime=nvidia \
    --gpus=all \
    -v "$(pwd):/home" \
    ultralytics/ultralytics:latest-jetson-jetpack6 \
    /bin/bash -lc "cd /home && python3 pt_to_trt.py"
```

After conversion, the `models/` directory contains:

```text
models/
├── yolo26n.pt
├── yolo26n.engine
├── yolo26n-cls.pt
├── yolo26n-cls.engine
├── yolo26n-pose.pt
├── yolo26n-pose.engine
├── yolo26n-seg.pt
├── yolo26n-seg.engine
├── yolo26n-obb.pt
└── yolo26n-obb.engine
```

The generated `.engine` files are used by the workload scripts.

## 6. Running the Experiments

### 6.1 Concurrent Execution with MPS

`run_all_mps_nsys.sh` executes all five YOLO workloads concurrently in separate Docker containers.

Before running the experiment, NVIDIA MPS must be running.

Make the script executable:

```bash
chmod +x run_all_mps_nsys.sh
```

Then run:

```bash
./run_all_mps_nsys.sh
```

The script launches the following five workloads concurrently:

```text
Classification
Detection
Pose Estimation
Segmentation
OBB
```

The workloads are synchronized to start at approximately the same time.

For the five-workload experiment, each workload is configured with:

```bash
CUDA_MPS_ENABLE_PER_CTX_DEVICE_MULTIPROCESSOR_PARTITIONING=1
CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=20
```

The current experiment applies the same 20% active-thread setting to each of the five CUDA contexts. This setting limits the portion of available CUDA execution resources that each context can use through MPS; it does not represent a fixed physical partition of 20% of the entire GPU.

During execution, the script collects:

- per-workload CPU utilization using `pidstat`
- per-workload memory usage using `pidstat`
- system-level CPU utilization using `tegrastats`
- system-level GPU utilization using `tegrastats`
- system-level memory usage using `tegrastats`
- CUDA execution traces using NVIDIA Nsight Systems

Nsight Systems profiles each workload using:

```bash
--trace=cuda,nvtx,osrt
```

A separate `.nsys-rep` file is generated for each workload.

### 6.2 Isolated Nsight Systems Profiling

`run_isolated_nsys.sh` is used to profile each workload individually without concurrent YOLO workloads.

Run:

```bash
chmod +x run_isolated_nsys.sh
./run_isolated_nsys.sh
```

Before running the isolated experiment, NVIDIA MPS must be disabled. The script verifies that no MPS control or server process is running and terminates if MPS is enabled. The five workloads are then executed sequentially, one at a time, with Nsight Systems enabled. This provides isolated profiling results that can be compared with concurrent execution.

The isolated results are stored separately under:

```text
isolated_logs/
```

## 7. Processing and Visualizing Results

The result-processing workflow is:

```text
Concurrent experiment
./run_all_mps_nsys.sh
          │
          ▼
      all_logs/
          │
          ├── Workload logs
          ├── pidstat logs
          ├── tegrastats log
          └── Nsight trace data
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

The same processing scripts also process results collected under `isolated_logs/`.

### 7.1 Parse Logs

After running the experiments, execute:

```bash
python3 parse_logs.py
```

For the concurrent experiment, the script generates:

```text
all_logs/parsed_logs_all.xlsx
```

For the isolated experiment, it generates:

```text
isolated_logs/parsed_logs_isolated.xlsx
```

The workbooks contain:

- `Inference_Time`: per-image inference time for each workload
- `Processing_Time`: preprocessing, inference, postprocessing, and total processing time
- `Tegrastats`: system-level CPU, GPU, and memory measurements
- workload-specific `pidstat` sheets: per-workload CPU and memory measurements

### 7.2 Generate Graphs

After parsing the logs, run:

```bash
python3 plot_graphs.py
```

The script generates inference-time and resource-utilization plots for the available concurrent and isolated experiment results.

## 8. Collected Metrics and Output

### Timing Metrics

For each image, each workload records preprocessing, inference, and postprocessing times, along with their sum as the total processing time:

```text
Preprocessing
     +
Inference
     +
Postprocessing
     =
Total Processing Time
```

The workload scripts additionally measure the wall-clock time of the complete model execution.

The following timing measurements are available:

- **Preprocessing time**: preprocessing time reported for each image
- **Inference time**: inference time reported for each image
- **Postprocessing time**: postprocessing time reported for each image
- **Total processing time**: sum of preprocessing, inference, and postprocessing times for each image
- **Total wall-clock time**: elapsed time of the complete model execution
- **Average wall-clock time**: total wall-clock time divided by the number of processed images

The reported processing time and measured wall-clock time cover different execution scopes and therefore can differ.

### Resource Metrics

`pidstat` provides workload-specific measurements, including:

- CPU utilization
- resident memory usage (RSS)

`tegrastats` provides system-level measurements, including:

- CPU utilization
- GPU utilization
- RAM usage

### Nsight Systems Trace Data

NVIDIA Nsight Systems generates a `.nsys-rep` report for each workload. Each report is also exported to an SQLite database containing the collected trace data.

Selected trace data are additionally exported to CSV files for direct inspection and analysis:

```text
nsys/
├── nsys_detection.nsys-rep
├── nsys_detection.sqlite
├── nsys_detection_kernel.csv
├── nsys_detection_runtime.csv
├── nsys_detection_memcpy.csv
├── nsys_detection_synchronization.csv
├── nsys_detection_nvtx.csv
├── nsys_detection_osrt.csv
└── ...
```

The CSV files contain:

- `*_kernel.csv`: CUDA kernel execution traces
- `*_runtime.csv`: CUDA Runtime API traces
- `*_memcpy.csv`: CUDA memory-copy traces
- `*_synchronization.csv`: CUDA synchronization traces
- `*_nvtx.csv`: NVTX events
- `*_osrt.csv`: OS runtime events

The `.nsys-rep` files can be opened using NVIDIA Nsight Systems for timeline-based analysis, while the SQLite and CSV files provide the underlying trace data in formats suitable for programmatic analysis.

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
├── parsed_logs_all.xlsx
├── inference_plot.png
├── tegrastats_boxplot.png
├── workload_boxplot.png
└── nsys/
    ├── nsys_classification.nsys-rep
    ├── nsys_classification.sqlite
    ├── nsys_classification_kernel.csv
    ├── ...
    ├── nsys_detection.nsys-rep
    ├── nsys_detection.sqlite
    ├── nsys_detection_kernel.csv
    └── ...
```

## 9. Notes

- The current concurrent experiment executes five YOLO workloads and uses `CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=20` for each workload.
- The MPS active-thread percentage controls the execution resources available to a CUDA context; it should not be interpreted as allocating a fixed 20% physical portion of the entire GPU to each workload.
- The isolated experiment runs the workloads sequentially with MPS disabled.
- Nsight Systems reports are generated separately because each workload is executed as an independent profiling target.
- Nsight Systems trace data are retained in `.nsys-rep` and SQLite formats, while selected trace categories are additionally exported to CSV.
- `pidstat` measures resource usage for individual workload processes, whereas `tegrastats` measures system-level resource utilization.
- Nsight Systems profiling introduces measurement overhead. Profiling results are primarily intended for analyzing CUDA execution behavior and should be interpreted with this overhead in mind.
- TensorRT engines are generated on the target Jetson platform from the `.pt` models provided under `models/`.
