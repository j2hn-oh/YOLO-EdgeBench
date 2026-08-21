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
- Python 3

The Docker image used for the YOLO workloads is:

```bash
ultralytics/ultralytics:latest-jetson-jetpack6
```

The current experiment scripts use NVIDIA Nsight Systems 2024.5.4 installed on the host and mount it into the workload containers.

## 4. Repository Structure

```text
YOLO-EdgeBench/
├── classification.py          # YOLO classification workload
├── detection.py               # YOLO object detection workload
├── estimation.py              # YOLO pose estimation workload
├── segmentation.py            # YOLO segmentation workload
├── obb.py                     # YOLO oriented bounding box workload
├── run_all_mps_nsys.sh        # Concurrent execution with MPS and profiling
├── run_isolated_nsight.sh     # Isolated Nsight Systems profiling
├── parse_logs.py              # Parse experiment logs into Excel
├── plot_graphs.py             # Generate plots from parsed results
└── README.md
```

## 5. Setup

### Input Images

The workload scripts use image datasets stored in the corresponding input directories. Make sure that the paths in each workload script match the local dataset locations before running the experiment.

The current scripts use directories such as:

```text
imagenet_images/
coco_images/
dota_images/
```

### TensorRT Engines

Place the required TensorRT engine files in the directory expected by the workload scripts:

```text
yolo26n-cls.engine
yolo26n.engine
yolo26n-pose.engine
yolo26n-seg.engine
yolo26n-obb.engine
```

TensorRT engine files are dependent on the target hardware and software environment. Engines should therefore be generated for the corresponding Jetson platform and TensorRT environment.

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

`run_isolated_nsight.sh` is used to profile each workload individually without concurrent YOLO workloads.

Run:

```bash
chmod +x run_isolated_nsight.sh
./run_isolated_nsight.sh
```

The script first disables NVIDIA MPS and verifies that no MPS control or server process remains. The five workloads are then executed sequentially, one at a time, with Nsight Systems enabled. This provides isolated profiling results that can be compared with concurrent execution.

The isolated results are stored separately under:

```text
isolated_logs/
```

## 7. Processing and Visualizing Results

For the concurrent experiment, the main result-processing workflow is:

```text
./run_all_mps_nsys.sh
          │
          ▼
   Raw experiment logs
   + Nsight reports
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

### 7.1 Parse Logs

After `run_all_mps_nsys.sh` finishes, run:

```bash
python3 parse_logs.py
```

The script parses the workload, `pidstat`, and `tegrastats` logs and generates:

```text
all_logs/parsed_logs_all.xlsx
```

The workbook contains the following data:

- `Inference_Time`: per-image inference time for each workload
- `Processing_Time`: preprocessing, inference, postprocessing, and total processing time
- `Tegrastats`: system-level CPU, GPU, and memory measurements
- workload-specific `pidstat` sheets: per-workload CPU and memory measurements

### 7.2 Generate Graphs

After generating `parsed_logs_all.xlsx`, run:

```bash
python3 plot_graphs.py
```

The script generates:

```text
all_logs/inference_plot.png
all_logs/tegrastats_boxplot.png
all_logs/workload_boxplot.png
```

These plots visualize inference time and system/per-workload resource utilization.

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

Therefore, the following timing measurements are available:

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

### Nsight Systems Profiles

NVIDIA Nsight Systems generates a separate profiling report for each workload:

```text
nsys_classification.nsys-rep
nsys_detection.nsys-rep
nsys_estimation.nsys-rep
nsys_segmentation.nsys-rep
nsys_obb.nsys-rep
```

These files can be opened using NVIDIA Nsight Systems for timeline-based analysis of CUDA execution and runtime behavior.

A typical concurrent experiment produces files such as:

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

## 9. Notes

- The current concurrent experiment executes five YOLO workloads and uses `CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=20` for each workload.
- The MPS active thread percentage controls the execution resources available to a CUDA context; it should not be interpreted simply as allocating a fixed 20% portion of the entire GPU to each workload.
- Nsight Systems reports are generated separately because each workload is executed as an independent profiling target.
- `pidstat` measures resource usage for individual workload processes, whereas `tegrastats` measures system-level resource utilization.
- Nsight Systems profiling introduces measurement overhead. Profiling results are primarily intended for analyzing CUDA execution behavior and should be interpreted with this overhead in mind.
- TensorRT engine files should be regenerated for the target Jetson platform and TensorRT environment when the hardware or software environment changes.
