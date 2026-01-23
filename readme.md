This repository provides an end-to-end oil palm plantation analysis pipeline for GeoTIFF imagery: tree detection, infrastructure segmentation, land cleanness grid analysis, and palm clustering.

## Repository description

- Main input: satellite or drone imagery in **TIFF/GeoTIFF** format.  
- Core pipeline: `palmAnalysisPipeline` in `algorithm/pipeline.py`, which orchestrates:
  - `PalmDetector` for palm tree detection (bounding boxes).  
  - `InfraDetector` for infrastructure semantic segmentation.  
  - `LandCleannessAnalyzer` for grid-based land cleanness computation.  
  - `ClusterPalm` for spatial clustering of palms and cluster polygon generation.  
- Main outputs: GeoJSON and GeoTIFF files ready for GIS tools (e.g., QGIS) or web mapping.

***

## Requirements & environment

### Core Python dependencies

The project assumes Python **3.9** with the following stack (matching the container setup):

- Core deep learning:
  - `torch==1.11.0+cu113`, `torchvision`, `torchaudio`  
  - `mmcv-full==1.5.0` (built for CUDA 11.3 + Torch 1.11.0)  
  - `transformers==4.35.2`, `accelerate==0.25.0`, `sentencepiece==0.1.99`  
  - `openmim`, `timm`, `fairscale==0.4.13`, `scipy==1.10.1`, `yapf==0.40.1`  
- TIFF + inference tooling:
  - `scikit-image`, `scikit-learn`, `opencv-python`, `pydantic`, `matplotlib`, `rasterio`, `sahi==0.11.14`  
- Orchestration (optional): `modal==0.65.65` for cloud execution (not required for local inference).

### Docker base image

The Dockerfile uses:

- `nvidia/cuda:11.3.1-cudnn8-devel-ubuntu20.04`  
- Python 3.9 installed via `apt` + `get-pip.py`  
- PyTorch 1.11.0 + CUDA 11.3 wheels  
- `mmcv-full==1.5.0` installed from the OpenMMLab index for torch 1.11 / cu113  
- All other dependencies installed with `pip`  
- Repository copied into `/workspace` and local package `packages/Co-DETR` installed in editable mode.

***

## Setup with Docker

1. **Build the image**

```bash
docker build -t croptic-palm-pipeline .
```

This will:

- Install Python 3.9 and system tools (FFmpeg, OpenCV deps, etc.).  
- Install PyTorch 1.11.0 CUDA 11.3 and `mmcv-full==1.5.0`.  
- Install all segmentation and TIFF/inference dependencies.  
- Copy the repository into `/workspace`.  
- Install the local `Co-DETR` package from `packages/Co-DETR`.

2. **Run the container (with GPU)**

Make sure the NVIDIA Docker runtime is installed and configured, then:

```bash
docker run --gpus all -it --rm \
  -v /path/to/data:/workspace/data \
  -v /path/to/output:/workspace/output \
  croptic-palm-pipeline bash
```

This mounts your local `data` and `output` directories into the container.

***

## Setup without Docker (local install)

For local installation, use `install.py`:

```bash
python install.py
```

The script will:

- Check Python version (requires ≥ 3.8).  
- Check that `torch` is installed; if not, it prints a suggested install command.  
- Install all dependencies from `requirements.txt` (if present).  
- Install the local package `packages/Co-DETR` in editable mode (`-e`).

If `torch` is missing, install a compatible version (e.g., `torch==1.11.0+cu113` and matching `torchvision`/`torchaudio`) before rerunning `install.py`.

***

## How to run the pipeline (CLI)

Use the `palm_cli.py` script to run local inference over a directory of TIFF files. Example:

```bash
python scripts/palm_cli.py \
  --input-dir data/tifs \
  --output-dir output/palm_local \
  --palm-detector-model models/palm_detector/latest.pth \
  --seg-model chribark/segformer-b3-finetuned-UAVid \
  --palm-detector-config-path models/palm_detector/config.py \
  --device cuda \
  --cluster-n 7 \
  --min-cluster 7 \
  --grid-size 30 30
```

Argument meaning:

- `--input-dir`: directory containing local `.tif` / `.tiff` files.  
- `--output-dir`: root directory where per-TIFF outputs will be written.  
- `--palm-detector-model`: path to the palm detector weights.  
- `--palm-detector-config-path`: detector config file (e.g., MMDetection config).  
- `--seg-model`: segmentation model name or path (Hugging Face / local).  
- `--device`: `cuda` or `cpu`.  
- `--cluster-n`, `--min-cluster`, `--grid-size`: clustering and grid configuration.

### Programmatic usage

You can also use the pipeline directly from Python:

```python
from algorithm.pipeline import palmAnalysisPipeline

pipeline = palmAnalysisPipeline(
    palm_detector_model_path="models/palm_detector/latest.pth",
    config_path="models/palm_detector/config.py",
    seg_model_path="models/segmentation", # or hungging face path
    grid_size=(30, 30),
    cluster_n=7,
    min_cluster=7,
    device="cuda",
)

pipeline.run("data/sample.tif", output_dir="output/sample")
```

***

## Outputs

For each input `.tif`, the pipeline writes a dedicated output folder (e.g., `output/sample/`) containing:

- `bbox_palm.geojson`  
  - GeoJSON with palm tree bounding boxes in georeferenced coordinates.  
- `polygon_infra.geojson`  
  - GeoJSON with infrastructure polygons for the selected semantic labels (default `["building"]`).  
- `cleanness_grid.tif`  
  - GeoTIFF raster representing cleanness/greenness values per grid cell (`grid_size`).  
- `clusters_map.geojson`  
  - GeoJSON containing palm cluster polygons for spatial grouping/visualization.

These files can be opened directly in GIS software (e.g., QGIS) or consumed by geospatial tooling in Python.

## 📚 Documentation
[Docs](docs/_build/index.html)