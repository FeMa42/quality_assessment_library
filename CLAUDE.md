# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Common Commands

### Installation
```bash
# Quick installation with models (recommended)
python scripts/install.py --develop

# Manual installation
pip install -e . --use-pep517
python -m scripts.download_models --output-dir ./car_quality_estimator/models
```

### MeshFleet Benchmark Setup
```bash
# Automated setup (downloads data and creates config template)
bash scripts/setup_meshfleet_eval.sh

# Download benchmark data only
python -m scripts.download_meshfleet_benchmark --output-dir ./data/meshfleet
```

### Running Evaluations
```bash
# Run full MeshFleet benchmark evaluation
python run_meshfleet_eval.py

# Skip preprocessing steps
python run_meshfleet_eval.py --skip-background-removal --skip-view-alignment --skip-scale-equalization

# Use custom config
python run_meshfleet_eval.py --config path/to/custom_config.yaml
```

## Architecture Overview

This library evaluates 3D model quality through image-based analysis, specifically designed for the MeshFleet benchmark of 3D vehicle models.

### Core Components

**Car Quality Estimator** (`car_quality_estimator/`):
- `CarQualityScore`: Main quality assessment class using trained classifiers
- `IndividualImageClassifier`: Per-image quality assessment with SigLIP/DINOv2 embeddings
- `EmbeddingModels`: Feature extraction using SigLIP and DINOv2 models
- `UncertaintyEstimation`: Monte Carlo dropout for quality confidence scores

**General Metrics** (`metrics/`):
- `Metrics`: Configurable semantic metrics (PSNR, SSIM, LPIPS, CLIP-S, etc.)
- `ImageBasedPromptEvaluator`: Text-image alignment evaluation
- `FlorenceWheelbaseOD`: Vehicle dimension analysis using Florence object detection
- Distribution metrics: FID, KID, Inception Score

**Preprocessing** (`preprocessing/`):
- Background removal using REMBG
- View alignment using DINO features
- Scale equalization for geometric consistency

### Key Data Flow

1. **Input**: Pairs of rendered images (ground truth vs generated)
2. **Preprocessing**: Background removal → view alignment → scale equalization
3. **Evaluation**: 
   - Semantic metrics (pixel-level comparisons)
   - Geometric metrics (bounding box, outline analysis)
   - Distribution metrics (FID, KID across image sets)
   - Prompt alignment (CLIP-S, ImageReward)
   - Vehicle-specific dimensions (wheelbase, aspect ratios)

### Configuration

- `meshfleet_benchmark/benchmark_config.yaml`: Main evaluation configuration
- `meshfleet_benchmark/config_meshfleet.json`: Metrics selection and parameters
- Models downloaded to `car_quality_estimator/models/`

### Image Requirements

The system expects 12 rendered views per object at fixed viewpoints:
- Azimuths: [0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330] degrees
- Elevations: [90] degrees (fixed)
- Distance: 1.5 units (approximate, handled by preprocessing)

### Package Structure

- `car_quality_estimator/`: Vehicle-specific quality assessment
- `metrics/`: General-purpose image quality metrics
- `preprocessing/`: Image preprocessing pipeline
- `scripts/`: Installation and data download utilities
- `meshfleet_benchmark/`: Benchmark configuration and metadata