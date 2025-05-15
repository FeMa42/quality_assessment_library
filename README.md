# Quality Assessment Library

This library provides a set of functions for assessing the quality of images and 3D objects. It is focused on estimating the quality of 3D objects using images.
It provides two main functionalities:

- General metrics: metrics that can be applied to any 3D object. These metrics include: MSE, CLIP-S, Spectral_MSE, D_lambda, ERGAS, PSNR, RASE, RMSE_wind, SAM, MS-SSIM, SSIM, UQI, VIF, LPIPS, SCC, FID, IS, KID. And are mainly based on torchmetrics.

- **This branch is specifically designed for the assessment of TRELLIS.** It enables the evaluation of 3D objects reconstructed with the TRELLIS model. For an evaluation, pairs of images (original rendered images & rendering of Trellis reconstructions) are required. It additionally includes the calculation of geometric metrics: Relative Difference of Bounding-Box Aspect Ratio, Relative Difference of occupied Pixel area, Relative Difference of Angle Difference of Silhouette Outline Normals (Pairwise and summed), Relative Difference of Squared Angle Difference of Silhouette Outline Normals (Pairwise and Summed). It will also compute the metrics from the Car Quality Classifier for the ground truth images, the generated images and their relative difference.

## Installation

### System Dependencies

Before installing the Python dependencies, ensure the following system dependencies are installed:

- Dependencies for 3D Metrics:
  - `libxrender1` (Linux: `sudo apt update && sudo apt install -y libxrender1`)
- Dependencies for 2D Metrics:
  - /

If you need this Package for 2D Metrics only, you will not need to install the dependencies for the 3D Metrics.

### Option 1: Quick installation with models

The easiest way to install the package with all required models is to use the provided installation script:

```bash
# Clone the repository
git clone https://github.com/FeMa42/quality_assessment_library.git
cd quality_assessment_library

# Run the installation script
python scripts/install.py --develop
```

This will install the package and download all required model files into the cloned repository.

### Option 2: Manual installation

You can also install the package and download models separately:

```bash
# Install the package
pip install -e . --use-pep517

# Download the models
python -m scripts.download_models --output-dir ./car_quality_estimator/models
```

## Data Preprocessing

`data_preprocessing.ipynb` provides a pipeline for processing generated Trellis viewpoints so that they can be compared to Ground Truth evaluation data. It includes functionalities for Background Removal, Equal Scaling and Visualization. It also includes methods to align the images in the folders. Front View Detection detects the front view of a car in a folder of car images and enables aligning the viewpoints.

## Usage of general metrics

Example usage of the metrics library can be found in `metrics.ipynb`.
There is an example provided to compute the semantic and geometric metrics separately for a given folder pair. The folder pair has to contain matching pairs of images, where the corresponding image pairs have the same identifier (name).
We also provide an option to compute the metrics for an entire subdirectory with evaluation data for multiple objects. With this, the metrics for an entire evaluation run of Trellis can be computed.

## Usage of car quality metrics

Example usage of the car quality metrics can be found in `car_quality_metrics.ipynb`.

## Basic usage

```python
from car_quality_estimator.car_quality_metric import load_car_quality_score
import PIL.Image
import glob
import os 

# Load the car quality metric
car_quality_metric = load_car_quality_score(
    use_combined_embedding_model=True
)

# Load some images to analyze
test_image_dir = "example_data/2c21b97ff3dc4fc3b1ef9e4bb0164318"
all_images = glob.glob(os.path.join(test_image_dir, "*.png"))
all_images = [PIL.Image.open(image).convert("RGB") for image in all_images]

# Compute quality scores
scores = car_quality_metric.compute_scores_no_reference(all_images)
print(scores)
```

## TRELLIS Benchmarking

## Further resources

- Collection of 3D Quality Assessment Repositories: [3DQA Databases](https://github.com/zzc-1998/Point-cloud-quality-assessment)
- Aesthetic Predictor used by TRELLIS: [improved-aesthetic-predictor](https://github.com/christophschuhmann/improved-aesthetic-predictor)
- Dataset and Classifier for classifying the quality of 3D vehicles: [MeshFleet](https://github.com/FeMa42/MeshFleet)
