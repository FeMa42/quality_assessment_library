# Quality Assessment Library

This library provides a set of functions for assessing the quality of images and 3D objects. It is focused on estimating the quality of 3D objects using images. 
It provides two main functionalities:
- General metrics: metrics that can be applied to any 3D object. These metrics include: MSE, CLIP-S, Spectral_MSE, D_lambda, ERGAS, PSNR, RASE, RMSE_wind, SAM, MS-SSIM, SSIM, UQI, VIF, LPIPS, SCC, FID, IS, KID. And are mainly based on torchmetrics.
- Car quality metrics: metrics that are specifically designed for cars using SigLIP and DINOv2 embeddings as features and a classifier trained on a dataset of quality-annotated cars (<https://huggingface.co/datasets/DamianBoborzi/CarQualityDataset>). The metrics include a quality score (the classifier score for a high quality car), the uncertainty of the quality score, a combined score that takes both the quality and the uncertainty into account, and metrics to assess the distribution of the quality scores. 

## Installation

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

## Usage of general metrics

Example usage of the metrics library can be found in `metrics.ipynb`.

## Usage of car quality metrics

Example usage of the car quality metrics can be found in `car_quality_metrics.ipynb`.

> Note: You have to download the model from [Huggingface](https://huggingface.co/DamianBoborzi/car_quality_estimator) and put them in the `./car_quality_estimator/models` folder (see [README_models.md](README_models.md) for more details).

### Basic usage:

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

## Further resources

- Collection of 3D Quality Assessment Repositories: [3DQA Databases](https://github.com/zzc-1998/Point-cloud-quality-assessment)
- Aesthetic Predictor used by TRELLIS: [improved-aesthetic-predictor](https://github.com/christophschuhmann/improved-aesthetic-predictor)
- Dataset and Classifier for classifying the quality of 3D vehicles: [MeshFleet](https://github.com/FeMa42/MeshFleet)
