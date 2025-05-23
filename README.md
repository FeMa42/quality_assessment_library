# Quality Assessment Library
This library provides a set of functions for assessing the quality of 3D objects. It is focused on estimating the quality of generated 3D objects using images. 
It provides two main functionalities:
- General metrics (Semantic Metrics): metrics that can be applied to any 3D object. These metrics include: MSE, CLIP-S, Spectral_MSE, D_lambda, ERGAS, PSNR, RASE, RMSE_wind, SAM, MS-SSIM, SSIM, UQI, VIF, LPIPS, SCC. And are mainly based on torchmetrics.
- Geometry Metrics:
  - Rel_BB_Aspect_Ratio_Diff: Bounding box aspect ratio difference
  - Rel_Pixel_Area_Diff: Relative pixel area difference
  - Squared_Outline_Normals_Angle_Diff: Squared difference of the angle between the outline normals
  - Squared_Summed_Outline_Normals_Angle_Diff: Squared difference of the summed outline normals
- Distribution Metrics:
  - Frechet Inception Distance (FID): Measures the distance between two distributions
  - Kernel Inception Distance (KID): Similar to FID but uses a different approach to measure distance
  - Inception Score (IS): Measures the quality of generated images
- Prompt alignment metrics:
  - CLIP-S: Measures the alignment of generated images with text prompts
  - ImageReward: Measures the quality of generated images using a reward model
- Vehicle Based Dimension comparison
  - Width and length comparison (normalized by height)
  - Vehicle wheelbase comparison (normalized by height). Uses Florence OD to detect the wheels.

**This repository is specifically designed for the assessment of Generative Models on the MeshFleet Benchmark.** 
It enables the evaluation of 3D objects reconstructed with a generative model like TRELLIS. For an evaluation, pairs of images (original rendered images & rendering of Trellis reconstructions) are required. It additionally includes the calculation of geometric metrics: Relative Difference of Bounding-Box Aspect Ratio, Relative Difference of occupied Pixel area, Relative Difference of Angle Difference of Silhouette Outline Normals (Pairwise and summed), Relative Difference of Squared Angle Difference of Silhouette Outline Normals (Pairwise and Summed). 


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

## MeshFleet Benchmark

### Dataset
To run the MeshFleet benchmark, you need to download the MeshFleet dataset (you only need the `benchmark_data` folder). The dataset can be downloaded from the following link: [MeshFleet Dataset](https://huggingface.co/datasets/DamianBoborzi/MeshFleet). You can place the downloaded dataset in the `data` folder of the repository or another location of your choice.

> Note: You can also finetune the model on the data referenced in the `meshfleet_benchmark/meshfleet_train.csv` file. Those images are not used for the evaluation. 

### Conditioning Images and Prompts
You can use the `meshfleet_cond_images_parquet` as the conditioning images for your model. For text to 3D you can use the `caption_3d_prompt` from the `meshfleet_benchmark/meshfleet_test.csv` file. For the evaluation, renders of the generated objects are reguired. Currently we use 12 images, rendered at horizontal viewpoints around the object with a distance of 30 degrees. The images are generated at a fixed elevation of 90 degrees. The azimuths and elevations are defined as follows:
```
azimuths =    [ 0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330]
elevations =  [90, 90, 90, 90,  90,  90,  90,  90,  90,  90,  90,  90]
```

### Evaluation
Lastly you need to set the reference and generated data (path to the rendered images of your objects) paths in `meshfleet_benchmark/benchmark_config.yaml` file. The paths should point to the folders containing the reference and generated images. The folder structure should look like this:

```
data/
├── meshfleet_test/
│   ├── 2c21b97ff3dc4fc3b1ef9e4bb0164318/
│   │   ├── 000.png
│   │   ├── 001.png
│   │   ├── 002.png
...
│   ├── 2c21b97ff3dc4fc3b1ef9e4bb0164318/
│   │   ├── 000.png
│   │   ├── 001.png
│   │   ├── 002.png
...
├── meshfleet_generated/
│   ├── 2c21b97ff3dc4fc3b1ef9e4bb0164318/
│   │   ├── 000.png
│   │   ├── 001.png
│   │   ├── 002.png
...
```

After downloading the dataset and setting the paths, you can run the benchmark by executing the following command:

```bash
python run_meshfleet_eval.py
```

By default the script will also preprocess the images (removing background, aligning the images and scaling the objects for the geometry metrics). The preprocessing can be skipped by setting the `--skip-all-preprocessing` flag or by setting the corresponding flags in the `meshfleet_benchmark/benchmark_config.yaml` file. 

## Data Preprocessing

`data_preprocessing.ipynb` provides a pipeline for processing generated Trellis viewpoints so that they can be compared to Ground Truth evaluation data. It includes functionalities for Background Removal, Equal Scaling and Visualization. It also includes methods to align the images in the folders. Front View Detection detects the front view of a car in a folder of car images and enables aligning the viewpoints.

## Usage of general metrics

Example usage of the metrics library can be found in `metrics.ipynb`.
There is an example provided to compute the semantic and geometric metrics separately for a given folder pair. The folder pair has to contain matching pairs of images, where the corresponding image pairs have the same identifier (name).
We also provide an option to compute the metrics for an entire subdirectory with evaluation data for multiple objects. With this, the metrics for an entire evaluation run of Trellis can be computed.

## Usage of car quality metrics

Example usage of the car quality metrics can be found in `car_quality_metrics.ipynb`.

## Basic usage:

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
