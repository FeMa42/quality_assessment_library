# Quality Assessment Library

This library provides a set of functions for assessing the quality of images and 3D objects. It is focused on estimating the quality of 3D objects using images. 
It provides two main functionalities:
- General metrics: metrics that can be applied to any 3D object. These metrics include: MSE, CLIP-S, Spectral_MSE, D_lambda, ERGAS, PSNR, RASE, RMSE_wind, SAM, MS-SSIM, SSIM, UQI, VIF, LPIPS, SCC, FID, IS, KID. And are mainly based on torchmetrics.
- Car quality metrics: metrics that are specifically designed for cars using SigLIP and DINOv2 embeddings as features and a classifier trained on a dataset of quality-annotated cars (<https://huggingface.co/datasets/DamianBoborzi/CarQualityDataset>). The metrics include a quality score (the classifier score for a high quality car), the uncertainty of the quality score, a combined score that takes both the quality and the uncertainty into account, and metrics to assess the distribution of the quality scores. 

## Installation

To install the dependencies, run the following command:

```bash
pip install -r requirements_min.txt
```

## Usage of general metrics

Example usage of the metrics library can be found in `metrics.ipynb`.

## Usage of car quality metrics

Download the model from [Huggingface](https://huggingface.co/DamianBoborzi/car_quality_estimator) and put them in the `models` folder. If you want to use the combined model also load the PCA model from the same repository. We provide an example on how to load the models in the `car_quality_metrics.ipynb` notebook. Example usage of the car quality metrics can also be found in `car_quality_metrics.ipynb`.

## Further resources

- Collection of 3D Quality Assessment Repositories: [3DQA Databases](https://github.com/zzc-1998/Point-cloud-quality-assessment)
- Aesthetic Predictor used by TRELLIS: [improved-aesthetic-predictor](https://github.com/christophschuhmann/improved-aesthetic-predictor)
- Dataset and Classifier for classifying the quality of 3D vehicles: [MeshFleet](https://github.com/FeMa42/MeshFleet)
