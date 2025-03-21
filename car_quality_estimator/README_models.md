# Model Files

This directory should contain the model files for the car_quality_estimator package.

## Required Model Files

When using the combined embedding model (`use_combined_embedding_model=True`):
- `car_quality_model_combined_transformer_individual.json`
- `car_quality_model_combined_transformer_individual.pt`
- `pca_model_DINOv2.pkl`

When using the SigLIP-only model (`use_combined_embedding_model=False`):
- `car_quality_model_siglip_transformer_individual.json`
- `car_quality_model_siglip_transformer_individual.pt`

## How to Download Model Files

The model files can be downloaded from HuggingFace. You can use the `scripts/download_models.py` script to download the models or use the following code:

```python
from huggingface_hub import snapshot_download
import os

# Create models directory if it doesn't exist
os.makedirs("./car_quality_estimator/models", exist_ok=True)

# Download the entire repository
repo_path = snapshot_download(
    repo_id="DamianBoborzi/CarQualityEstimator",
    local_dir="./car_quality_estimator/models",  # Where to save the files
    revision="main",  # Specific branch, tag, or commit (optional)
    repo_type="model",  # Can be "model", "dataset", or "space"
)
print(f"Models downloaded to: {repo_path}")
```

You can also download them manually from: https://huggingface.co/DamianBoborzi/CarQualityEstimator