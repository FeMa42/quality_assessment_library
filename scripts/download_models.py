#!/usr/bin/env python3
"""
Script to download model files for car_quality_estimator package.
"""

import os
import argparse
import sys

def download_models(output_dir="./models"):
    """
    Download model files from HuggingFace.
    
    Args:
        output_dir (str): Directory to save model files to
    """
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("Error: huggingface_hub package not found. Please install it with:")
        print("pip install huggingface_hub")
        sys.exit(1)
    
    # Create models directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Downloading models to {output_dir}...")
    try:
        # Download the entire repository
        repo_path = snapshot_download(
            repo_id="DamianBoborzi/CarQualityEstimator",
            local_dir=output_dir,  # Where to save the files
            revision="main",  # Specific branch, tag, or commit
            repo_type="model",  # Can be "model", "dataset", or "space"
        )
        print(f"Models successfully downloaded to: {repo_path}")
        
        # Verify that required files exist
        required_files = [
            "car_quality_model_combined_transformer_individual.json",
            "car_quality_model_combined_transformer_individual.pt",
            "car_quality_model_siglip_transformer_individual.json",
            "car_quality_model_siglip_transformer_individual.pt",
            "pca_model_DINOv2.pkl"
        ]
        
        for file in required_files:
            if not os.path.exists(os.path.join(output_dir, file)):
                print(f"Warning: Required file {file} not found in downloaded models.")
        
        return True
    except Exception as e:
        print(f"Error downloading models: {str(e)}")
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download model files for car_quality_estimator")
    parser.add_argument("--output-dir", "-o", type=str, default="./models", 
                        help="Directory to save model files to")
    args = parser.parse_args()
    
    success = download_models(args.output_dir)
    if success:
        print("Model download complete!")
    else:
        print("Failed to download models.")
        sys.exit(1)