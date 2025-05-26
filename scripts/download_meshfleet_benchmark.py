#!/usr/bin/env python3
"""
Download and unpack MeshFleet benchmark data from Hugging Face.
"""

import os
import sys
import argparse
import pandas as pd
from pathlib import Path
from PIL import Image
import io
import json
import shutil
from tqdm import tqdm
from huggingface_hub import snapshot_download, list_repo_files, hf_hub_download


def download_benchmark_data(repo_id="DamianBoborzi/MeshFleet", output_dir=None):
    """Download all files from the benchmark_data folder."""
    if output_dir is None:
        output_dir = os.environ.get("MESHFLEET_DATA_DIR", "./data/meshfleet")
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Downloading MeshFleet benchmark data to: {output_path}")
    
    # Create temporary directory for parquet files
    temp_dir = output_path / "temp_parquet"
    temp_dir.mkdir(exist_ok=True)
    
    # Download parquet files to temporary location
    print("Downloading parquet files...")
    try:
        snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            local_dir=temp_dir,
            allow_patterns="benchmark_data/*/*.parquet",
            cache_dir=None,
        )
        print("Parquet files downloaded successfully.")
    except Exception as e:
        print(f"Error downloading parquet files: {e}")
        sys.exit(1)
    
    # Also download the CSV files
    for csv_file in ["meshfleet_test.csv", "meshfleet_train.csv"]:
        try:
            hf_hub_download(
                repo_id=repo_id,
                repo_type="dataset",
                filename=csv_file,
                local_dir=output_path,
                cache_dir=None,
            )
            print(f"Downloaded {csv_file}")
        except Exception as e:
            print(f"Warning: Could not download {csv_file}: {e}")
    
    return output_path, temp_dir


def unpack_parquet_files(temp_dir, output_dir, parquet_subfolder, output_subfolder):
    """Unpack parquet files into the expected folder structure."""
    parquet_dir = temp_dir / "benchmark_data" / parquet_subfolder
    output_path = output_dir / "benchmark_data" / output_subfolder
    
    if not parquet_dir.exists():
        print(f"No parquet files found in: {parquet_dir}")
        # Check if the output directory already exists (files were already unpacked)
        if output_path.exists() and any(output_path.iterdir()):
            print(f"Output directory already exists with content: {output_path}")
            return True
        return False
    
    output_path.mkdir(parents=True, exist_ok=True)
    
    parquet_files = list(parquet_dir.glob("*.parquet"))
    print(f"\nUnpacking {len(parquet_files)} parquet files from {parquet_subfolder}...")
    
    for parquet_file in tqdm(parquet_files):
        try:
            # Read the parquet file
            df = pd.read_parquet(parquet_file)
            
            # Get the SHA256 (folder name) from the first row
            if len(df) > 0:
                sha256 = df.iloc[0]['sha256']
                folder_path = output_path / sha256
                folder_path.mkdir(exist_ok=True)
                
                # Save the metadata JSON if it exists
                if 'metadata_json' in df.columns:
                    metadata_json_str = df.iloc[0]['metadata_json']
                    json_path = folder_path / "metadata.json"
                    with open(json_path, 'w', encoding='utf-8') as f:
                        f.write(metadata_json_str)
                
                # Extract and save all images
                for idx, row in df.iterrows():
                    filename = row['filename']
                    image_bytes = row['image_bytes']
                    
                    # Save the image
                    image = Image.open(io.BytesIO(image_bytes))
                    image_path = folder_path / filename
                    image.save(image_path)
                
        except Exception as e:
            print(f"Error processing {parquet_file.name}: {e}")
            continue
    
    return True


def main():
    parser = argparse.ArgumentParser(description="Download and unpack MeshFleet benchmark data")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for the data (default: uses MESHFLEET_DATA_DIR env var or ./data/meshfleet)"
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip downloading and only unpack existing parquet files"
    )
    parser.add_argument(
        "--skip-unpack",
        action="store_true",
        help="Skip unpacking parquet files"
    )
    parser.add_argument(
        "--keep-parquet",
        action="store_true",
        help="Keep the temporary parquet files after unpacking"
    )
    
    args = parser.parse_args()
    
    # Get output directory
    output_dir = args.output_dir
    if output_dir is None:
        output_dir = os.environ.get("MESHFLEET_DATA_DIR", "./data/meshfleet")
    
    output_path = Path(output_dir)
    temp_dir = None
    
    # Download data
    if not args.skip_download:
        output_path, temp_dir = download_benchmark_data(output_dir=output_dir)
        print(f"\nData downloaded to: {output_path}")
    else:
        print(f"Skipping download, using existing data at: {output_path}")
        temp_dir = output_path / "temp_parquet"
    
    # Unpack parquet files if they exist
    if not args.skip_unpack and temp_dir and temp_dir.exists():
        print("\nChecking for parquet files to unpack...")
        
        # Unpack conditioning images
        cond_success = unpack_parquet_files(
            temp_dir,
            output_path,
            "meshfleet_cond_images",
            "meshfleet_cond_images"
        )
        
        # Unpack evaluation images
        eval_success = unpack_parquet_files(
            temp_dir,
            output_path,
            "meshfleet_eval_images",
            "meshfleet_eval_images"
        )
        
        if cond_success or eval_success:
            print("\nUnpacking complete!")
        
        # Clean up temporary parquet files
        if not args.keep_parquet and temp_dir.exists():
            print("\nCleaning up temporary parquet files...")
            shutil.rmtree(temp_dir)
            print("Temporary files removed.")
    
    # Print summary
    print("\n" + "="*60)
    print("DOWNLOAD COMPLETE")
    print("="*60)
    print(f"Data location: {output_path}")
    
    # Check what actually exists
    cond_images_path = output_path / "benchmark_data" / "meshfleet_cond_images"
    eval_images_path = output_path / "benchmark_data" / "meshfleet_eval_images"
    
    if cond_images_path.exists():
        num_cond = len(list(cond_images_path.glob("*")))
        print(f"  Conditioning images: {num_cond} folders in {cond_images_path}")
    
    if eval_images_path.exists():
        num_eval = len(list(eval_images_path.glob("*")))
        print(f"  Evaluation images: {num_eval} folders in {eval_images_path}")
    
    print(f"\nTo use with the evaluation script:")
    print(f"  1. Set ground_truth_folder in benchmark_config.yaml to:")
    print(f"     {output_path}/benchmark_data/meshfleet_eval_images/")
    print(f"  2. Place your generated images in a similar structure")
    print(f"  3. Run: python run_meshfleet_eval.py")


if __name__ == "__main__":
    main()
