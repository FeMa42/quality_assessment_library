import os
import json
import torch
import gc
import pandas as pd
import argparse
import logging
import warnings
import yaml
from pathlib import Path
from tqdm import tqdm

from metrics.metrics_eval import process_metrics_by_viewpoint, tensor_to_serializable
from metrics.metrics import ImageBasedPromptEvaluator
from metrics.helpers import preprocess_image, process_folder_with_metadata_file
from metrics.viewpoint_florence import FlorenceWheelbaseOD
from metrics.helpers import evaluate_vehicle_dimensions
from preprocessing.image_processing import remove_background_recursive
from preprocessing.image_processing import align_views, load_dino, get_dino_transform
from preprocessing.image_processing import restructure_images
from preprocessing.image_processing import process_equal_scaling_structure


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run evaluation for Meshfleet models with preprocessing options."
    )
    
    # Config file option (primary way to configure)
    parser.add_argument(
        '--config',
        type=str,
        default="meshfleet_benchmark/benchmark_config.yaml",
        help="Path to the benchmark configuration file (YAML format)"
    )
    
    # Override options for paths
    parser.add_argument('--gt-folder', type=str, help="Override ground truth folder path")
    parser.add_argument('--gen-folder', type=str, help="Override generated folder path")
    parser.add_argument('--output-folder', type=str, help="Override output folder path")
    parser.add_argument('--metadata-file', type=str, help="Override metadata file path")
    parser.add_argument('--metrics-config', type=str, help="Override metrics config path")
    
    # Override options for preprocessing
    parser.add_argument('--skip-background-removal', action='store_true', 
                        help='Override config to skip background removal')
    parser.add_argument('--skip-view-alignment', action='store_true',
                        help='Override config to skip view alignment')
    parser.add_argument('--skip-scale-equalization', action='store_true',
                        help='Override config to skip scale equalization')
    parser.add_argument('--force-preprocessing', action='store_true',
                        help='Override config to force preprocessing')
    parser.add_argument('--skip-all-preprocessing', action='store_true',
                        help='Override config to skip all preprocessing')
    
    # Override options for evaluation
    parser.add_argument('--skip-semantic-geometric', action='store_true',
                        help='Override config to skip semantic and geometric metrics')
    parser.add_argument('--skip-prompt-following', action='store_true',
                        help='Override config to skip prompt following')
    parser.add_argument('--skip-vehicle-dimensions', action='store_true',
                        help='Override config to skip vehicle dimensions')
    
    # Additional options
    parser.add_argument('--device', type=str, choices=['auto', 'cuda', 'cpu'],
                        help='Override device selection')
    parser.add_argument('--verbose', action='store_true',
                        help='Enable verbose logging')
    
    return parser.parse_args()


def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def merge_config_with_args(config, args):
    """Merge command-line arguments with config file, args take precedence."""
    # Override paths if provided
    if args.gt_folder:
        config['paths']['ground_truth_folder'] = args.gt_folder
    if args.gen_folder:
        config['paths']['generated_folder'] = args.gen_folder
    if args.output_folder:
        config['paths']['output_folder'] = args.output_folder
    if args.metadata_file:
        config['paths']['metadata_file'] = args.metadata_file
    if args.metrics_config:
        config['paths']['metrics_config'] = args.metrics_config
    
    # Override preprocessing settings
    if args.skip_all_preprocessing:
        config['preprocessing']['background_removal']['enabled'] = False
        config['preprocessing']['view_alignment']['enabled'] = False
        config['preprocessing']['scale_equalization']['enabled'] = False
    else:
        if args.skip_background_removal:
            config['preprocessing']['background_removal']['enabled'] = False
        if args.skip_view_alignment:
            config['preprocessing']['view_alignment']['enabled'] = False
        if args.skip_scale_equalization:
            config['preprocessing']['scale_equalization']['enabled'] = False
    
    if args.force_preprocessing:
        config['preprocessing']['background_removal']['force'] = True
        config['preprocessing']['view_alignment']['force'] = True
        config['preprocessing']['scale_equalization']['force'] = True
    
    # Override evaluation settings
    if args.skip_semantic_geometric:
        config['evaluation']['semantic_geometric']['enabled'] = False
    if args.skip_prompt_following:
        config['evaluation']['prompt_following']['enabled'] = False
    if args.skip_vehicle_dimensions:
        config['evaluation']['vehicle_dimensions']['enabled'] = False
    
    # Override runtime settings
    if args.device:
        config['runtime']['device'] = args.device
    
    return config


def setup_logging(verbose=False):
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )


def get_device(device_config):
    """Get the device to use based on configuration."""
    if device_config == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device_config


def check_geometric_metrics_requirements(config_path, gen_folder):
    """Check if geometric metrics are enabled and scaled images are available."""
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # Check if geometric metrics are enabled
        geometric_config = config.get('geometric', {})
        geometric_enabled = geometric_config.get('enabled', False)
        
        if geometric_enabled:
            parent_dir = Path(gen_folder).parent
            scaled_path = parent_dir / 'scaled' / Path(gen_folder).name
            if not scaled_path.exists():
                warnings.warn(
                    "Geometric metrics are enabled but no 'scaled' folder found. "
                    "Geometric metrics require scale-equalized images for accurate results. "
                    "Consider running with scale equalization enabled or disable geometric metrics.",
                    UserWarning
                )
                return False, gen_folder
            logging.info("Found scaled images folder, will use for geometric metrics computation")
            return True, str(scaled_path)
    except Exception as e:
        logging.warning(f"Could not check geometric metrics requirements: {e}")
    
    return False, gen_folder


def run_background_removal(gen_folder, force=False):
    """Run background removal on generated images."""
    logging.info("Running background removal...")
    
    for name in os.listdir(gen_folder):
        folder_path = os.path.join(gen_folder, name)
        if os.path.isdir(folder_path):
            # Check if already processed
            has_bg_removed = any('_no_bg' in f for f in os.listdir(folder_path) if f.endswith('.png'))
            if has_bg_removed and not force:
                logging.info(f"Background already removed for {name}, skipping...")
                continue
                
            logging.info(f"Processing {folder_path!r} for background removal...")
            remove_background_recursive(folder_path)


def run_view_alignment(gen_folder, gt_folder, device, force=False):
    """Run view alignment on generated images."""
    logging.info("Running view alignment...")
    
    # Load DINO model
    dino_model = load_dino(device)
    dino_transform = get_dino_transform(224)
    
    try:
        df = align_views(
            gen_folder, 
            detection_mode="dino", 
            dino_model=dino_model, 
            dino_transform=dino_transform, 
            ground_truth_parent=gt_folder, 
            device=device
        )
        
        logging.info("Restructuring images after alignment...")
        restructure_images(gen_folder)
        
    finally:
        # Clean up
        del dino_model
        del dino_transform
        torch.cuda.empty_cache()
        gc.collect()
        
    return df


def run_scale_equalization(gt_folder, gen_folder, force=False, canvas_size=(768, 768), fill_ratio=0.9):
    """Run scale equalization on images."""
    logging.info("Running scale equalization...")
    
    parent_dir = Path(gt_folder).parent
    scaled_folder = parent_dir / "scaled"
    
    # Check if already exists
    if scaled_folder.exists() and not force:
        logging.info("Scaled folder already exists, skipping scale equalization...")
        return
    
    process_equal_scaling_structure(
        ground_truth_parent=gt_folder,
        generated_parent=gen_folder,
        canvas_size=canvas_size,
        fill_ratio=fill_ratio
    )


def evaluate_semantic_geometric(gt_folder, gen_folder, config_path, device):
    """Evaluate semantic and geometric metrics."""
    logging.info("Calculating Semantic and Geometry metrics...")
    
    metrics_result = process_metrics_by_viewpoint(
        ground_truth_folder=gt_folder,
        generated_folder=gen_folder,
        device=device,
        config_path=config_path,
    )
    
    # Clear GPU memory
    torch.cuda.empty_cache()
    gc.collect()
    
    return metrics_result


def evaluate_prompt_following(gen_folder, metadata_file_path):
    """Evaluate prompt following metrics."""
    logging.info("Calculating Prompt Following metrics...")
    
    prompt_metric = ImageBasedPromptEvaluator()
    mean_scores, std_scores = process_folder_with_metadata_file(
        generated_base_folder=gen_folder,
        metadata_file_path=metadata_file_path,
        prompt_metric=prompt_metric,
        preprocess_image=preprocess_image
    )
    
    # Combine mean and std scores
    combined_scores = {}
    for key in mean_scores.keys():
        combined_scores[key] = {
            "mean": mean_scores[key],
            "std": std_scores[key]
        }
    
    # Clear GPU memory
    torch.cuda.empty_cache()
    gc.collect()
    
    return combined_scores


def evaluate_vehicle_dimensions_florence(gen_folder, metadata_file_path):
    """Evaluate vehicle dimensions metrics."""
    logging.info("Calculating Vehicle Dimensions metrics...")
    
    florence_wheelbase_od = FlorenceWheelbaseOD()
    average_diff, std_diff = evaluate_vehicle_dimensions(
        gen_folder, 
        metadata_file_path, 
        florence_wheelbase_od
    )
    combined_scores = {
        "average_diff": average_diff,
        "std_diff": std_diff
    }
    return combined_scores


def save_results(results, output_path):
    """Save results to JSON file."""
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=4)
    logging.info(f"Results saved to {output_path}")


def main():
    args = parse_arguments()
    setup_logging(args.verbose)
    
    # Load configuration
    try:
        config = load_config(args.config)
        logging.info(f"Loaded configuration from {args.config}")
    except Exception as e:
        logging.error(f"Failed to load config file: {e}")
        return
    
    # Merge with command-line overrides
    config = merge_config_with_args(config, args)
    
    logging.info("Starting Meshfleet evaluation pipeline...")
    
    # Setup environment
    os.environ["OMP_NUM_THREADS"] = config['runtime'].get('omp_num_threads', "1")
    os.makedirs(config['paths']['output_folder'], exist_ok=True)
    
    device = get_device(config['runtime']['device'])
    logging.info(f"Using device: {device}")
    
    # Extract settings from config
    gt_folder = config['paths']['ground_truth_folder']
    gen_folder = config['paths']['generated_folder']
    output_folder = config['paths']['output_folder']
    metadata_file = config['paths']['metadata_file']
    metrics_config_path = config['paths']['metrics_config']
    
    # Determine preprocessing steps
    preprocessing_steps = {
        'background_removal': config['preprocessing']['background_removal']['enabled'],
        'view_alignment': config['preprocessing']['view_alignment']['enabled'],
        'scale_equalization': config['preprocessing']['scale_equalization']['enabled']
    }
    do_preprocessing = any(preprocessing_steps.values())
    
    # Run preprocessing if needed
    if do_preprocessing:
        logging.info("Running preprocessing steps...")
        
        if preprocessing_steps['background_removal']:
            force = config['preprocessing']['background_removal']['force']
            run_background_removal(gen_folder, force=force)
        
        if preprocessing_steps['view_alignment']:
            force = config['preprocessing']['view_alignment']['force']
            run_view_alignment(gen_folder, gt_folder, device, force=force)
        
        if preprocessing_steps['scale_equalization']:
            force = config['preprocessing']['scale_equalization']['force']
            canvas_size = tuple(config['preprocessing']['scale_equalization']['canvas_size'])
            fill_ratio = config['preprocessing']['scale_equalization']['fill_ratio']
            run_scale_equalization(
                gt_folder, gen_folder, 
                force=force, canvas_size=canvas_size, fill_ratio=fill_ratio
            )
    
    # Determine which folders to use for evaluation
    if preprocessing_steps['scale_equalization']:
        parent_dir = Path(gt_folder).parent
        gt_dir_name = Path(gt_folder).name
        gen_dir_name = Path(gen_folder).name
        scaled_folder = parent_dir / "scaled"
        eval_gt_folder = str(scaled_folder / gt_dir_name)
        eval_gen_folder = str(scaled_folder / gen_dir_name)
    else:
        # Check if geometric metrics require scaled images
        use_scaled, suggested_folder = check_geometric_metrics_requirements(
            metrics_config_path, gen_folder
        )
        if use_scaled:
            parent_dir = Path(gt_folder).parent
            gt_dir_name = Path(gt_folder).name
            scaled_folder = parent_dir / "scaled"
            eval_gt_folder = str(scaled_folder / gt_dir_name)
            eval_gen_folder = suggested_folder
        else:
            eval_gt_folder = gt_folder
            eval_gen_folder = gen_folder
    
    # Initialize results dictionary
    all_results = {}
    
    # Run evaluations based on config
    if config['evaluation']['semantic_geometric']['enabled']:
        metrics_result = evaluate_semantic_geometric(
            eval_gt_folder, eval_gen_folder, metrics_config_path, device
        )
        all_results.update(tensor_to_serializable(metrics_result))
        
        # Save intermediate results
        output_file = os.path.join(output_folder, "meshfleet_semantic_geometric.json")
        save_results(tensor_to_serializable(metrics_result), output_file)
    
    if config['evaluation']['prompt_following']['enabled']:
        prompt_scores = evaluate_prompt_following(gen_folder, metadata_file)
        all_results["prompt_following"] = prompt_scores
        
        # Save intermediate results
        output_file = os.path.join(output_folder, "prompt_following_eval.json")
        save_results(prompt_scores, output_file)
    
    if config['evaluation']['vehicle_dimensions']['enabled']:
        dimension_scores = evaluate_vehicle_dimensions_florence(gen_folder, metadata_file)
        all_results["vehicle_dimensions"] = dimension_scores
        
        # Save intermediate results
        output_file = os.path.join(output_folder, "vehicle_dimensions_eval.json")
        save_results(dimension_scores, output_file)
    
    # Save combined results
    if all_results:
        output_file = os.path.join(output_folder, "meshfleet_combined_metrics.json")
        save_results(all_results, output_file)
        
        # Also save the configuration used for this run
        config_output = os.path.join(output_folder, "benchmark_config_used.yaml")
        with open(config_output, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        logging.info(f"Configuration used saved to {config_output}")
        
        logging.info("Evaluation completed successfully!")
    else:
        logging.warning("No evaluations were performed!")


if __name__ == "__main__":
    main()
