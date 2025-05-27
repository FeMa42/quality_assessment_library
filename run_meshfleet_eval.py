#!/usr/bin/env python3
"""
MeshFleet Evaluation Pipeline

A comprehensive evaluation pipeline for 3D model quality assessment with preprocessing
and multiple evaluation metrics including semantic, geometric, prompt following, and
vehicle dimension analysis.

Usage:
    python run_meshfleet_eval.py --config config.yaml
    python run_meshfleet_eval.py --dry-run  # Preview without execution
    python run_meshfleet_eval.py --validate-only  # Validate config only
"""

import os
import argparse
import logging
import yaml
from pathlib import Path

from meshfleet_benchmark import (
    ConfigValidator,
    load_config,
    merge_config_with_args,
    PreprocessingPipeline,
    EvaluationPipeline,
    ProgressTracker,
    get_device,
    setup_logging,
    print_dry_run_summary,
    print_final_summary
)
from meshfleet_benchmark.constants import PREPROCESSING_STAGES, EVALUATION_STAGES
from meshfleet_benchmark.utils import check_geometric_metrics_requirements


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run evaluation for Meshfleet models with preprocessing options.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --config benchmark_config.yaml
  %(prog)s --dry-run --config benchmark_config.yaml
  %(prog)s --validate-only --config benchmark_config.yaml
  %(prog)s --skip-all-preprocessing --config benchmark_config.yaml
  %(prog)s --continue-on-error --verbose --config benchmark_config.yaml
        """
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
                        help='Override config to enable and force preprocessing')
    parser.add_argument('--skip-all-preprocessing', action='store_true',
                        help='Override config to skip all preprocessing')
    
    # Override options for evaluation
    parser.add_argument('--skip-semantic-geometric', action='store_true',
                        help='Override config to skip semantic and geometric metrics')
    parser.add_argument('--skip-prompt-following', action='store_true',
                        help='Override config to skip prompt following')
    parser.add_argument('--skip-vehicle-dimensions', action='store_true',
                        help='Override config to skip vehicle dimensions')
    parser.add_argument('--filter-by-metadata', action='store_true',
                        help='Override config to enable filtering by metadata')
    parser.add_argument('--no-filter-by-metadata', action='store_true',
                        help='Override config to disable filtering by metadata')
    
    # New features
    parser.add_argument('--dry-run', action='store_true',
                        help='Preview what will be processed without execution')
    parser.add_argument('--validate-only', action='store_true',
                        help='Only validate configuration and paths, then exit')
    parser.add_argument('--continue-on-error', action='store_true',
                        help='Continue pipeline execution even if individual stages fail')
    
    # Additional options
    parser.add_argument('--device', type=str, choices=['auto', 'cuda', 'cpu'],
                        help='Override device selection')
    parser.add_argument('--verbose', action='store_true',
                        help='Enable verbose logging')
    
    return parser.parse_args()


def main():
    """Main execution function."""
    args = parse_arguments()
    setup_logging(args.verbose)
    
    try:
        # Load and validate configuration
        logging.info("Loading configuration...")
        config = load_config(args.config)
        config['_config_path'] = args.config  # Store for reference
        logging.info(f"Loaded configuration from {args.config}")
        
        # Validate configuration schema
        schema_errors = ConfigValidator.validate_config_schema(config)
        if schema_errors:
            logging.error("Configuration validation failed:")
            for error in schema_errors:
                logging.error(f"  - {error}")
            return 1
        
        # Merge with command-line overrides
        config = merge_config_with_args(config, args)
        
        # Validate paths
        path_errors = ConfigValidator.validate_paths(config)
        if path_errors:
            logging.error("Path validation failed:")
            for error in path_errors:
                logging.error(f"  - {error}")
            return 1
        
        if args.validate_only:
            logging.info("Configuration and paths validated successfully!")
            return 0
        
        # Validate folder structures
        logging.info("Validating folder structures...")
        gt_stats = ConfigValidator.validate_folder_structure(config['paths']['ground_truth_folder'])
        gen_stats = ConfigValidator.validate_folder_structure(config['paths']['generated_folder'])
        
        if gt_stats['total_objects'] == 0:
            logging.error("No objects found in ground truth folder!")
            return 1
        
        if gen_stats['total_objects'] == 0:
            logging.error("No objects found in generated folder!")
            return 1
        
        # Print warnings for data issues
        if gt_stats['invalid_objects']:
            logging.warning(f"Ground truth: {len(gt_stats['invalid_objects'])} objects have incorrect image count")
        
        if gen_stats['invalid_objects']:
            logging.warning(f"Generated data: {len(gen_stats['invalid_objects'])} objects have incorrect image count")
        
        # Handle dry-run mode
        if args.dry_run:
            print_dry_run_summary(config, gt_stats, gen_stats)
            return 0
        
        # Setup execution environment
        logging.info("Setting up execution environment...")
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
        filter_by_metadata = config['evaluation']['filter_by_metadata']['enabled']
        
        # Determine enabled stages
        enabled_preprocessing = [stage for stage in PREPROCESSING_STAGES 
                               if config['preprocessing'][stage]['enabled']]
        enabled_evaluation = [stage.replace('_', ' & ').title() + ' Evaluation' 
                            for stage in EVALUATION_STAGES 
                            if config['evaluation'][stage]['enabled']]
        
        all_stages = enabled_preprocessing + enabled_evaluation
        if not all_stages:
            logging.warning("No processing stages enabled!")
            return 0
        
        # Initialize progress tracker
        progress_tracker = ProgressTracker(all_stages)
        
        # Run preprocessing pipeline
        preprocessing_success = True
        if enabled_preprocessing:
            logging.info("Initializing preprocessing pipeline...")
            preprocessing_pipeline = PreprocessingPipeline(config, device, progress_tracker)
            
            try:
                preprocessing_success = preprocessing_pipeline.run_all(gt_folder, gen_folder)
                if not preprocessing_success and not args.continue_on_error:
                    logging.error("Preprocessing failed, stopping pipeline")
                    return 1
            except Exception as e:
                logging.error(f"Preprocessing pipeline failed: {e}")
                if not args.continue_on_error:
                    return 1
                preprocessing_success = False
        
        # Determine evaluation folders
        if config['preprocessing']['scale_equalization']['enabled'] and preprocessing_success:
            parent_dir = Path(gt_folder).parent
            gt_dir_name = Path(gt_folder).name
            gen_dir_name = Path(gen_folder).name
            scaled_folder = parent_dir / "scaled"
            eval_gt_folder = str(scaled_folder / gt_dir_name)
            eval_gen_folder = str(scaled_folder / gen_dir_name)
            logging.info("Using scaled images for evaluation")
        elif config['evaluation']['semantic_geometric']['enabled']:
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
                logging.info("Using existing scaled images for geometric metrics")
            else:
                eval_gt_folder = gt_folder
                eval_gen_folder = gen_folder
                logging.info("Using original images for evaluation")
        else:
            eval_gt_folder = gt_folder
            eval_gen_folder = gen_folder
            logging.info("Using original images for evaluation")
        
        # report evaluation folders
        logging.info("Starting Evaluation with:")
        logging.info(f"Evaluation ground truth folder: {eval_gt_folder}")
        logging.info(f"Evaluation generated folder: {eval_gen_folder}")

        # Run evaluation pipeline
        evaluation_success = True
        if enabled_evaluation:
            logging.info("Initializing evaluation pipeline...")
            evaluation_pipeline = EvaluationPipeline(config, device, progress_tracker)
            
            try:
                evaluation_success = evaluation_pipeline.run_all(
                    eval_gt_folder, eval_gen_folder, metrics_config_path, 
                    metadata_file, filter_by_metadata
                )
                
                if evaluation_success or args.continue_on_error:
                    # Save results
                    evaluation_pipeline.save_results(output_folder)
                    
                    # Save configuration used for this run
                    config_output = Path(output_folder) / "benchmark_config_used.yaml"
                    with open(config_output, 'w') as f:
                        # Remove internal keys before saving
                        config_to_save = {k: v for k, v in config.items() if not k.startswith('_')}
                        yaml.dump(config_to_save, f, default_flow_style=False)
                    logging.info(f"Configuration used saved to {config_output}")
                    
                    # Print final summary
                    print_final_summary(progress_tracker, evaluation_pipeline.results, config)
                    
                elif not args.continue_on_error:
                    logging.error("Evaluation failed, stopping pipeline")
                    return 1
                    
            except Exception as e:
                logging.error(f"Evaluation pipeline failed: {e}")
                if not args.continue_on_error:
                    return 1
                evaluation_success = False
        
        # Final status
        if preprocessing_success and evaluation_success:
            logging.info("Pipeline completed successfully!")
            return 0
        elif args.continue_on_error:
            logging.warning("Pipeline completed with some failures (continue-on-error mode)")
            return 0
        else:
            logging.error("Pipeline failed!")
            return 1
            
    except KeyboardInterrupt:
        logging.info("Pipeline interrupted by user")
        return 130
    except Exception as e:
        logging.error(f"Unexpected error: {e}")
        if args.verbose:
            import traceback
            logging.error(traceback.format_exc())
        return 1


if __name__ == "__main__":
    exit(main())
