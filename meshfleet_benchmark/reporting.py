"""
Reporting and summary functions for the MeshFleet benchmark pipeline.
"""

import time
from typing import Dict, Any

from .constants import PREPROCESSING_STAGES, EVALUATION_STAGES
from .progress import ProgressTracker


def print_dry_run_summary(config: Dict[str, Any], gt_stats: Dict[str, Any], gen_stats: Dict[str, Any]) -> None:
    """Print a dry-run summary of what would be processed."""
    print("\n" + "="*80)
    print("DRY RUN SUMMARY")
    print("="*80)
    
    # Configuration summary
    print(f"Configuration file: {config.get('_config_path', 'N/A')}")
    print(f"Ground truth folder: {config['paths']['ground_truth_folder']}")
    print(f"Generated folder: {config['paths']['generated_folder']}")
    print(f"Output folder: {config['paths']['output_folder']}")
    print(f"Metadata file: {config['paths']['metadata_file']}")
    print(f"Metrics config: {config['paths']['metrics_config']}")
    
    # Data summary
    print(f"\nGround Truth Data:")
    print(f"  Total objects: {gt_stats['total_objects']}")
    print(f"  Valid objects: {gt_stats['valid_objects']}")
    print(f"  Total images: {gt_stats['total_images']}")
    if gt_stats['invalid_objects']:
        print(f"  Objects with missing images: {len(gt_stats['invalid_objects'])}")
    
    print(f"\nGenerated Data:")
    print(f"  Total objects: {gen_stats['total_objects']}")
    print(f"  Valid objects: {gen_stats['valid_objects']}")
    print(f"  Total images: {gen_stats['total_images']}")
    if gen_stats['invalid_objects']:
        print(f"  Objects with missing images: {len(gen_stats['invalid_objects'])}")
    
    # Pipeline stages
    print(f"\nProcessing Pipeline:")
    enabled_preprocessing = [stage for stage in PREPROCESSING_STAGES 
                           if config['preprocessing'][stage]['enabled']]
    enabled_evaluation = [stage for stage in EVALUATION_STAGES 
                        if config['evaluation'][stage]['enabled']]
    
    print(f"  Preprocessing stages: {', '.join(enabled_preprocessing) if enabled_preprocessing else 'None'}")
    print(f"  Evaluation stages: {', '.join(enabled_evaluation) if enabled_evaluation else 'None'}")
    print(f"  Filter by metadata: {config['evaluation']['filter_by_metadata']['enabled']}")
    
    print("\nTo execute the pipeline, run without --dry-run flag.")
    print("="*80)


def print_final_summary(progress_tracker: ProgressTracker, results: Dict[str, Any], 
                       config: Dict[str, Any]) -> None:
    """Print final execution summary."""
    total_time = time.time() - progress_tracker.start_time
    
    print("\n" + "="*80)
    print("EVALUATION COMPLETE")
    print("="*80)
    print(f"Total execution time: {total_time:.1f}s")
    print(f"Stages completed: {len(progress_tracker.completed_stages)}")
    print(f"Output directory: {config['paths']['output_folder']}")
    
    if results:
        print(f"\nResults generated:")
        for metric_name in results.keys():
            print(f"  - {metric_name}")
        
        # Print key metrics if available
        if 'semantic_geometric' in results:
            sg_results = results['semantic_geometric']
            if 'mean_scores' in sg_results:
                print(f"\nKey Semantic/Geometric Metrics:")
                for metric, value in sg_results['mean_scores'].items():
                    if isinstance(value, (int, float)):
                        print(f"  {metric}: {value:.4f}")
        
        if 'prompt_following' in results:
            pf_results = results['prompt_following']
            print(f"\nPrompt Following Metrics:")
            for metric, stats in pf_results.items():
                if isinstance(stats, dict) and 'mean' in stats:
                    print(f"  {metric}: {stats['mean']:.4f} ± {stats['std']:.4f}")
    
    print("="*80)