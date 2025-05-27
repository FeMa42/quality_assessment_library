"""
Pipeline classes for preprocessing and evaluation stages.
"""

import logging
import json
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
from tqdm import tqdm

from .constants import DEFAULT_CANVAS_SIZE, DEFAULT_FILL_RATIO
from .utils import gpu_memory_manager
from .progress import ProgressTracker

# Import the actual processing functions
from metrics.metrics_eval import process_metrics_by_viewpoint, tensor_to_serializable
from metrics.metrics import ImageBasedPromptEvaluator
from metrics.helpers import preprocess_image, process_folder_with_metadata_file
from metrics.viewpoint_florence import FlorenceWheelbaseOD
from metrics.helpers import evaluate_vehicle_dimensions
from preprocessing.image_processing import (
    remove_background_recursive,
    remove_background_recursive_parallel,
    align_views, 
    load_dino, 
    get_dino_transform,
    restructure_images,
    process_equal_scaling_structure
)


class PreprocessingPipeline:
    """Handles all preprocessing steps."""
    
    def __init__(self, config: Dict[str, Any], device: str, progress_tracker: ProgressTracker):
        self.config = config
        self.device = device
        self.progress_tracker = progress_tracker
    
    def run_background_removal(self, gen_folder: str) -> bool:
        """Run background removal with error handling."""
        try:
            self.progress_tracker.start_stage("Background Removal")
            
            single_thread = self.config.get('single_thread', False)
            
            if single_thread:
                folders = [f for f in Path(gen_folder).iterdir() if f.is_dir()]
                for folder_path in tqdm(folders, desc="Removing backgrounds"):
                    remove_background_recursive(str(folder_path))
            else:
                remove_background_recursive_parallel(gen_folder, max_workers=12)
            
            self.progress_tracker.complete_stage("Background Removal")
            return True
            
        except Exception as e:
            logging.error(f"Background removal failed: {e}")
            return False
    
    def run_view_alignment(self, gen_folder: str, gt_folder: str) -> bool:
        """Run view alignment with proper resource management."""
        try:
            self.progress_tracker.start_stage("View Alignment")
            
            with gpu_memory_manager():
                # Load DINO model
                dino_model = load_dino(self.device)
                dino_transform = get_dino_transform(224)
                
                df = align_views(
                    gen_folder,
                    detection_mode="dino",
                    dino_model=dino_model,
                    dino_transform=dino_transform,
                    ground_truth_parent=gt_folder,
                    device=self.device
                )

            self.progress_tracker.complete_stage("View Alignment")
            return True
            
        except Exception as e:
            logging.error(f"View alignment failed: {e}")
            return False
    
    def run_scale_equalization(self, gt_folder: str, gen_folder: str, force: bool = False,
                             canvas_size: Tuple[int, int] = DEFAULT_CANVAS_SIZE,
                             fill_ratio: float = DEFAULT_FILL_RATIO) -> bool:
        """Run scale equalization with validation."""
        try:
            self.progress_tracker.start_stage("Scale Equalization")
            
            parent_dir = Path(gt_folder).parent
            scaled_folder = parent_dir / "scaled"
            if scaled_folder.exists(): 
                scaled_gen_folder = scaled_folder / Path(gen_folder).name
                scaled_gt_folder = scaled_folder / Path(gt_folder).name
                if scaled_gen_folder.exists() and scaled_gt_folder.exists() and not force:
                    logging.info("Scaled folder already exists, skipping scale equalization...")
                    self.progress_tracker.complete_stage("Scale Equalization")
                    return True
            
            process_equal_scaling_structure(
                ground_truth_parent=gt_folder,
                generated_parent=gen_folder,
                canvas_size=canvas_size,
                fill_ratio=fill_ratio
            )
            
            self.progress_tracker.complete_stage("Scale Equalization")
            return True
            
        except Exception as e:
            logging.error(f"Scale equalization failed: {e}")
            return False
    
    def run_all(self, gt_folder: str, gen_folder: str) -> bool:
        """Run all enabled preprocessing steps."""
        success = True
        
        if self.config['preprocessing']['background_removal']['enabled']:
            if not self.run_background_removal(gen_folder):
                success = False
        
        if self.config['preprocessing']['view_alignment']['enabled']:
            if not self.run_view_alignment(gen_folder, gt_folder):
                success = False
        
        if self.config['preprocessing']['scale_equalization']['enabled']:
            force = self.config['preprocessing']['scale_equalization'].get('force', False)
            canvas_size = tuple(self.config['preprocessing']['scale_equalization'].get(
                'canvas_size', DEFAULT_CANVAS_SIZE))
            fill_ratio = self.config['preprocessing']['scale_equalization'].get(
                'fill_ratio', DEFAULT_FILL_RATIO)
            if not self.run_scale_equalization(gt_folder, gen_folder, force, canvas_size, fill_ratio):
                success = False
        
        return success


class EvaluationPipeline:
    """Handles all evaluation steps."""
    
    def __init__(self, config: Dict[str, Any], device: str, progress_tracker: ProgressTracker):
        self.config = config
        self.device = device
        self.progress_tracker = progress_tracker
        self.results = {}
    
    def evaluate_semantic_geometric(self, gt_folder: str, gen_folder: str, 
                                  config_path: str, metadata_file_path: Optional[str] = None) -> bool:
        """Evaluate semantic and geometric metrics with error handling."""
        try:
            self.progress_tracker.start_stage("Semantic & Geometric Evaluation")
            
            with gpu_memory_manager():
                metrics_result = process_metrics_by_viewpoint(
                    ground_truth_folder=gt_folder,
                    generated_folder=gen_folder,
                    device=self.device,
                    config_path=config_path,
                    metadata_file_path=metadata_file_path
                )
                
                self.results['semantic_geometric'] = tensor_to_serializable(metrics_result)
            
            self.progress_tracker.complete_stage("Semantic & Geometric Evaluation")
            return True
            
        except Exception as e:
            logging.error(f"Semantic/geometric evaluation failed: {e}")
            return False
    
    def evaluate_prompt_following(self, gen_folder: str, metadata_file_path: str) -> bool:
        """Evaluate prompt following metrics with error handling."""
        try:
            self.progress_tracker.start_stage("Prompt Following Evaluation")
            
            with gpu_memory_manager():
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
                
                self.results['prompt_following'] = combined_scores
            
            self.progress_tracker.complete_stage("Prompt Following Evaluation")
            return True
            
        except Exception as e:
            logging.error(f"Prompt following evaluation failed: {e}")
            return False
    
    def evaluate_vehicle_dimensions(self, gen_folder: str, metadata_file_path: str) -> bool:
        """Evaluate vehicle dimensions with error handling."""
        try:
            self.progress_tracker.start_stage("Vehicle Dimensions Evaluation")
            
            florence_wheelbase_od = FlorenceWheelbaseOD()
            average_diff, std_diff = evaluate_vehicle_dimensions(
                gen_folder,
                metadata_file_path,
                florence_wheelbase_od
            )
            
            self.results['vehicle_dimensions'] = {
                "average_diff": average_diff,
                "std_diff": std_diff
            }
            
            self.progress_tracker.complete_stage("Vehicle Dimensions Evaluation")
            return True
            
        except Exception as e:
            logging.error(f"Vehicle dimensions evaluation failed: {e}")
            return False
    
    def run_all(self, gt_folder: str, gen_folder: str, metrics_config_path: str, 
               metadata_file: str, filter_by_metadata: bool) -> bool:
        """Run all enabled evaluation steps."""
        success = True
        
        if self.config['evaluation']['semantic_geometric']['enabled']:
            metadata_file_path_vp = metadata_file if filter_by_metadata else None
            if not self.evaluate_semantic_geometric(gt_folder, gen_folder, metrics_config_path, metadata_file_path_vp):
                success = False
        
        if self.config['evaluation']['prompt_following']['enabled']:
            if not self.evaluate_prompt_following(gen_folder, metadata_file):
                success = False
        
        if self.config['evaluation']['vehicle_dimensions']['enabled']:
            if not self.evaluate_vehicle_dimensions(gen_folder, metadata_file):
                success = False
        
        return success
    
    def save_results(self, output_folder: str) -> None:
        """Save all results to files."""
        output_folder = Path(output_folder)
        
        # Save individual results
        for metric_name, result in self.results.items():
            output_file = output_folder / f"{metric_name}_eval.json"
            with open(output_file, 'w') as f:
                json.dump(result, f, indent=4)
            logging.info(f"Saved {metric_name} results to {output_file}")
        
        # Save combined results
        if self.results:
            combined_file = output_folder / "meshfleet_combined_metrics.json"
            with open(combined_file, 'w') as f:
                json.dump(self.results, f, indent=4)
            logging.info(f"Saved combined results to {combined_file}")