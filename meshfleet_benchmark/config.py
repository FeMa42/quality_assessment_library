"""
Configuration validation and management for the MeshFleet benchmark pipeline.
"""

import yaml
from pathlib import Path
from typing import Dict, List, Any
from dataclasses import dataclass

from .constants import (
    PREPROCESSING_STAGES, 
    EVALUATION_STAGES, 
    DEFAULT_EXPECTED_IMAGES
)


@dataclass
class PipelineConfig:
    """Configuration for the evaluation pipeline."""
    paths: Dict[str, str]
    preprocessing: Dict[str, Any]
    evaluation: Dict[str, Any]
    runtime: Dict[str, Any]


class ConfigValidator:
    """Validates configuration files and paths."""
    
    @staticmethod
    def validate_config_schema(config: Dict[str, Any]) -> List[str]:
        """Validate configuration schema and return list of errors."""
        errors = []
        
        # Required top-level sections
        required_sections = ['paths', 'preprocessing', 'evaluation', 'runtime']
        for section in required_sections:
            if section not in config:
                errors.append(f"Missing required section: '{section}'")
        
        # Validate paths section
        if 'paths' in config:
            required_paths = ['ground_truth_folder', 'generated_folder', 'output_folder', 
                            'metadata_file', 'metrics_config']
            for path_key in required_paths:
                if path_key not in config['paths']:
                    errors.append(f"Missing required path: 'paths.{path_key}'")
        
        # Validate preprocessing section
        if 'preprocessing' in config:
            for stage in PREPROCESSING_STAGES:
                if stage not in config['preprocessing']:
                    errors.append(f"Missing preprocessing stage: 'preprocessing.{stage}'")
                elif not isinstance(config['preprocessing'][stage], dict):
                    errors.append(f"Invalid preprocessing stage format: 'preprocessing.{stage}' must be a dict")
                elif 'enabled' not in config['preprocessing'][stage]:
                    errors.append(f"Missing 'enabled' flag in 'preprocessing.{stage}'")
        
        # Validate evaluation section
        if 'evaluation' in config:
            for stage in EVALUATION_STAGES:
                if stage not in config['evaluation']:
                    errors.append(f"Missing evaluation stage: 'evaluation.{stage}'")
                elif not isinstance(config['evaluation'][stage], dict):
                    errors.append(f"Invalid evaluation stage format: 'evaluation.{stage}' must be a dict")
                elif 'enabled' not in config['evaluation'][stage]:
                    errors.append(f"Missing 'enabled' flag in 'evaluation.{stage}'")
        
        return errors
    
    @staticmethod
    def validate_paths(config: Dict[str, Any]) -> List[str]:
        """Validate that required paths exist."""
        errors = []
        
        if 'paths' not in config:
            return ["Missing 'paths' section in config"]
        
        paths = config['paths']
        
        # Check ground truth folder
        if 'ground_truth_folder' in paths:
            gt_path = Path(paths['ground_truth_folder'])
            if not gt_path.exists():
                errors.append(f"Ground truth folder does not exist: {gt_path}")
            elif not gt_path.is_dir():
                errors.append(f"Ground truth path is not a directory: {gt_path}")
        
        # Check generated folder
        if 'generated_folder' in paths:
            gen_path = Path(paths['generated_folder'])
            if not gen_path.exists():
                errors.append(f"Generated folder does not exist: {gen_path}")
            elif not gen_path.is_dir():
                errors.append(f"Generated path is not a directory: {gen_path}")
        
        # Check metadata file
        if 'metadata_file' in paths:
            metadata_path = Path(paths['metadata_file'])
            if not metadata_path.exists():
                errors.append(f"Metadata file does not exist: {metadata_path}")
            elif not metadata_path.is_file():
                errors.append(f"Metadata path is not a file: {metadata_path}")
        
        # Check metrics config
        if 'metrics_config' in paths:
            metrics_path = Path(paths['metrics_config'])
            if not metrics_path.exists():
                errors.append(f"Metrics config file does not exist: {metrics_path}")
            elif not metrics_path.is_file():
                errors.append(f"Metrics config path is not a file: {metrics_path}")
        
        return errors
    
    @staticmethod
    def validate_folder_structure(folder_path: str, expected_images: int = DEFAULT_EXPECTED_IMAGES) -> Dict[str, Any]:
        """Validate folder structure and return statistics."""
        folder_path = Path(folder_path)
        stats = {
            'total_objects': 0,
            'valid_objects': 0,
            'invalid_objects': [],
            'missing_images': [],
            'total_images': 0
        }
        
        if not folder_path.exists():
            return stats
        
        for obj_folder in folder_path.iterdir():
            if obj_folder.is_dir():
                stats['total_objects'] += 1
                images = list(obj_folder.glob('*.png')) + list(obj_folder.glob('*.jpg'))
                stats['total_images'] += len(images)
                
                if len(images) == expected_images:
                    stats['valid_objects'] += 1
                else:
                    stats['invalid_objects'].append({
                        'object': obj_folder.name,
                        'images_found': len(images),
                        'expected': expected_images
                    })
                    if len(images) < expected_images:
                        stats['missing_images'].append(obj_folder.name)
        
        return stats


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def merge_config_with_args(config: Dict[str, Any], args) -> Dict[str, Any]:
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
        config['preprocessing']['background_removal']['enabled'] = True
        config['preprocessing']['view_alignment']['enabled'] = True
        config['preprocessing']['scale_equalization']['enabled'] = True
        config['preprocessing']['scale_equalization']['force'] = True
    
    # Override evaluation settings
    if args.skip_semantic_geometric:
        config['evaluation']['semantic_geometric']['enabled'] = False
    if args.skip_prompt_following:
        config['evaluation']['prompt_following']['enabled'] = False
    if args.skip_vehicle_dimensions:
        config['evaluation']['vehicle_dimensions']['enabled'] = False
    if args.filter_by_metadata:
        config['evaluation']['filter_by_metadata']['enabled'] = True
    if args.no_filter_by_metadata:
        config['evaluation']['filter_by_metadata']['enabled'] = False
    
    # Override runtime settings
    if args.device:
        config['runtime']['device'] = args.device
    
    return config