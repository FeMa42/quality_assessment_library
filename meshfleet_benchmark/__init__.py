"""
MeshFleet Benchmark Package

A modular evaluation pipeline for 3D model quality assessment.
"""

from .config import ConfigValidator, load_config, merge_config_with_args
from .pipeline import PreprocessingPipeline, EvaluationPipeline
from .progress import ProgressTracker
from .utils import gpu_memory_manager, get_device, setup_logging
from .reporting import print_dry_run_summary, print_final_summary

__version__ = "1.0.0"
__all__ = [
    "ConfigValidator",
    "load_config", 
    "merge_config_with_args",
    "PreprocessingPipeline",
    "EvaluationPipeline", 
    "ProgressTracker",
    "gpu_memory_manager",
    "get_device",
    "setup_logging",
    "print_dry_run_summary",
    "print_final_summary"
]