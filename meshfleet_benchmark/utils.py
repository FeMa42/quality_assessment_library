"""
Utility functions for the MeshFleet benchmark pipeline.
"""

import gc
import torch
import logging
import json
import warnings
from pathlib import Path
from contextlib import contextmanager
from typing import Tuple


@contextmanager
def gpu_memory_manager():
    """Context manager for GPU memory cleanup."""
    try:
        yield
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()


def get_device(device_config: str) -> str:
    """Get the device to use based on configuration."""
    if device_config == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device_config


def setup_logging(verbose: bool = False) -> None:
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )


def check_geometric_metrics_requirements(config_path: str, gen_folder: str) -> Tuple[bool, str]:
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