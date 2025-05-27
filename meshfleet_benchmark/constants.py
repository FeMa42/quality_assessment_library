"""
Constants used throughout the MeshFleet benchmark pipeline.
"""

# Image processing constants
DEFAULT_CANVAS_SIZE = (768, 768)
DEFAULT_FILL_RATIO = 0.9
DEFAULT_EXPECTED_IMAGES = 12

# Pipeline stage definitions
PREPROCESSING_STAGES = ['background_removal', 'view_alignment', 'scale_equalization']
EVALUATION_STAGES = ['semantic_geometric', 'prompt_following', 'vehicle_dimensions']