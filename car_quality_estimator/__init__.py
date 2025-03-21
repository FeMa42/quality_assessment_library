"""Car Quality Estimator package for evaluating 3D car models."""

from car_quality_estimator.car_quality_metric import CarQualityScore, load_car_quality_score
from car_quality_estimator.embedding_models import (
    generate_siglip_embedding_model,
    generate_dino_embedding_model_for_individual_images
)
from car_quality_estimator.individual_image_classifier import (
    IndividualImageClassifier,
    IndividualImageTransformer,
    generate_individual_siglip_estimator,
    generate_individual_combined_estimator
)

__version__ = "0.1.0"
