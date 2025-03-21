import torch
import numpy as np
from tqdm import tqdm
import json
import pickle
import os

from car_quality_estimator.individual_image_classifier import (
    generate_individual_siglip_estimator,
    generate_individual_combined_estimator
)
from car_quality_estimator.uncertrainty_estimation import prepare_model_for_uncertainty_estimation
from car_quality_estimator.divergences_and_distances import compute_distribution_metrics, compute_kid_from_embeddings


class CarQualityScore:
    """
    A metric for evaluating the quality of generated 3D vehicle models.
    
    This metric uses the trained car quality classifier to assess both the quality and 
    consistency of generated 3D car models. It provides both a direct quality score and 
    uncertainty estimates, which can be combined into a single metric.
    
    Args:
        config_path (str): Path to the JSON configuration file for the model architecture
        weights_path (str): Path to the model weights file (.pt)
        pca_model_path (str): Path to the PCA model for DINOv2 embeddings
        model_type (str, optional): Type of model to build. Defaults to "transformer".
        use_combined_embedding_model (bool, optional): Whether to use combined (SigLIP + DINOv2) embeddings. Defaults to False.
        device (str, optional): Device to run inference on. Defaults to "cuda" if available.
        mc_samples (int, optional): Number of Monte Carlo samples for uncertainty estimation. Defaults to 50.
        batch_size (int, optional): Batch size for embedding generation. Defaults to 32.
    """

    def __init__(self, config_path, weights_path, pca_model_path=None,
                 model_type="transformer", use_combined_embedding_model=False,
                 device=None, mc_samples=50, batch_size=32):
        # Set device
        self.device = device if device else (
            "cuda" if torch.cuda.is_available() else "cpu")
        self.mc_samples = mc_samples
        self.batch_size = batch_size
        self.use_combined_embedding_model = use_combined_embedding_model

        if use_combined_embedding_model:
            # Load the individual image classifier - this directly processes individual images
            self.classifier = generate_individual_combined_estimator(
                config_path=config_path,
                weights_path=weights_path,
                pca_file_name=pca_model_path,
                device=self.device,
                model_type=model_type
            )
        else:
            self.classifier = generate_individual_siglip_estimator(
                config_path=config_path, 
                weights_path=weights_path,
                device=self.device,
                model_type=model_type
            )

        # Create uncertainty-enabled model for MC dropout
        self.uncertainty_model = prepare_model_for_uncertainty_estimation(self.classifier.classifier)

    def generate_embeddings(self, images, batch_size=32):
        if len(images) > batch_size:
            # split into batches
            batches = [images[i:i+batch_size] for i in range(0, len(images), batch_size)]
            embeddings = []
            for batch in batches:
                embeddings.append(self.generate_embeddings_batch(batch))
            return torch.cat(embeddings, dim=0)
        else:
            return self.generate_embeddings_batch(images)

    def generate_embeddings_batch(self, images):
        """
        Generate embeddings for a batch of images
        
        Args:
            images (list): List of PIL images
            
        Returns:
            torch.Tensor: Embeddings for the images
        """
            
        if self.use_combined_embedding_model:
            # For combined embeddings, we need to generate both types
            with torch.no_grad():
                # Extract SigLIP embeddings
                siglip_embeddings = self.classifier.siglip_embedding_fnc(images)
                # Extract DINOv2 embeddings
                dino_embeddings = self.classifier.dino_embedding_fnc(images)
            
            # Concatenate embeddings
            embeddings = torch.cat((siglip_embeddings, dino_embeddings), dim=1)
        else:
            # For SigLIP-only embeddings
            with torch.no_grad():
                embeddings = self.classifier.embed_image(images)
        
        return embeddings

    def compute_quality_score_from_embeddings(self, embeddings):
        """
        Compute the quality score for a model using its pre-computed embeddings
        
        Args:
            embeddings (torch.Tensor): Tensor of shape [num_views, embedding_dim]
                containing embeddings for different views of a model
            
        Returns:
            dict: Dictionary containing quality scores and uncertainty metrics
        """

        # Compute uncertainty using MC dropout
        uncertainty_metrics = self._compute_uncertainty_from_embeddings(embeddings)
        
        # Return all metrics
        return {
            "quality_score": uncertainty_metrics["avg_score"].cpu().numpy(),
            "entropy": uncertainty_metrics["entropy"].cpu().numpy(),
            "combined_score": self._calculate_combined_score(uncertainty_metrics["avg_score"], uncertainty_metrics["entropy"])
        }
    
    def _compute_uncertainty_from_embeddings(self, embeddings):
        """
        Compute uncertainty metrics using Monte Carlo dropout from pre-computed embeddings
        
        Args:
            embeddings (torch.Tensor): Tensor of shape [num_views, embedding_dim]
            
        Returns:
            dict: Dictionary containing uncertainty metrics
        """
        from car_quality_estimator.uncertrainty_estimation import predictive_entropy
        
        all_outputs = []
        
        # Run multiple MC forward passes with dropout enabled
        mc_outputs = []
        for _ in range(self.mc_samples):
            with torch.no_grad():
                output = self.uncertainty_model(embeddings)
                mc_outputs.append(output)
            
        # Average the outputs for this MC sample
        avg_output = torch.mean(torch.stack(mc_outputs), dim=0)
        avg_score = avg_output[:, 1]
        
        # Stack outputs from multiple MC samples
        all_outputs = torch.stack(mc_outputs, dim=0)
        
        # Compute uncertainty metrics
        entropy = predictive_entropy(all_outputs)
        
        return {
            "avg_score": avg_score,
            "entropy": entropy
        }
    
    def _calculate_combined_score(self, quality_score, uncertainty_entropy):
        """
        Calculate a combined score that incorporates both quality and certainty.
        
        Args:
            quality_score (float): Direct quality prediction
            uncertainty_metrics (dict): Dictionary of uncertainty metrics
            
        Returns:
            float: Combined score
        """
        # Weighting for uncertainty penalty
        alpha = 0.3

        # Combined score: quality_score * (1 - α * normalized_uncertainty)
        # Lower entropy means higher certainty, thus higher score
        max_entropy = np.log(2)  # Max entropy for binary classification
        normalized_uncertainty = uncertainty_entropy / max_entropy

        combined_score = torch.multiply(
            quality_score, 1 - alpha * normalized_uncertainty)
        return combined_score.cpu().numpy()

    def aggregate_metrics(self, metrics_dict):
        avg_quality = np.mean([metrics_dict["quality_score"]])
        avg_entropy = np.mean([metrics_dict["entropy"]])
        avg_combined = np.mean([metrics_dict["combined_score"]])

        # Calculate distribution statistics
        quality_std = np.std(metrics_dict["quality_score"])

        gen_metrics = {
            "avg_quality_score": avg_quality,
            "avg_entropy": avg_entropy,
            "avg_combined_score": avg_combined,
            "quality_std": quality_std,
            "num_samples": len(metrics_dict["quality_score"])
        }
        return gen_metrics
    
    def compute_scores_no_reference(self, generated_data):
        """
        Compute quality scores for generated models without reference data.
        
        Args:
            generated_data: List of individual images that will be randomly sampled to create batches (is_batched=False)

        Returns:
            dict: Dictionary containing quality scores and uncertainty metrics
        """
        # Generate embeddings for the generated data
        generated_embeddings = self.generate_embeddings(generated_data)
        
        # Compute quality scores using the pre-computed embeddings
        generated_scores = self.compute_quality_score_from_embeddings(generated_embeddings)
        gen_metrics = self.aggregate_metrics(generated_scores)

        return gen_metrics
        
    
    def compare_with_reference(self, generated_data, reference_data, compute_kid=True):
        """
        Compare generated models with a reference set (e.g., MeshFleet high-quality cars).
        
        Args:
            generated_data: List of individual images that will be randomly sampled to create batches (is_batched=False)
            reference_data: Same format as generated_data
            compute_kid (bool): Whether to compute KID metrics (can be slow for large batches)
            is_batched (bool): Whether the input data is already batched into groups of views
            num_batches (int, optional): Number of batches to create if not batched. If None, creates as many as possible.
            seed (int): Random seed for batch creation to ensure the same sampling for both sets
            
        Returns:
            dict: Comparison metrics
        """

        # Evaluate both sets and get embeddings along with metrics
        generated_embeddings = self.generate_embeddings(generated_data)
        reference_embeddings = self.generate_embeddings(reference_data)

        # Compute quality scores using the pre-computed embeddings
        generated_scores = self.compute_quality_score_from_embeddings(generated_embeddings)
        gen_metrics = self.aggregate_metrics(generated_scores)

        reference_scores = self.compute_quality_score_from_embeddings(reference_embeddings)
        ref_metrics = self.aggregate_metrics(reference_scores)

        # Calculate quality gap
        quality_gap = ref_metrics["avg_quality_score"] - gen_metrics["avg_quality_score"]

        # Compute improved distribution metrics
        distribution_metrics = compute_distribution_metrics(
            generated_scores["quality_score"], reference_scores["quality_score"])
        
        results = {
            "generated_metrics": gen_metrics,
            "reference_metrics": ref_metrics,
            "quality_gap": quality_gap,
            "score_distribution_metrics": distribution_metrics
        }

        # Compute KID metrics if requested (reuse the already computed embeddings)
        if compute_kid:
            kid_metrics = compute_kid_from_embeddings(generated_embeddings, reference_embeddings)
            results["kid_metrics"] = kid_metrics

        return results

# def load_car_quality_score(model_dir="../models", device=None, use_combined_embedding_model=False, batch_size=32):
#     """
#     Factory function to load a CarQualityScore instance with the appropriate configuration.
    
#     Args:
#         model_dir (str): Directory containing model files
#         device (str, optional): Device to run on. Defaults to None (auto-detect).
#         use_combined_embedding_model (bool, optional): Whether to use the combined embedding model. Defaults to False.
#         batch_size (int, optional): Batch size for embedding generation. Defaults to 32.
    
#     Returns:
#         CarQualityScore: Configured instance
#     """
#     if device is None:
#         device = "cuda" if torch.cuda.is_available() else "cpu"
#         # check for mps
#         if torch.backends.mps.is_available():
#             device = "mps"
    
#     if use_combined_embedding_model:
#         config_path = os.path.join(model_dir, "car_quality_model_combined_transformer_individual.json")
#         weights_path = os.path.join(model_dir, "car_quality_model_combined_transformer_individual.pt")
#         # PCA model path is the same for both types
#         pca_model_path = os.path.join(model_dir, "pca_model_DINOv2.pkl")
#         if not os.path.exists(pca_model_path):
#             raise FileNotFoundError(f"Could not find PCA model at {pca_model_path}")
#     else:
#         config_path = os.path.join(model_dir, "car_quality_model_siglip_transformer_individual.json")
#         weights_path = os.path.join(model_dir, "car_quality_model_siglip_transformer_individual.pt")
#         pca_model_path = None
#     model_type = "transformer"  # Default to transformer architecture
    
#     # Create and return the CarQualityScore instance
#     return CarQualityScore(
#         config_path=config_path,
#         weights_path=weights_path,
#         pca_model_path=pca_model_path,
#         model_type=model_type,
#         use_combined_embedding_model=use_combined_embedding_model,
#         device=device,
#         batch_size=batch_size
#     )


def load_car_quality_score(model_dir=None, device=None, use_combined_embedding_model=False, batch_size=32):
    """
    Factory function to load a CarQualityScore instance with the appropriate configuration.
    
    Args:
        model_dir (str): Directory containing model files. If models aren't found, will look in package directory.
        device (str, optional): Device to run on. Defaults to None (auto-detect).
        use_combined_embedding_model (bool, optional): Whether to use the combined embedding model. Defaults to False.
        batch_size (int, optional): Batch size for embedding generation. Defaults to 32.
    
    Returns:
        CarQualityScore: Configured instance
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        # check for mps
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = "mps"



    # If models don't exist in the provided directory, try to find them in the package
    if model_dir is None:
        try:
            # Get the package installation path
            # package_path = pkg_res.resource_filename('car_quality_estimator.car_quality_metric', '')
            file_path = os.path.dirname(os.path.abspath(__file__))
            # But models are in the car_quality_estimator submodule
            default_model_dir = os.path.join(file_path, 'models')
            if not os.path.exists(default_model_dir):
                from scripts.download_models import download_models
                download_models(default_model_dir)

            if use_combined_embedding_model:
                config_path = os.path.join(
                    default_model_dir, "car_quality_model_combined_transformer_individual.json")
                weights_path = os.path.join(
                    default_model_dir, "car_quality_model_combined_transformer_individual.pt")
                pca_model_path = os.path.join(
                    default_model_dir, "pca_model_DINOv2.pkl")

                if not (os.path.exists(config_path) and os.path.exists(weights_path) and os.path.exists(pca_model_path)):
                    raise FileNotFoundError(
                        f"Could not find model files in {model_dir} or {default_model_dir}")
            else:
                config_path = os.path.join(
                    default_model_dir, "car_quality_model_siglip_transformer_individual.json")
                weights_path = os.path.join(
                    default_model_dir, "car_quality_model_siglip_transformer_individual.pt")

                if not (os.path.exists(config_path) and os.path.exists(weights_path)):
                    raise FileNotFoundError(
                        f"Could not find model files in {model_dir} or {default_model_dir}")

            print(
                f"Using models from package installation directory: {default_model_dir}")
        except Exception as e:
            print(f"Error finding models in package directory: {str(e)}")
            if use_combined_embedding_model:
                raise FileNotFoundError(
                    f"Could not find model files at {config_path}, {weights_path}, or {pca_model_path}")
            else:
                raise FileNotFoundError(
                    f"Could not find model files at {config_path} or {weights_path}")
    else:
        if use_combined_embedding_model:
            config_path = os.path.join(
                model_dir, "car_quality_model_combined_transformer_individual.json")
            weights_path = os.path.join(
                model_dir, "car_quality_model_combined_transformer_individual.pt")
            pca_model_path = os.path.join(model_dir, "pca_model_DINOv2.pkl")
            if not (os.path.exists(config_path) and os.path.exists(weights_path) and os.path.exists(pca_model_path)):
                raise FileNotFoundError(
                    f"Could not find model files at {model_dir}")
        else:
            config_path = os.path.join(
                model_dir, "car_quality_model_siglip_transformer_individual.json")
            weights_path = os.path.join(
                model_dir, "car_quality_model_siglip_transformer_individual.pt")
            if not (os.path.exists(config_path) and os.path.exists(weights_path)):
                raise FileNotFoundError(
                    f"Could not find model files at {model_dir}")
    
    model_type = "transformer"  # Default to transformer architecture

    # Create and return the CarQualityScore instance
    return CarQualityScore(
        config_path=config_path,
        weights_path=weights_path,
        pca_model_path=pca_model_path,
        model_type=model_type,
        use_combined_embedding_model=use_combined_embedding_model,
        device=device,
        batch_size=batch_size
    )
