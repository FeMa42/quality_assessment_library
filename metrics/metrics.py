import os 
import glob
import PIL.Image
from PIL import Image
import gc
from typing import List, Optional, Union
import numpy as np
import torch
import torch.utils.checkpoint
import torch.nn.functional as F
from torchvision.transforms import ToTensor
from rembg import remove
import cv2
import contextlib
from metrics.helpers import preprocess_image, load_images_from_dir_new, compute_bounding_box, compare_bounding_boxes, compare_px_area, compute_outline_normals, compare_outline_normals, compare_summed_outline_normals
from car_quality_estimator.car_quality_metric import load_car_quality_score
from functools import partial

#######################################
#### Definition of Metrics classes ####
#######################################

class Metrics:
    """
    A class to compute various image quality metrics between input and target images.
    Includes framewise metrics (aggregated per image) as well as distribution metrics that 
    are computed over all images. Distribution metrics are only computed if 
    compute_distribution_metrics=True.
    """
    def __init__(
        self,
        device: Optional[str] = "cuda",
        clip_model_path: str = "openai/clip-vit-large-patch14",
        clip_cache_dir: str = None,
        compute_distribution_metrics: bool = True,  
    ):
        from torchmetrics.functional.image import (
            peak_signal_noise_ratio,  # PSNR, higher
            learned_perceptual_image_patch_similarity,  # LPIPS, lower
            structural_similarity_index_measure,  # SSIM, higher
            spectral_distortion_index,  # D_lambda, lower
            error_relative_global_dimensionless_synthesis,  # ERGAS, lower
            relative_average_spectral_error,  # RASE, lower
            root_mean_squared_error_using_sliding_window,  # RMSE wind, lower
            spectral_angle_mapper,  # SAM, lower
            multiscale_structural_similarity_index_measure,  # MS SSIM, higher
            universal_image_quality_index,  # higher
            visual_information_fidelity,  # higher
            spatial_correlation_coefficient,  # higher
        )
        # Fréchet inception distance (FID)
        from torchmetrics.image.fid import FrechetInceptionDistance
        # Inception Score (IS)
        from torchmetrics.image.inception import InceptionScore
        # Kernel Inception Distance (KID)
        from torchmetrics.image.kid import KernelInceptionDistance
        from transformers import CLIPImageProcessor, CLIPModel, CLIPTokenizer

        self.device = device
        self.compute_distribution_metrics = compute_distribution_metrics

        if clip_cache_dir is None:
            clip_cache_dir = os.path.join(os.path.dirname(__file__), "models", "clip-vit-large-patch14")
        else:
            clip_cache_dir = os.path.join(clip_cache_dir, "clip-vit-large-patch14")

        self.clip_model = CLIPModel.from_pretrained(clip_model_path, cache_dir=clip_cache_dir).to(device)
        self.clip_preprocess = CLIPImageProcessor.from_pretrained(clip_model_path, cache_dir=clip_cache_dir)

        def _CLIP_score(input, target, device=device):
            # If input has 4 channels (RGBA), drop the alpha channel.
            if input.shape[1] == 4:
                input = input[:, :3, :, :]
                target = target[:, :3, :, :]
            with torch.no_grad():
                _input = self.clip_preprocess(
                    input * 0.5 + 0.5, do_rescale=False, return_tensors="pt"
                )["pixel_values"]
                _target = self.clip_preprocess(
                    target * 0.5 + 0.5, do_rescale=False, return_tensors="pt"
                )["pixel_values"]
                emb_input = self.clip_model.get_image_features(_input.to(device))
                emb_target = self.clip_model.get_image_features(_target.to(device))
                cos_sim = torch.nn.functional.cosine_similarity(emb_input, emb_target)
                if len(cos_sim) > 1:
                    cos_sim = cos_sim.mean()
                del emb_input, emb_target, _input, _target
                torch.cuda.empty_cache()
                return cos_sim.item()

        def spectral_mse(input, target):
            orig_dtype = input.dtype
            fft1 = torch.fft.fft2(input.to(torch.float32))
            fft2 = torch.fft.fft2(target.to(torch.float32))
            return ((fft1.abs() - fft2.abs()) ** 2).mean().to(orig_dtype)

        def _image_mean_squared_error(input, target):
            orig_dtype = input.dtype
            return F.mse_loss(input.to(torch.float32), target.to(torch.float32)).to(orig_dtype)

        # Initialize the distribution metrics only if requested.
        if self.compute_distribution_metrics:
            self.fid_metric = FrechetInceptionDistance(feature=64, normalize=True).to(device)
            self.is_metric = InceptionScore(normalize=True).to(device)
            self.kid_metric = KernelInceptionDistance(subset_size=21, normalize=True).to(device)
            self.distribution_metrics = {
                "FID": self._FID_score, 
                "IS": self._IS_score, 
                "KID": self._KID_score
            }
        else:
            self.distribution_metrics = {}

        self.metrics = {
            "MSE": _image_mean_squared_error,
            "CLIP-S": _CLIP_score,
            "Spectral_MSE": spectral_mse,
            "D_lambda": spectral_distortion_index,
            "ERGAS": error_relative_global_dimensionless_synthesis,
            "PSNR": peak_signal_noise_ratio,
            "RASE": relative_average_spectral_error,
            "RMSE_wind": root_mean_squared_error_using_sliding_window,
            "SAM": spectral_angle_mapper,
            "MS-SSIM": multiscale_structural_similarity_index_measure,
            "SSIM": structural_similarity_index_measure,
            "UQI": universal_image_quality_index,
            "VIF": visual_information_fidelity,
            "LPIPS": learned_perceptual_image_patch_similarity,
            "SCC": spatial_correlation_coefficient
        }
        self.result = torch.zeros(len(self.metrics), device=device)
        # Only initialize distribution results if we compute them.
        self.result_distribution = (
            torch.zeros(len(self.distribution_metrics), device=device)
            if self.compute_distribution_metrics
            else None
        )
        self.total = 0

    # Distribution metric implementations that get used in the distribution_metrics dict.
    def _FID_score(self, input=None, target=None, device=None):
        if input is None and target is None:
            fid_score = self.fid_metric.compute()
            return fid_score
        orig_dtype = input.dtype
        self.fid_metric.update(target.to(self.device), real=True)
        self.fid_metric.update(input.to(self.device), real=False)
        fid_score = self.fid_metric.compute()
        return fid_score.to(orig_dtype)

    def _IS_score(self, input=None, target=None, device=None):
        if input is None:
            is_score, _ = self.is_metric.compute()
            return is_score
        orig_dtype = input.dtype
        self.is_metric.update(input.to(self.device))
        is_score, _ = self.is_metric.compute()
        return is_score.to(orig_dtype)

    def _KID_score(self, input=None, target=None, device=None):
        if input is None:
            kid_score, _ = self.kid_metric.compute()
            return kid_score
        orig_dtype = input.dtype
        self.kid_metric.update(target.to(self.device), real=True)
        self.kid_metric.update(input.to(self.device), real=False)
        kid_score, _ = self.kid_metric.compute()
        return kid_score.to(orig_dtype)

    def reset_fid(self):
        if self.compute_distribution_metrics:
            self.fid_metric.reset()

    def reset_kid(self):
        if self.compute_distribution_metrics:
            self.kid_metric.reset()

    def reset_is(self):
        if self.compute_distribution_metrics:
            from torchmetrics.image.inception import InceptionScore
            self.is_metric = InceptionScore(normalize=True).to(self.device)

    def compute_image(self, input, target):
        """
        Compute image metrics framewise and aggregate over image.
        Also update the distribution metrics if enabled.
        """
        assert input.shape == target.shape
        num_frames = input.shape[0]
        input = input.to(self.device) * 2 - 1
        target = target.to(self.device) * 2 - 1

        imagewise_output = torch.zeros(len(self.metrics), device=self.device)
        for i in range(num_frames):
            framewise_output_dict = {}
            framewise_output_dict["frame"] = i
            for name, metric in self.metrics.items():
                framewise_output_dict[name] = metric(
                    input[i].unsqueeze(0), target[i].unsqueeze(0)
                )
            imagewise_output += torch.tensor(
                [v for k, v in framewise_output_dict.items() if k != "frame"], device=self.device
            )
        if self.compute_distribution_metrics:
            distrib_dict = {}
            for name, metric in self.distribution_metrics.items():
                distrib_dict[name] = metric(input, target)
            result_distribution = torch.tensor(
                [v for k, v in distrib_dict.items()], device=self.device
            )
        else:
            result_distribution = None
        self.result += imagewise_output / num_frames
        self.total += 1
        metrics_out = dict(zip(self.metrics.keys(), self.result / self.total))
        if self.compute_distribution_metrics and result_distribution is not None:
            self.result_distribution += result_distribution
            metrics_out.update(
                dict(zip(self.distribution_metrics.keys(), self.result_distribution / self.total))
            )
        return metrics_out

    def reset_distribution_metrics(self):
        if self.compute_distribution_metrics:
            self.reset_fid()
            self.reset_kid()
            self.reset_is()

    def get_total_metrics(self):
        if self.compute_distribution_metrics:
            distrib_dict = {}
            for name, metric in self.distribution_metrics.items():
                distrib_dict[name] = metric()
            result_distribution = torch.tensor(
                [v for k, v in distrib_dict.items()], device=self.device
            )
            self.reset_distribution_metrics()
            self.result_distribution += result_distribution
            metrics_out = dict(zip(self.metrics.keys(), self.result / self.total))
            metrics_out.update(
                dict(zip(self.distribution_metrics.keys(), self.result_distribution / self.total))
            )
        else:
            metrics_out = dict(zip(self.metrics.keys(), self.result / self.total))
        return metrics_out
    
class GeometryMetrics:
    """
    A class to compute and aggregate geometric metrics between pairs of preprocessed images.
    
    Metrics computed:
      - Object area differences (relative pixel area difference using compare_px_area)
      - Bounding box differences (absolute and percentual differences using compare_bounding_boxes)
      - Outline differences (average angle difference using compare_outline_normals)
      
    Usage:
      gm = GeometryMetrics(num_points=100)
      gm.compute_image_pair(image1, image2)
      # ... process more pairs ...
      avg_metrics = gm.get_average_metrics()
      gm.reset()  # to start over
    """
    def __init__(self, num_points=100):
        self.num_points = num_points
        self.reset()
        
    def reset(self):
        """Reset accumulated metrics."""
        self.area_diffs = []
        self.bbox_metrics = []
        self.outline_angle_diffs = []
        self.summed_outline_angle_diffs = []
        self.outline_angle_diffs_squared = []            
        self.summed_outline_angle_diffs_squared = [] 
        self.total_pairs = 0

    def compute_image_pair(self, image1: Image.Image, image2: Image.Image):
        """
        Compute the geometric metrics for a pair of images.
        Assumes images have been preprocessed (e.g. with preprocess_image_rgba) into RGBA.
        """
        area_result = compare_px_area(image1, image2)
        self.area_diffs.append(area_result["relative_difference"])
        
        bbox_result = compare_bounding_boxes(image1, image2)
        self.bbox_metrics.append(bbox_result["aspect_percent"])
        
        outline_result = compare_outline_normals(image1, image2, num_points=self.num_points)
        self.outline_angle_diffs.append(outline_result["average_angle_difference_degrees"])
        
        summed_outline_result = compare_summed_outline_normals(image1, image2, num_points=self.num_points)
        self.summed_outline_angle_diffs.append(summed_outline_result["average_summed_angle_difference_degrees"])

        self.outline_angle_diffs_squared.append(outline_result["average_angle_difference_squared_degrees"])
        
        self.summed_outline_angle_diffs_squared.append(summed_outline_result["average_summed_angle_difference_squared_degrees"])

        self.total_pairs += 1
        
        result = {}
        result.update(area_result)
        result.update(bbox_result)
        result.update(outline_result)
        result.update(summed_outline_result)

        return result

    def get_average_metrics(self):
        """
        Returns a dictionary with the average metrics computed over all processed image pairs.
        """
        avg_area = np.mean(self.area_diffs) if self.area_diffs else None
        avg_outline = np.mean(self.outline_angle_diffs) if self.outline_angle_diffs else None
        avg_summed_outline = np.mean(self.summed_outline_angle_diffs) if self.summed_outline_angle_diffs else None
        avg_outline_squared = np.mean(self.outline_angle_diffs_squared) if self.outline_angle_diffs_squared else None
        avg_summed_outline_squared = np.mean(self.summed_outline_angle_diffs_squared) if self.summed_outline_angle_diffs_squared else None
        avg_bbox = np.mean(self.bbox_metrics) if self.bbox_metrics else None
                
        return {
            "Rel_BB_Aspect_Ratio_Diff": avg_bbox,
            "Rel_Pixel_Area_Diff": avg_area,
            "Outline_Normals_Angle_Diff": avg_outline,
            "Squared_Outline_Normals_Angle_Diff": avg_outline_squared,
            "Summed_Outline_Normals_Angle_Diff": avg_summed_outline,
            "Squared_Summed_Outline_Normals_Angle_Diff": avg_summed_outline_squared,
            "Image_Pairs": self.total_pairs
        }


class CarQualityMetrics:
    """
    Wraps the no‑reference CarQualityScore so we can
    get its five aggregated metrics on a folder of PNGs.
    """
    def __init__(
        self,
        use_combined_embedding_model: bool = True,
        device: str = None,
        batch_size: int = 32
    ):
        # Forward device and batch_size into the factory
        self.metric = load_car_quality_score(
            device=device,
            use_combined_embedding_model=use_combined_embedding_model,
            batch_size=batch_size
        )

    def compute_folder_metrics(self, folder: str) -> dict:
        """
        - Finds all .png in `folder`
        - Runs compute_scores_no_reference (returns a dict of 5 metrics)
        - Coerces them to Python floats/ints
        """
        paths = sorted(glob.glob(os.path.join(folder, "*.png")))
        if not paths:
            raise ValueError(f"No PNG images found in {folder!r}")
        imgs = [Image.open(p).convert("RGB") for p in paths]

        raw = self.metric.compute_scores_no_reference(imgs)

        cleaned = {}
        for k, v in raw.items():
            # num_samples is integer, everything else float-like
            if k == "num_samples":
                cleaned[k] = int(v)
            else:
                # in case v is a numpy scalar or array
                cleaned[k] = float(np.array(v))
        return cleaned



class ImageBasedPromptEvaluator:
    """
    A class to evaluate image-based prompts using CLIP and ImageReward.
    Uses ImageReward from: https://github.com/THUDM/ImageReward
    Args:
        model_name_or_path (str): Path to the CLIP model or model name from Hugging Face Hub.
    """
    def __init__(self, model_name_or_path="openai/clip-vit-base-patch16"):
        """
        Initialize the evaluator with a CLIP model and ImageReward model.
        Args:
            model_name_or_path (str): Path to the CLIP model or model name from Hugging Face Hub.
        """
        from torchmetrics.functional.multimodal import clip_score
        import ImageReward as RM
        self.model_name_or_path = model_name_or_path
        self.clip_score_fn = partial(clip_score, model_name_or_path=self.model_name_or_path)
        self.reward_model = RM.load("ImageReward-v1.0")

    def calculate_clip_score_PIL(self, images, prompts):
        """
        Calculate the CLIP score for a batch of images and a batch of prompts.
        Args:
            images (list): List of PIL.Image images to evaluate.  
            prompts (list): List of prompts corresponding to the images. Has to be the same length as images.
        Returns:
            float: The average CLIP score for the batch.
        """
        np_images = np.array(images, dtype=np.uint8)
        image_tensor = torch.from_numpy(np_images).permute(0, 3, 1, 2)
        return self.calculate_clip_score(image_tensor, prompts)
    
    def calculate_clip_score(self, image_tensor, prompts):
        """
        Calculate the CLIP score for a batch of images and a batch of prompts.
        Args:
            image_tensor (torch.Tensor): Tensor of images to evaluate.
            prompts (list): List of prompts corresponding to the images.
        Returns:
            float: The average CLIP score for the batch.
        """
        with torch.no_grad():
            prompt_clip_score = self.clip_score_fn(image_tensor, prompts).detach()
        return round(float(prompt_clip_score), 4)

    def calculate_reward(self, images, prompt):
        """
        Calculate the reward for a batch of images and single prompt using the ImageReward model.
        Args:
            images (list, PIL.Image): List of PIL.Image images to evaluate or a single image.
            prompts (str): Single prompt corresponding to the images. 
        Returns:
            torch.Tensor: The average reward for the batch.
        """
        with torch.no_grad():
            rewards = self.reward_model.score(prompt, images)
            # Convert to numpy array
            rewards = np.array(rewards)
            # mean over the batch
            rewards = np.mean(rewards)
        return round(float(rewards), 4)

    def evaluate(self, images, prompts):
        """
        Evaluate a batch of images using CLIP and ImageReward. 
        It expects a batch of images and a batch of prompts with the same length.
        Args:
            images (list): List of PIL.Image images to evaluate. 
            prompts (list): List of prompts corresponding to the images. Has to be the same length as images.
        Returns:
            dict: Dictionary containing the CLIP score and ImageReward score.
        """
        clip_score = self.calculate_clip_score_PIL(images, prompts)
        rewards = []
        for prompt in prompts:
            reward = self.calculate_reward(images, prompt)
            rewards.append(reward)
        reward = np.mean(rewards)
        reward = round(float(reward), 4)
        return {
            "clip_score": clip_score,
            "image_reward": reward
        }

    def evaluate_one_prompt_n_images(self, images, prompt):
        """
        Evaluate a batch of images using CLIP and ImageReward. 
        It expects a batch of images and a single prompt since it is used for renders of a 3D model.
        Args:
            images (list): List of PIL.Image images to evaluate. 
            prompts (str): Single prompt corresponding to the images.
        Returns:
            dict: Dictionary containing the CLIP score and ImageReward score.
        """
        prompts = [prompt] * len(images)
        clip_score = self.calculate_clip_score_PIL(images, prompts)
        reward = self.calculate_reward(images, prompt)
        return {
            "clip_score": clip_score,
            "image_reward": reward
        }