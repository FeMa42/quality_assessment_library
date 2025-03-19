import gc
from typing import List, Optional, Union
import numpy as np
import torch
import torch.utils.checkpoint
import torch.nn.functional as F

class Metrics:
    """
    A class to compute various image quality metrics between input and target images.
    Metrics are computed framewise and logged locally via the provided csv logger.
    The metrics are aggregated per image and the total mean (over all made computations) can be obtained using `get_total_metrics`.
    """

    def __init__(
        self,
        device: Optional[str] = "cuda",
        clip_model_path: str = "openai/clip-vit-large-patch14",
        clip_cache_dir: str = "./models/clip-vit-large-patch14",
    ):
        from torchmetrics.functional.image import (
            peak_signal_noise_ratio,  # PSNR, higher
            learned_perceptual_image_patch_similarity,  # LPIPS, lower
            structural_similarity_index_measure,  # SSIM, higher
            spectral_distortion_index,  # D_lambda, lower
            error_relative_global_dimensionless_synthesis,  # ERGAS, lower
            relative_average_spectral_error,  # RASE, lower
            root_mean_squared_error_using_sliding_window,  # RMSE wind, lower
            spectral_angle_mapper,  # SAM, absolute value of the spectral angle, lower
            multiscale_structural_similarity_index_measure,  # MS SSIM, higher
            universal_image_quality_index,  # higher
            visual_information_fidelity,  # higher
            spatial_correlation_coefficient,  # higher
        )
        # Fréchet inception distance (FID) 
        from torchmetrics.image.fid import FrechetInceptionDistance
        # Inception Score (IS) which is used to access how realistic generated images are
        from torchmetrics.image.inception import InceptionScore
        # Kernel Inception Distance (KID) which is used to access the quality of generated images
        from torchmetrics.image.kid import KernelInceptionDistance
        from torchmetrics.functional.regression import mean_squared_error
        from transformers import CLIPImageProcessor, CLIPModel, CLIPTokenizer

        clip_model = CLIPModel.from_pretrained(
            clip_model_path, cache_dir=clip_cache_dir
        ).to(device)
        clip_preprocess = CLIPImageProcessor.from_pretrained(
            clip_model_path, cache_dir=clip_cache_dir
        )

        def _CLIP_score(input, target, device="cuda"):
            # if input.shape[0] > 1:
            #     return np.nan

            # Calculate the embeddings for the images using the CLIP model
            with torch.no_grad():
                _input = clip_preprocess(
                    input * 0.5 + 0.5, do_rescale=False, return_tensors="pt"
                )["pixel_values"]
                _target = clip_preprocess(
                    target * 0.5 + 0.5, do_rescale=False, return_tensors="pt"
                )["pixel_values"]

                emb_input = clip_model.get_image_features(_input.to(device))
                emb_target = clip_model.get_image_features(_target.to(device))

                # Calculate the cosine similarity between the embeddings
                cos_sim = torch.nn.functional.cosine_similarity(emb_input, emb_target)
                if len(cos_sim) > 1:
                    cos_sim = cos_sim.mean()

                # emb_input, emb_target, _input, _target = emb_input.cpu(), emb_target.cpu(), _input.cpu(), _target.cpu()
                del emb_input, emb_target, _input, _target
                gc.collect()
                torch.cuda.empty_cache()
                return cos_sim.item()

        def spectral_mse(input, target):
            orig_dtype = input.dtype

            fft1 = torch.fft.fft2(input.to(torch.float32))
            fft2 = torch.fft.fft2(target.to(torch.float32))
            return ((fft1.abs() - fft2.abs()) ** 2).mean().to(orig_dtype)

        self.fid_metric = FrechetInceptionDistance(feature=64, normalize=True).to(device)
        def _FID_score(input, target, device="cuda"):
            orig_dtype = input.dtype
            self.fid_metric.update(target.to(device), real=True)
            self.fid_metric.update(input.to(device), real=False)
            if self.fid_metric.real_features_num_samples > 1 and self.fid_metric.fake_features_num_samples > 1:
                fid_score = self.fid_metric.compute()
            else:
                fid_score = torch.tensor(float(0.0))
            return fid_score.to(orig_dtype)

        self.is_metric = InceptionScore(normalize=True).to(device)
        def _IS_score(input, target, device="cuda"):
            orig_dtype = input.dtype
            is_score = self.is_metric.update(input.to(device))
            is_score, _ = self.is_metric.compute()
            return is_score.to(orig_dtype)

        self.kid_metric = KernelInceptionDistance(subset_size=1, normalize=True).to(device)
        def _KID_score(input, target, device="cuda"):
            orig_dtype = input.dtype
            self.kid_metric.update(target.to(device), real=True)
            self.kid_metric.update(input.to(device), real=False)
            if len(self.kid_metric.real_features) > 1 and len(self.kid_metric.fake_features) > 1:
                kid_score, _ = self.kid_metric.compute()
            else:
                kid_score = torch.tensor(float(0.0))
            return kid_score.to(orig_dtype)

        self.metrics = {
            "MSE": mean_squared_error,
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
            "SCC": spatial_correlation_coefficient,
            "FID": _FID_score,
            "IS": _IS_score,
            "KID": _KID_score,
        }
        self.result = torch.zeros(len(self.metrics), device=device)
        self.total = 0
        self.device = device

    def reset_fid(self):
        self.fid_metric.reset()

    def reset_kid(self):
        self.kid_metric.reset()

    def reset_is(self):
        from torchmetrics.image.inception import InceptionScore
        self.is_metric = InceptionScore(normalize=True).to(self.device)

    def compute_image(self, input, target, csv_logger, log_dict={}):
        """
        compute image metrics framewise and aggregate over image

        Args:
            input (torch.Tensor): Input tensor of shape (num_frames, channels, height, width)
            target (torch.Tensor): Target tensor of shape (num_frames, channels, height, width)
            csv_logger (CSVLogger): Logger object to log metrics in a CSV file
            log_dict (dict, optional): Additional key-value pairs to log in the CSV file

        Returns:
            dict: A dictionary containing the average values of the metrics computed over all frames / for a single image
        """
        # compute metrics framewise
        assert input.shape == target.shape
        num_frames = input.shape[0]
        # normalize images
        input = input.to(self.device) * 2 - 1
        target = target.to(self.device) * 2 - 1

        imagewise_output = torch.zeros(len(self.metrics), device=self.device)
        for i in range(num_frames):
            framewise_output_dict = log_dict.copy()
            framewise_output_dict["frame"] = i
            # compute all metrics for the frame
            for name, metric in self.metrics.items():
                framewise_output_dict[name] = metric(
                    input[i].unsqueeze(0), target[i].unsqueeze(0)
                )
            # save values as one row in csv
            csv_logger.log_metrics(framewise_output_dict)
            csv_logger.save()
            # aggregate metrics over all frames
            imagewise_output += torch.tensor([v for k, v in framewise_output_dict.items() if k not in log_dict.keys() and k != "frame"], device=self.device)
            del framewise_output_dict
            gc.collect()
            torch.cuda.empty_cache()
        self.reset_fid()
        self.reset_kid()
        self.reset_is()
        self.result += imagewise_output / num_frames
        self.total += 1
        gc.collect()
        torch.cuda.empty_cache()
        return dict(zip(self.metrics.keys(), imagewise_output / self.total))

    def get_total_metrics(self):
        """
        Get the average values of the metrics computed over all images.
        Returns:
            dict: A dictionary containing the average values of the metrics computed over all images.
        """
        return dict(zip(self.metrics.keys(), self.result / self.total))
