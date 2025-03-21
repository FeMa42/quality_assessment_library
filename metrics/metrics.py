import os 
import glob
import PIL.Image
import gc
from typing import List, Optional, Union
import numpy as np
import torch
import torch.utils.checkpoint
import torch.nn.functional as F
from torchvision.transforms import ToTensor
from rembg import remove
import cv2

def preprocess_image(
    image, image_frame_ratio=None, resize_image=True, output_tensor=False
):
    """
    Preprocess an image for further processing.

    Args:
        image (PIL.Image.Image): The input image to be preprocessed.
        image_frame_ratio (float or None): The ratio of the object's size to the image frame size.
            If provided, the object will be resized to fit within a square of size `max(w, h) / image_frame_ratio`.
            If `None`, the object will be resized to the original frame ratio.
        resize_image (bool): Whether to resize the object to fit within a bounding box around the object or not.
        output_tensor (bool): Whether to return the preprocessed image as a PyTorch tensor or a PIL Image.

    Returns:
        PIL.Image.Image or torch.Tensor: The preprocessed image.
    """
    if image.mode == "RGBA":
        pass
    else:
        # remove bg
        image.thumbnail([768, 768], PIL.Image.Resampling.LANCZOS)
        image = remove(image.convert("RGBA"), alpha_matting=True)

    # resize object in frame
    if resize_image:
        image_arr = np.array(image)
        in_w, in_h = image_arr.shape[:2]
        ret, mask = cv2.threshold(
            np.array(image.split()[-1]), 0, 255, cv2.THRESH_BINARY
        )
        x, y, w, h = cv2.boundingRect(mask)
        max_size = max(w, h)
        side_len = (
            int(max_size / image_frame_ratio) if image_frame_ratio is not None else in_w
        )
        padded_image = np.zeros((side_len, side_len, 4), dtype=np.uint8)
        center = side_len // 2
        padded_image[
            center - h // 2 : center - h // 2 + h,
            center - w // 2 : center - w // 2 + w,
        ] = image_arr[y : y + h, x : x + w]
        # resize frame to 576x576
        image = PIL.Image.fromarray(padded_image).resize((576, 576), PIL.Image.LANCZOS)
    # white bg
    rgba_arr = np.array(image) / 255.0
    rgb = rgba_arr[..., :3] * rgba_arr[..., -1:] + (1 - rgba_arr[..., -1:])
    input_image = PIL.Image.fromarray((rgb * 255).astype(np.uint8))

    if output_tensor:
        return ToTensor()(input_image).float()
    return input_image

class Metrics:
    """
    A class to compute various image quality metrics between input and target images.
    Includes Metrics that are computed framewise and logged as well as distribution metrics that are computed over all images.
    The framewise metrics are aggregated per image and the total mean (over all made computations) can be obtained using `get_total_metrics`.
    The distribution metrics are calculated at each call to `get_total_metrics` and are reset after each call. 
    """

    def __init__(
        self,
        device: Optional[str] = "cuda",
        clip_model_path: str = "openai/clip-vit-large-patch14",
        clip_cache_dir: str = None,
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
        # from torchmetrics.functional.regression import mean_squared_error
        from transformers import CLIPImageProcessor, CLIPModel, CLIPTokenizer

        if clip_cache_dir is None:
            # "./metrics/models/clip-vit-large-patch14"
            clip_cache_dir = os.path.join(os.path.dirname(__file__), "models", "clip-vit-large-patch14")
        else:
            clip_cache_dir = os.path.join(clip_cache_dir, "clip-vit-large-patch14")

        clip_model = CLIPModel.from_pretrained(
            clip_model_path, cache_dir=clip_cache_dir
        ).to(device)
        clip_preprocess = CLIPImageProcessor.from_pretrained(
            clip_model_path, cache_dir=clip_cache_dir
        )

        def _CLIP_score(input, target, device=device):
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
                cos_sim = torch.nn.functional.cosine_similarity(
                    emb_input, emb_target)
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

        def _image_mean_squared_error(input, target):
            orig_dtype = input.dtype
            return F.mse_loss(input.to(torch.float32), target.to(torch.float32)).to(orig_dtype)

        self.fid_metric = FrechetInceptionDistance(
            feature=64, normalize=True).to(device)

        def _FID_score(input=None, target=None, device=device):
            """
            Compute the Fréchet Inception Distance (FID) between the input and target images.
            If input and target are None, the FID score is computed and returned.
            If input and target are not None, they are added to the FID metric before computing the score.
            """
            if input is None and target is None:
                fid_score = self.fid_metric.compute()
                return fid_score
            orig_dtype = input.dtype
            self.fid_metric.update(target.to(device), real=True)
            self.fid_metric.update(input.to(device), real=False)
            fid_score = self.fid_metric.compute()
            return fid_score.to(orig_dtype)

        self.is_metric = InceptionScore(normalize=True).to(device)

        def _IS_score(input=None, target=None, device=device):
            """
            Compute the Inception Score (IS) between the input and target images.
            If input and target are None, the IS score is computed and returned.
            If input and target are not None, they are added to the IS metric before computing the score.
            """
            if input is None:
                is_score, _ = self.is_metric.compute()
                return is_score
            orig_dtype = input.dtype
            is_score = self.is_metric.update(input.to(device))
            is_score, _ = self.is_metric.compute()
            return is_score.to(orig_dtype)

        self.kid_metric = KernelInceptionDistance(
            subset_size=21, normalize=True).to(device)

        def _KID_score(input=None, target=None, device=device):
            """
            Compute the Kernel Inception Distance (KID) between the input and target images.
            If input and target are None, the KID score is computed and returned.
            If input and target are not None, they are added to the KID metric before computing the score.
            """
            if input is None:
                kid_score, _ = self.kid_metric.compute()
                return kid_score
            orig_dtype = input.dtype
            self.kid_metric.update(target.to(device), real=True)
            self.kid_metric.update(input.to(device), real=False)
            kid_score, _ = self.kid_metric.compute()
            return kid_score.to(orig_dtype)

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
        self.distribution_metrics = {
            "FID": _FID_score,
            "IS": _IS_score,
            "KID": _KID_score
        }
        self.result = torch.zeros(len(self.metrics), device=device)
        self.result_distribution = torch.zeros(len(self.distribution_metrics), device=device)
        self.total = 0
        self.device = device

    def reset_fid(self):
        self.fid_metric.reset()

    def reset_kid(self):
        self.kid_metric.reset()

    def reset_is(self):
        from torchmetrics.image.inception import InceptionScore
        self.is_metric = InceptionScore(normalize=True).to(self.device)

    def compute_image(self, input, target):
        """
        Compute image metrics framewise and aggregate over image. Adds the images to the distribution metrics.

        Args:
            input (torch.Tensor): Input tensor of shape (num_frames, channels, height, width)
            target (torch.Tensor): Target tensor of shape (num_frames, channels, height, width)

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
            framewise_output_dict = {}
            framewise_output_dict["frame"] = i
            # compute all metrics for the frame
            for name, metric in self.metrics.items():
                framewise_output_dict[name] = metric(
                    input[i].unsqueeze(0), target[i].unsqueeze(0)
                )
            # aggregate metrics over all frames
            imagewise_output += torch.tensor([v for k, v in framewise_output_dict.items(
            ) if k != "frame"], device=self.device)
            del framewise_output_dict
            gc.collect()
            torch.cuda.empty_cache()
        distribtuion_output = torch.zeros(len(self.distribution_metrics), device=self.device)
        distribtuion_dict = {}
        for name, metric in self.distribution_metrics.items():
            distribtuion_dict[name] = metric(input, target)
        result_distribution = torch.tensor([v for k, v in distribtuion_dict.items()], device=self.device)
        del distribtuion_dict
        gc.collect()
        torch.cuda.empty_cache()
        self.result += imagewise_output / num_frames
        self.total += 1
        gc.collect()
        torch.cuda.empty_cache()
        metrics = dict(zip(self.metrics.keys(), imagewise_output / self.total))
        metrics.update(dict(zip(self.distribution_metrics.keys(), result_distribution / self.total)))
        return metrics

    def reset_distribution_metrics(self):
        """
        Reset the distribution metrics.
        """
        self.reset_fid()
        self.reset_kid()
        self.reset_is()
        gc.collect()
        torch.cuda.empty_cache()

    def get_total_metrics(self):
        """
        Get the average values of the metrics computed over all images.
        Returns:
            dict: A dictionary containing the average values of the metrics computed over all images.
        """
        distribtuion_dict = {}
        for name, metric in self.distribution_metrics.items():
            distribtuion_dict[name] = metric()
        result_distribution = torch.tensor([v for k, v in distribtuion_dict.items()], device=self.device)
        del distribtuion_dict
        self.reset_distribution_metrics()
        self.result_distribution += result_distribution 
        metrics = dict(zip(self.metrics.keys(), self.result / self.total))
        metrics.update(dict(zip(self.distribution_metrics.keys(), self.result_distribution / self.total)))
        return metrics