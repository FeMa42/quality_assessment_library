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

#############################
#### Image Preprocessing ####
#############################
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

def preprocess_image_rgba(
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
    image.thumbnail([768, 768], PIL.Image.Resampling.LANCZOS)

    # Surpress weird scipy warning
    @contextlib.contextmanager
    def suppress_stdout_stderr():
        with open(os.devnull, 'w') as devnull:
            with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
                yield
    with suppress_stdout_stderr():
        image = remove(image, alpha_matting=True)

    # resize object in frame
    if resize_image:
        image_arr = np.array(image)
        in_w, in_h = image_arr.shape[:2]
        
        alpha_channel = image_arr[..., 3]

        _, mask = cv2.threshold(alpha_channel, 0, 255, cv2.THRESH_BINARY)
        x, y, w, h = cv2.boundingRect(mask)
        max_size = max(w, h)
        side_len = (
            int(max_size / image_frame_ratio) if image_frame_ratio is not None else in_w
        )
        padded_image = np.zeros((side_len, side_len, 4), dtype=np.uint8)
        center = side_len // 2
        offset_y = center - h // 2
        offset_x = center - w // 2
        padded_image[offset_y : offset_y + h, offset_x : offset_x + w] = image_arr[y : y + h, x : x + w]

        # resize frame to 576x576
        image = PIL.Image.fromarray(padded_image, mode="RGBA").resize((576, 576), PIL.Image.LANCZOS)
    
    """    
    # white bg
    rgba_arr = np.array(image) / 255.0
    rgb = rgba_arr[..., :3] * rgba_arr[..., -1:] + (1 - rgba_arr[..., -1:])
    input_image = PIL.Image.fromarray((rgb * 255).astype(np.uint8))
    """
    if output_tensor:
        return ToTensor()(image).float()
    return image

def load_images_from_dir(image_dir: str, device):
    """
    Load all images from a directory and return them as a tensor with shape (num_frames, channels, height, width)
    """
    all_images = glob.glob(os.path.join(image_dir, "*.png"))
    all_images = [preprocess_image(Image.open(image)) for image in all_images]
    torch_image_tensor = torch.tensor(np.array(all_images), dtype=torch.float32)
    # (num_frames, channels, height, width)
    torch_image_tensor = torch_image_tensor.permute(0, 3, 1, 2)
    target = torch_image_tensor.to(device)
    return target

def load_images_from_dir_new(image_dir: str, device, preprocess_func):
    """
    Load all images from a directory, preprocess them with preprocess_func,
    and return a tensor of shape (num_frames, channels, height, width).
    """
    files = sorted(glob.glob(os.path.join(image_dir, "*.png")))
    images = [preprocess_func(Image.open(f)) for f in files]
    arr = np.array([np.array(img) for img in images])
    tensor = np.transpose(arr, (0, 3, 1, 2))
    tensor = torch.tensor(tensor, dtype=torch.float32).to(device)
    return tensor

def process_folder_new(original_folder, generated_folder, preprocess_func, metric_class, device=None, **metric_kwargs):
    """
    Process a pair of folders using a common interface.
    
    Args:
      original_folder (str): Folder with original images.
      generated_folder (str): Folder with generated images.
      preprocess_func (callable): Preprocessing function to apply to each image.
      metric_class (type): Either Metrics or GeometryMetrics.
      device: For semantic metrics, the torch device (e.g. "cuda" or "cpu").
      metric_kwargs: Additional keyword arguments to initialize the metric class.
      
    Returns:
      dict: Averaged metrics.
    """
    # For semantic metrics (Metrics class), assume it works on full image tensors.
    if metric_class.__name__ == "Metrics":
        # For semantic metrics, we load all images into tensors.
        semantic_metric = metric_class(**metric_kwargs, device=device)
        input_tensor = load_images_from_dir_new(original_folder, device, preprocess_func)
        target_tensor = load_images_from_dir_new(generated_folder, device, preprocess_func)
        semantic_metric.compute_image(input_tensor, target_tensor)
        return semantic_metric.get_total_metrics()
    
    # For geometry metrics (GeometryMetrics class), process pairwise.
    elif metric_class.__name__ == "GeometryMetrics":
        geo_metric = metric_class(**metric_kwargs)
        orig_files = sorted(glob.glob(os.path.join(original_folder, "*.png")))
        gen_files = sorted(glob.glob(os.path.join(generated_folder, "*.png")))
        # Build a lookup for generated images by filename.
        gen_dict = {os.path.basename(f): f for f in gen_files}
        for orig_path in orig_files:
            fname = os.path.basename(orig_path)
            if fname not in gen_dict:
                print(f"Warning: No generated image for {fname}")
                continue
            orig_img = Image.open(orig_path)
            gen_img = Image.open(gen_dict[fname])
            orig_img = preprocess_func(orig_img)
            gen_img = preprocess_func(gen_img)
            geo_metric.compute_image_pair(orig_img, gen_img)
        return geo_metric.get_average_metrics()
    
    else:
        raise ValueError("Unsupported metric_class. Must be either Metrics or GeometryMetrics.")
    
def process_folder(original_folder, generated_folder, num_points=100, preprocess_func=None):
    """
    Loop over image pairs (matched by filename) from two folders,
    preprocess them with preprocess_func (e.g. preprocess_image_rgba), 
    compute geometric metrics for each pair, and return averaged results.
    
    Args:
        original_folder (str): Folder containing the original images.
        generated_folder (str): Folder containing the generated images.
        num_points (int): Number of points to use for outline comparison.
        preprocess_func (callable): Function to preprocess each image. Should return an RGBA image.
    
    Returns:
        dict: Averaged geometric metrics.
    """
    orig_files = sorted(glob.glob(os.path.join(original_folder, "*.png")))
    gen_files = sorted(glob.glob(os.path.join(generated_folder, "*.png")))
    
    gen_dict = {os.path.basename(f): f for f in gen_files}
    
    gm = GeometryMetrics(num_points=num_points)
    
    for orig_path in orig_files:
        file_name = os.path.basename(orig_path)
        gen_path = gen_dict.get(file_name)
        if not gen_path:
            print(f"Warning: No generated image for {file_name}")
            continue
        
        orig_img = Image.open(orig_path)
        gen_img = Image.open(gen_path)
        
        if preprocess_func:
            orig_img = preprocess_func(orig_img)
            gen_img = preprocess_func(gen_img)
        
        gm.compute_image_pair(orig_img, gen_img)
    
    return gm.get_average_metrics()


####################################################
#### Definition of Custom Metrics Preprocessing ####
####################################################

def compute_bounding_box(image: Image.Image):
    """
    Compute the bounding box (x, y, width, height) of the object in an RGBA image.
    Assumes that the image background is transparent (alpha=0) where the object is absent.
    """
    image_arr = np.array(image)
    
    alpha_channel = image_arr[..., 3]
    _, binary_mask = cv2.threshold(alpha_channel, 0, 255, cv2.THRESH_BINARY)
    
    x, y, w, h = cv2.boundingRect(binary_mask)
    return x, y, w, h

def compare_bounding_boxes(image1: Image.Image, image2: Image.Image):
    """
    Compare the bounding boxes of two preprocessed RGBA images.
    
    Returns:
        A dictionary with the absolute differences in width, height, area, 
        and aspect ratio (width/height) between the two bounding boxes.
    """
    _, _, w1, h1 = compute_bounding_box(image1)
    _, _, w2, h2 = compute_bounding_box(image2)
    
    width_diff = abs(w1 - w2)
    height_diff = abs(h1 - h2)
    
    area1 = w1 * h1
    area2 = w2 * h2
    area_diff = abs(area1 - area2)
    
    aspect1 = w1 / h1 if h1 != 0 else 0
    aspect2 = w2 / h2 if h2 != 0 else 0
    aspect_ratio_diff = abs(aspect1 - aspect2)

    percent_width = (width_diff / ((w1 + w2)/2))*100
    percent_height = (height_diff / ((h1 + h2)/2))*100
    percent_area = (area_diff / ((area1 + area2)/2))*100
    percent_aspect = (aspect_ratio_diff / ((aspect1 + aspect2)/2))*100

    
    return {
        "area_percent": round(percent_area,2) ,
        "aspect_percent": round(percent_aspect,2) ,
    }

def compare_px_area(image1: Image.Image, image2: Image.Image):
    arr1 = np.array(image1)
    arr2 = np.array(image2)

    alpha1 = arr1[...,3]
    alpha2 = arr2[...,3]

    _, binary_mask1 = cv2.threshold(alpha1, 0, 255, cv2.THRESH_BINARY)
    _, binary_mask2 = cv2.threshold(alpha2, 0, 255, cv2.THRESH_BINARY)
    mask1 = binary_mask1 // 255  
    mask2 = binary_mask2 // 255

    area1 = int(np.sum(mask1))
    area2 = int(np.sum(mask2))
    
    absolute_difference = abs(area1 - area2)
    relative_difference = absolute_difference / ((area1 + area2) / 2) if (area1 + area2) > 0 else 0

    return {
        "area_image1": area1,
        "area_image2": area2,
        "absolute_difference": absolute_difference,
        "relative_difference": relative_difference,
    }

def compute_outline_normals(image, num_points=100):
    """
    Given a preprocessed RGBA image (with background removed), compute the resampled
    outline (contour) points and their corresponding normal vectors.

    Args:
        image (PIL.Image.Image): Input image with transparent background.
        num_points (int): Number of points to sample along the contour.

    Returns:
        tuple: (resampled_points, normals)
          - resampled_points: numpy array of shape (num_points, 2)
          - normals: numpy array of shape (num_points, 2)
    """
    image_arr = np.array(image)
    alpha = image_arr[..., 3]
    _, binary_mask = cv2.threshold(alpha, 0, 255, cv2.THRESH_BINARY) # Binary Mask
    contours, _ = cv2.findContours(binary_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        raise ValueError("No contour found in image")
    contour = max(contours, key=lambda cnt: cv2.contourArea(cnt)) #Contour/Outline
    contour = contour[:, 0, :] 
    
    distances = np.zeros(len(contour))
    distances[1:] = np.cumsum(np.linalg.norm(np.diff(contour, axis=0), axis=1))
    total_length = distances[-1]
    if total_length == 0:
        raise ValueError("Contour has zero length")

    sample_dists = np.linspace(0, total_length, num_points, endpoint=False) # Sample points on outline
    resampled_points = []
    for d in sample_dists:
        idx = np.searchsorted(distances, d)
        if idx == 0:
            resampled_points.append(contour[0])
        else:
            # Linear interpolation between contour[idx-1] and contour[idx]
            t = (d - distances[idx-1]) / (distances[idx] - distances[idx-1])
            point = (1 - t) * contour[idx-1] + t * contour[idx]
            resampled_points.append(point)
    resampled_points = np.array(resampled_points) 
    
    # Compute tangent and normal at each resampled point.
    normals = []
    center = np.mean(resampled_points, axis=0)
    for i in range(num_points):
        prev_point = resampled_points[i - 1]
        next_point = resampled_points[(i + 1) % num_points]
        tangent = next_point - prev_point
        norm = np.linalg.norm(tangent)
        if norm == 0:
            tangent_norm = np.array([0.0, 0.0])
        else:
            tangent_norm = tangent / norm
        normal = np.array([-tangent_norm[1], tangent_norm[0]])
        vec_from_center = resampled_points[i] - center
        if np.dot(normal, vec_from_center) < 0:
            normal = -normal
        normals.append(normal)
    normals = np.array(normals)
    return resampled_points, normals

def compare_outline_normals(image1, image2, num_points=100):
    """
    Compare the object outlines of two preprocessed images by computing
    the normals along their silhouettes and measuring the average angle difference.

    Args:
        image1 (PIL.Image.Image): First preprocessed image (RGBA, background removed).
        image2 (PIL.Image.Image): Second preprocessed image.
        num_points (int): Number of points to sample along the contour.

    Returns:
        dict: A dictionary containing the average angle difference (in degrees)
              and other detailed arrays.
    """
    points1, normals1 = compute_outline_normals(image1, num_points=num_points)
    points2, normals2 = compute_outline_normals(image2, num_points=num_points)
    
    angle_diffs = []
    for p, n in zip(points1, normals1): 
        dists = np.linalg.norm(points2 - p, axis=1)
        nearest_idx = np.argmin(dists)
        n2 = normals2[nearest_idx]
        dot = np.clip(np.dot(n, n2), -1.0, 1.0)
        angle = np.arccos(dot)  
        angle_deg = np.degrees(angle)
        if angle_deg > 90:
            angle_deg = 180 - angle_deg
        angle_diffs.append(angle_deg)
    average_angle_diff = np.mean(angle_diffs)
    
    squared_normals1 = np.square(normals1)
    squared_normals2 = np.square(normals2)

    norms1 = np.linalg.norm(squared_normals1, axis=1, keepdims=True)
    norms2 = np.linalg.norm(squared_normals2, axis=1, keepdims=True)

    norms1[norms1 == 0] = 1
    norms2[norms2 == 0] = 1
    squared_normals1_normed = squared_normals1 / norms1
    squared_normals2_normed = squared_normals2 / norms2

    angle_diffs_squared = []
    for p, sq_n in zip(points1, squared_normals1_normed):
        dists = np.linalg.norm(points2 - p, axis=1)
        nearest_idx = np.argmin(dists)
        sq_n2 = squared_normals2_normed[nearest_idx]
        dot_sq = np.clip(np.dot(sq_n, sq_n2), -1.0, 1.0)
        angle_sq = np.arccos(dot_sq)
        angle_sq_deg = np.degrees(angle_sq)
        if angle_sq_deg > 90:
            angle_sq_deg = 180 - angle_sq_deg
        angle_diffs_squared.append(angle_sq_deg)
    average_angle_diff_squared = np.mean(angle_diffs_squared)

    return {
        "average_angle_difference_degrees": average_angle_diff,
        "angle_differences": angle_diffs,
        "average_angle_difference_squared_degrees": average_angle_diff_squared,
        "angle_differences_squared": angle_diffs_squared
    }

def compare_summed_outline_normals(image1, image2, num_points=100):
    """
    Compare two objects by summing all their outline normals into a single resultant
    vector per shape, then measuring the angle between these two resultant vectors.

    Returns:
        dict with:
            - angle_diff_degrees: angle in degrees between the two summed vectors
            - magnitude_diff: absolute difference in magnitudes
            - sum_vector1, sum_vector2: the resultant vectors
    """
    _, normals1 = compute_outline_normals(image1, num_points=num_points)
    _, normals2 = compute_outline_normals(image2, num_points=num_points)
    
    sum_vector1 = np.sum(normals1, axis=0)
    sum_vector2 = np.sum(normals2, axis=0)
    
    mag1 = np.linalg.norm(sum_vector1)
    mag2 = np.linalg.norm(sum_vector2)
    if mag1 == 0 or mag2 == 0:
        angle_diff = 0.0
    else:
        dot = np.clip(np.dot(sum_vector1, sum_vector2) / (mag1 * mag2), -1.0, 1.0)
        angle_diff = np.degrees(np.arccos(dot))
    
    magnitude_diff = abs(mag1 - mag2)
    
    squared_normals1 = np.square(normals1)
    squared_normals2 = np.square(normals2)
    sum_vector1_sq = np.sum(squared_normals1, axis=0)
    sum_vector2_sq = np.sum(squared_normals2, axis=0)
    mag1_sq = np.linalg.norm(sum_vector1_sq)
    mag2_sq = np.linalg.norm(sum_vector2_sq)
    if mag1_sq == 0 or mag2_sq == 0:
        angle_diff_sq = 0.0
    else:
        dot_sq = np.clip(np.dot(sum_vector1_sq, sum_vector2_sq) / (mag1_sq * mag2_sq), -1.0, 1.0)
        angle_diff_sq = np.degrees(np.arccos(dot_sq))
    magnitude_diff_sq = abs(mag1_sq - mag2_sq)

    return {
        "average_summed_angle_difference_degrees": angle_diff,
        "magnitude_diff": magnitude_diff,
        "sum_vector1": sum_vector1,
        "sum_vector2": sum_vector2,
        "average_summed_angle_difference_squared_degrees": angle_diff_sq,
        "magnitude_diff_squared": magnitude_diff_sq,
        "sum_vector1_squared": sum_vector1_sq,
        "sum_vector2_squared": sum_vector2_sq,
    }

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
