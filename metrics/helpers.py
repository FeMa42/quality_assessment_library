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
import pandas as pd
from tqdm import tqdm

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

def load_images_from_dir_legacy(image_dir: str, device):
    """
    Load all images from a directory and return them as a tensor with shape (num_frames, channels, height, width)
    """
    all_images = glob.glob(os.path.join(image_dir, "*.png"))
    all_images = [preprocess_image(Image.open(image)) for image in all_images]
    torch_image_tensor = torch.tensor(np.array(all_images), dtype=torch.float32)
    # (num_frames, channels, height, width)
    torch_image_tensor = torch_image_tensor.permute(0, 3, 1, 2)
    return torch_image_tensor.to(device)

def load_images_from_dir(image_dir: str, device, preprocess_func):
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

def process_folder(original_folder, generated_folder, preprocess_func, metric_class, device=None, **metric_kwargs):
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
    print(f"[process_folder] class={metric_class.__name__}, kwargs={metric_kwargs}")

    if metric_class.__name__ == "Metrics":
        semantic_metric = metric_class(device=device, **metric_kwargs)
        input_tensor  = load_images_from_dir(original_folder, device, preprocess_func)
        target_tensor = load_images_from_dir(generated_folder,  device, preprocess_func)
        return semantic_metric.compute_image(input_tensor, target_tensor)
    
    elif metric_class.__name__ == "GeometryMetrics":
        geo_metric = metric_class(**metric_kwargs)
        orig_files = sorted(glob.glob(os.path.join(original_folder, "*.png")))
        gen_files = sorted(glob.glob(os.path.join(generated_folder, "*.png")))
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
    _, binary_mask = cv2.threshold(alpha, 0, 255, cv2.THRESH_BINARY) 
    contours, _ = cv2.findContours(binary_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        raise ValueError("No contour found in image")
    contour = max(contours, key=lambda cnt: cv2.contourArea(cnt)) 
    contour = contour[:, 0, :] 
    
    distances = np.zeros(len(contour))
    distances[1:] = np.cumsum(np.linalg.norm(np.diff(contour, axis=0), axis=1))
    total_length = distances[-1]
    if total_length == 0:
        raise ValueError("Contour has zero length")

    sample_dists = np.linspace(0, total_length, num_points, endpoint=False) 
    resampled_points = []
    for d in sample_dists:
        idx = np.searchsorted(distances, d)
        if idx == 0:
            resampled_points.append(contour[0])
        else:
            t = (d - distances[idx-1]) / (distances[idx] - distances[idx-1])
            point = (1 - t) * contour[idx-1] + t * contour[idx]
            resampled_points.append(point)
    resampled_points = np.array(resampled_points) 
    
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


##############################################
#### Definition of Custom Metrics Classes ####
##############################################

def get_caption_from_metadata(metadata_df, sha256: str, caption_col: str = "caption"):
    """
    Get the caption from the metadata DataFrame for a given sha256.
    """
    metadata_row = metadata_df[metadata_df["sha256"] == sha256]
    if metadata_row.empty:
        print(f"No metadata found for {sha256}. Skipping.")
        return None
    metadata_row = metadata_row.iloc[0]
    metadata_caption = metadata_row[caption_col]
    if isinstance(metadata_caption, str):
        pass
    elif isinstance(metadata_caption, list):
        metadata_caption = " ".join(metadata_caption)
    else:
        raise ValueError(f"Unexpected type for caption: {type(metadata_caption)}. Expected str or list.")
    return metadata_caption

def process_folder_with_metadata_file(generated_base_folder, metadata_file_path, prompt_metric, preprocess_image):
    """
    Process a folder of generated images and compute the prompt metric using the metadata from the metadata file.
    This function loads images from the specified folder, preprocesses them using the provided function,
    and evaluates them against the given object prompt using the provided prompt metric.
    Args:
        generated_base_folder (str): Path to the folder containing generated images.
        metadata_file_path (str): Path to the metadata file.
        prompt_metric: function which estimates the prompt metrics. E.g. Instance of ImageBasedPromptEvaluator.
        preprocess_image: function which preprocesses the image. E.g. preprocess_image_rgba.
    """
    # Load metadata file
    metadata_file = pd.read_csv(metadata_file_path)
    gen_object_folder = os.listdir(generated_base_folder)

    all_scores = {}
    for i in tqdm(range(len(gen_object_folder))):
        obj_folder = gen_object_folder[i]
        object_prompt = get_caption_from_metadata(metadata_file, obj_folder.strip(), "caption_3d_prompt")
        if object_prompt is None:
            continue
        promp_score = process_folder_with_prompt(generated_folder=os.path.join(generated_base_folder, obj_folder),
                                                object_prompts=object_prompt,
                                                preprocess_func=preprocess_image,
                                                prompt_metric=prompt_metric)
        for key, value in promp_score.items():
            if key not in all_scores:
                all_scores[key] = []
            all_scores[key].append(value)


    # get mean and standard deviation of all_scores
    mean_scores = {}
    std_scores = {}
    for key, value in all_scores.items():
        std_scores[key] = pd.Series(value).std()
        mean_scores[key] = pd.Series(value).mean()
    return mean_scores, std_scores

def load_images_from_dir_to_pil(image_dir: str, preprocess_func):
    """
    Load all images from a directory, preprocess them with preprocess_func,
    and return a tensor of shape (num_frames, channels, height, width).
    """
    files = sorted(glob.glob(os.path.join(image_dir, "*.png")))
    images = [preprocess_func(Image.open(f)) for f in files]
    return images

def load_prompts_from_dir(prompt_dir: str):
    prompt_files = sorted(glob.glob(os.path.join(prompt_dir, "*.txt")))
    object_prompts = []
    for prompt_file in prompt_files:
        with open(prompt_file, "r") as f:
            # each file is one prompt 
            object_prompts.append(f.read().strip())
    return object_prompts

def process_folder_with_prompt_files(
    generated_folder: str,
    preprocess_func,
    prompt_metric):
    """
    Process a folder of generated images and compute the prompt metric using the metadata from the metadata file. 
    This function loads images from the specified folder, preprocesses them using the provided function,
    and evaluates them against the given object prompt using the provided prompt metric.
    Args:
        generated_folder (str): Path to the folder containing generated images.
        object_prompts (Union[str, List[str]]): Object prompts to evaluate against.
        preprocess_func: Preprocessing function for images.
        prompt_metric: function which estimates the prompt metrics. E.g. Instance of ImageBasedPromptEvaluator.
    """
    generated_images = load_images_from_dir_to_pil(generated_folder, preprocess_func)  
    object_prompts = load_prompts_from_dir(generated_folder)
    # Check if the number of prompts matches the number of images
    if len(object_prompts) != len(generated_images):
        raise ValueError(f"Number of prompts ({len(object_prompts)}) does not match number of images ({len(generated_images)}).")
    
    image_scores = prompt_metric.evaluate(generated_images, object_prompts)
    return image_scores

def process_folder_with_prompt(
    generated_folder: str,
    object_prompts: Union[str, List[str]],
    preprocess_func,
    prompt_metric):
    """
    Process a folder of generated images and compute the prompt metric using the metadata from the metadata file. 
    This function loads images from the specified folder, preprocesses them using the provided function,
    and evaluates them against the given object prompt using the provided prompt metric.
    Args:
        generated_folder (str): Path to the folder containing generated images.
        object_prompts (Union[str, List[str]]): Object prompts to evaluate against.
        preprocess_func: Preprocessing function for images.
        prompt_metric: function which estimates the prompt metrics. E.g. Instance of ImageBasedPromptEvaluator.
    """
    generated_images = load_images_from_dir_to_pil(generated_folder, preprocess_func)  
    if isinstance(object_prompts, str):
        object_prompts = [object_prompts] * len(generated_images)
    elif len(object_prompts) != len(generated_images):
        raise ValueError(f"Number of prompts ({len(object_prompts)}) does not match number of images ({len(generated_images)}).")
    
    image_scores = prompt_metric.evaluate(generated_images, object_prompts)
    return image_scores

def evaluate_vehicle_dimensions(generated_base_folder, metadata_file_path, florence_wheelbase_od):
    """
    Evaluate the vehicle dimensions of generated objects against the metadata file.
    Args:
        generated_base_folder (str): Path to the folder containing generated objects.
        metadata_file_path (str): Path to the metadata file.
        florence_wheelbase_od (FlorenceWheelbaseOD): Instance of the FlorenceWheelbaseOD class.
    """
    # Load the metadata file
    metadata_file = pd.read_csv(metadata_file_path)
    all_obj_sha256 = os.listdir(generated_base_folder)
    dimension_differences = {
        "length_difference": [],
        "width_difference": [],
        "wheelbase_difference": []
    }
    for sha256 in all_obj_sha256:
        # Check if the folder exists
        generated_folder = os.path.join(generated_base_folder, sha256)
        if not os.path.exists(generated_folder):
            print(f"Folder {generated_folder} does not exist.")
            continue
        generated_vehicle_data = florence_wheelbase_od.get_vehicle_dimensions_from_folder(generated_folder, normalize=True)
        metadata_row = metadata_file[metadata_file["sha256"] == sha256]
        if metadata_row.empty:
            print(f"Metadata for {sha256} not found.")
        else:
            # Extract the metadata values
            metadata_row = metadata_row.iloc[0]
            length = metadata_row["normalized_depth_of_object"]
            width = metadata_row["normalized_width_of_object"]
            wheelbase = metadata_row["normalized_wheelbase"]
            # compare the generated vehicle data with the metadata
            length_diff = abs(generated_vehicle_data["depth_of_object"] - length)
            width_diff = abs(generated_vehicle_data["width_of_object"] - width)
            wheelbase_diff = abs(generated_vehicle_data["wheelbase"] - wheelbase)
            # Store the differences in a dictionary for lengths, widths, and wheelbases
            dimension_differences["length_difference"].append(length_diff)
            dimension_differences["width_difference"].append(width_diff)
            dimension_differences["wheelbase_difference"].append(wheelbase_diff)
    # Calculate the average of the differences


            
    # Calculate the average of the average differences
    overall_average_diff = {}
    standard_deviation = {}
    for key, value in dimension_differences.items():
        if len(value) > 0:
            overall_average_diff[key] = np.mean(value)
            standard_deviation[key] = np.std(value)
        else:
            overall_average_diff[key] = None
            standard_deviation[key] = None
    
    return overall_average_diff, standard_deviation
