import os
import cv2
import glob
import numpy as np
from PIL import Image
import scipy.signal
import pandas as pd
from rembg import remove
import torch
import torch.nn.functional as F
import timm
import torchvision.transforms as T
from tqdm import tqdm
from metrics.helpers import compute_bounding_box

# =============================================================================
# DINOv2 Feature Extraction
# =============================================================================

def load_dino(device='cuda'):
    model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitg14').to(device)
    model.eval()
    return model

def get_dino_transform(image_size=224):

    return T.Compose([
        T.Resize(image_size),
        T.CenterCrop(image_size),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225])
    ])

def load_dino_model(model_name="dino_vits16", device="cuda"):
    model = timm.create_model(model_name, pretrained=True)
    model.eval()
    model.to(device)
    return model

def compute_dino_features(image, dino_model, transform, device):
    """
    Compute a normalized feature vector for a given PIL image using the DINO model.
    
    Args:
        image (PIL.Image): Input image.
        dino_model: The DINOv2 model loaded (e.g., via timm).
        transform: The torchvision transform to apply.
        device (str): "cuda" or "cpu".
    
    Returns:
        feature (torch.Tensor): Normalized 1-D feature vector.
    """
    image_rgb = image.convert("RGB")
    input_tensor = transform(image_rgb).unsqueeze(0).to(device)  
    with torch.no_grad():
        output = dino_model(input_tensor)
        if isinstance(output, dict):
            if "features" in output:
                features = output["features"]
            elif "embeddings" in output:
                features = output["embeddings"]
            else:
                raise KeyError("Output dictionary does not contain 'features' or 'embeddings'.")
        else:
            features = output
    if features.dim() == 4:
        features = features.mean(dim=[2, 3])
    features = torch.nn.functional.normalize(features, dim=1)
    return features.squeeze(0)

# =============================================================================
# Utility Functions (Common)
# =============================================================================

def calc_multiview_bbox_dim(pil_imgs):

    ws, hs = [], []
    for img in pil_imgs:
        try:
            alpha = np.array(img.split()[-1])
        except Exception:
            alpha = np.array(img.convert("L"))
        _, mask = cv2.threshold(alpha, 0, 255, cv2.THRESH_BINARY)
        _, _, w, h = cv2.boundingRect(mask)
        ws.append(w)
        hs.append(h)
    return ws, hs

def find_min(arr):

    n = len(arr)
    arr_padded = np.concatenate(([arr[-1]], arr, [arr[0]]))
    minima_indices = scipy.signal.find_peaks(-arr_padded)[0] - 1
    return minima_indices % n

# =============================================================================
# Front View Detection & Image Reordering
# =============================================================================

def is_front(img, model, processor, prompt, device):

    inputs = processor(images=img, text=prompt, return_tensors="pt").to(device)
    outputs = model.generate(
        **inputs,
        do_sample=False,
        num_beams=5,
        max_length=60,
        min_length=1,
        top_p=0.9,
        repetition_penalty=1.5,
        length_penalty=1.0,
        temperature=1,
    )
    answer = processor.batch_decode(outputs, skip_special_tokens=True)[0]
    if "?" in answer:
        answer = answer.split("?", 1)[-1]
    answer = answer.strip().lower()
    return "front" in answer and "rear" not in answer

def determine_front_candidate_vlm(ws, pil_imgs, model, processor, prompt, device, skip_if_not_two_candidates=True):
    """
    Determine the candidate index for the front view using VLM.
    This function uses bounding box minima to select two candidates and queries the VLM.
    
    Args:
        ws (list): Bounding box widths.
        pil_imgs (list): List of candidate images.
        model, processor, prompt, device: VLM parameters.
        skip_if_not_two_candidates (bool): Return -1 if not exactly two minima.
        
    Returns:
        int: Candidate index for front view (or -1 on failure).
    """
    minima = find_min(np.array(ws))
    if len(minima) != 2 and skip_if_not_two_candidates:
        return -1
    candidate1, candidate2 = minima[0], minima[1]
    
    if is_front(pil_imgs[candidate1], model, processor, prompt, device):
        print(f"Candidate index {candidate1} confirmed as front by VLM.")
        return candidate1
    elif is_front(pil_imgs[candidate2], model, processor, prompt, device):
        print(f"Candidate index {candidate2} confirmed as front by VLM.")
        return candidate2
    else:
        print("Neither candidate confirmed as front by VLM. Defaulting to candidate 1.")
        return candidate1

def determine_front_candidate_dino(ws, pil_imgs, ground_truth_img, dino_model, dino_transform, device, skip_if_not_two_candidates=True):
    """
    Determine the front view candidate by comparing DINOv2 features of the candidates with a ground truth front view image.
    
    Args:
        ws (list): List of bounding box widths.
        pil_imgs (list): List of candidate images.
        ground_truth_img (PIL.Image): GT front view image (assumed to be 000.png).
        dino_model: Loaded DINOv2 model.
        dino_transform: Transform for DINO feature extraction.
        device (str): Device identifier.
    
    Returns:
        int: Candidate index with higher cosine similarity to GT.
    """
    minima = find_min(np.array(ws))
    if len(minima) != 2 and skip_if_not_two_candidates:
        return -1
    candidate1, candidate2 = minima[0], minima[1]
    
    gt_features = compute_dino_features(ground_truth_img, dino_model, dino_transform, device)
    feat1 = compute_dino_features(pil_imgs[candidate1], dino_model, dino_transform, device)
    feat2 = compute_dino_features(pil_imgs[candidate2], dino_model, dino_transform, device)
    
    sim1 = F.cosine_similarity(gt_features.unsqueeze(0), feat1.unsqueeze(0), dim=1).item()
    sim2 = F.cosine_similarity(gt_features.unsqueeze(0), feat2.unsqueeze(0), dim=1).item()
    
    # print(f"DINO similarity: Candidate1: {sim1:.4f}, Candidate2: {sim2:.4f}")
    return candidate1 if sim1 >= sim2 else candidate2

def align_subfolder(subfolder, detection_mode="vlm",
                    vlm_model=None, vlm_processor=None, prompt=None,
                    dino_model=None, dino_transform=None, ground_truth_parent=None,
                    device="cuda"):
    """
    Process a subfolder (corresponding to a single object) of reconstructed images,
    detect the front view candidate and reorder the images so that the candidate becomes 000.png.
    
    The parameter `detection_mode` chooses which method to use:
      - "vlm": Use the vision-language model. (Requires: vlm_model, vlm_processor, prompt)
      - "dino": Use DINOv2 feature comparison. (Requires: ground_truth_parent, dino_model, dino_transform)
    
    Args:
        subfolder (str): Path to the subfolder.
        detection_mode (str): "vlm" or "dino".
        vlm_model, vlm_processor, prompt: VLM parameters (only needed for "vlm" mode).
        dino_model, dino_transform, ground_truth_parent: DINO parameters and GT folder (only needed for "dino" mode).
        device (str): "cuda" or "cpu".
    
    Returns:
        int or None: The candidate index (in the original sorted order) of the detected front view, or None on failure.
    """
    image_files = sorted([f for f in os.listdir(subfolder) if f.lower().endswith('.png')])
    if not image_files:
        print(f"No PNG images found in {subfolder}. Skipping.")
        return None
    pil_imgs = [Image.open(os.path.join(subfolder, f)).convert("RGBA") for f in image_files]
    ws, _ = calc_multiview_bbox_dim(pil_imgs)
    
    if detection_mode.lower() == "dino":
        if not (ground_truth_parent and dino_model and dino_transform):
            print("DINO mode selected but required parameters not provided. Aborting.")
            return None
        gt_folder = os.path.join(ground_truth_parent, os.path.basename(subfolder))
        gt_img_path = os.path.join(gt_folder, "000.png")
        if not os.path.exists(gt_img_path):
            print(f"GT front view not found for {subfolder} in {gt_folder}. Aborting.")
            return None
        ground_truth_img = Image.open(gt_img_path).convert("RGB")
        front_index = determine_front_candidate_dino(ws, pil_imgs, ground_truth_img, dino_model, dino_transform, device)
    else:  
        if not (vlm_model and vlm_processor and prompt):
            print("VLM mode selected but required parameters not provided. Aborting.")
            return None
        front_index = determine_front_candidate_vlm(ws, pil_imgs, vlm_model, vlm_processor, prompt, device)
    
    if front_index == -1:
        print(f"Candidate extraction failed in {subfolder}.")
        return None
    
    # print(f"[{os.path.basename(subfolder)}] Detected front view index: {front_index} (original file: {image_files[front_index]})")
    
    new_order = image_files[front_index:] + image_files[:front_index]
    temp_prefix = "temp_rename_"
    for f in image_files:
        os.rename(os.path.join(subfolder, f), os.path.join(subfolder, temp_prefix + f))
    for i, f in enumerate(new_order):
        os.rename(os.path.join(subfolder, temp_prefix + f), os.path.join(subfolder, f"{i:03d}.png"))
    
    return front_index

def align_views(parent_folder, detection_mode="vlm",
                vlm_model=None, vlm_processor=None, prompt=None,
                dino_model=None, dino_transform=None, ground_truth_parent=None,
                device="cuda"):
    """
    Process every subfolder (each corresponding to an object) within parent_folder.
    For each subfolder, detect the front view candidate and reorder the images so that
    the candidate becomes 000.png.
    
    The function uses the selected detection_mode ("vlm" or "dino"). Only the parameters relevant
    to the chosen mode need to be provided.
    
    Args:
        parent_folder (str): Path containing subfolders of reconstructed images.
        detection_mode (str): "vlm" or "dino".
        vlm_model, vlm_processor, prompt: VLM parameters (for "vlm" mode).
        dino_model, dino_transform, ground_truth_parent: DINO parameters (for "dino" mode).
        device (str): "cuda" or "cpu".
    
    Returns:
        DataFrame: A pandas DataFrame summarizing each subfolder (uid) and the detected front view index.
    """
    subfolders = [os.path.join(parent_folder, d) for d in os.listdir(parent_folder)
                  if os.path.isdir(os.path.join(parent_folder, d))]
    results = []
    for subf in tqdm(subfolders, desc="Aligning views", unit="subfolder"):
        front_idx = align_subfolder(subf, detection_mode=detection_mode,
                                    vlm_model=vlm_model, vlm_processor=vlm_processor, prompt=prompt,
                                    dino_model=dino_model, dino_transform=dino_transform, ground_truth_parent=ground_truth_parent,
                                    device=device)
        if front_idx is not None:
            results.append((os.path.basename(subf), front_idx))
    df = pd.DataFrame(results, columns=["uid", "front_view_index"])
    csv_path = os.path.join(parent_folder, "front_view_alignment.csv")
    df.to_csv(csv_path, index=False)
    print(f"Alignment complete. Summary saved to {csv_path}")
    return df

# =============================================================================
# Background Removal Functions
# =============================================================================

def remove_background_from_folder(folder, verbose=True):

    valid_exts = ('.png',)
    image_files = [f for f in os.listdir(folder) if f.lower().endswith(valid_exts)]
    for filename in image_files:
        path = os.path.join(folder, filename)
        im = Image.open(path)
        output_im = remove(im)
        output_im.save(path)
    if verbose:
        print(f"Background removal complete in folder: {folder}")

def remove_background_recursive(parent_folder):

    for root, dirs, files in os.walk(parent_folder):
        pngs = [f for f in files if f.lower().endswith('.png')]
        if pngs:
            print(f"Processing background removal in folder: {root}")
            remove_background_from_folder(root)

# Define this function at module level (outside of any other function)
def process_with_progress(folder):
    """
    Helper function to process a folder and return any errors.
    
    Args:
        folder (str): Path to the folder to process
        
    Returns:
        tuple: (folder_path, error_message or None)
    """
    try:
        # from preprocessing.image_processing import remove_background_from_folder
        remove_background_from_folder(folder, verbose=False)
        return folder, None
    except Exception as e:
        return folder, str(e)

def remove_background_recursive_parallel(parent_folder, max_workers=None):
    """
    Process background removal recursively on all PNG files using multiple processes
    with tqdm progress bars.
    
    Args:
        parent_folder (str): Root folder to start the recursive search
        max_workers (int, optional): Maximum number of worker processes. 
                                    If None, it will use the number of CPU cores.
    """
    import concurrent.futures
    import multiprocessing
    from tqdm import tqdm
    
    if max_workers is None:
        max_workers = multiprocessing.cpu_count()
    
    # First, collect all folders that need processing
    folders_to_process = []
    for root, dirs, files in os.walk(parent_folder):
        pngs = [f for f in files if f.lower().endswith('.png')]
        if pngs:
            folders_to_process.append(root)
    
    total_folders = len(folders_to_process)
    print(f"Found {total_folders} folders with PNG images to process using {max_workers} worker processes")
    
    # Create a progress bar for the overall process
    pbar = tqdm(total=total_folders, desc="Removing backgrounds", unit="folder")
    
    # Process folders in parallel
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Map is simpler and handles the results in order of completion
        for folder, error in executor.map(process_with_progress, folders_to_process):
            pbar.update(1)
            if error:
                pbar.write(f"Error in {folder}: {error}")
    
    pbar.close()
    print(f"Background removal complete. Processed {total_folders} folders.")

# =============================================================================
# Equal Image Scaling Functions
# =============================================================================
def crop_object(image: Image.Image):

    x, y, w, h = compute_bounding_box(image)
    return image.crop((x, y, x + w, y + h)), w, h

def scale_image(image: Image.Image, scale_factor: float):

    orig_width, orig_height = image.size
    new_width = int(orig_width * scale_factor)
    new_height = int(orig_height * scale_factor)
    return image.resize((new_width, new_height), resample=Image.LANCZOS)

def center_on_canvas(image: Image.Image, canvas_size=(576, 576)):
    """
    Place the given image at the center of a new blank (transparent) canvas.
    
    Args:
        image (PIL.Image.Image): Image to be centered (assumed RGBA).
        canvas_size (tuple): Size of the canvas (width, height).
    
    Returns:
        PIL.Image.Image: The new image with the object centered.
    """
    canvas = Image.new("RGBA", canvas_size, (0, 0, 0, 0))
    img_width, img_height = image.size
    canvas_width, canvas_height = canvas_size

    left = (canvas_width - img_width) // 2
    top = (canvas_height - img_height) // 2
    canvas.paste(image, (left, top), mask=image)
    return canvas


def process_equal_scaling(original_folder, generated_folder, 
                          out_original_folder, out_generated_folder,
                          canvas_size=(768,768), fill_ratio=0.95):
    """
    Process pairs of images from two folders so that the generated object's scale matches
    the original object's scale, and both objects are scaled to fill only a given percentage
    of the canvas width (fill_ratio) when centered.
    
    Steps for each pair:
      1. Crop each image to its object (using bounding box).
      2. Scale the generated object so its width equals the original object's width.
      3. Compute a secondary scaling factor so that the object's width becomes fill_ratio * canvas_width.
      4. Apply this secondary scaling to both the original and the generated images.
      5. Center both on a new transparent canvas.
      6. Save the results in new output folders.
    
    Args:
        original_folder (str): Folder with original images.
        generated_folder (str): Folder with generated images.
        out_original_folder (str): Folder to save processed original images.
        out_generated_folder (str): Folder to save processed generated images.
        canvas_size (tuple): Size of the canvas (width, height).
        fill_ratio (float): Fraction of the canvas width to be occupied by the object.
    """
    os.makedirs(out_original_folder, exist_ok=True)
    os.makedirs(out_generated_folder, exist_ok=True)
    
    orig_files = sorted(glob.glob(os.path.join(original_folder, "*.png")))
    gen_files = sorted(glob.glob(os.path.join(generated_folder, "*.png")))
    
    gen_dict = {os.path.basename(f): f for f in gen_files}
    
    canvas_width = canvas_size[0]
    target_width = fill_ratio * canvas_width  
    
    for orig_path in orig_files:
        fname = os.path.basename(orig_path)
        gen_path = gen_dict.get(fname)
        if not gen_path:
            print(f"Warning: No matching generated image for {fname}")
            continue
        
        orig_img = Image.open(orig_path).convert("RGBA")
        gen_img = Image.open(gen_path).convert("RGBA")
        
        orig_crop, orig_w, orig_h = crop_object(orig_img)
        gen_crop, gen_w, gen_h = crop_object(gen_img)
        
        if gen_w == 0:
            print(f"Warning: Generated image {fname} has zero width object.")
            continue
        scale_factor_gen = orig_w / gen_w
        gen_scaled = scale_image(gen_crop, scale_factor_gen)
        
        scale_factor_final = target_width / orig_w
        
        orig_final = scale_image(orig_crop, scale_factor_final)
        gen_final = scale_image(gen_scaled, scale_factor_final)
        
        orig_centered = center_on_canvas(orig_final, canvas_size=canvas_size)
        gen_centered = center_on_canvas(gen_final, canvas_size=canvas_size)
        
        orig_centered.save(os.path.join(out_original_folder, fname))
        gen_centered.save(os.path.join(out_generated_folder, fname))
        print(f"Processed {fname}: orig_w={orig_w}, gen_w={gen_w}, "
              f"scale_factor_gen={scale_factor_gen:.3f}, scale_factor_final={scale_factor_final:.3f}")
        

def process_equal_scaling_structure(ground_truth_parent, generated_parent, 
                                    canvas_size=(768, 768), fill_ratio=0.95):
    """
    Process images from corresponding subfolders (object IDs) under the ground truth and generated folders,
    so that both images are equally scaled based on the ground truth object. The processed images are 
    saved under a "scaled" folder automatically generated in the common parent directory of the inputs.
    
    Expected folder structure:
      ground_truth_parent/
          <object_id_1>/
              000.png, 001.png, ..., 011.png
          <object_id_2>/
              ...
      generated_parent/
          <object_id_1>/
              000.png, 001.png, ..., 011.png
          <object_id_2>/
              ...
    
    Processing steps for each corresponding image pair:
      1. Crop each image to its object using crop_object().
      2. Scale the generated object's crop so its width equals that of the ground truth crop.
      3. Compute a secondary scaling factor so that the ground truth object's width becomes 
         (fill_ratio * canvas_width).
      4. Apply this final scaling to both images.
      5. Center each scaled image on a new transparent canvas of size canvas_size.
      6. Save the processed images in a folder structure that mirrors the input,
         stored under a "scaled" subfolder in the common parent directory.
    
    Args:
        ground_truth_parent (str): Path to the folder containing ground truth object subfolders.
        generated_parent (str): Path to the folder containing generated object subfolders.
        canvas_size (tuple): Canvas dimensions (width, height) for the final images.
        fill_ratio (float): Fraction of the canvas width to be occupied by the object.
    """
    
    parent_dir = os.path.dirname(ground_truth_parent)
    scaled_folder = os.path.join(parent_dir, "scaled")
    os.makedirs(scaled_folder, exist_ok=True)
    
    out_orig_folder = os.path.join(scaled_folder, os.path.basename(ground_truth_parent))
    out_gen_folder = os.path.join(scaled_folder, os.path.basename(generated_parent))
    os.makedirs(out_orig_folder, exist_ok=True)
    os.makedirs(out_gen_folder, exist_ok=True)
    
    canvas_width = canvas_size[0]
    target_width = fill_ratio * canvas_width  
    
    for obj_folder in tqdm(sorted(os.listdir(ground_truth_parent)), desc="Processing objects", unit="object"):
        gt_obj_path = os.path.join(ground_truth_parent, obj_folder)
        if not os.path.isdir(gt_obj_path):
            continue
        
        gen_obj_path = os.path.join(generated_parent, obj_folder)
        if not os.path.isdir(gen_obj_path):
            print(f"Warning: No corresponding generated folder for object {obj_folder}")
            continue

        out_orig_obj = os.path.join(out_orig_folder, obj_folder)
        out_gen_obj = os.path.join(out_gen_folder, obj_folder)
        os.makedirs(out_orig_obj, exist_ok=True)
        os.makedirs(out_gen_obj, exist_ok=True)
        
        gt_files = sorted(glob.glob(os.path.join(gt_obj_path, "*.png")))
        for gt_path in gt_files:
            fname = os.path.basename(gt_path)
            gen_path = os.path.join(gen_obj_path, fname)
            if not os.path.exists(gen_path):
                print(f"Warning: In object {obj_folder} no matching generated image for {fname}")
                continue
            
            gt_img = Image.open(gt_path).convert("RGBA")
            gen_img = Image.open(gen_path).convert("RGBA")
            
            gt_crop, gt_w, gt_h = crop_object(gt_img)
            gen_crop, gen_w, gen_h = crop_object(gen_img)
            
            if gen_w == 0:
                print(f"Warning: Generated image {fname} in object {obj_folder} has zero width object.")
                continue
            
            scale_factor_gen = gt_w / gen_w
            gen_scaled = scale_image(gen_crop, scale_factor_gen)
            
            scale_factor_final = target_width / gt_w
            
            gt_final = scale_image(gt_crop, scale_factor_final)
            gen_final = scale_image(gen_scaled, scale_factor_final)
            
            gt_centered = center_on_canvas(gt_final, canvas_size=canvas_size)
            gen_centered = center_on_canvas(gen_final, canvas_size=canvas_size)
            
            gt_centered.save(os.path.join(out_orig_obj, fname))
            gen_centered.save(os.path.join(out_gen_obj, fname))
            
            # print(f"Processed {obj_folder}/{fname}: gt_w={gt_w}, gen_w={gen_w}, "
            #       f"scale_factor_gen={scale_factor_gen:.3f}, scale_factor_final={scale_factor_final:.3f}")

# =============================================================================
# GPU Memory Management
# =============================================================================

def release_gpu(*args):

    for obj in args:
        del obj
    torch.cuda.empty_cache()
    print("GPU memory has been released.")

# =============================================================================
# Concat Images into Grid for Visualization 
# =============================================================================

def concatenate_images(
        folder_path, 
        output_image='concatenated.png', 
        rows = 1, 
        target_identifier = None, 
        crop_width_percent=0,
        crop_height_percent=0):

    image_paths = []
    valid_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.gif')
    
    if target_identifier is None:
        for fname in os.listdir(folder_path):
            if fname.lower().endswith(valid_extensions):
                image_paths.append(os.path.join(folder_path, fname))
    else:
        for subfolder in os.listdir(folder_path):
            subfolder_path = os.path.join(folder_path, subfolder)
            if os.path.isdir(subfolder_path):
                candidate = os.path.join(subfolder_path, target_identifier)
                if os.path.exists(candidate):
                    image_paths.append(candidate)
    
    
    if not image_paths:
        print("No images found to concatenate.")
        return
    
    images = []
    for path in image_paths:
        try:
            im = Image.open(path)
        except Exception as e:
            print(f"Error opening image {path}: {e}")
            continue

        if im.mode == 'RGBA':
            background = Image.new('RGB', im.size, (255, 255, 255))
            background.paste(im, mask=im.split()[3])
            im = background
        else:
            im = im.convert('RGB')
    
        if crop_width_percent > 0 or crop_height_percent > 0:
            width, height = im.size
            new_width = int(width * (1 - crop_width_percent / 100))
            new_height = int(height * (1 - crop_height_percent / 100))
            left = (width - new_width) // 2
            top = (height - new_height) // 2
            right = left + new_width
            bottom = top + new_height
            im = im.crop((left, top, right, bottom))
            
        images.append(im)
    if not images:
        print("No valid images loaded.")
        return

    total_images = len(images)
    
    if rows < 1:
        print("The number of rows must be at least 1.")
        return
        
    cols = math.ceil(total_images / rows) if rows > 1 else total_images

    grid = []
    for i in range(rows):
        row_images = images[i * cols : (i + 1) * cols]
        if row_images:
            grid.append(row_images)
  
    row_widths = []
    row_heights = []
    for row in grid:
        widths = [img.width for img in row]
        heights = [img.height for img in row]
        row_widths.append(sum(widths))
        row_heights.append(max(heights))
        
    final_width = max(row_widths)
    final_height = sum(row_heights)

    concatenated_image = Image.new('RGB', (final_width, final_height), color=(255, 255, 255))

    y_offset = 0
    for row, row_height in zip(grid, row_heights):
        x_offset = 0
        for img in row:
            concatenated_image.paste(img, (x_offset, y_offset))
            x_offset += img.width
        y_offset += row_height

    concatenated_image.save(output_image)
    print(f"Concatenated image saved as '{output_image}'")

# =============================================================================
# Manual Image Reordering
# =============================================================================

def reorder_images_in_folder(folder, new_first_index):
    """
    Reorder PNG images in the given folder so that the image at new_first_index becomes 000.png.
    
    Args:
        folder (str): Path to folder containing PNG images.
        new_first_index (int): Zero-based index (in sorted order) to become 000.png.
    """
    image_files = sorted([f for f in os.listdir(folder) if f.lower().endswith('.png')])
    if not image_files:
        print("No PNG images found in folder:", folder)
        return
    num_files = len(image_files)
    if new_first_index < 0 or new_first_index >= num_files:
        print(f"Index {new_first_index} out of range; must be between 0 and {num_files-1}.")
        return

    new_order = image_files[new_first_index:] + image_files[:new_first_index]
    temp_prefix = "temp_rename_"
    for f in image_files:
        os.rename(os.path.join(folder, f), os.path.join(folder, temp_prefix + f))
    for i, f in enumerate(new_order):
        os.rename(os.path.join(folder, temp_prefix + f), os.path.join(folder, f"{i:03d}.png"))
    print("Reordering complete in folder:", folder)


def swap_files(dir_path, file1, file2):
    """
    Swap two files in the directory using a temporary filename.

    Args:
        dir_path (str): Path to the directory containing the files.
        file1 (str): Filename of the first image.
        file2 (str): Filename of the second image.
    """
    path1 = os.path.join(dir_path, file1)
    path2 = os.path.join(dir_path, file2)
    
    # Verify that both files exist before swapping.
    if not (os.path.exists(path1) and os.path.exists(path2)):
        print(f"Warning: {file1} or {file2} not found in {dir_path}. Skipping swap.")
        return

    # Use a temporary file name that avoids conflicts.
    tmp_path = os.path.join(dir_path, f"tmp_swap_{file1}")
    
    # Perform the swapping operation.
    os.rename(path1, tmp_path)
    os.rename(path2, path1)
    os.rename(tmp_path, path2)
    # print(f"Swapped {file1} and {file2} in {dir_path}")

def restructure_images(parent_folder):

    # Check that the parent folder exists.
    if not os.path.isdir(parent_folder):
        print(f"Error: The directory {parent_folder} does not exist or is not a folder.")
        return

    # Iterate over all items in the parent directory.
    for item in tqdm(os.listdir(parent_folder, desc="Restructuring images", unit="item")):
        subfolder_path = os.path.join(parent_folder, item)
        
        # Process only subfolders.
        if os.path.isdir(subfolder_path):
            # print(f"Processing folder: {subfolder_path}")
            
            # Define swap pairs
            swap_pairs = [
                ("001.png", "011.png"),
                ("002.png", "010.png"),
                ("003.png", "009.png"),
                ("004.png", "008.png"),
                ("005.png", "007.png")
            ]
            
            # Execute swaps for each pair in the current subfolder.
            for file1, file2 in swap_pairs:
                swap_files(subfolder_path, file1, file2)
    print("Restructuring complete.")