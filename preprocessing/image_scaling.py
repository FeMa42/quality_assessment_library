import os
import glob
import numpy as np
import cv2
from PIL import Image
from metrics.metrics import compute_bounding_box


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
    
    for obj_folder in sorted(os.listdir(ground_truth_parent)):
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
            
            print(f"Processed {obj_folder}/{fname}: gt_w={gt_w}, gen_w={gen_w}, "
                  f"scale_factor_gen={scale_factor_gen:.3f}, scale_factor_final={scale_factor_final:.3f}")
