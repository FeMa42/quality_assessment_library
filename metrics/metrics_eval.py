import json
import pandas as pd
import os
import shutil
import tempfile
import torch
import glob
from PIL import Image
from metrics.metrics import process_folder_new, Metrics, GeometryMetrics, preprocess_image, preprocess_image_rgba
from torchvision.transforms.functional import to_tensor

def process_object_metrics_old(root_folder,  device="cuda"):
    original_folder = os.path.join(root_folder, "original")
    if not os.path.isdir(original_folder):
        raise ValueError(f"Original folder not found in {root_folder}")
    
    subfolders = [d for d in os.listdir(root_folder)
                  if os.path.isdir(os.path.join(root_folder, d)) 
                  and d.lower() not in ["original", "scaled"]]
    
    results = {}
    for sub in subfolders:
        generated_folder = os.path.join(root_folder, sub)
        print(f"Processing subfolder: {sub}")
        
        sem_metrics = process_folder_new(
            original_folder=original_folder,
            generated_folder=generated_folder,
            preprocess_func=preprocess_image,  
            metric_class=Metrics,
            device=device,  
            compute_distribution_metrics=False
        ) 
        geom_metrics = process_folder_new(
            original_folder=original_folder,
            generated_folder=generated_folder,
            preprocess_func=preprocess_image_rgba,  
            metric_class=GeometryMetrics,
            num_points=100
        )
        results[sub] = {
            "semantic_metrics": sem_metrics,
            "geometric_metrics": geom_metrics
        }
    return results

def convert_to_float(x):
    if torch.is_tensor(x):
        return x.item()
    try:
        return float(x)
    except Exception:
        return x
    
def process_metrics_by_viewpoint(ground_truth_folder, generated_folder, device="cuda"):
    """
    Processes metrics for a single generation run over a hierarchical folder structure.
    
    Expected folder structure:
      ground_truth_folder/
          <object_id_1>/
              000.png, 001.png, ... 011.png
          <object_id_2>/
              ...
      generated_folder/
          <object_id_1>/
              000.png, 001.png, ... 011.png
          <object_id_2>/
              ...

    For each viewpoint (e.g. "000.png"):
      1. Scans every object folder in both ground_truth_folder and generated_folder.
      2. Copies each matching file into temporary flat folders (one for ground truth, one for generated),
         renaming them with the object id as a prefix.
      3. Calls process_folder_new on these folders to compute detailed metrics for that viewpoint.
      4. Stores these per-viewpoint metric dictionaries.
      
    Finally, it computes an overall average over all viewpoints (averaging each metric key separately)
    and returns a dictionary structured as:
    
    {
      "per_viewpoint": {
          "000": { "semantic_metrics": { ... }, "geometric_metrics": { ... } },
          "001": { "semantic_metrics": { ... }, "geometric_metrics": { ... } },
          ...
      },
      "overall_semantic_metrics": { ... },
      "overall_geometric_metrics": { ... }
    }
    
    Temporary directories are automatically removed.
    """
    # 1. Determine all viewpoint filenames from ground truth.
    viewpoint_set = set()
    obj_ids = [d for d in os.listdir(ground_truth_folder) if os.path.isdir(os.path.join(ground_truth_folder, d))]
    for obj in obj_ids:
        obj_path = os.path.join(ground_truth_folder, obj)
        for file_path in glob.glob(os.path.join(obj_path, "*.png")):
            viewpoint_set.add(os.path.basename(file_path))
    if not viewpoint_set:
        raise ValueError("No viewpoint images found in the ground truth folder.")
    
    per_viewpoint = {}
    sem_metrics_list = []
    geom_metrics_list = []
    
    for vp in sorted(viewpoint_set):
        print(f"Processing viewpoint: {vp}")
        with tempfile.TemporaryDirectory() as orig_temp, tempfile.TemporaryDirectory() as gen_temp:
            for obj in obj_ids:
                gt_obj_dir = os.path.join(ground_truth_folder, obj)
                gen_obj_dir = os.path.join(generated_folder, obj)
                src_gt = os.path.join(gt_obj_dir, vp)
                src_gen = os.path.join(gen_obj_dir, vp) if os.path.isdir(gen_obj_dir) else None
                if not os.path.exists(src_gt):
                    print(f"Warning: In object '{obj}', ground truth image {vp} not found.")
                    continue
                if not (src_gen and os.path.exists(src_gen)):
                    print(f"Warning: In object '{obj}', generated image {vp} not found.")
                    continue
                dst_gt = os.path.join(orig_temp, f"{obj}_{vp}")
                dst_gen = os.path.join(gen_temp, f"{obj}_{vp}")
                shutil.copy2(src_gt, dst_gt)
                shutil.copy2(src_gen, dst_gen)
            
            temp_gt_files = glob.glob(os.path.join(orig_temp, "*.png"))
            temp_gen_files = glob.glob(os.path.join(gen_temp, "*.png"))
            if not temp_gt_files or not temp_gen_files:
                print(f"Skipping viewpoint {vp} due to insufficient images.")
                continue
            
            sem = process_folder_new(
                original_folder=orig_temp,
                generated_folder=gen_temp,
                preprocess_func=preprocess_image,
                metric_class=Metrics,
                device=device,
                compute_distribution_metrics=True
            )
            print(sem)
            geom = process_folder_new(
                original_folder=orig_temp,
                generated_folder=gen_temp,
                preprocess_func=preprocess_image_rgba,
                metric_class=GeometryMetrics,
                num_points=100
            )
            print(geom)
            
            vp_key = os.path.splitext(vp)[0]  
            per_viewpoint[vp_key] = {
                "semantic_metrics": sem,
                "geometric_metrics": geom
            }
            sem_metrics_list.append(sem)
            geom_metrics_list.append(geom)
    
    def average_metric_dicts(dict_list):
        averaged = {}
        keys = dict_list[0].keys()
        for k in keys:
            total = sum(convert_to_float(m[k]) for m in dict_list)
            averaged[k] = total / len(dict_list)
        return averaged

    overall_semantic = average_metric_dicts(sem_metrics_list) if sem_metrics_list else None
    overall_geometric = average_metric_dicts(geom_metrics_list) if geom_metrics_list else None

    """
    overall_metrics = {}
    if overall_semantic:
        overall_metrics.update(overall_semantic)
    if overall_geometric:
        overall_metrics.update(overall_geometric)
    """
    results = {
        "per_viewpoint": per_viewpoint,
        "overall_semantic_metrics": overall_semantic,
        "overall_geometric_metrics": overall_geometric,
    }
    return results

def tensor_to_serializable(obj):
    if isinstance(obj, torch.Tensor):
        return obj.item() if obj.ndim == 0 else obj.tolist()
    elif isinstance(obj, dict):
        return {k: tensor_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [tensor_to_serializable(i) for i in obj]
    else:
        return obj   

def json_file_to_combined_table(json_filepath):
    """
    Convert the JSON metrics file into a combined Pandas DataFrame.
    """
    with open(json_filepath, 'r') as f:
        data = json.load(f)
    
    combined_data = {}
    overall_sem = data.get("overall_semantic_metrics", {})
    overall_geom = data.get("overall_geometric_metrics", {})
    combined = {**overall_sem, **overall_geom}
    # Build a single-column DataFrame for the overall metrics.
    df = pd.DataFrame(combined, index=[0])
    return df