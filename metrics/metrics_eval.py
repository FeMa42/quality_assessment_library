import json
import os
import glob
import shutil
import tempfile
import torch
import pandas as pd

from metrics.metrics import Metrics, GeometryMetrics, CarQualityMetrics, compute_global_distribution_metrics
from metrics.helpers import process_folder, preprocess_image, preprocess_image_rgba


def tensor_to_serializable(obj):
    if isinstance(obj, torch.Tensor):
        return obj.item() if obj.ndim == 0 else obj.tolist()
    if isinstance(obj, dict):
        return {k: tensor_to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [tensor_to_serializable(v) for v in obj]
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
    df = pd.DataFrame(combined, index=[0])
    return df

def process_metrics_by_viewpoint(
    ground_truth_folder: str,
    generated_folder: str,
    device: str = "cuda",
    config_path: str = None,
    metadata_file_path: str = None,
):
    cfg = {}
    if config_path:
        with open(config_path, "r") as f:
            cfg = json.load(f)

    sem_cfg = cfg.get("semantic", {})
    semantic_enabled = sem_cfg.get("enabled", True)
    semantic_list = sem_cfg.get("metrics", None)

    dist_cfg = cfg.get("distribution", {})
    distribution_enabled = dist_cfg.get("enabled", True) and semantic_enabled
    distribution_list = dist_cfg.get("metrics", None)

    geom_cfg = cfg.get("geometric", {})
    geometric_enabled = geom_cfg.get("enabled", True)
    geometric_list = geom_cfg.get("metrics", None)

    car_cfg = cfg.get("car_quality", {})
    car_enabled = car_cfg.get("enabled", True)
    car_list = car_cfg.get("metrics", None)

    obj_ids = [
        d for d in os.listdir(ground_truth_folder)
        if os.path.isdir(os.path.join(ground_truth_folder, d))
    ]

    if metadata_file_path is not None:
        metadata_df = pd.read_csv(metadata_file_path) # metadata_df["sha256"] 
        metadata_df = metadata_df[metadata_df["sha256"].isin(obj_ids)]
        obj_ids = metadata_df["sha256"].tolist()
        if not obj_ids:
            raise ValueError("No valid object IDs found in the metadata file.")

    viewpoint_set = set()
    for obj in obj_ids:
        for fn in glob.glob(os.path.join(ground_truth_folder, obj, "*.png")):
            viewpoint_set.add(os.path.basename(fn))
    if not viewpoint_set:
        raise ValueError("No viewpoint images found in the ground truth folder.")

    per_viewpoint = {}
    sem_acc = []
    geom_acc = []
    car_acc = []

    car_metric = None
    if car_enabled:
        car_metric = CarQualityMetrics(
            device=device,
            metrics_list=car_list,
            )

    for vp in sorted(viewpoint_set):
        print(f"Processing viewpoint {vp}...")

        with tempfile.TemporaryDirectory() as gt_tmp, tempfile.TemporaryDirectory() as gen_tmp:
            for obj in obj_ids:
                src_gt = os.path.join(ground_truth_folder, obj, vp)
                src_gen = os.path.join(generated_folder,   obj, vp)
                if os.path.exists(src_gt) and os.path.exists(src_gen):
                    shutil.copy(src_gt, os.path.join(gt_tmp,  f"{obj}_{vp}"))
                    shutil.copy(src_gen, os.path.join(gen_tmp, f"{obj}_{vp}"))

            if not os.listdir(gt_tmp):
                continue

            entry = {}

            # semantic
            if semantic_enabled:
                sem = process_folder(
                    original_folder=gt_tmp,
                    generated_folder=gen_tmp,
                    preprocess_func=preprocess_image,
                    metric_class=Metrics,
                    device=device,
                    compute_distribution_metrics=distribution_enabled,
                    metric_list=semantic_list,
                    distribution_list=distribution_list,
                )
                sem_acc.append(sem)
                entry["semantic_metrics"] = sem

            # geometric
            if geometric_enabled:
                geom = process_folder(
                    original_folder=gt_tmp,
                    generated_folder=gen_tmp,
                    preprocess_func=preprocess_image_rgba,
                    metric_class=GeometryMetrics,
                    num_points=100,
                    metric_list=geometric_list,
                )
                geom_acc.append(geom)
                entry["geometric_metrics"] = geom


            # car quality
            if car_enabled:
                o = car_metric.compute_folder_metrics(gt_tmp)
                g = car_metric.compute_folder_metrics(gen_tmp)
                rel = {
                    k: None if o[k] == 0 else (g.get(k, 0) - o[k]) / o[k]
                    for k in o
                }
                flat = {}
                for k, v in o.items():
                    flat[f"orig_{k}"] = float(v)
                for k, v in g.items():
                    flat[f"gen_{k}"]  = float(v)
                for k, v in rel.items():
                    flat[f"rel_{k}"]  = None if v is None else float(v)
                car_acc.append(flat)
                entry["car_quality_metrics"] = {
                    "orig_score": o,
                    "gen_score":  g,
                    "rel_diff":   rel,
                }

            per_viewpoint[os.path.splitext(vp)[0]] = entry


    def avg(dl):
        if not dl:
            return None
        keys = dl[0].keys()
        return {k: sum(d[k] for d in dl) / len(dl) for k in keys}

    overall_semantic = avg(sem_acc)
    overall_geometric = avg(geom_acc)
    overall_car_quality = avg(car_acc)

    global_cfg     = cfg.get("global_distribution", {})
    global_enabled = global_cfg.get("enabled", False)
    global_list    = global_cfg.get("metrics", None)
    batch_size     = global_cfg.get("batch_size", 128)
    num_workers    = global_cfg.get("num_workers", 4)
    compute_on_cpu = global_cfg.get("compute_on_cpu", False)

    global_dist = None
    if global_enabled:
        print(f"[GLOBAL DISTRIBUTION] computing {global_list} "
              f"(batch={batch_size}, cpu_only={compute_on_cpu})")
        global_dist = compute_global_distribution_metrics(
            ground_truth_folder,
            generated_folder,
            global_list,
            device=device,
            batch_size=batch_size,
            num_workers=num_workers,
            compute_on_cpu=compute_on_cpu,
        )

    results = {"per_viewpoint": per_viewpoint}

    if semantic_enabled:
        results["overall_semantic_metrics"] = overall_semantic
    if geometric_enabled:
        results["overall_geometric_metrics"] = overall_geometric
    if car_enabled:
        results["overall_car_quality_metrics"] = overall_car_quality
    if global_enabled:
        results["global_distribution_metrics"] = global_dist

    return tensor_to_serializable(results)
