import json
from metrics_3d.metrics_3d import (
    Metrics3D,
    process_mesh_folder_fr,
    process_mesh_folder_nr,
)


def process_3d_metrics(
    ground_truth_folder: str,
    generated_folder: str,
    config_path: str,
    logging: bool = True,
):
    # Load config
    cfg = {}
    if config_path:
        with open(config_path, "r") as f:
            cfg = json.load(f)

    metrics_3d_cfg = cfg.get("metrics_3d", {})
    metrics_3d_enabled = metrics_3d_cfg.get("enabled", True)
    metrics_3d_fr_list = metrics_3d_cfg.get("fr_metrics", None)
    metrics_3d_nr_list = metrics_3d_cfg.get("nr_metrics", None)
    metrics_3d_spacing = metrics_3d_cfg.get("spacing", [1.0, 1.0, 1.0])
    metrics_3d_nsd_tau = metrics_3d_cfg.get("nsd_tau", 1.0)
    metrics_3d_biou_tau = metrics_3d_cfg.get("biou_tau", 1.0)
    metrics_3d_hd_percentile = metrics_3d_cfg.get("hd_percentile", 95.0)

    if not metrics_3d_enabled:
        raise ValueError("3D metrics are not enabled in the config.")

    # Instantiate metric class
    metrics_3D = Metrics3D(
        metric_fr_list=metrics_3d_fr_list,
        metric_nr_list=metrics_3d_nr_list,
        spacing=metrics_3d_spacing,
        nsd_tau=metrics_3d_nsd_tau,
        biou_tau=metrics_3d_biou_tau,
        hd_percentile=metrics_3d_hd_percentile,
    )

    # Compute full reference metrics for all mesh pairs
    fr_results_per_object = process_mesh_folder_fr(
        ground_truth_folder, generated_folder, metrics_3D, logging=logging
    )

    # TODO: Compute no reference metrics for all meshes in folder
    nr_results_per_object = process_mesh_folder_nr(
        generated_folder, metrics_3D, logging=logging
    )

    # Combine full reference and no reference results
    results_per_object = fr_results_per_object.copy()
    for obj, nr_metrics in nr_results_per_object.items():
        if obj in results_per_object:
            results_per_object[obj].update(nr_metrics)
        else:
            results_per_object[obj] = nr_metrics

    # Aggregate (average) results
    all_metrics = list(results_per_object.values())
    if all_metrics:
        keys = all_metrics[0].keys()
        # For each metric key, compute the mean across all objects,
        # skipping None values (which indicate missing or failed metrics).
        # If all values are None for a metric, the denominator is set to 1 to avoid division by zero.
        overall = {
            k: sum(d[k] for d in all_metrics if d[k] is not None)
            / max(1, sum(1 for d in all_metrics if d[k] is not None))
            for k in keys
        }
    else:
        overall = {}

    results = {
        "per_object": results_per_object,
        "overall_3d_metrics": overall,
    }
    return results
