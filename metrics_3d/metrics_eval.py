import json
from metrics_3d.metrics_3d import MeshMetrics3D, process_mesh_folder


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
    metrics_3d_list = metrics_3d_cfg.get("metrics", None)
    metrics_3d_spacing = metrics_3d_cfg.get("spacing", [1.0, 1.0, 1.0])
    metrics_3d_nsd_tau = metrics_3d_cfg.get("nsd_tau", 1.0)
    metrics_3d_biou_tau = metrics_3d_cfg.get("biou_tau", 1.0)
    metrics_3d_hd_percentile = metrics_3d_cfg.get("hd_percentile", 95.0)

    if not metrics_3d_enabled:
        raise ValueError("3D metrics are not enabled in the config.")

    # Instantiate metric class
    mesh_metrics_3d = MeshMetrics3D(
        metric_list=metrics_3d_list,
        spacing=metrics_3d_spacing,
        nsd_tau=metrics_3d_nsd_tau,
        biou_tau=metrics_3d_biou_tau,
        hd_percentile=metrics_3d_hd_percentile,
    )

    # Compute metrics for all mesh pairs
    results_per_object = process_mesh_folder(
        ground_truth_folder, generated_folder, mesh_metrics_3d, logging=logging
    )

    # Aggregate (average) results
    all_metrics = list(results_per_object.values())
    if all_metrics:
        keys = all_metrics[0].keys()
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
