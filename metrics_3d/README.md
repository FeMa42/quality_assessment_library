# `metrics_3d` Module

A comprehensive Python library for evaluating 3D mesh quality using various geometric metrics. This module provides full-reference metrics for comparing ground truth and generated 3D meshes.

## 📋 Table of Contents

1. [Supported File Formats](#1-supported-file-formats)
2. [Quick Start](#2-quick-start)
3. [Ideal Workflow](#3-ideal-workflow)
4. [Configuration](#4-configuration)
5. [Usage Examples](#5-usage-examples)
6. [Available Metrics](#6-available-metrics)
7. [Metric Interpretation Guide](#7-metric-interpretation-guide)
8. [3D Mesh Preprocessing](#8-3d-mesh-preprocessing)
9. [Troubleshooting](#9-troubleshooting)

## 1. Supported File Formats

- `.obj` - Wavefront OBJ
- `.stl` - STereoLithography
- `.ply` - Polygon File Format
- `.glb` - Binary glTF

## 2. Quick Start

```python
from metrics_3d.metrics_eval import process_3d_metrics

# Evaluate 3D meshes
results = process_3d_metrics(
    ground_truth_folder="./ground_truth/",
    generated_folder="./generated/",
    config_path="./config.json",
    logging=True
)

print("3D Metrics Results:", results)
```

## 3. Ideal Workflow

1. **Preprocess Ground Truth Meshes**: Use `preprocessing_3d` to standardize and normalize meshes.
2. **Generate New Meshes**: Create new meshes using your generation pipeline.
3. **Evaluate Metrics**: Use `metrics_eval` to compute metrics between ground truth and generated meshes.

## 4. Configuration

### Basic Configuration File

Create a `config.json` file with the following structure for 3D metrics:

```json
{
  "metrics_3d": {
    "enabled": true,
    
    "fr_metrics": [
      "Hausdorff",
      "Hausdorff_Percentile", 
      "MASD",
      "ASSD",
      "NSD",
      "BIoU"
    ],
    
    "fr_pc_metrics": [
      "Chamfer_Distance",
      "Hausdorff_PC",
      "Hausdorff_Percentile_PC",
      "Convex_Hull_Volume_Difference",
      "Bounding_Box_Volume_Difference",
      "Point_Density_Volume_Difference"
    ],
    
    "nr_metrics": [],
    
    "nsd_tau": 1.0,
    "biou_tau": 1.0,
    "Hausdorff_Percentile": 95,
    "pc_n_samples": 10000,
    "pc_sample_ratio_expensive": 0.1,

    "align": false,
    "alignment_method": "overlap",
    "alignment_axis": 0,
    "normalize_mesh_scale": true,
    "normalize_method": "largest_dimension",
    "norm_scale": 1.0
  }
}
```

### Configuration Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `enabled` | bool | `true` | Enable/disable 3D metrics computation |
| `fr_metrics` | list | `[]` | Full-reference surface metrics (requires watertight meshes) |
| `fr_pc_metrics` | list | `[]` | Full-reference point cloud metrics (works with any mesh) |
| `nr_metrics` | list | `[]` | No-reference metrics (not implemented yet) |
| `spacing` | list | `[1.0, 1.0, 1.0]` | Voxel spacing for surface metrics |
| `nsd_tau` | float | `1.0` | Tolerance threshold for NSD metric |
| `biou_tau` | float | `1.0` | Tolerance threshold for BIoU metric |
| `Hausdorff_Percentile` | int | `95` | Percentile for Hausdorff distance |
| `pc_n_samples` | int | `10000` | Number of points to sample for point cloud metrics |
| `pc_n_sample_ratio_expensive` | float | `0.1` | Ratio of points to sample for expensive metrics like EMD and RMSE (compared to pc_n_samples) |
| `align` | bool | `false` | Apply mesh alignment "postprocessing" |
| `alignment_method` | string | `"overlap"` | Method for mesh alignment |
| `alignment_axis` | int | `0` | Axis to align to, if the method is not plane based (0 for x-axis, 1 for y-axis, 2 for z-axis) |
| `normalize_mesh_scale` | bool | `true` | Apply mesh scaling "postprocessing" |
| `normalize_method` | string | `"largest_dimension"` | Method for mesh scaling |
| `norm_scale` | float | `1.0` | Size of the unit cube to scale meshes to |

### Alignment Methods

By default both the reference and the ground truth objects are aligned to have the biggest overlap possible. This works great for our use case of car models, but might most definetely not work on other kinds of models.
For other cases it is might be better to align the calculated axis to the global x-axis or the xy-plane depending on the method used.
This can be changed by providing an axis value in the `alignment_axis` parameter.
0 for x-axis, 1 for y-axis, and 2 for z-axis.

⚠️ **WARNING:**

Automatic alignment will NOT work for all meshes as well as for the meshes we tested it on (mainly cars) -> always check the results visually and try different methods if needed!

- `"overlap"` - Align both models to maximize overlap
- `"longest_dimension"` - Align to longest axis-aligned dimension
- `"pca"` - Align using Principal Component Analysis
- `"longest_oriented_dimension"` - Align to longest oriented bounding box dimension
- `"xy_plane_shortest_axis"` - Align shortest axis to Z (objects lie flat)
- `"xy_plane_pca"` - Align to XY plane using PCA

### Normalization Methods

By default the meshes are scaled to fit within a unit cube.
This can be changed by providing a different value for the `norm_scale` parameter.

- `"largest_oriented_dimension"` - Scale by largest oriented bounding box dimension
- `"largest_dimension"` - Scale by largest axis-aligned dimension  
- `"axis"` - Scale by specific axis dimension

## 5. Usage Examples

### Basic Evaluation

```python
import os
from metrics_3d.metrics_eval import process_3d_metrics

# Set up paths
ground_truth_folder = os.path.abspath("./example_data/metrics_3d/Ground_Truth")
generated_folder = os.path.abspath("./example_data/metrics_3d/Comparison")
config_path = os.path.abspath("./config.json")

# Run evaluation
results = process_3d_metrics(
    ground_truth_folder=ground_truth_folder,
    generated_folder=generated_folder,
    config_path=config_path,
    logging=True
)

# Save results
import json
with open("metrics_3d_results.json", "w") as f:
    json.dump(results, f, indent=2)
```

## 6. Available Metrics

### Full-Reference Surface Metrics (`fr_metrics`)

Requires both meshes to be ``watertight``!

| Metric | Description | Range | Best Value |
|--------|-------------|-------|------------|
| `Hausdorff` | Maximum distance between surfaces | [0, ∞) | 0 |
| `Hausdorff_Percentile` | Percentile Hausdorff distance (e.g., 95th) | [0, ∞) | 0 |
| `MASD` | Mean Average Surface Distance | [0, ∞) | 0 |
| `ASSD` | Average Symmetric Surface Distance | [0, ∞) | 0 |
| `NSD` | Normalized Surface Dice | [0, 1] | 1 |
| `BIoU` | Boundary Intersection over Union | [0, 1] | 1 |

### Full-Reference Point Cloud Metrics (`fr_pc_metrics`)

Works with any mesh

| Metric | Description | Range | Best Value |
|--------|-------------|-------|------------|
| `Chamfer_Distance` | Average nearest-neighbor distance | [0, ∞) | 0 |
| `Hausdorff_PC` | Maximum distance between point clouds | [0, ∞) | 0 |
| `Hausdorff_Percentile_PC` | Percentile Hausdorff for point clouds | [0, ∞) | 0 |
| `Earth_Movers_Distance` | **⚠️** Optimal transport distance | [0, ∞) | 0 |
| `Point_to_Surface_RMSE` | **⚠️** Root mean square surface distance | [0, ∞) | 0 |
| `Convex_Hull_Volume_Difference` | Relative convex hull volume difference | [0, ∞) | 0 |
| `Bounding_Box_Volume_Difference` | Relative bounding box volume difference | [0, ∞) | 0 |
| `Point_Density_Volume_Difference` | Estimated volume difference using point density | [0, ∞) | 0 |

**⚠️ Performance Warning:**

 `Earth_Movers_Distance` and `Point_to_Surface_RMSE` are computationally expensive metrics. For large meshes (~100MB+), these metrics can significantly increase computation time and memory usage. Consider excluding them for large-scale evaluations or reduce sample size of those 2 methods in your configuration by changing the `pc_n_sample_ratio_expensive` parameter. (Default is 0.1, meaning 10% of `pc_n_samples` will be used for these metrics).

## 7. Metric Interpretation Guide

### Distance-Based Metrics

Lower values indicate better similarity

| Metric | Best For | Use Case | Interpretation |
|--------|----------|----------|----------------|
| ``Hausdorff`` | Detecting worst-case differences | Quality control, outlier detection | Maximum distance between surfaces - sensitive to single bad points |
| ``Hausdorff_Percentile`` | Robust shape comparison | General shape similarity assessment | 95th percentile distance - ignores outliers, more stable than Hausdorff |
| ``MASD`` | Overall surface accuracy | Fine detail preservation evaluation | Mean distance - good for average surface fidelity |
| ``ASSD`` | Symmetric shape comparison | Bidirectional shape similarity | Average of both directions - balanced view of shape differences |
| ``Chamfer_Distance`` | Point cloud comparison | Non-watertight mesh evaluation | Nearest-neighbor distance - works with any mesh topology |
| `Earth_Movers_Distance` | **⚠️** Global shape similarity | High-precision shape analysis | Optimal transport distance - considers global point distribution |
| `Point_to_Surface_RMSE` | **⚠️** Surface accuracy | Precise surface fidelity measurement | RMS distance from points to nearest surface |

### Overlap-Based Metrics

Higher values indicate better overlap

| Metric | Best For | Use Case | Interpretation |
|--------|----------|----------|----------------|
| ``NSD`` | Volume-based similarity | Medical imaging, solid objects | **Normalized Surface Dice**: Measures how much of the surface area overlaps within a tolerance distance (`nsd_tau`).|
| ``BIoU`` | Boundary accuracy | Edge preservation, fine details | **Boundary Intersection over Union**: Calculates the ratio of overlapping surface area to total surface area within a tolerance (`biou_tau`). Similar to IoU for 2D images but applied to 3D surfaces.|

### Volume-Based Metrics

Lower values indicate more similar volumes

| Metric | Best For | Use Case | Interpretation |
|--------|----------|----------|----------------|
| ``Convex_Hull_Volume_Difference`` | Overall shape similarity | Coarse shape comparison | Relative difference in convex hulls - captures overall form |
| ``Bounding_Box_Volume_Difference`` | Size consistency | Scale validation, preprocessing check | Relative difference in axis-aligned bounding boxes |
| ``Point_Density_Volume_Difference`` | Estimated volume comparison | Non-watertight volume estimation | Volume estimated from point cloud density |

### Error Values

- **`-1.0`**: Computation failed (e.g., mesh loading error, invalid geometry)
- **`None`**: Metric not computed (e.g., disabled in config, incompatible mesh)
- **`inf` or very large values**: Extreme geometric differences or numerical issues

## 8. 3D Mesh Preprocessing

The `preprocessing_3d` module provides tools for standardizing meshes before the generation process.
All available preprocessing methods are explained in more detail in the [configuration parameters](#configuration-parameters) section under ``Alignment Methods`` and ``Normalization Methods``.
The file itself is referenced [here](./preprocessing_3d.py).

**⚠️** `overlap` can not be used for alignment in at the preprocessing stage, as there have to be two meshes to compare and overlap.

### Basic Preprocessing

```python
from metrics_3d.preprocessing_3d import process_ground_truth_folder

# Process folder without subfolders
results = process_ground_truth_folder(
    input_folder="./raw_meshes/",
    output_folder="./processed_meshes/",
    normalize_scale=True,
    center=True,
    logging=True
)

print(f"Processed: {results['processed']} meshes")
print(f"Failed: {results['failed']} meshes")
```

### Advanced Preprocessing with Alignment

```python
# Process with alignment and custom scaling
results = process_ground_truth_folder(
    input_folder="./raw_meshes/",
    output_folder="./processed_meshes/",
    center=True,
    alignment_method="xy_plane_shortest_axis",
    normalize_scale=True,
    normalizing_method="largest_oriented_dimension",
    normalizing_size=1.0,
    logging=True
)
```

### Processing Folder Structure with Subfolders

```python
# For folder structure:
# ground_truth_parent/
#     <object_id_1>/
#         object_id_1.glb, 000.png, 001.png, ..., 011.png
#     <object_id_2>/
#         ...

results = process_ground_truth_folder(
    input_folder="./ground_truth_parent/",
    output_folder="./processed_output/",
    normalize_scale=True,
    center=True,
    has_subfolders=True,
    logging=True
)
```

### Single Mesh Processing

```python
from metrics_3d.preprocessing_3d import process_single_mesh

success = process_single_mesh(
    mesh_path="./input/model.obj",
    output_path="./output/model_processed.obj",
    center=True,
    normalize_scale=True,
    normalizing_method="largest_oriented_dimension",
    normalizing_size=1.0
)

print(f"Processing successful: {success}")
```

## 9. Troubleshooting

### Common Issues

#### 1. **Mesh Loading Failures**

```python
# Error: "Mesh has zero extents (degenerate)"
# Solution: Check if mesh contains valid geometry
from metrics_3d.helpers import safe_load_trimesh

try:
    mesh = safe_load_trimesh("problematic_mesh.obj", logging=True)
    print(f"Vertices: {len(mesh.vertices)}, Faces: {len(mesh.faces)}")
except Exception as e:
    print(f"Loading failed: {e}")
```

#### 2. **Watertightness Issues**

```python
# Check mesh watertightness
mesh = safe_load_trimesh("mesh.obj")
print(f"Watertight: {mesh.is_watertight}")

if not mesh.is_watertight:
    print("⚠️ Mesh is not watertight - surface metrics may fail")
    print("💡 Use point cloud metrics instead")
```

A watertight mesh is `required` for full-reference surface metrics. If the mesh is not watertight, consider using point cloud metrics instead or always compute point cloud based metrics.

#### 3. **Memory Issues with Large Meshes**

```python
# Reduce point cloud sample size for large meshes
config = {
    "metrics_3d": {
        "pc_n_samples": 5000,  # Reduced from 10000
        # ... other settings
    }
}
```

### Performance Tips

1. **Reduce `pc_n_samples`** in the config for faster computation (beware of accuracy loss)
2. **Preprocess meshes** before generating similar meshes to ensure consistency

---

For implemented examples and advanced usage, see the provided Jupyter notebooks:

- [`metrics_3d_example.ipynb`](../metrics_3d_example.ipynb) - Complete evaluation workflow with example data and configuration
- [`data_3d_preprocessing.ipynb`](../data_3d_preprocessing.ipynb) - Mesh preprocessing examples including normalization, centering, and folder structure handling

## References

- [MeshMetrics Github](https://github.com/gasperpodobnik/MeshMetrics) / [Mesh Metrics Paper](https://arxiv.org/abs/2410.02630) - Provides detailed explanations of the metrics and their applications
