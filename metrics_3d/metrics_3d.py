import os
from MeshMetrics.metrics import DistanceMetrics
from metrics_3d.helpers import (
    trimesh_to_vtk,
    safe_load_trimesh,
    load_trimesh_to_obj,
    load_obj_with_vtk,
)
import vtk
import trimesh


class MeshMetrics3D:
    """
    Compute a configurable subset of 3D mesh metrics using MeshMetrics' DistanceMetrics.
    """

    def __init__(
        self,
        metric_list=None,
        spacing=(1.0, 1.0, 1.0),
        nsd_tau=1.0,
        biou_tau=1.0,
        hd_percentile=95.0,
    ):
        # Default to all MeshMetrics metrics if not specified
        self.metric_list = metric_list or [
            "Hausdorff",
            "Hausdorff_Percentile",
            "MASD",
            "ASSD",
            "NSD",
            "BIoU",
        ]
        self.spacing = spacing
        self.nsd_tau = nsd_tau
        self.biou_tau = biou_tau
        self.hd_percentile = hd_percentile
        self.available_metrics = {
            "Hausdorff": self._hausdorff,
            "Hausdorff_Percentile": self._hausdorff_percentile,
            "MASD": self._masd,
            "ASSD": self._assd,
            "NSD": self._nsd,
            "BIoU": self._biou,
        }

    def _prepare(self, pred_mesh_path: str, gt_mesh_path: str):
        """
        Load and convert meshes to vtkPolyData for distance metrics computation.
        Args:
            pred_mesh_path (str): Path to the predicted mesh file.
            gt_mesh_path (str): Path to the ground truth mesh file.
        Returns:
            tuple: A tuple containing the ground truth and predicted meshes as vtkPolyData.
        """
        pred = safe_load_trimesh(pred_mesh_path)
        gt = safe_load_trimesh(gt_mesh_path)
        pred_vtk = trimesh_to_vtk(pred)
        gt_vtk = trimesh_to_vtk(gt)
        # pred_obj_path = load_trimesh_to_obj(pred_mesh_path)
        # gt_obj_path = load_trimesh_to_obj(gt_mesh_path)
        # pred_vtk = load_obj_with_vtk(pred_obj_path)
        # gt_vtk = load_obj_with_vtk(gt_obj_path)
        return gt_vtk, pred_vtk

    def compute_mesh_pair(self, pred_mesh_path: str, gt_mesh_path: str):
        """
        Compute metrics for a pair of meshes (predicted and ground truth).
        Args:
            pred_mesh_path (str): Path to the predicted mesh file.
            gt_mesh_path (str): Path to the ground truth mesh file.
        Returns:
            dict: A dictionary containing the computed metrics.
        """
        gt, pred = self._prepare(pred_mesh_path, gt_mesh_path)
        dm = DistanceMetrics()
        dm.set_input(gt, pred, spacing=self.spacing)
        results = {}
        for name in self.metric_list:
            if name in self.available_metrics:
                try:
                    if name == "Hausdorff_Percentile":
                        results[name] = self.available_metrics[name](
                            dm, percentile=self.hd_percentile
                        )
                    else:
                        results[name] = self.available_metrics[name](dm)
                except Exception as e:
                    results[name] = None
                    print(
                        f"[MeshMetrics3D] Error computing {name} for {pred_mesh_path}: {e}"
                    )
        return results

    # MeshMetrics3D methods for each metric
    def _hausdorff(self, dm: DistanceMetrics) -> float:
        """
        Hausdorff Distance (MeshMetrics)
        - Measures the largest geometric error between two surfaces.
        - For every point on Mesh A, finds the nearest point on Mesh B (and vice versa).
        - The symmetric Hausdorff is the single largest gap anywhere between the two surfaces.
        - Useful for detecting the largest geometric error anywhere on the model.
        - Does not indicate the location of the error, only its size.

        Returns:
            float: the maximum Hausdorff distance/ deviation between the two meshes.
        """
        return dm.hd(percentile=100.0)

    def _hausdorff_percentile(
        self, dm: DistanceMetrics, percentile: float = 95.0
    ) -> float:
        """
        Percentile Hausdorff Distance (e.g., HD_95)
        - Same as Hausdorff, but returns the distance at a given percentile (e.g., 95th).
        - Reduces sensitivity to outliers or tiny spikes.

        Returns:
            float: the Hausdorff distance at the specified percentile.
        """
        return dm.hd(percentile=percentile)

    def _masd(self, dm: DistanceMetrics) -> float:
        """
        Mean Average Surface Distance (MASD, MeshMetrics)
        - Computes two one-way average distances:
            1. From every point on the reference surface to its nearest point on the test.
            2. From every point on the test surface to its nearest point on the reference.
        - Represents the average surface deviation across the entire mesh.
        - Gives equal weight to each direction, regardless of vertex count.
        - MASD and ASSD are equal if vertex counts are the same.

        Returns:
            float: the mean average surface distance between the two meshes.
        """
        return dm.masd()

    def _assd(self, dm: DistanceMetrics) -> float:
        """
        Average Symmetric Surface Distance (ASSD, MeshMetrics)
        - Pools all one-way point-to-surface distances (both directions) into a single set, then computes the mean.
        - Weights each individual sample point equally (proportional to vertex count).
        - Useful for detecting widespread surface deviations.
        - MASD and ASSD are equal if vertex counts are the same.

        Returns:
            float: the average symmetric surface distance between the two meshes.
        """
        return dm.assd()

    def _nsd(self, dm: DistanceMetrics) -> float:
        """
        Normalized Surface Dice (NSD, MeshMetrics)
        - Boundary-overlap (surface-Dice) metric.
        - Evaluates what fraction of the surfaces lies under a tolerance τ of the other mesh’s surface.
        - Counts how many distances fall below tolerance τ from each side, returns percentage within τ.
        - Useful for verifying how many gaps between test and reference are below τ (e.g., 3mm), as a percentage.
        - τ has to be specified in the scale of the mesh (e.g., 1.0 for 1mm).

        Returns:
            float: the normalized surface Dice score between the two meshes.
        """
        return dm.nsd(tau=self.nsd_tau)

    def _biou(self, dm: DistanceMetrics) -> float:
        """
        Boundary IoU (BIoU, MeshMetrics)
        - Boundary-overlap (Intersection-over-Union) metric.
        - Quantifies how well the boundary regions of two meshes overlap within tolerance τ.
        - Similar to NSD but stricter: counts intersection once, divides by union of both bands.
        - Penalizes extra points in either band more heavily.
        - Ideal when any extra or missing boundary points beyond τ should meaningfully lower your score.
        - τ has to be specified in the scale of the mesh (e.g., 1.0 for 1mm).

        Returns:
            float: the boundary IoU score between the two meshes.
        """
        return dm.biou(tau=self.biou_tau)


def process_mesh_folder(gt_folder: str, pred_folder: str, metric_class) -> dict:
    """
    Process a folder of meshes, computing metrics for each pair of ground truth and predicted meshes.
    Currently supports .obj and .glb files.
    Args:
        gt_folder (str): Path to the folder containing ground truth meshes.
        pred_folder (str): Path to the folder containing predicted meshes.
        metric_class: An instance of MeshMetrics3D or similar class for computing metrics.
    Returns:
        dict: A dictionary mapping mesh filenames to their computed metrics.
    """
    results = {}
    for file in os.listdir(gt_folder):
        if file.endswith(".obj") or file.endswith(".glb"):
            gt_path = os.path.join(gt_folder, file)
            pred_path = os.path.join(pred_folder, file)
            if os.path.exists(pred_path):
                gt_mesh = safe_load_trimesh(gt_path)
                if not (gt_mesh.is_watertight and gt_mesh.is_winding_consistent):
                    print(
                        f"Skipping {file}: not a closed/manifold mesh. Watertight={gt_mesh.is_watertight}, Winding Consistent={gt_mesh.is_winding_consistent}"
                    )
                    continue
                # gt_mesh = load_trimesh_to_obj(gt_path)
                # gt_trimesh = trimesh.load_mesh(gt_mesh)
                # if (
                #     not gt_trimesh.is_watertight
                #     and not gt_trimesh.is_winding_consistent
                # ):
                #     print(f"Skipping {file}: not a watertight mesh.")
                #     continue
                try:
                    print("Computing metrics for:", file)
                    results[file] = metric_class.compute_mesh_pair(pred_path, gt_path)
                    print("Successfully computed metrics for:", file)
                except AssertionError as e:
                    print(f"Skipping {file}: {e}")
    return results
