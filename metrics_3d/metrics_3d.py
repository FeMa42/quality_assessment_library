import os
from MeshMetrics.metrics import DistanceMetrics
from metrics_3d.helpers import (
    trimesh_to_vtk,
    safe_load_trimesh,
    estimate_volume_from_points,
    align_mesh,
    scale_mesh,
    CorruptedMeshError,
)
from tqdm.auto import tqdm
import numpy as np
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
from scipy.spatial import ConvexHull
import trimesh
import vtk
from typing import Tuple


class Metrics3D:
    """
    Compute a configurable subset of 3D mesh metrics using MeshMetrics' DistanceMetrics and MM_PCQA.
    """

    def __init__(
        self,
        metric_fr_list=None,
        metric_fr_pc_list=None,
        metric_nr_list=None,
        spacing=(1.0, 1.0, 1.0),
        nsd_tau=1.0,
        biou_tau=1.0,
        hd_percentile=95.0,
        pc_n_samples=10000,
        pc_n_sample_ratio_expensive=0.1,
        align=False,
        alignment_method="xy_plane_shortest_axis",  # or "longest_dimension"
        alignment_axis=0,
        normalize_mesh_scale=False,
        normalize_method="largest_dimension",  # or "largest_dimension"
        norm_scale=1.0,  # Scale factor for normalization
    ):
        self.metric_fr_list = metric_fr_list or None
        self.metric_nr_list = metric_nr_list or None
        self.metric_fr_pc_list = metric_fr_pc_list or None
        self.spacing = spacing
        self.nsd_tau = nsd_tau
        self.biou_tau = biou_tau
        self.hd_percentile = hd_percentile
        self.pc_n_samples = pc_n_samples
        self.pc_n_sample_ratio_expensive = pc_n_sample_ratio_expensive
        self.align = align
        self.alignment_method = alignment_method
        self.alignment_axis = alignment_axis
        self.normalize_mesh_scale = normalize_mesh_scale
        self.normalize_method = normalize_method
        self.norm_scale = norm_scale

        self.available_metrics = {
            # full reference metrics:
            "Hausdorff": self._hausdorff,
            "Hausdorff_Percentile": self._hausdorff_percentile,
            "MASD": self._masd,
            "ASSD": self._assd,
            "NSD": self._nsd,
            "BIoU": self._biou,
            # Point Cloud metrics (work with any mesh):
            "Chamfer_Distance": self._chamfer_distance,
            "Hausdorff_PC": self._hausdorff_pc,
            "Hausdorff_Percentile_PC": self._hausdorff_percentile_pc,
            "Point_to_Surface_RMSE": self._point_to_surface_rmse,
            "Earth_Mover_Distance": self._earth_mover_distance,
            "Convex_Hull_Volume_Difference": self._convex_hull_volume_difference,
            "Bounding_Box_Volume_Difference": self._bbox_volume_difference,
            "Point_Density_Volume_Difference": self._point_density_volume_difference,
            # no reference metrics:
            "MM_PCQA": self._MM_PCQA,  # Placeholder for MM_PCQA metric
        }

    def _prepare(
        self, pred_mesh_path: str, gt_mesh_path: str, logging: bool = True
    ) -> Tuple[
        vtk.vtkPolyData, vtk.vtkPolyData, trimesh.Trimesh, trimesh.Trimesh, bool
    ]:
        """
        Load and convert meshes to vtkPolyData for distance metrics computation.
        Applies preprocessing such as watertightness checks and mesh alignment if specified as well as normalization of mesh scale.

        Args:
            pred_mesh_path (str): Path to the predicted mesh file.
            gt_mesh_path (str): Path to the ground truth mesh file.
        Returns:
            tuple: A tuple containing the ground truth and predicted meshes as vtkPolyData and Trimesh, as well as a boolean depicting watertightness.
        """
        watertight = True

        # Safe load meshes using trimesh
        try:
            pred_trimesh = safe_load_trimesh(pred_mesh_path, logging=logging)
        except CorruptedMeshError as e:
            if logging:
                print(f"\t [Error] Predicted mesh corrupted: {e}")
            raise  # Re-raise to be caught in compute_mesh_pair
        if not pred_trimesh.is_watertight:
            watertight = False

        try:
            gt_trimesh = safe_load_trimesh(gt_mesh_path, logging=logging)
        except CorruptedMeshError as e:
            if logging:
                print(f"\t [Error] Ground Truth mesh corrupted: {e}")
            raise  # Re-raise to be caught in compute_mesh_pair

        if not gt_trimesh.is_watertight:
            watertight = False

        # rotate mesh to align to x-axis (0) or x-y plane if specified
        if self.align:
            pred_trimesh = align_mesh(
                self.alignment_method, pred_trimesh, axis=self.alignment_axis
            )
            gt_trimesh = align_mesh(
                self.alignment_method, gt_trimesh, axis=self.alignment_axis
            )

        # normalize mesh scale if specified
        if self.normalize_mesh_scale:
            pred_trimesh = scale_mesh(
                self.normalize_method,
                pred_trimesh,
                target_axis=self.alignment_axis,
                target_size=self.norm_scale,
            )
            gt_trimesh = scale_mesh(
                self.normalize_method,
                gt_trimesh,
                target_axis=self.alignment_axis,
                target_size=self.norm_scale,
            )

        # Convert to vtkPolyData for mesh based distance metrics
        pred_vtk = trimesh_to_vtk(pred_trimesh)
        gt_vtk = trimesh_to_vtk(gt_trimesh)

        return gt_vtk, pred_vtk, gt_trimesh, pred_trimesh, watertight

    def compute_mesh_pair(
        self, pred_mesh_path: str, gt_mesh_path: str, logging: bool = True
    ) -> tuple:
        """
        Compute metrics for a pair of meshes (predicted and ground truth).
        Args:
            pred_mesh_path (str): Path to the predicted mesh file.
            gt_mesh_path (str): Path to the ground truth mesh file.
            logging (bool): Whether to log the process.
        Returns:
            tuple: A tuple containing a dictionary with the computed metrics and if the metric caluclation was successful.
        """

        success = True
        results = {}

        # convert meshes to Trimesh and vtkPolyData
        try:
            gt_vtk, pred_vtk, gt_trimesh, pred_trimesh, watertight = self._prepare(
                pred_mesh_path, gt_mesh_path, logging=logging
            )

            # 1. Compute MeshMetrics (require watertight meshes)
            if self.metric_fr_list:
                if watertight:
                    try:
                        dm = DistanceMetrics()
                        dm.set_input(gt_vtk, pred_vtk, spacing=self.spacing)

                        for name in self.metric_fr_list:
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
                                    if logging:
                                        tqdm.write(
                                            f"[Metrics3D] Error computing {name}: {e}"
                                        )
                                    success = False
                    except Exception as e:
                        if logging:
                            tqdm.write(
                                f"\t [Error] MeshMetrics initialization failed: {e}"
                            )
                        # Fill all FR metrics with None
                        for name in self.metric_fr_list:
                            results[name] = None
                        success = False
                else:
                    # Set MeshMetrics to None if not watertight
                    for name in self.metric_fr_list:
                        if name in self.available_metrics:
                            results[name] = None
                    success = False

            # 2. Compute Point Cloud metrics (works with any mesh)
            if self.metric_fr_pc_list:
                try:
                    # Create point clouds from meshes using fixed seeds
                    np.random.seed(42)  # For reproducibility
                    pred_pc = pred_trimesh.sample(self.pc_n_samples)
                    np.random.seed(42)  # For reproducibility
                    gt_pc = gt_trimesh.sample(self.pc_n_samples)
                    np.random.seed(None)  # Reset seed

                    for name in self.metric_fr_pc_list:
                        if name in self.available_metrics:
                            try:
                                if name == "Hausdorff_Percentile_PC":
                                    results[name] = self.available_metrics[name](
                                        pred_pc, gt_pc, percentile=self.hd_percentile
                                    )
                                elif name == "Point_to_Surface_RMSE":
                                    results[name] = self.available_metrics[name](
                                        pred_trimesh, gt_trimesh
                                    )
                                elif name == "Earth_Mover_Distance":
                                    results[name] = self.available_metrics[name](
                                        pred_trimesh, gt_trimesh
                                    )
                                else:
                                    # Default Case for point cloud metrics
                                    results[name] = self.available_metrics[name](
                                        pred_pc, gt_pc
                                    )
                            except Exception as e:
                                results[name] = None
                                if logging:
                                    tqdm.write(
                                        f"[Metrics3D] Error computing {name}: {e}"
                                    )
                                success = False
                except Exception as e:
                    if logging:
                        tqdm.write(f"\t [Error] Point Cloud sampling failed: {e}")
                    # Fill all PC metrics with None
                    for name in self.metric_fr_pc_list:
                        results[name] = None
                    success = False

            # 3. Compute no-reference metrics TODO
            if self.metric_nr_list:
                if logging:
                    tqdm.write(
                        "[Metrics3D] No Reference Metrics currently not implemented"
                    )
                for name in self.metric_nr_list:
                    if name in self.available_metrics:
                        results[name] = None

        except (CorruptedMeshError, Exception) as e:
            if logging:
                print(f"\t [SKIPPED] Corrupted mesh detected: {e}")

            # Return empty results with failure flag
            success = False

            # Fill results with None values for all expected metrics
            if self.metric_fr_list:
                for name in self.metric_fr_list:
                    results[name] = None

            if self.metric_fr_pc_list:
                for name in self.metric_fr_pc_list:
                    results[name] = None

            if self.metric_nr_list:
                for name in self.metric_nr_list:
                    results[name] = None

        return results, success

    def compute_no_reference_metrics(self, mesh_path: str) -> tuple:
        """
        NOT IMPLEMENTED YET.
        Compute no-reference metrics for a single mesh.
        Args:
            mesh_path (str): Path to the mesh file.
        Returns:
            tuple: A tuple containing a dictionary with the computed metrics and if the metric caluclation was successful.
        """
        success = True
        results = {}
        if self.metric_nr_list is None or len(self.metric_nr_list) == 0:
            print(
                "[Metrics3D] No no-reference metrics specified, skipping computation."
            )
            success = False
            return results, success

        for name in self.metric_nr_list:
            if name in self.available_metrics:
                try:
                    results[name] = self.available_metrics[name](mesh_path)
                except Exception as e:
                    results[name] = None
                    print(f"[Metrics3D] Error computing {name} for {mesh_path}: {e}")
                    success = False
        return results, success

    ################################################################
    ################ Mesh Metric Functions #########################
    ################################################################

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
        Percentile Hausdorff Distance (e.g., HD_95) (MeshMetrics)
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

    ################################################################
    ################ Point Cloud Functions #########################
    ################################################################

    def _chamfer_distance(
        self, pred: Tuple[float, int], gt: Tuple[float, int]
    ) -> float:
        """
        Chamfer Distance using point cloud sampling

        - Measures the AVERAGE distance from each point in one cloud to the nearest point in the other cloud.
        - Sensitive to outliers (large errors get amplified).
        - Useful for evaluating the overall shape similarity between two point clouds.

        Output Interpretation:
        - Range: [0, ∞) - always non-negative
        - 0 = perfect match (predicted points lie exactly on GT points)
        - Lower values = better prediction quality
        - Higher values = worse prediction quality

        Args:
            pred (Tuple[float, int]): Predicted mesh points (sample_size, 3).
            gt (Tuple[float, int]): Ground truth mesh points (sample_size, 3).
        Returns:
            float: Chamfer distance between the predicted and ground truth point clouds.
        """

        dist_matrix = cdist(pred, gt)
        chamfer = np.mean(np.min(dist_matrix, axis=1)) + np.mean(
            np.min(dist_matrix, axis=0)
        )
        return chamfer / 2  # Average of both directions

    def _hausdorff_pc(self, pred: Tuple[float, int], gt: Tuple[float, int]) -> float:
        """
        Hausdorff Distance using point cloud sampling

        - Measures the MAXIMUM distance from each point in one cloud to the nearest point in the other cloud.
        - Sensitive to outliers (large errors get amplified).
        - Useful for evaluating the worst-case distance between two point clouds.

        Output Interpretation:
        - Range: [0, ∞) - always non-negative
        - 0 = perfect match (predicted points lie exactly on GT points)
        - Lower values = better prediction quality
        - Higher values = worse prediction quality
        - No percentile, always returns the maximum distance.

        Args:
            pred (Tuple[float, int]): Predicted mesh points (sample_size, 3).
            gt (Tuple[float, int]): Ground truth mesh points (sample_size, 3).
        Returns:
            float: Hausdorff distance between the predicted and ground truth point clouds.
        """
        dist_matrix = cdist(pred, gt)
        return max(
            np.max(np.min(dist_matrix, axis=1)), np.max(np.min(dist_matrix, axis=0))
        )

    def _hausdorff_percentile_pc(
        self, pred: Tuple[float, int], gt: Tuple[float, int], percentile: float = 95.0
    ) -> float:
        """
        Percentile Hausdorff Distance using point cloud sampling

        - Computes the Hausdorff distance at a specified percentile (e.g., 95th).
        - Reduces sensitivity to outliers or tiny spikes in the point cloud.
        - Useful for evaluating the worst-case distance at a given percentile.

        Output Interpretation:
        - Range: [0, ∞) - always non-negative
        - 0 = perfect match (predicted points lie exactly on GT points)
        - Lower values = better prediction quality
        - Higher values = worse prediction quality
        - Percentile determines how much of the worst-case distance is considered
          (e.g., 95% of distances are below this value).

        Args:
            pred (Tuple[float, int]): Predicted mesh points (sample_size, 3).
            gt (Tuple[float, int]): Ground truth mesh points (sample_size, 3).
            percentile (float): Percentile to compute the Hausdorff distance at.
        Returns:
            float: Hausdorff distance at the specified percentile between the predicted and ground truth point clouds
        """

        dist_matrix = cdist(pred, gt)
        dist1 = np.min(dist_matrix, axis=1)
        dist2 = np.min(dist_matrix, axis=0)
        return max(
            np.percentile(dist1, self.hd_percentile),
            np.percentile(dist2, self.hd_percentile),
        )

    def _point_to_surface_rmse(
        self, pred: trimesh.Trimesh, gt: trimesh.Trimesh
    ) -> float:
        """
        Point-to-surface RMSE

        - Samples points from predicted mesh surface
        - Finds nearest point on ground truth surface for each sample
        - Computes RMSE of all distances (asymmetric: pred → GT only)
        - Sensitive to outliers (large errors get amplified)
        - Measures how accurately prediction captures true surface

        Output Interpretation:
        - Range: [0, ∞) - always non-negative
        - 0 = perfect match (predicted points lie exactly on GT surface)
        - Lower values = better prediction quality
        - Higher values = worse prediction quality

        Args:
            pred (trimesh.Trimesh): Predicted mesh.
            gt (trimesh.Trimesh): Ground truth mesh.
        Returns:
            float: Root Mean Square Error (RMSE) of the distances from predicted points to the nearest point on the ground truth surface.
        """
        np.random.seed(42)  # For reproducibility
        pred_pc = pred.sample(
            int(self.pc_n_samples * self.pc_n_sample_ratio_expensive)
        )  # Smaller Sample Size for RMSE
        np.random.seed(None)  # Reset seed
        # distances = []
        # for point in pred_pc:
        #     _, distance, __ = gt.nearest.on_surface([point])
        #     distances.append(distance[0])
        _, distances, _ = gt.nearest.on_surface(pred_pc)
        return np.sqrt(np.mean(np.array(distances) ** 2))

    def _earth_mover_distance(
        self, pred: trimesh.Trimesh, gt: trimesh.Trimesh
    ) -> float:
        """
        Earth Mover's Distance (or Wasserstein Distance)

        - Samples equal numbers of points from both meshes
        - Finds optimal matching between point sets (min-cost assignment)
        - Computes average cost to transform one point cloud to another
        - Measures shape correspondence and deformation quality
        - Uses smaller sample size (1/10th) for computational efficiency (O(n³))

        Output Interpretation:
        - Range: [0, ∞) - always non-negative
        - 0 = perfect match (identical point clouds)
        - -Inf = computation failure (e.g., empty point cloud)
        - Lower values = better shape correspondence
        - Higher values = more deformation/shape difference needed
        - Return scale depends on the scale of the input meshes
        - Typical "good" values: < 10% of mesh bounding box diagonal
        - Returns -inf on computation failure

        Args:
            pred (trimesh.Trimesh): Predicted mesh.
            gt (trimesh.Trimesh): Ground truth mesh.
        Returns:
            float: Earth Mover's Distance between the predicted and ground truth point clouds.
        """
        try:
            # Sample equal numbers of points
            np.random.seed(42)  # For reproducibility
            pred_pc = pred.sample(
                int(self.pc_n_samples * self.pc_n_sample_ratio_expensive)
            )  # Smaller Sample Size for EMD
            np.random.seed(42)  # For reproducibility
            gt_pc = gt.sample(len(pred_pc))
            np.random.seed(None)  # Reset seed

            cost_matrix = cdist(pred_pc, gt_pc)
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            return cost_matrix[row_ind, col_ind].sum() / len(pred_pc)
        except Exception:
            return -np.inf  # EMD can be expensive/fail for large point clouds

    def _convex_hull_volume_difference(
        self, pred: Tuple[float, int], gt: Tuple[float, int]
    ) -> float:
        """
        Convex Hull Volume Difference (Point Cloud Based)

        - Samples points from both meshes
        - Computes convex hull volumes from point clouds
        - Good approximation for convex or nearly-convex objects

        Output Interpretation:
        - Range: [0, ∞) - always non-negative
        - 0 = identical convex hull volumes
        - -1.0 = error indicator (e.g., mesh loading failed)
        - Lower values = more similar overall size/shape
        - Higher values = significant size differences
        - Scale: relative difference (0.5 = 50% difference)

        Args:
            pred_mesh (trimesh.Trimesh): Predicted mesh
            gt_mesh (trimesh.Trimesh): Ground truth mesh
        Returns:
            float: Relative difference in convex hull volumes
        """
        try:
            # Compute convex hulls and their volumes

            pred_hull = ConvexHull(pred)
            gt_hull = ConvexHull(gt)

            pred_volume = pred_hull.volume
            gt_volume = gt_hull.volume

            if gt_volume == 0:
                return float("inf") if pred_volume > 0 else 0.0

            return abs(pred_volume - gt_volume) / gt_volume

        except Exception:
            return -1.0  # Error indicator

    def _bbox_volume_difference(
        self, pred: Tuple[float, int], gt: Tuple[float, int]
    ) -> float:
        """
        Bounding Box Volume Difference (Point Cloud Based)

        - Samples points from both meshes
        - Computes axis-aligned bounding box volumes
        - Fast and robust for size change detection
        - Works well for detecting scale/size differences

        Output Interpretation:
        - Range: [0, ∞) - always non-negative
        - 0 = identical bounding box volumes
        - 1.0 = 100% volume difference (e.g., doubled height)
        - -1.0 = error indicator (e.g., mesh loading failed)
        - Good for detecting overall size changes

        Args:
            pred_mesh (trimesh.Trimesh): Predicted mesh
            gt_mesh (trimesh.Trimesh): Ground truth mesh
        Returns:
            float: Relative difference in bounding box volumes
        """
        try:
            # Compute bounding boxes
            pred_min, pred_max = np.min(pred, axis=0), np.max(pred, axis=0)
            gt_min, gt_max = np.min(gt, axis=0), np.max(gt, axis=0)

            # Compute volumes
            pred_volume = np.prod(pred_max - pred_min)
            gt_volume = np.prod(gt_max - gt_min)

            if gt_volume == 0:
                return float("inf") if pred_volume > 0 else 0.0

            return abs(pred_volume - gt_volume) / gt_volume

        except Exception:
            return -1.0  # Error indicator

    def _point_density_volume_difference(
        self, pred: Tuple[float, int], gt: Tuple[float, int]
    ) -> float:
        """
        Point Density Volume Difference (Advanced PC Method)

        - Samples points uniformly from mesh surfaces
        - Estimates volume using point density in 3D grid
        - More accurate than bounding box, works with complex shapes
        - Good balance between accuracy and robustness

        Output Interpretation:
        - Range: [0, ∞) - always non-negative
        - 0 = similar estimated volumes
        - -1.0 = error indicator (e.g., mesh loading failed)
        - Values depend on voxel resolution and mesh complexity
        - Better for detecting actual volume changes vs just size

        Args:
            pred_mesh (trimesh.Trimesh): Predicted mesh
            gt_mesh (trimesh.Trimesh): Ground truth mesh
        Returns:
            float: Relative difference in estimated volumes
        """
        try:
            # Create 3D voxel grids for volume estimation
            pred_volume = estimate_volume_from_points(pred)
            gt_volume = estimate_volume_from_points(gt)

            if gt_volume == 0:
                return float("inf") if pred_volume > 0 else 0.0

            return abs(pred_volume - gt_volume) / gt_volume

        except Exception:
            return -1.0  # Error indicator

    ################################################################
    ################ No Reference Metric Functions #################
    ################################################################

    def _MM_PCQA(self, mesh_path: str) -> float:
        """
        NOT IMPLEMENTED YET.
        MM_PCQA
        - Placeholder for MM_PCQA metric, which is not implemented in this class.
        - This method can be extended to include the MM_PCQA metric if needed.

        Returns:
            float: Not implemented, returns None.
        """
        raise NotImplementedError("MM_PCQA metric is not implemented in Metrics3D.")


################################################################
################ Folder Processing Functions ###################
################################################################
def process_mesh_folder_fr(
    gt_folder: str, pred_folder: str, metric_class: Metrics3D, logging=True
) -> dict:
    """
    Process a folder of meshes, computing FULL REFERENCE metrics for each pair of ground truth and predicted meshes.
    Currently supports obj/glb/ply/stl files.
    Args:
        gt_folder (str): Path to the folder containing ground truth meshes.
        pred_folder (str): Path to the folder containing predicted meshes.
        metric_class: An instance of Metrics3D for computing metrics.
    Returns:
        dict: A dictionary mapping mesh filenames to their computed metrics.
    """
    results = {}
    mesh_files = [
        file
        for file in os.listdir(gt_folder)
        if file.endswith(".obj")
        or file.endswith(".glb")
        or file.endswith(".ply")
        or file.endswith(".stl")
    ]
    for file in tqdm(mesh_files, desc="Processing meshes", unit="file"):
        gt_path = os.path.join(gt_folder, file)
        pred_path = os.path.join(pred_folder, file)
        if os.path.exists(pred_path):
            try:
                if logging:
                    tqdm.write(f"Computing full reference metrics for: {file}")
                results[file], success = metric_class.compute_mesh_pair(
                    pred_path, gt_path, logging=logging
                )
                if logging:
                    tqdm.write(f"\t Metrics computation success: {success} for: {file}")
            except AssertionError as e:
                if logging:
                    tqdm.write(f"\t Assertion error for {file}: {e}")
    return results


def process_mesh_folder_nr(
    mesh_folder: str, metric_class: Metrics3D, logging=True
) -> dict:
    """
    NOT IMPLEMENTED YET.
    Process a folder of meshes, computing NO REFERENCE metrics for each mesh.
    Currently supports obj/glb/ply/stl files.
    Args:
        mesh_folder (str): Path to the folder containing meshes.
        metric_class: An instance of Metrics3D for computing metrics.
    Returns:
        dict: A dictionary mapping mesh filenames to their computed metrics.
    """
    results = {}
    # check if any no-reference metrics are specified
    if metric_class.metric_nr_list is None or len(metric_class.metric_nr_list) == 0:
        print(
            "[Metrics3D] No no-reference metrics specified in config file, skipping computation."
        )
        success = False
        return results

    mesh_files = [
        file
        for file in os.listdir(mesh_folder)
        if file.endswith(".obj")
        or file.endswith(".glb")
        or file.endswith(".ply")
        or file.endswith(".stl")
    ]
    for file in tqdm(mesh_files, desc="Processing meshes", unit="file"):
        mesh_path = os.path.join(mesh_folder, file)
        if os.path.exists(mesh_path):
            try:
                if logging:
                    tqdm.write(f"Computing no-reference metrics for: {file}")
                results[file], success = metric_class.compute_no_reference_metrics(
                    mesh_path
                )
                if logging:
                    tqdm.write(f"\t Metrics computation success: {success} for: {file}")
            except AssertionError as e:
                if logging:
                    tqdm.write(f"\t Assertion error for {file}: {e}")
    return results
