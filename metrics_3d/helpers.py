import vtk
import trimesh
import numpy as np
from scipy.spatial.transform import Rotation
from scipy.spatial import cKDTree
from typing import Optional


class CorruptedMeshError(Exception):
    """Exception raised when a mesh file is corrupted or invalid."""

    pass


def safe_load_trimesh(mesh_path: str, logging: bool = True) -> trimesh.Trimesh:
    """
    Load a mesh file using trimesh, ensuring it is a valid Trimesh object.
    Args:
        mesh_path (str): Path to the mesh file.
    Returns:
        trimesh.Trimesh: A valid Trimesh object.
    Raises:
        ValueError: If the loaded mesh is not a valid Trimesh object.
        CorruptedMeshError: If the mesh file is corrupted or invalid.
    """
    try:
        mesh = trimesh.load(mesh_path)
    except Exception as e:
        raise CorruptedMeshError(f"Failed to load mesh file {mesh_path}: {e}")

    if isinstance(mesh, trimesh.Scene):
        try:
            mesh = mesh.to_mesh()
        except Exception as e:
            raise CorruptedMeshError(
                f"Failed to convert scene to mesh for {mesh_path}: {e}"
            )

    if not isinstance(mesh, trimesh.Trimesh):
        raise ValueError(f"File {mesh_path} did not yield a Trimesh object.")

    # Validation checks - raise CorruptedMeshError for corrupted files

    if mesh.vertices is None or len(mesh.vertices) == 0:
        raise CorruptedMeshError(f"Mesh has no vertices: {mesh_path}")

    if mesh.faces is None or len(mesh.faces) == 0:
        raise CorruptedMeshError(f"Mesh has no faces: {mesh_path}")

    # Check for degenerate mesh (all vertices at same point)
    if mesh.extents.max() < 1e-8:
        raise CorruptedMeshError(f"Mesh has zero extents (degenerate): {mesh_path}")

    # Validating Face Indices BEFORE any operations are done
    max_vertex_index = len(mesh.vertices) - 1
    max_face_index = np.max(mesh.faces) if len(mesh.faces) > 0 else -1

    if max_face_index > max_vertex_index:
        raise CorruptedMeshError(
            f"Invalid face indices: max face index {max_face_index} "
            f"exceeds vertex count {len(mesh.vertices)} in {mesh_path}"
        )

    if np.any(mesh.faces < 0):
        raise CorruptedMeshError(f"Negative face indices found in {mesh_path}")

    if mesh.visual is not None:
        mesh.visual.uv = None  # Strip UVs if they exist, since it almost always causes watertightness issues

    # Ensure the mesh is watertight and clean (might not work for all meshes)
    if not mesh.is_watertight:
        if logging:
            print(
                f"\t [Warning] {mesh_path}: Mesh is not watertight, attempting to clean it."
            )

        mesh.update_faces(mesh.nondegenerate_faces())
        mesh.remove_unreferenced_vertices()
        mesh.merge_vertices()
        watertight = mesh.fill_holes()
        if logging:
            print("\t watertight after cleaning:", watertight)

    # Check the mesh once more after cleaning
    if len(mesh.vertices) == 0:
        if logging:
            print(
                f"\t [Warning] {mesh_path}: Mesh has no valid vertices after cleaning."
            )
        raise CorruptedMeshError(
            f"Most vertices seemed to be at the same point -> Mesh has no valid vertices after cleaning: {mesh_path}"
        )

    return mesh


def trimesh_to_vtk(mesh: trimesh.Trimesh) -> vtk.vtkPolyData:
    """
    Convert a trimesh.Trimesh object to vtkPolyData.
    Args:
        mesh (trimesh.Trimesh): The Trimesh object to convert.
    Returns:
        vtk.vtkPolyData: The converted mesh as vtkPolyData.
    """
    points = vtk.vtkPoints()
    for v in mesh.vertices:
        points.InsertNextPoint(float(v[0]), float(v[1]), float(v[2]))

    polys = vtk.vtkCellArray()
    for face in mesh.faces:
        polys.InsertNextCell(3)
        polys.InsertCellPoint(int(face[0]))
        polys.InsertCellPoint(int(face[1]))
        polys.InsertCellPoint(int(face[2]))

    polydata = vtk.vtkPolyData()
    polydata.SetPoints(points)
    polydata.SetPolys(polys)
    return polydata


def center_mesh(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """
    Center a trimesh object to its center of mass.
    Args:
        mesh (trimesh.Trimesh): The mesh to center.
    Returns:
        trimesh.Trimesh: The centered mesh.
    """
    centered_vertices = mesh.vertices - mesh.center_mass
    centered_mesh = mesh.copy()
    centered_mesh.vertices = centered_vertices
    return centered_mesh


def estimate_volume_from_points(points: tuple[float, int], voxel_size=0.1):
    # Find bounding box
    min_coords = np.min(points, axis=0)

    # Assign points to voxels
    voxel_coords = ((points - min_coords) / voxel_size).astype(int)

    # Count unique occupied voxels
    unique_voxels = np.unique(voxel_coords, axis=0)
    occupied_volume = len(unique_voxels) * (voxel_size**3)

    return occupied_volume


# General Mesh alignment and scaling methods
def align_mesh(
    align_method: str,
    mesh: trimesh.Trimesh,
    axis: int = 0,
) -> trimesh.Trimesh:
    """
    Align meshes based on the specified alignment method.
    Args:
        mesh (trimesh.Trimesh): Mesh to align.
        target_axis (int): Target axis for alignment (0=x, 1=y, 2=z).
    Returns:
        trimesh.Trimesh: Aligned mesh.
    """
    if align_method == "longest_dimension":
        # Align to longest dimension
        mesh = align_mesh_to_axis_by_longest_dimension(mesh, target_axis=axis)
    elif align_method == "pca":
        # Align using PCA
        mesh = align_mesh_to_axis_pca(mesh, target_axis=axis)
    elif align_method == "longest_oriented_dimension":
        # Align to longest oriented dimension
        mesh = align_mesh_to_axis_by_longest_oriented_dimension(mesh, target_axis=axis)
    elif align_method == "xy_plane_shortest_axis":
        # Align to XY plane using shortest axis
        mesh = align_mesh_to_xy_plane_shortest_axis(mesh)
    elif align_method == "xy_plane_pca":
        # Align to XY plane using PCA
        mesh = align_mesh_to_xy_plane_pca(mesh)
    else:
        raise ValueError(f"Unknown allignment method: {align_method}")

    return mesh


def scale_mesh(
    normalize_method: str,
    mesh: trimesh.Trimesh,
    target_size: float = 1.0,
    target_axis: int = 0,
) -> trimesh.Trimesh:
    """
    Scale meshes based on the specified scaling method.
    Args:
        mesh (trimesh.Trimesh): Mesh to scale.
        target_axis (int): Target axis for scaling (0=x, 1=y, 2=z).
    Returns:
        trimesh.Trimesh: Scaled mesh.
    """
    if normalize_method == "largest_oriented_dimension":
        # Scale to unit length along largest dimension of the rotated bounding box
        mesh = scale_by_largest_oriented_dimension(mesh, target_size=target_size)
    elif normalize_method == "largest_dimension":
        # Scale to unit length along largest dimension
        mesh = scale_by_largest_dimension(mesh, target_size=target_size)
    elif normalize_method == "axis":
        mesh = scale_by_axis(mesh, axis=target_axis, target_size=target_size)
    else:
        # No scaling applied
        pass
    return mesh


################################################################
################ Scaling Functions #############################
################################################################


def scale_by_axis(
    mesh: trimesh.Trimesh, axis: int = 0, target_size: float = 1.0
) -> trimesh.Trimesh:
    """
    Scale a trimesh object along a specified axis to fit within a target size.
    Args:
        mesh (trimesh.Trimesh): The mesh to scale.
        axis (int): The axis to scale along (0 for x, 1 for y, 2 for z).
        target_size (float): The target size for the specified axis.
    Returns:
        trimesh.Trimesh: The scaled mesh.
    """
    # Calculate the length of the specified axis
    axis_length = mesh.extents[axis]

    # Calculate the scaling factor
    scale_factor = target_size / axis_length

    # Scale the mesh
    scaled_mesh = mesh.copy()
    scaled_mesh.apply_scale(scale_factor)

    return scaled_mesh


def scale_by_largest_dimension(
    mesh: trimesh.Trimesh, target_size: float = 1.0
) -> trimesh.Trimesh:
    """
    Scale a trimesh object to fit within a target size based on its largest dimension.
    Args:
        mesh (trimesh.Trimesh): The mesh to scale.
        target_size (float): The target size for the largest dimension.
    Returns:
        trimesh.Trimesh: The scaled mesh.
    """
    # Calculate the largest dimension
    max_dimension = np.max(mesh.extents)
    print(f"Max dimension: {max_dimension}")

    # Calculate the scaling factor
    scale_factor = target_size / max_dimension
    print(f"Scale factor: {scale_factor}")

    # Scale the mesh
    scaled_mesh = mesh.copy()
    scaled_mesh.apply_scale(scale_factor)

    return scaled_mesh


def scale_by_largest_oriented_dimension(
    mesh: trimesh.Trimesh, target_size: float = 1.0
) -> trimesh.Trimesh:
    """
    Scale a trimesh object to fit within a target size based on its largest oriented dimension.
    Args:
        mesh (trimesh.Trimesh): The mesh to scale.
        target_size (float): The target size for the largest oriented dimension.
    Returns:
        trimesh.Trimesh: The scaled mesh.
    """
    # Get the oriented bounding box
    bbox = mesh.bounding_box_oriented

    # Calculate the largest oriented dimension
    max_dimension = np.max(bbox.primitive.extents)

    # Calculate the scaling factor
    scale_factor = target_size / max_dimension

    # Scale the mesh
    scaled_mesh = mesh.copy()
    scaled_mesh.apply_scale(scale_factor)

    return scaled_mesh


################################################################
################ Alignment Functions ###########################
################################################################


def align_mesh_to_axis_by_longest_oriented_dimension(
    mesh: trimesh.Trimesh, target_axis: int = 0
) -> trimesh.Trimesh:
    """Align a trimesh object to a target axis based on its longest oriented dimension.
    Args:
        mesh (trimesh.Trimesh): The mesh to align.
        target_axis (int): The target axis (0 for x, 1 for y, 2 for z).
    Returns:
        trimesh.Trimesh: The aligned mesh.
    """
    if target_axis < 0 or target_axis > 2:
        raise ValueError("target_axis must be 0, 1, or 2.")

    # Get the extents of the oriented mesh as the object might not be axis-aligned
    bbox = mesh.bounding_box_oriented
    bbox_extents = bbox.primitive.extents
    longest_axis_index = np.argmax(bbox_extents)

    # Get the direction vector of the longest axis in mesh coordinates
    longest_axis_vector = bbox.primitive.transform[:3, longest_axis_index]
    longest_axis_vector = longest_axis_vector / np.linalg.norm(longest_axis_vector)

    # create target vector from specified target axis
    target_vector = np.eye(3)[:, target_axis]

    # Compute rotation matrix to align with the target axis
    if np.allclose(longest_axis_vector, target_vector):
        R = np.eye(3)
    elif np.allclose(longest_axis_vector, -target_vector):
        R = -np.eye(3)
    else:
        # Compute rotation axis and angle
        rotation_axis = np.cross(longest_axis_vector, target_vector)
        rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)
        angle = np.arccos(np.clip(np.dot(longest_axis_vector, target_vector), -1, 1))

        # Create rotation object
        rot = Rotation.from_rotvec(angle * rotation_axis)
        R = rot.as_matrix()

    # Center mesh for rotation
    centered_vertices = mesh.vertices - mesh.center_mass
    aligned_vertices = centered_vertices @ R.T
    aligned_mesh = mesh.copy()
    aligned_mesh.vertices = aligned_vertices + mesh.center_mass

    return aligned_mesh


def align_mesh_to_axis_by_longest_dimension(
    mesh: trimesh.Trimesh, target_axis: int = 0
) -> trimesh.Trimesh:
    """
    Align a trimesh object to a target axis based on its longest dimension.
    Args:
        mesh (trimesh.Trimesh): The mesh to align.
        target_axis (int): The target axis (0 for x, 1 for y, 2 for z).
    Returns:
        trimesh.Trimesh: The aligned mesh.
    """
    # Get the extents of the mesh
    extents = mesh.extents

    # Find the index of the longest axis
    longest_axis_index = np.argmax(extents)

    # Get the direction vector of the longest axis in mesh coordinates
    bbox = mesh.bounding_box_oriented
    axis_vector = bbox.primitive.transform[:3, longest_axis_index]
    axis_vector = axis_vector / np.linalg.norm(axis_vector)

    # Target axis vector (e.g., [0,0,1] for Z)
    target_vector = np.eye(3)[:, target_axis]

    # Rotation matrix calculation
    if np.allclose(axis_vector, target_vector):
        R = np.eye(3)
    elif np.allclose(axis_vector, -target_vector):
        R = -np.eye(3)
    else:
        # Compute rotation axis and angle
        rotation_axis = np.cross(axis_vector, target_vector)
        rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)
        angle = np.arccos(np.clip(np.dot(axis_vector, target_vector), -1, 1))

        # Create rotation object
        rot = Rotation.from_rotvec(angle * rotation_axis)
        R = rot.as_matrix()

    # Center mesh for rotation
    centered_vertices = mesh.vertices - mesh.center_mass
    aligned_vertices = centered_vertices @ R.T
    aligned_mesh = mesh.copy()
    aligned_mesh.vertices = aligned_vertices + mesh.center_mass

    return aligned_mesh


def align_mesh_to_axis_pca(
    mesh: trimesh.Trimesh, target_axis: int = 0
) -> trimesh.Trimesh:
    """
    Align a trimesh object to a target axis using PCA.
    Args:
        mesh (trimesh.Trimesh): The mesh to align.
        target_axis (int): The target axis (0 for x, 1 for y, 2 for z).
    Returns:
        trimesh.Trimesh: The aligned mesh.
    """
    # Center and compute PCA
    centered_vertices = mesh.vertices - mesh.center_mass
    cov = np.cov(centered_vertices.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    principal_axis = eigvecs[:, np.argmax(eigvals)]

    # Target axis
    target_vector = np.eye(3)[:, target_axis]

    # Rotation matrix calculation
    if np.allclose(principal_axis, target_vector):
        R = np.eye(3)
    elif np.allclose(principal_axis, -target_vector):
        R = -np.eye(3)
    else:
        # Compute rotation axis and angle
        rotation_axis = np.cross(principal_axis, target_vector)
        rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)
        angle = np.arccos(np.clip(np.dot(principal_axis, target_vector), -1, 1))

        # Create rotation object
        rot = Rotation.from_rotvec(angle * rotation_axis)
        R = rot.as_matrix()

    # Apply rotation
    aligned_vertices = centered_vertices @ R.T
    aligned_mesh = mesh.copy()
    aligned_mesh.vertices = aligned_vertices + mesh.center_mass

    return aligned_mesh


def align_mesh_to_xy_plane_shortest_axis(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """
    Align mesh to XY plane by making the shortest axis point along Z.
    This makes the object lie "flat" with its thinnest dimension vertical.
    """
    # Get oriented bounding box
    bbox = mesh.bounding_box_oriented
    extents = bbox.primitive.extents

    # Find the shortest axis (this should become Z)
    shortest_axis_index = np.argmin(extents)

    # Get the direction vector of the shortest axis
    shortest_axis_vector = bbox.primitive.transform[:3, shortest_axis_index]
    shortest_axis_vector = shortest_axis_vector / np.linalg.norm(shortest_axis_vector)

    # Target: align shortest axis to Z
    target_vector = np.array([0, 0, 1])

    # Compute rotation
    if np.allclose(shortest_axis_vector, target_vector, atol=1e-6):
        R = np.eye(3)
    elif np.allclose(shortest_axis_vector, -target_vector, atol=1e-6):
        R = -np.eye(3)
    else:
        rotation_axis = np.cross(shortest_axis_vector, target_vector)
        rotation_axis_norm = np.linalg.norm(rotation_axis)
        if rotation_axis_norm < 1e-8:
            R = np.eye(3)  # Already aligned
        else:
            rotation_axis = rotation_axis / rotation_axis_norm
            angle = np.arccos(
                np.clip(np.dot(shortest_axis_vector, target_vector), -1, 1)
            )
            rot = Rotation.from_rotvec(angle * rotation_axis)
            R = rot.as_matrix()

    # Apply rotation
    centered_vertices = mesh.vertices - mesh.center_mass
    aligned_vertices = centered_vertices @ R.T
    aligned_mesh = mesh.copy()
    aligned_mesh.vertices = aligned_vertices + mesh.center_mass

    return aligned_mesh


def align_mesh_to_xy_plane_pca(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """
    Align mesh to XY plane using PCA - align smallest principal component to Z.
    Good for objects with clear directional variance.
    """
    # Center and compute PCA
    centered_vertices = mesh.vertices - mesh.center_mass
    cov = np.cov(centered_vertices.T)
    eigvals, eigvecs = np.linalg.eigh(cov)

    # Smallest eigenvalue corresponds to the direction of least variance
    # (this should become Z to make object lie flat)
    smallest_pc = eigvecs[:, np.argmin(eigvals)]

    # Target: align smallest PC to Z
    target_vector = np.array([0, 0, 1])

    # Compute rotation
    if np.allclose(smallest_pc, target_vector, atol=1e-6):
        R = np.eye(3)
    elif np.allclose(smallest_pc, -target_vector, atol=1e-6):
        R = -np.eye(3)
    else:
        rotation_axis = np.cross(smallest_pc, target_vector)
        rotation_axis_norm = np.linalg.norm(rotation_axis)
        if rotation_axis_norm < 1e-8:
            R = np.eye(3)
        else:
            rotation_axis = rotation_axis / rotation_axis_norm
            angle = np.arccos(np.clip(np.dot(smallest_pc, target_vector), -1, 1))
            rot = Rotation.from_rotvec(angle * rotation_axis)
            R = rot.as_matrix()

    # Apply rotation
    aligned_vertices = centered_vertices @ R.T
    aligned_mesh = mesh.copy()
    aligned_mesh.vertices = aligned_vertices + mesh.center_mass

    return aligned_mesh


def align_with_eachother_overlap(
    mesh1: trimesh.Trimesh,
    mesh2: trimesh.Trimesh,
    n_samples: int = 50000,
    max_iters: int = 50,
    tol: float = 1e-6,
    reject_percentile: float = 95.0,
    pca_init: bool = True,
    return_transform: bool = False,
) -> tuple[trimesh.Trimesh, trimesh.Trimesh, Optional[np.ndarray]]:
    """Align mesh2 onto mesh1 with point-to-point ICP (optionally PCA-initialised).

    Steps:
        1. Sample up to n_samples surface points from each mesh (area-weighted).
        2. (Optional but recommended) PCA orientation disambiguation by enumerating right-handed
           sign combinations; pick lowest mean NN distance to reduce 180° flips.
        3. Iterate ICP:
            a. Nearest neighbour matching (moving -> reference)
            b. Optional robust trimming via reject_percentile
            c. Solve Kabsch (SVD) for incremental (R_inc, t_inc)
            d. Accumulate transform; update working points
            e. Early stop if RMSE improvement < tol
        4. Apply final accumulated transform to full-resolution moving mesh.

    Args:
        mesh1: Reference (fixed) mesh.
        mesh2: Moving mesh to align.
        n_samples: Surface samples per mesh for ICP correspondence.
        max_iters: Max ICP iterations.
        tol: Minimum absolute RMSE improvement to continue.
        reject_percentile: Keep matches with distance <= percentile (100 disables).
        pca_init: Use PCA-based coarse orientation search before ICP.
        return_transform: Return 4x4 homogeneous transform if True.

    Returns:
        (ref_mesh, aligned_moving_mesh, 4x4_transform)
    """
    # --- Validation -------------------------------------------------------
    if len(mesh1.vertices) == 0 or len(mesh2.vertices) == 0:
        raise ValueError("Meshes must have vertices.")

    # Copy to avoid mutating originals
    ref_mesh = mesh1.copy()
    mov_mesh = mesh2.copy()

    # rng = np.random.default_rng(random_seed)
    # np.random.seed(random_seed)

    # Sample points
    P_ref = ref_mesh.sample(min(n_samples, len(ref_mesh.vertices)))
    P_mov_original = mov_mesh.sample(min(n_samples, len(mov_mesh.vertices)))

    # Center both (translation handled separately)
    c_ref = P_ref.mean(axis=0)  # type: ignore
    c_mov = P_mov_original.mean(axis=0)  # type: ignore
    P_ref_c = P_ref - c_ref
    P_mov_c = P_mov_original - c_mov

    # Optional PCA init (align principal axes)
    if pca_init:

        def pca_frame(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            cov = np.cov(points.T)
            vals, vecs = np.linalg.eigh(cov)
            # Sort eigenvectors: largest variance first
            order = np.argsort(vals)[::-1]
            vals = vals[order]
            vecs = vecs[:, order]
            # Enforce right-handed frame (no reflection)
            if np.linalg.det(vecs) < 0:
                vecs[:, -1] *= -1
            return vals, vecs

        # Principal frames for reference and moving sets
        vals_ref, F_ref = pca_frame(P_ref_c)
        vals_mov, F_mov_raw = pca_frame(P_mov_c)

        # Generate all right-handed sign combinations for moving frame
        sign_options = []
        for sx in (-1, 1):
            for sy in (-1, 1):
                for sz in (-1, 1):
                    S = np.diag([sx, sy, sz])
                    # Keep only right-handed combinations
                    if np.linalg.det(F_mov_raw @ S) > 0:
                        sign_options.append(S)

        # Build KD-tree once on centered reference for quick distance tests
        tree_ref = cKDTree(P_ref_c)
        # Subsample moving points for speed in orientation scoring
        subsample = min(5000, len(P_mov_c))
        mov_idx = np.random.choice(len(P_mov_c), subsample, replace=False)

        best_err = np.inf
        best_R0 = None

        # Score each sign combination: apply candidate rotation and measure mean NN distance
        for S in sign_options:
            F_mov = F_mov_raw @ S
            # Rotation that maps moving PCA frame to reference PCA frame
            R_test = F_ref @ F_mov.T
            # Rotate subset of moving points
            P_mov_test = (R_test @ P_mov_c[mov_idx].T).T
            # Nearest neighbour distances in reference set
            dists, _ = tree_ref.query(P_mov_test, k=1)
            err = dists.mean()
            if err < best_err:
                best_err = err
                best_R0 = R_test

        # Selected coarse rotation
        R0 = best_R0 if best_R0 is not None else np.eye(3)

    else:
        R0 = np.eye(3)

    t0 = c_ref - R0 @ c_mov

    # Apply initial transform to moving samples
    P_mov = (R0 @ P_mov_original.T).T + t0  # type: ignore

    # Accumulate transform
    R_acc = R0.copy()
    t_acc = t0.copy()

    prev_rmse = np.inf

    tree = cKDTree(P_ref)

    # --- ICP refinement loop ---------------------------------------------
    for it in range(max_iters):
        # 1. Find closest reference point for every transformed moving point
        dists, idx = tree.query(P_mov, k=1)
        # 2. Optional robust trimming: discard largest distance matches
        if reject_percentile < 100:
            thresh = np.percentile(dists, reject_percentile)
            mask = dists <= thresh
            Q_ref = P_ref[idx[mask]]  # matched reference points
            Q_mov = P_mov[mask]  # corresponding moving points
            d_use = dists[mask]  # distances used for RMSE
        else:
            Q_ref = P_ref[idx]
            Q_mov = P_mov
            d_use = dists

        # Safety: need enough correspondences to solve
        if len(Q_ref) < 10:
            break

        # 3. Compute centroids of the matched subsets
        mu_ref = Q_ref.mean(axis=0)
        mu_mov = Q_mov.mean(axis=0)

        # 4. Form centered correlation matrices
        X = Q_mov - mu_mov
        Y = Q_ref - mu_ref

        # 5. Kabsch: optimal rotation minimizing ||R*X - Y||
        H = X.T @ Y
        U, S, Vt = np.linalg.svd(H)
        R_inc = Vt.T @ U.T
        if np.linalg.det(R_inc) < 0:
            Vt[-1, :] *= -1
            R_inc = Vt.T @ U.T

        # 6. Incremental translation
        t_inc = mu_ref - R_inc @ mu_mov

        # 7. Update accumulated transform (R_acc * newR, t_acc + new)
        R_acc = R_inc @ R_acc
        t_acc = R_inc @ t_acc + t_inc

        # 8. Apply to original moving samples for next iter
        P_mov = (R_inc @ P_mov.T).T + t_inc

        # 9. Compute RMSE of (kept) correspondences (point-to-point)
        rmse = np.sqrt((d_use**2).mean())

        # 10. Early stop if improvement below tolerance
        if prev_rmse - rmse < tol:
            break
        prev_rmse = rmse

    # --- Final application to full-resolution moving mesh ----------------
    mov_vertices = (R_acc @ mov_mesh.vertices.T).T + t_acc
    aligned_mov = mov_mesh.copy()
    aligned_mov.vertices = mov_vertices

    # Optional homogeneous 4x4 transform return
    if return_transform:
        T = np.eye(4)
        T[:3, :3] = R_acc
        T[:3, 3] = t_acc
        return ref_mesh, aligned_mov, T

    return ref_mesh, aligned_mov, None
