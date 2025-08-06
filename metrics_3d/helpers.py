import vtk
import trimesh
import numpy as np
from scipy.spatial.transform import Rotation


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
    align_method: str, mesh: trimesh.Trimesh, axis: int = 0
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
    normalizing_method: str,
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
    if normalizing_method == "largest_oriented_dimension":
        # Scale to unit length along largest dimension of the rotated bounding box
        mesh = scale_by_largest_oriented_dimension(mesh, target_size=target_size)
    elif normalizing_method == "largest_dimension":
        # Scale to unit length along largest dimension
        mesh = scale_by_largest_dimension(mesh, target_size=target_size)
    elif normalizing_method == "axis":
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
