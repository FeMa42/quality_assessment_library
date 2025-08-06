import os

from typing import Optional, List
from tqdm.auto import tqdm
from .helpers import (
    safe_load_trimesh,
    scale_mesh,
    align_mesh,
    center_mesh,
)


def process_single_mesh(
    mesh_path: str,
    output_path: str,
    center: bool = True,
    alignment_method: Optional[str] = None,
    target_axis: int = 0,
    normalize_scale: bool = True,
    normalizing_method: str = "largest_oriented_dimension",
    normalizing_size: float = 1.0,
    logging: bool = True,
) -> bool:
    """
    Process a single mesh file with scaling and alignment.

    Args:
        mesh_path (str): Path to input mesh file
        output_path (str): Path to save processed mesh
        center (bool): Whether to center the mesh
        alignment_method (str): Alignment method or None
        target_axis (int): Target axis for alignment (0=x, 1=y, 2=z)
        normalize_scale (bool): Whether to normalize mesh scale
        normalizing_method (str): Method for scaling (e.g., "largest_oriented_dimension")
        normalizing_size (float): Target size for scaling
        logging (bool): Enable logging

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        if logging:
            print(f"Processing: {mesh_path}")

        # Load mesh
        mesh = safe_load_trimesh(mesh_path, logging=logging)

        if center:
            mesh = center_mesh(mesh)

        # Apply alignment if specified
        if alignment_method:
            mesh = align_mesh(alignment_method, mesh, target_axis)

        # Apply scaling if specified
        if normalize_scale:
            mesh = scale_mesh(
                normalizing_method,
                mesh,
                target_size=normalizing_size,
                target_axis=target_axis,
            )

        # Create output directory
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Save processed mesh
        mesh.export(output_path)

        if logging:
            print(f"Saved to: {output_path}")

        return True

    except Exception as e:
        if logging:
            print(f"Error processing {mesh_path}: {e}")
        return False


def get_mesh_files(folder_path: str) -> List[str]:
    """
    Get all mesh files from a folder.

    Args:
        folder_path (str): Path to folder

    Returns:
        List[str]: List of mesh file paths
    """
    mesh_extensions = [".obj", ".stl", ".ply", ".glb"]
    mesh_files = []

    for file in os.listdir(folder_path):
        if any(file.lower().endswith(ext) for ext in mesh_extensions):
            mesh_files.append(os.path.join(folder_path, file))

    return mesh_files


def process_folder_no_subfolders(
    input_folder: str,
    output_folder: str,
    center: bool = True,
    alignment_method: Optional[str] = None,
    target_axis: int = 0,
    normalize_scale: bool = True,
    normalizing_method: str = "largest_oriented_dimension",
    normalizing_size: float = 1.0,
    logging: bool = True,
) -> dict:
    """
    Process all mesh files in a folder (no subfolders).

    Args:
        input_folder (str): Path to input folder
        output_folder (str): Path to output folder
        alignment_method (str): Alignment method or None
        target_axis (int): Target axis for alignment (0=x, 1=y, 2=z)
        normalize_scale (bool): Whether to normalize mesh scale
        normalizing_method (str): Method for scaling (e.g., "largest_oriented_dimension")
        normalizing_size (float): Target size for scaling
        logging (bool): Enable logging

    Returns:
        dict: Results with success/failure counts
    """
    if not os.path.exists(input_folder):
        raise ValueError(f"Input folder does not exist: {input_folder}")

    os.makedirs(output_folder, exist_ok=True)

    mesh_files = get_mesh_files(input_folder)

    if not mesh_files:
        if logging:
            print(f"No mesh files found in {input_folder}")
        return {"processed": 0, "failed": 0, "files": []}

    if logging:
        print(f"Found {len(mesh_files)} mesh files to process")

    results = {"processed": 0, "failed": 0, "files": []}

    for mesh_file in tqdm(mesh_files, desc="Processing meshes", unit="file"):
        filename = os.path.basename(mesh_file)
        output_path = os.path.join(output_folder, filename)

        success = process_single_mesh(
            mesh_file,
            output_path,
            center,
            alignment_method,
            target_axis,
            normalize_scale,
            normalizing_method,
            normalizing_size,
            logging,
        )

        if success:
            results["processed"] += 1
        else:
            results["failed"] += 1

        results["files"].append({
            "input": mesh_file,
            "output": output_path,
            "success": success,
        })

    if logging:
        print("\nProcessing complete:")
        print(f"Processed: {results['processed']}")
        print(f"Failed: {results['failed']}")

    return results


def process_folder_with_subfolders(
    input_folder: str,
    output_folder: str,
    center: bool = True,
    alignment_method: Optional[str] = None,
    target_axis: int = 0,
    normalize_scale: bool = True,
    normalizing_method: str = "largest_oriented_dimension",
    normalizing_size: float = 1.0,
    logging: bool = True,
) -> dict:
    """
    Process mesh files in a folder structure with subfolders.

    Structure:
        input_folder/
            <object_id_1>/
                object_id_1.glb/obj/stl/ply, 000.png, 001.png, ..., 011.png
            <object_id_2>/
                ...

    Args:
        input_folder (str): Path to input folder
        output_folder (str): Path to output folder
        center (bool): Whether to center the mesh
        alignment_method (str): Alignment method or None
        target_axis (int): Target axis for alignment (0=x, 1=y, 2=z)
        normalize_scale (bool): Whether to normalize mesh scale
        normalizing_method (str): Method for scaling (e.g., "largest_oriented_dimension")
        normalizing_size (float): Target size for scaling
        logging (bool): Enable logging

    Returns:
        dict: Results with success/failure counts per subfolder
    """
    if not os.path.exists(input_folder):
        raise ValueError(f"Input folder does not exist: {input_folder}")

    os.makedirs(output_folder, exist_ok=True)

    # Get all subfolders
    subfolders = [
        f
        for f in os.listdir(input_folder)
        if os.path.isdir(os.path.join(input_folder, f))
    ]

    if not subfolders:
        if logging:
            print(f"No subfolders found in {input_folder}")
        return {"processed": 0, "failed": 0, "subfolders": {}}

    if logging:
        print(f"Found {len(subfolders)} subfolders to process")

    total_results = {"processed": 0, "failed": 0, "subfolders": {}}

    for subfolder in tqdm(subfolders, desc="Processing subfolders", unit="subfolder"):
        if logging:
            print(f"\nProcessing subfolder: {subfolder}")

        input_subfolder = os.path.join(input_folder, subfolder)
        output_subfolder = os.path.join(output_folder, subfolder)

        # Process this subfolder
        subfolder_results = process_folder_no_subfolders(
            input_subfolder,
            output_subfolder,
            center,
            alignment_method,
            target_axis,
            normalize_scale,
            normalizing_method,
            normalizing_size,
            logging,
        )

        total_results["processed"] += subfolder_results["processed"]
        total_results["failed"] += subfolder_results["failed"]
        total_results["subfolders"][subfolder] = subfolder_results

    if logging:
        print("\nAll subfolders processed:")
        print(f"Total processed: {total_results['processed']}")
        print(f"Total failed: {total_results['failed']}")

    return total_results


def process_ground_truth_folder(
    input_folder: str,
    output_folder: str,
    center: bool = True,
    alignment_method: Optional[str] = None,
    target_axis: int = 0,
    normalize_scale: bool = True,
    normalizing_method: str = "largest_oriented_dimension",
    normalizing_size: float = 1.0,
    has_subfolders: bool = False,
    logging: bool = True,
) -> dict:
    """
    Main function to process ground truth folder with or without subfolders.

    Args:
        input_folder (str): Path to input folder
        output_folder (str): Path to output folder
        center (bool): Whether to center the mesh
        alignment_method (str): Alignment method or None
        target_axis (int): Target axis for alignment (0=x, 1=y, 2=z)
        normalize_scale (bool): Whether to normalize mesh scale
        normalizing_method (str): Method for scaling (e.g., "largest_oriented_dimension")
        normalizing_size (float): Target size for scaling
        has_subfolders (bool): Whether input has subfolders
        logging (bool): Enable logging

    Returns:
        dict: Results summary
    """
    if logging:
        print("Starting ground truth preprocessing...")
        print(f"Input folder: {input_folder}")
        print(f"Output folder: {output_folder}")
        print(f"Target size: {normalizing_size}")
        print(f"Alignment method: {alignment_method}")
        print(f"Normalize scale: {normalize_scale}")
        print(f"Has subfolders: {has_subfolders}")

    if has_subfolders:
        return process_folder_with_subfolders(
            input_folder,
            output_folder,
            center,
            alignment_method,
            target_axis,
            normalize_scale,
            normalizing_method,
            logging,
        )
    else:
        return process_folder_no_subfolders(
            input_folder,
            output_folder,
            center,
            alignment_method,
            target_axis,
            normalize_scale,
            normalizing_method,
            normalizing_size,
            logging,
        )
