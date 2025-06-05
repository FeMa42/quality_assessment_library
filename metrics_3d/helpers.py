import vtk
import os
import trimesh


def load_trimesh_to_obj(mesh_path):
    """
    Load a mesh file using trimesh and export it to an OBJ file.
    Args:
        mesh_path (str): Path to the mesh file.
    Returns:
        str: Path to the exported OBJ file.
    """
    # if the mesh is already in OBJ format, return the path directly without exporting
    if mesh_path.endswith(".obj"):
        return mesh_path

    export_dir = os.path.join(os.path.dirname(mesh_path), "exported_objects")
    os.makedirs(export_dir, exist_ok=True)
    # Build the export path with the same base name but .obj extension
    base_name = os.path.splitext(os.path.basename(mesh_path))[0]
    export_path = os.path.join(export_dir, base_name + ".obj")

    # check if the export file path already exists
    if os.path.exists(export_path):
        print(f"Exported file {export_path} already exists, skipping export.")
        return export_path

    # Load the mesh and export it to OBJ format
    try:
        mesh = trimesh.load(mesh_path)
        mesh.export(export_path, file_type="obj")
        return export_path
    except Exception as e:
        print(f"Error exporting {mesh_path} to {export_path}: {e}")


def safe_load_trimesh(mesh_path):
    """
    Load a mesh file using trimesh, ensuring it is a valid Trimesh object.
    Args:
        mesh_path (str): Path to the mesh file.
    Returns:
        trimesh.Trimesh: A valid Trimesh object.
    Raises:
        ValueError: If the loaded mesh is not a valid Trimesh object.
    """
    mesh = trimesh.load(mesh_path)
    mesh.export(mesh_path + ".obj", file_type="obj")
    if isinstance(mesh, trimesh.Scene):
        print("is scene")

        # Merge all geometries into one mesh
        mesh = trimesh.util.concatenate([
            g for g in mesh.geometry.values() if isinstance(g, trimesh.Trimesh)
        ])
    if not isinstance(mesh, trimesh.Trimesh):
        raise ValueError(f"File {mesh_path} did not yield a Trimesh object.")

    # Ensure the mesh is watertight and clean (might not work for all meshes)
    if not mesh.is_watertight:
        print(f"[Warning] {mesh_path}: Mesh is not watertight, attempting to clean it.")
        watertight = mesh.fill_holes()
        print("watertight after fill_holes:", watertight)
        mesh.update_faces(mesh.nondegenerate_faces())
        mesh.remove_unreferenced_vertices()
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


def load_obj_with_vtk(filename):
    reader = vtk.vtkOBJReader()
    reader.SetFileName(filename)
    reader.Update()
    polydata = reader.GetOutput()
    return polydata
