"""Wrapper for easy handling of Gmsh's Python API"""

import logging
from pathlib import Path

import gmsh
import numpy as np

import fox


def _create_mesh(
    output_dir: Path,
    dim: int = 3,
    mesh_size: float | bool = False,
    recombine_all: bool = True,
    quasi_structured: bool = False,
    element_order: int = 1,
    smoothing: int = 100,
    transfinite_automatic: bool = False,
) -> None:
    """Mesh created geometry with sane defaults"""

    if mesh_size:
        gmsh.option.set_number("Mesh.MeshSizeFromPoints", False)
        gmsh.option.set_number("Mesh.MeshSizeMin", mesh_size)
        gmsh.option.set_number("Mesh.MeshSizeMax", mesh_size)
    gmsh.option.set_number("Mesh.RecombineAll", recombine_all)
    if quasi_structured:
        gmsh.option.set_number("Mesh.Algorithm", 11)  # quasi-structured
    gmsh.option.set_number("Mesh.ElementOrder", element_order)
    gmsh.option.set_number("Mesh.Smoothing", smoothing)
    if transfinite_automatic:
        gmsh.model.mesh.set_transfinite_automatic()
    gmsh.model.mesh.generate(dim=dim)

    # save mesh
    mesh_file = output_dir / Path("mesh.msh")
    gmsh.write(mesh_file.as_posix())

    # get nodes
    node_tags, node_coords, _ = gmsh.model.mesh.get_nodes()
    node_tags -= 1
    nodes = node_coords.reshape(-1, 3)
    num_nodes = nodes.shape[0]
    logging.info(f"{num_nodes = }")

    # get elements
    element_types, _, element_node_tags_list = gmsh.model.mesh.get_elements(dim=dim)
    if (element_types.shape[0] != 1) or (element_types[0] != 5):
        raise NotImplementedError(f"Currently only supporting hexahedral elements. Mesh needs to be changed. element_types = {element_types}")
    elements = np.int64(element_node_tags_list[0].reshape(-1, 8) - 1)  # for compatibility with PyVista, make sure to use int64 (by default, you get uint64 here)
    num_elements = elements.shape[0]
    logging.info(f"{num_elements = }")

    return mesh_file


def _show_geometry(
    points: bool = True,
    lines: bool = True,
    surfaces: bool = True,
    point_numbers: bool = False,
    line_numbers: bool = False,
    surface_numbers: bool = False,
) -> None:
    """Open GUI to show created geometry"""

    gmsh.option.set_number("Geometry.Points", points)
    gmsh.option.set_number("Geometry.Lines", lines)
    gmsh.option.set_number("Geometry.Surfaces", surfaces)
    gmsh.option.set_number("Geometry.PointNumbers", point_numbers)
    gmsh.option.set_number("Geometry.LineNumbers", line_numbers)
    gmsh.option.set_number("Geometry.SurfaceNumbers", surface_numbers)
    gmsh.fltk.run()


def _show_mesh(
    mesh_file: Path,
    node_numbers: bool = False,
    element_numbers: bool = False,
    element_surfaces: bool = True,
) -> None:
    """Open GUI to show created mesh"""

    gmsh.open(mesh_file.as_posix())
    gmsh.option.set_number("Geometry.Points", False)
    gmsh.option.set_number("Geometry.Lines", False)
    gmsh.option.set_number("Geometry.Surfaces", False)
    gmsh.option.set_number("Mesh.SurfaceFaces", element_surfaces)
    gmsh.option.set_number("Mesh.PointNumbers", node_numbers)
    gmsh.option.set_number("Mesh.SurfaceNumbers", element_numbers)
    gmsh.fltk.run()


def create_cube(
    output_dir: Path,
    width: float = 1.0,
    height: float = 1.0,
    thickness: float = 1.0,
    dim: int = 3,
    mesh_size: float | bool = False,
    num_elements_thickness: int = 7,
    recombine_all: bool = True,
    quasi_structured: bool = False,
    element_order: int = 1,
    smoothing: int = 100,
    transfinite_automatic: bool = False,
    show_geometry: bool = False,
    show_mesh: bool = False,
) -> None:
    section = "Cube"
    fox.log.start(section)

    # Initialization
    gmsh.initialize()
    gmsh.model.add(section)
    gmsh.option.set_number("General.Terminal", False)
    gmsh.option.set_number("General.Tooltips", False)

    # Geometry
    x, y, z = 0.0, 0.0, 0.0  # position of bottom left point of rectangle
    plane = gmsh.model.occ.add_rectangle(x, y, z, width, height)
    volume = gmsh.model.occ.extrude([(2, plane)], dx=0.0, dy=0.0, dz=thickness, numElements=[num_elements_thickness], recombine=True)[1][1]  # get tag of 3D entity
    gmsh.model.occ.synchronize()  # needs to be called before any use of functions outside of the OCC kernel
    if show_geometry:
        _show_geometry()

    # Mesh
    mesh_file = _create_mesh(
        output_dir=output_dir,
        dim=dim,
        mesh_size=mesh_size,
        recombine_all=recombine_all,
        quasi_structured=quasi_structured,
        element_order=element_order,
        smoothing=smoothing,
        transfinite_automatic=transfinite_automatic,
    )
    if show_mesh:
        _show_mesh(mesh_file)

    gmsh.model.add_physical_group(dim, [volume])

    fox.log.end(section)


def create_notched_specimen(
    output_dir: Path,
    width: float = 8.0,
    height: float = 3.0,
    thickness: float = 0.5,
    radius: float = 0.3,
    dim: int = 3,
    mesh_size: float = 0.2,
    num_elements_thickness: int = 3,
    recombine_all: bool = True,
    quasi_structured: bool = False,
    element_order: int = 1,
    smoothing: int = 100,
    transfinite_automatic: bool = True,
    show_geometry: bool = False,
    show_mesh: bool = False,
) -> None:
    section = "Notched specimen"
    fox.log.start(section)

    # Initialization
    gmsh.initialize()
    gmsh.model.add(section)
    gmsh.option.set_number("General.Terminal", False)
    gmsh.option.set_number("General.Tooltips", False)

    # Geometry
    x, y, z = 0.0, 0.0, 0.0  # position of bottom left point of rectangle
    rec = gmsh.model.occ.add_rectangle(x, y, z, width, height)
    cyl1 = gmsh.model.occ.add_cylinder(x + width / 2.0, y, z - 0.5, 0.0, 0.0, 1.0, radius)
    cyl2 = gmsh.model.occ.add_cylinder(x + width / 2.0, y + height, z - 0.5, 0.0, 0.0, 1.0, radius)
    plane = gmsh.model.occ.cut([(2, rec)], [(3, cyl1), (3, cyl2)])[0][0][1]
    volume = gmsh.model.occ.extrude([(2, plane)], dx=0.0, dy=0.0, dz=thickness, numElements=[num_elements_thickness], recombine=True)[1][1]  # get tag of 3D entity
    gmsh.model.occ.synchronize()  # needs to be called before any use of functions outside of the OCC kernel
    if show_geometry:
        _show_geometry()

    # Mesh
    mesh_file = _create_mesh(
        output_dir=output_dir,
        dim=dim,
        mesh_size=mesh_size,
        recombine_all=recombine_all,
        quasi_structured=quasi_structured,
        element_order=element_order,
        smoothing=smoothing,
        transfinite_automatic=transfinite_automatic,
    )
    if show_mesh:
        _show_mesh(mesh_file)

    gmsh.model.add_physical_group(dim, [volume])

    fox.log.end(section)
