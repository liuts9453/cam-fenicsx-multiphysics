from collections import namedtuple
from pathlib import Path

import gmsh
import jax
import numpy as np
from dolfinx.io import gmshio
from mpi4py import MPI

from ADTVPkin import ADTVPkin as TVP_kin
from Multiphysics.Kernels import HeatConduction, PushForwardCauchy
from Multiphysics.Kernels import TotalLagrangianStressDivergence
from Multiphysics.Postprocessors import PlasticStrain
from Multiphysics.Simulation import Monolithic as Simulation
from Multiphysics.tiliuTools.data import load_tables_csv
from Multiphysics.tiliuTools.fenicsx import constantInitialValue
from Multiphysics.tiliuTools.math import SplineInterpolatorOpt as SplineInterpolator


jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")

CASE_DIR = Path(__file__).resolve().parent
mesh_path = CASE_DIR / "dogbone_bak.msh"
data_table = CASE_DIR / "PA6_shen.csv"
output_path = CASE_DIR / "result" / "example_thermoviscoplastic_dog.bp"


def read_msh_verbose(filename, comm=MPI.COMM_WORLD):
    """Read a Gmsh mesh and print compact element statistics on rank 0."""
    filename = str(filename)
    if comm.rank == 0:
        gmsh.initialize()
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.open(filename)

        print(f"\n==== Mesh Summary for '{filename}' ====")
        summary = {}
        for dim, tag in gmsh.model.getEntities():
            try:
                element_types, element_tags, _ = gmsh.model.mesh.getElements(dim, tag)
            except Exception:
                continue

            for element_type, elements in zip(element_types, element_tags):
                name = gmsh.model.mesh.getElementProperties(element_type)[0]
                summary[name] = summary.get(name, 0) + len(elements)

        for name, count in summary.items():
            print(f"  {name:<20} : {count:>6} elements")
        print("=======================================\n")
        gmsh.finalize()

    return gmshio.read_from_msh(filename, comm)


# ================================================================
#                       Parameters
# ================================================================
tables = load_tables_csv(str(data_table))
T_table = tables["T"] + 273.15

E_interp = SplineInterpolator(T_table, tables["E"])
sigY1_interp = SplineInterpolator(T_table, tables["sigY1"])
sigY2_interp = SplineInterpolator(T_table, tables["sigY2"])
beta_interp = SplineInterpolator(T_table, tables["beta"])
H_interp = SplineInterpolator(T_table, tables["H"])
alpha_interp = SplineInterpolator(T_table, tables["alpha"])
c_interp = SplineInterpolator(T_table, tables["c"])
b_interp = SplineInterpolator(T_table, tables["b"])
E2_interp = SplineInterpolator(T_table, tables["E2"])
tau_interp = SplineInterpolator(T_table, tables["tau"])
phi_interp = SplineInterpolator(T_table, tables["phi"])
delta_interp = SplineInterpolator(T_table, tables["delta"])

chi = 1.0
nu = 0.4
init_temp = 24.5 + 273.15
ux_disp_max = 30.0
n_steps = 120 * 10
eps_dot = 2.0
dt = ux_disp_max / (eps_dot * n_steps)
n_steps_fine = 0
fine_steps = 20
time_steps = [
    *([dt] * (n_steps - n_steps_fine)),
    *([dt / fine_steps] * (n_steps_fine * fine_steps)),
]

time = np.cumsum(time_steps)
disps = time * eps_dot

MaterialParameters = namedtuple(
    "MaterialParameters",
    [
        "E",
        "sigY1",
        "sigY2",
        "alpha",
        "beta",
        "H",
        "E2",
        "tau",
        "phi",
        "delta",
        "c",
        "b",
        "lM",
        "nu",
        "chi",
    ],
)

material_parameters = MaterialParameters(
    E=lambda temp: E_interp.evaluate(temp) * chi,
    sigY1=lambda temp: chi * sigY1_interp.evaluate(temp),
    sigY2=lambda temp: sigY2_interp.evaluate(temp),
    alpha=lambda temp: alpha_interp.evaluate(temp),
    beta=lambda temp: chi * beta_interp.evaluate(temp),
    H=lambda temp: chi * H_interp.evaluate(temp),
    E2=lambda temp: E2_interp.evaluate(temp),
    tau=lambda temp: tau_interp.evaluate(temp),
    phi=lambda temp: phi_interp.evaluate(temp),
    delta=lambda temp: delta_interp.evaluate(temp),
    c=lambda temp: c_interp.evaluate(temp),
    b=lambda temp: b_interp.evaluate(temp),
    lM=2,
    nu=nu,
    chi=chi,
)


# ================================================================
#                   Initialize simulation
# ================================================================
sim = Simulation(
    read_msh_verbose(mesh_path, MPI.COMM_WORLD),
    variables=[
        ("displacement", 3),
        ("temperature", 1),
    ],
    time_steps=time_steps,
)
var_displacement = sim.fields.getVariable("displacement")
var_temperature = sim.fields.getVariable("temperature")

stress_output = sim.fields.register("stress", shape=6)
disp_output = sim.fields.register("Displacement", shape=3)
T = sim.fields.register("Temperature", shape=1)
plas_stretch = sim.fields.register("plastic_stretch", shape=6)

constantInitialValue(T, init_temp)
constantInitialValue(var_temperature, init_temp)

amg_petsc_options = {
    "snes_type": "newtonls",
    "snes_linesearch_type": "basic",
    "snes_atol": 1e-9,
    "snes_rtol": 1e-6,
    "snes_max_it": 15,
    "ksp_type": "gmres",
    "ksp_rtol": 1e-4,
    "ksp_max_it": 500,
    "pc_type": "gamg",
    "pc_gamg_type": "agg",
    "pc_gamg_agg_nsmooths": 1,
}

sim.initialize(
    outputs=[
        (T, var_temperature),
        (stress_output, None),
        (disp_output, var_displacement),
        (plas_stretch, None),
    ],
    material=(TVP_kin, material_parameters),
    kernels=[
        (TotalLagrangianStressDivergence, var_displacement),
        (
            HeatConduction,
            var_temperature,
            {
                "unit": 0,
                "temperature_old": T,
                "thermal_conductivity": 0.28,
            },
        ),
    ],
    bc=[
        (var_displacement[1], 1, 41, 0.0, 1),
        (var_displacement[0], 1, 41, 0.0, 1),
        (var_displacement[0], 1, -41, 0.0, 1),
        (var_displacement[1], 1, -41, 0.0, 1),
        (var_displacement[2], 1, -41, 0.0, 1),
        (
            var_temperature,
            -1,
            -0.025 * (var_temperature - init_temp) * sim.dt,
            2,
            "Tag",
        ),
    ],
    load_steps=[
        (0, disps),
    ],
    aux_kernels=[
        (PushForwardCauchy, stress_output),
    ],
    postprocessors=[
        PlasticStrain,
    ],
    petsc_options=amg_petsc_options,
)


if __name__ == "__main__":
    output_path.parent.mkdir(exist_ok=True)
    sim.run(str(output_path))
