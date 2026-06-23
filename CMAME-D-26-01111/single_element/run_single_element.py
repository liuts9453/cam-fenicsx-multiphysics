#   _________ _____ ___
#  / ___/ __ `/ __ `__ \
# / /__/ /_/ / / / / / /
# \___/\__,_/_/ /_/ /_/ Computational Applied Mechanics,
#                               University of Wuppertal
#
# Authors:
#     - Tiansheng Liu (tiliu@uni-wuppertal.de)
# ------------------------------------------------------------------------------
#    Single element test

# Import modules
import numpy as np
from pathlib import Path

from PS import PlasticStretch
import jax
from Multiphysics.tiliuTools.data import load_tables_csv
from mpi4py import MPI
from dolfinx import mesh
from TVPkinIFT_heat import TVPkinIFT_heat as TVPkin
from Multiphysics.Kernels import (
    TotalLagrangianStressDivergence,
    ConstanScalerVariable,
    HeatConduction,
)
from Multiphysics.Kernels import PushForwardCauchy, DeformationGradient
from Multiphysics.tiliuTools.fenicsx import constantInitialValue
from Multiphysics.tiliuTools.math import SplineInterpolatorOpt as SplineInterpolator
from collections import namedtuple
from Multiphysics.Postprocessors import HistvarAtNode

jax.config.update("jax_enable_x64", True)  # Enable 64-bit arithmetic in JAX
jax.config.update("jax_platform_name", "cpu")
CASE_DIR = Path(__file__).resolve().parent
# ================================================================
#                       Parameters
# ================================================================

data_table = CASE_DIR / "PA6_tvp.csv"
# ================================================================
#                       Parameters
# ================================================================
tables = load_tables_csv(str(data_table))
# Temperature sampling nodes
T_table = tables["T"] + 273.15

# Material property tables
E_table = tables["E"]
sigY1_table = tables["sigY1"]
sigY2_table = tables["sigY2"]
beta_table = tables["beta"]
H_table = tables["H"]
alpha_table = tables["alpha"]
E2_table = tables["E2"]
tau_table = tables["tau"]
phi_table = tables["phi"]
delta_table = tables["delta"]
c_table = tables["c"]
b_table = tables["b"]

# Interpolator instances
E_interp = SplineInterpolator(T_table, E_table)
sigY1_interp = SplineInterpolator(T_table, sigY1_table)
sigY2_interp = SplineInterpolator(T_table, sigY2_table)
beta_interp = SplineInterpolator(T_table, beta_table)
H_interp = SplineInterpolator(T_table, H_table)
alpha_interp = SplineInterpolator(T_table, alpha_table)
c_interp = SplineInterpolator(T_table, c_table)
b_interp = SplineInterpolator(T_table, b_table)

E2_interp = SplineInterpolator(T_table, E2_table)
tau_interp = SplineInterpolator(T_table, tau_table)
phi_interp = SplineInterpolator(T_table, phi_table)
delta_interp = SplineInterpolator(T_table, delta_table)


chi = 0.23
eps_dot = 0.35 / 60

init_temp = 23.18 + 273.15
ux_disp_max = 0.18  # Maximum displacement of the top surface in x-direction
n_steps = 50 # Number of loading steps
dt = ux_disp_max / (eps_dot * n_steps)
time_steps = [dt] * n_steps
time = np.cumsum(np.asarray(time_steps))
disps = time * eps_dot
iso_thermo = False

# 1. Define the material-parameter namedtuple once.
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

# 2. Instantiate the material parameters once during setup.
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
    c=lambda temp: c_interp.evaluate(temp) / 0.23 * chi,
    b=lambda temp: b_interp.evaluate(temp),
    lM=2,
    nu=0.35,
    chi=chi,
)
# ================================================================
#                            Mesh
# ================================================================

# A 3-D cube
l = 1
w = 1
h = 1
j1 = 1
j2 = 1
j3 = 1
domain = mesh.create_box(
    MPI.COMM_WORLD,
    [[0.0, 0.0, 0.0], [l, w, h]],
    [j1, j2, j3],
    mesh.CellType.hexahedron,
)


# ================================================================
#                   Initialize simulation
# ================================================================
from Multiphysics.Simulation import Monolithic as Simulation

sim = Simulation(
    domain,
    variables=[
        ("displacement", 3),  # Register the displacement unknowns.
        ("temperature", 1),
    ],
    time_steps=time_steps,
)
var_displacement = sim.fields.getVariable("displacement")
var_temperature = sim.fields.getVariable("temperature")
# ================================================================
#                   Auxiliary quantities
# ================================================================
stress_output = sim.fields.register("stress", shape=6)
rep = sim.fields.register("rep", shape=1)
rp = sim.fields.register("rp", shape=1)
rev = sim.fields.register("rev", shape=1)
rv = sim.fields.register("rv", shape=1)
pstrain_output = sim.fields.register("plastic_stretch", shape=6)
F_output = sim.fields.register("F", shape=9)
disp_output = sim.fields.register("Displacement", shape=3)
T = sim.fields.register("Temperature", shape=1)
# ================================================================
#                   Initial Conditions
# ================================================================
constantInitialValue(T, init_temp)
constantInitialValue(var_temperature, init_temp)

# ================================================================
#             PETSc options for the Krylov solver with AMG preconditioning
# ================================================================
amg_petsc_options = {
    # 1. Nonlinear solver settings.
    "snes_type": "newtonls",
    "snes_linesearch_type": "basic", 
    "snes_atol": 1e-9,
    "snes_rtol": 1e-6,
    "snes_max_it": 15,
    
    # 2. Use GMRES for the linear solve.
    "ksp_type": "gmres",               
    "ksp_rtol": 1e-4,                  # Relative tolerance for the linear solve.
    "ksp_max_it": 500,                 # Allow additional inner Krylov iterations.
    
    # 3. Use algebraic multigrid (GAMG) as preconditioner.
    "pc_type": "gamg",                 
    "pc_gamg_type": "agg",             # Use smoothed aggregation.
    "pc_gamg_agg_nsmooths": 1,         # Number of aggregation smoothing sweeps.
    
    # Optional solver monitors can be enabled for solver inspection.
    # "snes_monitor": "",
    # "ksp_monitor": "",               
}

sim.initialize(
    # ================================================================
    #                       Set outputs
    # ================================================================
    outputs=[
        (T, var_temperature),  # Outputs
        (stress_output, None),
        (F_output, None),
        (rep, None),
        (rp, None),
        (rev, None),
        (rv, None),
        (pstrain_output, None),
        (disp_output, var_displacement),
    ],
    # ================================================================
    #                       Materials
    # ================================================================
    material=(TVPkin, material_parameters),
    # ================================================================
    #                       Kernels
    # ================================================================
    kernels=[
        (TotalLagrangianStressDivergence, var_displacement),  # Kernels
        (
            (
                ConstanScalerVariable,
                var_temperature,
                {
                    "value": init_temp,
                },
            )
            if iso_thermo is True
            else (
                HeatConduction,
                var_temperature,
                {
                    "temperature_old": T,
                    "thermal_conductivity": 0.27,
                    "unit": 0,
                },
            )
        ),
    ],
    # ================================================================
    #                   Boundary conditions
    # ================================================================
    bc=[
        (var_displacement[0], 0, 0, 0.0, 1),
        (var_displacement[0], 0, 1, 0.0, 1),
        (var_displacement[1], 1, 0, 0.0, 1),
        (var_displacement[2], 2, 0, 0.0, 1),
        (var_temperature, 1, 0, -0.025 * (var_temperature - init_temp) * dt, 2),
        (var_temperature, 1, 1, -0.025 * (var_temperature - init_temp) * dt, 2),
        (var_temperature, 2, 0, -0.025 * (var_temperature - init_temp) * dt, 2),
        (var_temperature, 2, 1, -0.025 * (var_temperature - init_temp) * dt, 2),
        (var_temperature, 0, 0, -0.025 * (var_temperature - init_temp) * dt, 2),
        (var_temperature, 0, 1, -0.025 * (var_temperature - init_temp) * dt, 2),
    ],  # BC
    # ================================================================
    #           Loadsteps, auxiliary kernels, postprocessors
    # ================================================================
    load_steps=[
        (1, disps),
    ],  # Loadsteps
    aux_kernels=[
        (PushForwardCauchy, stress_output),
        (PlasticStretch, pstrain_output),
    ],  # Aux Kernel
    postprocessors=[
        (HistvarAtNode, {"index": 21, "field": "rep"}),
        (HistvarAtNode, {"index": 22, "field": "rp"}),
        (HistvarAtNode, {"index": 23, "field": "rev"}),
        (HistvarAtNode, {"index": 24, "field": "rv"}),
    ],  # Custom processor
    
    # === Apply the Krylov solver configuration with AMG preconditioning ===
    petsc_options=amg_petsc_options,
)


if __name__ == "__main__":

    output_path = CASE_DIR / "result" / f"Felder_ele{eps_dot:.4f}.bp"
    output_path.parent.mkdir(exist_ok=True)
    sim.run(str(output_path))
