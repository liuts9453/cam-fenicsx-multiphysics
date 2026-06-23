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
from Multiphysics.Kernels import PushForwardCauchy
from Multiphysics.tiliuTools.fenicsx import constantInitialValue
from Multiphysics.tiliuTools.math import SplineInterpolatorOpt as SplineInterpolator
from Multiphysics.Postprocessors import PlasticStrain, HistvarAtNode, Extractor
from collections import namedtuple

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")
CASE_DIR = Path(__file__).resolve().parent

# Material data
data_table = CASE_DIR / "PA6_tvp.csv"
tables = load_tables_csv(str(data_table))
T_table = tables["T"] + 273.15
E_interp = SplineInterpolator(T_table, tables["E"])
sigY1_interp = SplineInterpolator(T_table, tables["sigY1"])
sigY2_interp = SplineInterpolator(T_table, tables["sigY2"])
beta_interp = SplineInterpolator(T_table, tables["beta"])
H_interp = SplineInterpolator(T_table, tables["H"])
alpha_interp = SplineInterpolator(T_table, tables["alpha"])
E2_interp = SplineInterpolator(T_table, tables["E2"])
tau_interp = SplineInterpolator(T_table, tables["tau"])
phi_interp = SplineInterpolator(T_table, tables["phi"])
delta_interp = SplineInterpolator(T_table, tables["delta"])
c_interp = SplineInterpolator(T_table, tables["c"])
b_interp = SplineInterpolator(T_table, tables["b"])

chi = 0.23
eps_dot = 0.35 / 60
init_temp = 23.18 + 273.15
ux_disp_max = 0.18
n_steps = 50
dt = ux_disp_max / (eps_dot * n_steps)
time_steps = [dt] * n_steps
time = np.cumsum(np.asarray(time_steps))
disps = time * eps_dot

MaterialParameters = namedtuple("MaterialParameters", ["E", "sigY1", "sigY2", "alpha", "beta", "H", "E2", "tau", "phi", "delta", "c", "b", "lM", "nu", "chi"])
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
    lM=2, nu=0.35, chi=chi
)

domain = mesh.create_box(MPI.COMM_WORLD, [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], [1, 1, 1], mesh.CellType.hexahedron)

from Multiphysics.Simulation import Monolithic as Simulation
sim = Simulation(domain, variables=[("displacement", 3), ("temperature", 1)], time_steps=time_steps)
var_displacement = sim.fields.getVariable("displacement")
var_temperature = sim.fields.getVariable("temperature")

stress_output = sim.fields.registerQp("stress", shape=6) 
T_field = sim.fields.register("Temperature", shape=1)

constantInitialValue(T_field, init_temp)
constantInitialValue(var_temperature, init_temp)

amg_petsc_options = {
    "snes_type": "newtonls", "snes_linesearch_type": "basic", "snes_atol": 1e-9, "snes_rtol": 1e-6, "snes_max_it": 15,
    "ksp_type": "gmres", "ksp_rtol": 1e-4, "ksp_max_it": 500, "pc_type": "gamg", "pc_gamg_type": "agg", "pc_gamg_agg_nsmooths": 1,
}

sim.initialize(
    outputs=[(T_field, var_temperature)], 
    material=(TVPkin, material_parameters),
    kernels=[
        (TotalLagrangianStressDivergence, var_displacement),
        (HeatConduction, var_temperature, {"temperature_old": T_field, "thermal_conductivity": 0.27, "unit": 0})
    ],
    bc=[
        (var_displacement[0], 0, 0, 0.0, 1), 
        (var_displacement[0], 0, 1, 0.0, 1),
        (var_displacement[1], 1, 0, 0.0, 1), 
        (var_displacement[2], 2, 0, 0.0, 1),
        (var_temperature, 1, 0, -0.025 * (var_temperature - init_temp) * dt, 2),
    ],
    load_steps=[(1, disps)],
    aux_kernels=[(PushForwardCauchy, stress_output)],
    postprocessors=[
        (Extractor, {
            "filename": str(CASE_DIR / "result" / "point_extraction.csv"),
            "items": [
                {"field": "u", "node_index": 7, "component": 0, "label": "ux_pull"},
                {"field": "u", "node_index": 7, "component": 3, "label": "T_pull"},
                {"field": "stress", "qp_index": 7, "component": 0, "label": "sxx_gp7"},
            ]
        }),
    ],
    petsc_options=amg_petsc_options,
)

if __name__ == "__main__":
    output_path = CASE_DIR / "result" / "test_extract.bp"
    output_path.parent.mkdir(exist_ok=True)
    sim.run(str(output_path))
