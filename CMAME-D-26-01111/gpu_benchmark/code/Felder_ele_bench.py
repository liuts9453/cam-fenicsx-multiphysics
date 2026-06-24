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
import os


import jax
from Multiphysics.tiliuTools.data import load_tables_csv
from mpi4py import MPI
from dolfinx import mesh
from TVPkinIFT_heat import TVPkinIFT_heat as TVPkinIFT
from Multiphysics.Kernels import (
    TotalLagrangianStressDivergence,
    ConstanScalerVariable,
    HeatConduction,
)
from Multiphysics.Kernels import PushForwardCauchy, DeformationGradient
from Multiphysics.tiliuTools.fenicsx import constantInitialValue
from Multiphysics.tiliuTools.math import SplineInterpolatorOpt as SplineInterpolator
from Multiphysics.Postprocessors import PlasticStrain
import pprint
import Multiphysics
from collections import namedtuple
from Multiphysics.Postprocessors import HistvarAtNode

import sys
import argparse

sys.argv.append("-log_view")

parser = argparse.ArgumentParser()
parser.add_argument("--mesh_size", type=int, default=30)
parser.add_argument("--platform", type=str, default="cpu", choices=["cpu", "gpu"])
parser.add_argument("--steps", type=int, default=50)
args, unknown = parser.parse_known_args()

jax.config.update("jax_enable_x64", True)  # Enable 64-bit arithmetic in JAX
jax.config.update("jax_platform_name", args.platform)
output_dir = list(Multiphysics.__path__)[0]
# ================================================================
#                       Parmeters
# ================================================================

data_table = "PA6_tvp.csv"
# ================================================================
#                       Parmeters
# ================================================================
tables = load_tables_csv(data_table)
pprint.pprint(tables)
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
n_steps = args.steps # Number of loading steps
dt = ux_disp_max / (eps_dot * 50)  # Use 50 to keep step size identical to original
time_steps = [dt] * n_steps
time = np.cumsum(np.asarray(time_steps))
disps = time * eps_dot
iso_thermo = False

# Define the material-parameter namedtuple type once.
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

# Instantiate the material parameters once during initialization.
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
j1 = args.mesh_size
j2 = args.mesh_size
j3 = args.mesh_size
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
        ("displacement", 3),  # Set the unkowns to the system
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
#             PETSc options: AMG-preconditioned Krylov solver
# ================================================================
amg_petsc_options = {
    # 1. Nonlinear solver (SNES): unchanged from the benchmark setup.
    "snes_type": "newtonls",
    "snes_linesearch_type": "basic", 
    "snes_atol": 1e-9,
    "snes_rtol": 1e-6,
    "snes_max_it": 15,
    
    # 2. Linear solver (KSP): GMRES.
    "ksp_type": "gmres",               
    "ksp_rtol": 1e-4,                  # Relative tolerance for each linear solve
    "ksp_max_it": 500,                 # Allow a larger number of inner Krylov iterations
    
    # 3. Use algebraic multigrid (GAMG) as the preconditioner.
    "pc_type": "gamg",                 
    "pc_gamg_type": "agg",             # Smoothed aggregation
    "pc_gamg_agg_nsmooths": 1,         # Number of aggregation smoothing sweeps
    
    "log_view": "",                    # Enable PETSc logging
    # Debug monitors. Uncomment to inspect SNES/KSP convergence histories.
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
    material=(TVPkinIFT, material_parameters),
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
    ],  # Aux Kernel
    postprocessors=[
        (HistvarAtNode, {"index": 21, "field": "rep"}),
        (HistvarAtNode, {"index": 22, "field": "rp"}),
        (HistvarAtNode, {"index": 23, "field": "rev"}),
        (HistvarAtNode, {"index": 24, "field": "rv"}),
    ],  # Custom processor
    
    # === Inject the AMG-preconditioned Krylov solver configuration ===
    petsc_options=amg_petsc_options,
)


# ================================================================
#             Monkey Patch to measure JAX Constitutive Time & Comm
# ================================================================
from Multiphysics.Materials.ExoMaterial import ExoMaterial
import time
import jax
import jax.numpy as jnp
import numpy as np
from mpi4py import MPI

prof_h2d = []
prof_comp = []
prof_d2h = []

def patched_evaluateTangent(self, variables):
    vars_ = variables.reshape((-1, self.driving_force.ufl_shape[0]))
    old_hist = self.hist.x.array[:].reshape((-1, self.hist_len))
    self.computeGlobalValues()

    # 1. H2D (Host to Device) Transfer
    t0 = time.time()
    vars_d = jax.device_put(vars_)
    old_hist_d = jax.device_put(old_hist)
    global_vals_d = jax.device_put(self.global_vals)
    jax.block_until_ready((vars_d, old_hist_d, global_vals_d))
    t_h2d = time.time() - t0

    # 2. GPU/CPU Compute (including JIT on first call)
    t1 = time.time()
    cijkl_, aux = self.jac(vars_d, old_hist_d, global_vals_d, self.par)
    jax.block_until_ready((cijkl_, aux))
    t_comp = time.time() - t1

    new_hist = aux[0]
    n_aux = len(aux)
    n_groups = (n_aux - 1) // 3
    local_ok = True

    for i in range(n_groups):
        conv_i = aux[1 + 3 * i]
        niter_i = aux[1 + 3 * i + 1]
        history_i = aux[1 + 3 * i + 2]
        if int(jnp.max(niter_i)) > 0 and self.domain.comm.rank == 0:
            self.printConvergeHistory(conv_i, niter_i, history_i)
        if not jnp.all(conv_i):
            local_ok = False
        if not jnp.isfinite(history_i).all():
            local_ok = False

    comm = self.domain.comm
    sendbuf = np.array(1 if local_ok else 0, dtype=np.int32)
    recvbuf = np.array(0, dtype=np.int32)
    comm.Allreduce(sendbuf, recvbuf, op=MPI.MIN)
    global_ok = bool(recvbuf)

    if not global_ok:
        raise RuntimeError("Local constitutive update did not converge on some rank")

    # 3. D2H (Device to Host) Bulk Transfer
    t2 = time.time()
    cijkl_np = np.nan_to_num(np.array(cijkl_), nan=0.0, posinf=0.0, neginf=0.0)
    new_hist_np = np.nan_to_num(np.array(new_hist), nan=0.0, posinf=0.0, neginf=0.0)
    t_d2h = time.time() - t2
    
    prof_h2d.append(("Tangent", t_h2d))
    prof_comp.append(("Tangent", t_comp))
    prof_d2h.append(("Tangent", t_d2h))

    return cijkl_np.reshape(-1), new_hist_np.reshape(-1)

def patched_evaluateFlux(self, variables):
    vars_ = variables.reshape((-1, self.driving_force.ufl_shape[0]))
    hist_ = self.hist.x.array[:].reshape((-1, self.hist_len))
    self.computeGlobalValues()

    # 1. H2D
    t0 = time.time()
    vars_d = jax.device_put(vars_)
    hist_d = jax.device_put(hist_)
    global_vals_d = jax.device_put(self.global_vals)
    jax.block_until_ready((vars_d, hist_d, global_vals_d))
    t_h2d = time.time() - t0

    # 2. Compute
    t1 = time.time()
    fl_, _ = self._flux(vars_d, hist_d, global_vals_d, self.par)
    jax.block_until_ready(fl_)
    t_comp = time.time() - t1

    # 3. D2H
    t2 = time.time()
    fl_np = np.array(fl_)
    t_d2h = time.time() - t2
    
    prof_h2d.append(("Flux", t_h2d))
    prof_comp.append(("Flux", t_comp))
    prof_d2h.append(("Flux", t_d2h))

    return fl_np.reshape(-1)

ExoMaterial.evaluateTangent = patched_evaluateTangent
ExoMaterial.evaluateFlux = patched_evaluateFlux

if __name__ == "__main__":

    output_path = f"result/bench_{args.platform}_{args.mesh_size}.bp"
    sim_t0 = time.time()
    sim.run(output_path)
    sim_t1 = time.time()
    
    print(f"\n[Profiling] Platform: {args.platform}, Mesh: {args.mesh_size}^3")
    
    if len(prof_comp) > 0:
        first_comp_time = prof_comp[0][1] + (prof_comp[1][1] if len(prof_comp) > 1 else 0.0)
        
        pure_comp_time = sum(t for _, t in prof_comp) - first_comp_time
        pure_comp_calls = max(1, len(prof_comp) - 2)
        avg_comp = pure_comp_time / pure_comp_calls
        
        total_h2d = sum(t for _, t in prof_h2d)
        avg_h2d = total_h2d / len(prof_h2d)
        
        total_d2h = sum(t for _, t in prof_d2h)
        avg_d2h = total_d2h / len(prof_d2h)
        
        total_comm = total_h2d + total_d2h
        
        print(f"[Profiling] JAX First Call (JIT Compilation + Init Compute): {first_comp_time:.4f} s")
        print(f"[Profiling] JAX Pure Compute (Total without JIT): {pure_comp_time:.4f} s")
        print(f"[Profiling] JAX Pure Compute (Avg per call): {avg_comp:.6f} s")
        
        print(f"[Profiling] Communication H2D (Total): {total_h2d:.4f} s | Avg: {avg_h2d:.6f} s")
        print(f"[Profiling] Communication D2H (Total): {total_d2h:.4f} s | Avg: {avg_d2h:.6f} s")
        print(f"[Profiling] Total Communication (H2D + D2H): {total_comm:.4f} s")
        
        total_jax_overall = sum(t for _, t in prof_comp) + total_comm
        total_assembly_solve = (sim_t1 - sim_t0) - total_jax_overall
        print(f"[Profiling] Global Assembly & Solve (Total Approx): {total_assembly_solve:.4f} s")
    else:
        print("[Profiling] No JAX calls recorded.")
