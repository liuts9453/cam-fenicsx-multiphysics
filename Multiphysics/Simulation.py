import time
import traceback
import signal
import os
from functools import partial
import psutil
import dolfinx
import numpy as np
import ufl
from dolfinx import fem
from dolfinx.fem import form
from dolfinx.fem.petsc import (
    NonlinearProblem,
    assemble_vector,
    apply_lifting,
    set_bc,
)
from mpi4py import MPI
from petsc4py import PETSc
from ufl import as_vector

from Multiphysics.Kernels.Base import ActionManager
from Multiphysics.Materials import ExoMaterial
from Multiphysics.SimulationManagers.AuxSolverManager import AuxSolverManager
from Multiphysics.SimulationManagers.BCBuilder import BcsBuilder
from Multiphysics.SimulationManagers.FailureReport import FailureReport, formatFailure
from Multiphysics.SimulationManagers.SimulationIO import SimulationIO
from Multiphysics.tiliuTools.fenicsx import FieldManager

# Import the separated cutback-strategy module
from Multiphysics.SimulationManagers.CutbackStrategies import (
    BaseCutbackStrategy,
    AdaptiveCutbackStrategy,
)

import sys

# ==============================================================================
# PETSc error handling and global interrupt tracking
# ==============================================================================
PETSc.Sys.pushErrorHandler("ignore")

_PETSC_ERR_MAP = {}
for name in dir(PETSc):
    if name.startswith("ERR_"):
        try:
            val = getattr(PETSc, name)
            if isinstance(val, int):
                _PETSC_ERR_MAP[val] = name
        except Exception:
            pass

class _GlobalInterruptTracker:
    """Global interrupt tracker that prevents PETSc from hiding Ctrl+C as an error code."""
    def __init__(self):
        self.interrupted = False
        try:
            self._original_handler = signal.getsignal(signal.SIGINT)
            signal.signal(signal.SIGINT, self._handler)
        except ValueError:
            self._original_handler = None

    def _handler(self, sig, frame):
        self.interrupted = True
        if callable(self._original_handler):
            self._original_handler(sig, frame)
        else:
            raise KeyboardInterrupt()

INTERRUPT_TRACKER = _GlobalInterruptTracker()

class _SolveState:
    """Internal error tracker used to expose model exceptions hidden behind PETSc errors."""
    def __init__(self):
        self.internal_error = None

SOLVE_STATE = _SolveState()
# ==============================================================================


def _assemble_residual_with_callback(
    u: fem.Function,
    F_form,
    J_form,
    bcs,
    external_callback,
    snes: PETSc.SNES,
    x: PETSc.Vec,
    b: PETSc.Vec,
) -> None:
    global SOLVE_STATE
    try:
        x.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
        x.copy(u.x.petsc_vec)
        u.x.scatter_forward()

        if external_callback is not None:
            external_callback()

        with b.localForm() as b_local:
            b_local.set(0.0)
        assemble_vector(b, F_form)

        apply_lifting(b, [J_form], [bcs], [x], -1.0)
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        set_bc(b, bcs, x, -1.0)

        res_norm = b.norm()

        if not np.isfinite(res_norm) or res_norm > 1e10:
            with b.localForm() as b_local:
                b_local.set(np.nan)
            print(
                f"\033[33m[Divergence Guard] Rank {b.comm.rank} injected NaN to force global divergence. (Local norm: {res_norm})\033[0m",
                flush=True,
            )
            
    except KeyboardInterrupt:
        raise
    except Exception as e:
        # Record the original internal exception so it can be reported outside PETSc.
        SOLVE_STATE.internal_error = f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}"
        raise


class Monolithic:
    def __init__(self, mesh, variables, time_steps: list = [1]):
        if not isinstance(time_steps, (list, tuple, np.ndarray)):
            raise TypeError("time_steps must be list/tuple/numpy.ndarray")

        if isinstance(mesh, dolfinx.mesh.Mesh):
            self.domain = mesh
        else:
            self.domain, self.mesh_markers, self.facets = mesh

        self.fields = FieldManager(self.domain)

        u_size = sum(size for _, size in variables)
        self.u = self.fields.register("u", shape=u_size, degree=1)

        counter = 0
        for name, size in variables:
            if u_size == 1:
                self.fields.registerVariable(name, self.u)
            else:
                if size > 1:
                    self.fields.registerVariable(
                        name, as_vector([self.u[counter + i] for i in range(size)])
                    )
                    counter += size
                elif size == 1:
                    self.fields.registerVariable(name, self.u[counter])
                    counter += 1

        self.t_steps = time_steps
        self.dt = fem.Constant(self.domain, PETSc.ScalarType(float(self.t_steps[0])))
        self.t_fem = fem.Constant(self.domain, PETSc.ScalarType(0.0))
        
        self._load_value = fem.Constant(self.domain, 0.0)
        self.variables = variables

        self.material_data = None
        self.kernels = None
        self.bc = None
        self.load_steps = None
        self.aux_kernels = []
        self.io = None
        self.auxSolverManager = None
        self.strategy = None

        self._petsc_options_prefix = "mp_"
        self._petsc_options = None

    def initialize(
        self,
        outputs,
        material,
        kernels,
        bc,
        load_steps,
        aux_kernels=[],
        postprocessors=[],
        petsc_options: dict | None = None,
        petsc_options_prefix: str = "mp_",
        cutback_strategy: BaseCutbackStrategy | None = None,
    ):
        self.material_data = material
        self.kernels = kernels
        self.bc = bc
        self.load_steps = load_steps
        self.aux_kernels = aux_kernels

        self._petsc_options_prefix = petsc_options_prefix
        self._petsc_options = petsc_options

        if cutback_strategy is None:
            self.strategy = AdaptiveCutbackStrategy(
                maxCuts=20, growthFactor=1.25, recoverFactor=2.0, fastConvIts=5
            )
        else:
            self.strategy = cutback_strategy

        self.auxSolverManager = AuxSolverManager(self.domain)
        self.io = SimulationIO(self.domain, outputs, postprocessors)

    def run(self, path="./result/output.bp", jump_material=False):
        self.io.log("Multiphysics base on Dolfinx10")
        rank = self.domain.comm.rank

        self.io.setupOutput(path)
        start = time.perf_counter()

        if not jump_material:
            MatClass, mat_params = self.material_data
            self._mat = MatClass(mat_params, self.fields, self.dt)

        self._buildPhysics()
        self._buildSolver()
        self.io.setupPostprocessing(path, self.fields, self._mat, self.action)
        self.io.printAndSaveParams(path, self._mat, self.kernels)

        self._setupAuxSolvers()
        self._checkLoadSteps()
        self._solveLoop()

        end = time.perf_counter()
        nsteps = self.n_steps if self.static else len(self.step_table)
        self.io.log(f"Total {nsteps} steps, Time: {end - start:.4f} s", flush=True)

        self.io.finalize()
        self.io.teardownOutput()

    def _solveLoop(self):
        rank = self.domain.comm.rank
        t = 0.0
        self.io.writeStep(t)

        strategy = self.strategy
        lastSuccessTargets = None

        for k, macro in enumerate(self.step_table):
            dtMacroVal = float(macro["dt"])
            endTargets = dict(macro["targets"])

            if lastSuccessTargets is None:
                startTargets = {bcIdx: 0.0 for bcIdx in endTargets.keys()}
            else:
                startTargets = {
                    bcIdx: float(lastSuccessTargets.get(bcIdx, 0.0))
                    for bcIdx in endTargets.keys()
                }

            strategy.reset(dtMacroVal)

            while not strategy.isDone():
                dtTry, alphaNext = strategy.getStepInfo()

                subTargets = {}
                for bcIdx, vEnd in endTargets.items():
                    v0 = startTargets.get(bcIdx, 0.0)
                    subTargets[bcIdx] = v0 + alphaNext * (float(vEnd) - float(v0))

                if rank == 0 and not self.static:
                    cuts = getattr(strategy, "cuts", 0)
                    self._logTrialStep(k, subTargets, lastSuccessTargets, dtTry, cuts)

                self.io.postman.run("preprocess")

                stepData = {"dt": dtTry, "targets": subTargets}
                success, tNew, nits, report = self._trialStep(t, stepData)

                isUb = self._isUnknownBlock(report)
                try:
                    strategy.update(success, n_its=nits, is_unknown_block=isUb)
                except RuntimeError as e:
                    if rank == 0:
                        self.io.log(f"\033[31mAborting: {e}\033[0m")
                    raise

                if success:
                    t = tNew
                    lastSuccessTargets = dict(subTargets)
                else:
                    if rank == 0:
                        self._logFailure(k, isUb, report)

    def _trialStep(self, t_old, step_data):
        t_start_step = time.perf_counter()
        comm = self.domain.comm
        dt = float(step_data["dt"])
        targets = step_data["targets"]
        
        t_new = t_old + dt
        self.t_fem.value = PETSc.ScalarType(t_new)
        self.dt.value = PETSc.ScalarType(dt)
        
        for bc_idx, val in targets.items():
            self.BCsManager.update(bc_idx, val)

        u_backup = self.u.x.array.copy()
        mat_backup = None
        if hasattr(self, "_mat") and hasattr(self._mat, "dump_state"):
            mat_backup = self._mat.dump_state()

        local_ok = 0
        local_num_its = -1
        local_report = None

        global SOLVE_STATE
        SOLVE_STATE.internal_error = None
        # Clear any stale terminal interrupt state.
        INTERRUPT_TRACKER.interrupted = False

        try:
            self.problem.solve()
            snes = self.problem.solver

            try:
                its = int(snes.getIterationNumber())
            except Exception:
                its = -1

            try:
                snes_reason = int(snes.getConvergedReason())
            except Exception:
                snes_reason = 0

            local_num_its = its

            if snes_reason > 0 and its >= 0:
                local_ok = 1
                local_report = None
            else:
                local_ok = 0
                try:
                    ksp_reason = int(snes.getKSP().getConvergedReason())
                except:
                    ksp_reason = "N/A"

                local_report = FailureReport(
                    source="SNES",
                    summary="Not Converged or Invalid Iterations",
                    details={
                        "snes_reason": snes_reason,
                        "ksp_reason": ksp_reason,
                        "its": its,
                        "note": (
                            "Iterations is -1 implies solve logic broken"
                            if its == -1
                            else ""
                        ),
                    },
                )

            if local_ok == 1:
                has_nan = not np.isfinite(self.u.x.array).all()
                if has_nan:
                    local_ok = 0
                    local_report = FailureReport(
                        source="NaNCheck",
                        summary="NaN detected in solution",
                        details={},
                    )
                    
        except KeyboardInterrupt:

            if comm.rank == 0:
                print("\n\033[31m[Interrupt] External interrupt: Ctrl+C detected; terminating the global simulation...\033[0m", flush=True)
            comm.Abort(130)
            os._exit(130)

        except PETSc.Error as e:
            err_code = e.ierr
            err_name = _PETSC_ERR_MAP.get(err_code, "UNKNOWN_ERR")

            # Use a PETSc-independent interrupt tracker to identify masked error-code failures.
            if INTERRUPT_TRACKER.interrupted or err_code == 59 or err_name == "ERR_SIG":
                if comm.rank == 0:
                    print(f"\n\033[31m[Interrupt] External interrupt: Ctrl+C (masked PETSc {err_code})...\033[0m", flush=True)
                # Use os._exit() to avoid deadlocks during forced termination.
                comm.Abort(130)
                os._exit(130)

            local_ok = 0
            # Recover the original code or material-model error hidden by PETSc.
            if SOLVE_STATE.internal_error:
                if comm.rank == 0:
                    print(f"\n\033[31m[Internal Error] Local update raised an exception (masked by PETSc {err_code} masked):\n{SOLVE_STATE.internal_error}\033[0m", flush=True)
                
                local_report = FailureReport(
                    source="PythonCallback",
                    summary="Exception in constitutive update or callback",
                    details={"traceback": SOLVE_STATE.internal_error},
                )
            else:
                if comm.rank == 0:
                    print(
                        f"\033[31m[PETSc Error] Code {err_code} ({err_name})\033[0m",
                        flush=True,
                    )
                local_report = FailureReport(
                    source="PETSc",
                    summary=f"Error code {err_code} ({err_name})",
                    details={"original_msg": str(e)},
                )
            
        except Exception as e:
            if isinstance(e, SystemExit):
                raise
                
            local_ok = 0
            local_report = FailureReport(
                source="Exception", summary=f"{type(e).__name__}: {e}", details={}
            )

        # Final interrupt guard.
        if INTERRUPT_TRACKER.interrupted:
            if comm.rank == 0:
                print("\n\033[31m[Interrupt] External interrupt: Ctrl+C detected; terminating the global simulation...\033[0m", flush=True)
            comm.Abort(130)
            os._exit(130)

        global_ok = comm.allreduce(local_ok, op=MPI.MIN)

        if global_ok == 0:
            self._log_memory("Start backup")
            self.u.x.array[:] = u_backup
            self.u.x.scatter_forward()
            if mat_backup is not None:
                self._mat.load_state(mat_backup)
            try:
                if hasattr(self._mat, "flux") and hasattr(
                    self._mat.flux, "ref_coefficient"
                ):
                    self._mat.flux.ref_coefficient.x.array[:] = 0.0
                    self._mat.flux.ref_coefficient.x.scatter_forward()
            except Exception:
                pass

            self._log_memory("Before Destroy")
            try:
                self.resetSolver()
            except Exception as e:
                if comm.rank == 0:
                    print(f"Warning: Failed to reset solver: {e}")

            self._log_memory("After Re-Build")
            return False, t_old, local_num_its, local_report

        if hasattr(self._mat, "postUpdate"):
            self._mat.postUpdate()
        self.u.x.scatter_forward()

        self.auxSolverManager.solveAll()

        self.io.writeStep(t_new)
        t_end_step = time.perf_counter()
        self.io.log(
            f"      Nonlinear Solving \033[32mConverged\033[0m after {local_num_its} iterations. (Time: {t_end_step - t_start_step:.4f} s)",
            flush=True,
        )

        return True, t_new, local_num_its, None

    def _setupAuxSolvers(self):
        self.auxSolverManager = AuxSolverManager(self.domain)

        for i, aux_setup in enumerate(self.aux_kernels):
            var = aux_setup[1]
            aux_action = ActionManager(var, self._mat, aux_setup)
            aux_action._dt = self.dt
            aux_action.computeResidual()
            aux_action.computeJacobian()
            prefix = f"aux{i}_"
            aux_problem = dolfinx.fem.petsc.NonlinearProblem(
                form(aux_action.residual),
                var,
                J=form(aux_action.jacobian),
                petsc_options_prefix=prefix,
            )
            self.auxSolverManager.register(var, aux_problem)

    def _checkLoadSteps(self):
        if self.load_steps[0] == "STATIC":
            self.static = True
            self.n_steps = int(self.load_steps[1])
            self.step_table = [
                {"dt": self.t_steps[k], "targets": {}} for k in range(self.n_steps)
            ]
        else:
            self.static = False
            n = len(self.load_steps[0][1])
            if n != len(self.t_steps):
                raise ValueError(
                    f"Inconsistent load steps ({n}) vs time steps ({len(self.t_steps)})"
                )

            self.step_table = []
            for k in range(n):
                targets_k = {bc_idx: arr[k] for bc_idx, arr in self.load_steps}
                self.step_table.append({"dt": self.t_steps[k], "targets": targets_k})

    def _isUnknownBlock(self, report):
        if not report:
            return False
        s = getattr(report, "summary", "").lower()
        if "unknown block" in s:
            return True
        details = getattr(report, "details", {}) or {}
        for v in details.values():
            if isinstance(v, str) and "unknown block" in v.lower():
                return True
        return False

    def _logTrialStep(self, stepIdx, subTargets, lastSuccessTargets, dtTry, cuts):
        if lastSuccessTargets is None:
            refTargets = {bcIdx: 0.0 for bcIdx in subTargets.keys()}
        else:
            refTargets = {
                bcIdx: float(lastSuccessTargets.get(bcIdx, 0.0))
                for bcIdx in subTargets.keys()
            }

        self.io.log(f"\033[35mSTEP {stepIdx+1} Load\033[0m", end="", flush=True)
        for bcIdx, v in subTargets.items():
            dval = float(v) - float(refTargets.get(bcIdx, 0.0))
            self.io.log(
                f"\033[35m BC[{bcIdx}]=>{float(v):.6g}, Δ={dval:+.6g}\033[0m",
                end="",
                flush=True,
            )
        self.io.log(f"\033[35m (dt={float(dtTry):.6g}, cut={cuts})\033[0m", flush=True)

    def _logFailure(self, stepIdx, isUb, report):
        self.io.log(
            f"      \033[31mNot converged: cutback inside STEP {stepIdx+1} (unknownblock={isUb}).\033[0m",
            flush=True,
        )
        self.io.log(
            f"      \033[31mFailure report:\033[0m {formatFailure(report)}",
            flush=True,
        )

    def _buildPhysics(self):
        self.domain.topology.create_connectivity(
            self.domain.topology.dim - 1, self.domain.topology.dim
        )

        self.action = ActionManager(self.u, self._mat, *self.kernels)
        self.action._dt = self.dt

        builder = BcsBuilder(
            self.domain, self.u.function_space, self.action.test_global
        )

        for bc in self.bc:
            if isinstance(bc[-1], str) and bc[-1] == "Tag":
                var, tag_id, bc_val, typ, _ = bc
                dof = (
                    -1
                    if len(var.ufl_operands) == 0
                    else int(var.ufl_operands[1].indices()[0])
                )
                builder.addCondition(
                    dof=dof,
                    tag_id=tag_id,
                    bc_value=bc_val,
                    bc_type=typ,
                    tags=True,
                    tag_facets=self.facets.find(tag_id),
                )
            else:
                var, ax, val, bc_val, typ = bc
                dof = (
                    -1
                    if len(var.ufl_operands) == 0
                    else int(var.ufl_operands[1].indices()[0])
                )
                builder.addCondition(
                    axis=ax, value=val, dof=dof, bc_value=bc_val, bc_type=typ
                )

        self.bcs = builder.build()
        self.BCsManager = builder

        self.action.computeResidual()
        self.action.residual += builder.residual
        self.action.computeJacobian()

    def _buildSolver(self):
        if self._petsc_options is None:
            petsc_opts = {
                "snes_type": "newtonls",
                "snes_linesearch_type": "basic",
                "snes_atol": 1e-9,
                "snes_rtol": 1e-6,
                "snes_max_it": 10,
                "ksp_type": "preonly",
                "pc_type": "lu",
                "pc_factor_mat_solver_type": "mumps",
            }
        else:
            petsc_opts = dict(self._petsc_options)

        self.problem = NonlinearProblem(
            self.action.residual,
            self.u,
            bcs=self.bcs,
            J=self.action.jacobian,
            petsc_options_prefix=self._petsc_options_prefix,
            petsc_options=petsc_opts,
        )

        if isinstance(self._mat, ExoMaterial):
            external_cb = self._mat.constitutiveUpdate(self.action.jacobian_ex)
        else:
            external_cb = None

        res_cb = partial(
            _assemble_residual_with_callback,
            self.problem.u,
            self.problem.F,
            self.problem.J,
            self.bcs,
            external_cb,
        )
        self.problem.solver.setFunction(res_cb, self.problem.b)

    def _destroySolver(self):
        if self.problem is not None:
            if hasattr(self.problem, "solver"):
                try:
                    self.problem.solver.setFunction(None, None)
                    self.problem.solver.setJacobian(None, None)
                except Exception:
                    pass
                self.problem.solver.destroy()

            if hasattr(self.problem, "A") and self.problem.A is not None:
                self.problem.A.destroy()
            if hasattr(self.problem, "b") and self.problem.b is not None:
                self.problem.b.destroy()

            self.problem = None

        import gc

        gc.collect()

    def resetSolver(self):
        self._destroySolver()
        self._buildSolver()

    def _log_memory(self, tag=""):
        if self.domain.comm.rank == 0:
            process = psutil.Process(os.getpid())
            mem_mb = process.memory_info().rss / 1024 / 1024
            print(f"\033[33m[MEM CHECK] {tag}: {mem_mb:.2f} MB\033[0m", flush=True)

