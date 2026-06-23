import numpy as np
from mpi4py import MPI


class AuxSolverManager:
    """
    Manages the solving process for auxiliary variables (e.g. heat, history vars).
    Compatible with dolfinx 0.10+ fem.petsc.NonlinearProblem (SNES wrapper).

    Features:
      - solve each aux problem sequentially
      - NaN/Inf detection
      - global ok synchronization
      - rollback on failure without crashing the main simulation
    """

    def __init__(self, domain):
        self.domain = domain
        self.comm = domain.comm
        self._problems = []  # list[(var, problem)]

    def register(self, variable, problem):
        """
        variable: dolfinx.fem.Function
        problem: dolfinx.fem.petsc.NonlinearProblem  (dolfinx 0.10 interface)
        """
        self._problems.append((variable, problem))

    def solveAll(self):
        for var, problem in self._problems:
            self._solveSingle(var, problem)

    def _solveSingle(self, var, problem):
        comm = self.comm

        # 1) Backup state
        var_backup = var.x.array.copy()

        ok_local = 1
        its_local = -1
        reason_local = None

        try:
            # Ensure starting vector is consistent across ranks (optional, safe)
            try:
                var.x.scatter_forward()
            except Exception:
                pass

            # 2) Solve using the new NonlinearProblem wrapper
            # dolfinx.fem.petsc.NonlinearProblem provides:
            #   - problem.solve()
            #   - problem.solver (PETSc.SNES)
            if not hasattr(problem, "solve") or not hasattr(problem, "solver"):
                raise TypeError(
                    f"AuxSolverManager expects dolfinx.fem.petsc.NonlinearProblem, got {type(problem)}"
                )

            problem.solve()

            # 3) Read SNES status
            snes = problem.solver
            its_local = int(snes.getIterationNumber())
            reason_local = int(snes.getConvergedReason())

            # reason_local > 0 means converged in PETSc
            converged = reason_local > 0

            # 4) Sync and NaN check
            var.x.scatter_forward()
            has_nan = not np.isfinite(var.x.array).all()

            if (not converged) or has_nan:
                ok_local = 0

        except Exception as e:
            ok_local = 0
            reason_local = f"{type(e).__name__}: {e}"
            if comm.rank == 0:
                print(
                    f"[WARN] Aux solve raised exception: {type(e).__name__}: {e}",
                    flush=True,
                )

        # 5) Global decision: any rank fails => rollback everywhere
        ok_global = comm.allreduce(ok_local, op=MPI.MIN)

        if ok_global == 0:
            var.x.array[:] = var_backup
            var.x.scatter_forward()
            try:
                snes = problem.solver  # PETSc.SNES
                ksp = snes.getKSP()
                pc = ksp.getPC()
                ksp.reset()
                pc.reset()
                #snes.reset()
            except Exception:
                pass
            if comm.rank == 0:
                print(
                    f"[WARN] Aux solve failed or produced NaN. "
                    f"Rollback applied. (its={its_local}, reason={reason_local})",
                    flush=True,
                )
        else:
            if comm.rank == 0:
                print(
                    f"[INFO] Aux solve converged. (its={its_local}, reason={reason_local})",
                    flush=True,
                )
