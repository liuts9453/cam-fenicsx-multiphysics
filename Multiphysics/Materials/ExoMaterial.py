import jax
import jax.numpy as jnp
from dolfinx_external_operator import (
    FEMExternalOperator,
    evaluate_external_operators,
    evaluate_operands,
)
from dolfinx import fem
import basix
import numpy as np
from fox.utils import timeit
from functools import partial
from mpi4py import MPI


class ExoMaterial:
    """
    Abstract base class for materials using ExternalOperators (FEniCSx + JAX).

    This structure is inspired by generalized constitutive modeling, where:
      - 'driving_force' or gradient potential represents the state input (e.g., strain, temperature gradient).
      - 'flux' represents the material response (e.g., stress, heat flux).

    Key functionality:
      - `computeDrivingForce()`: Define UFL input to the constitutive model.
      - `computeQpFlux()`: Define per-quadrature-point flux computation.
      - `Flux`: FEMExternalOperator used in UFL variational forms.
      - `hist`: Stores internal variables at quadrature points.
      - `computeJacobian()`: Builds and registers flux and tangent operator.

    To use:
      - Subclass ExoMaterial.
      - Implement `outputShape()`, `initHistoryArray()`, `computeDrivingForce()`, `computeQpFlux()`.
    """

    def __init__(self, mat_pars, fields, dt, max_local=10):
        """
        Parameters:
            domain: dolfinx.mesh.Mesh
            fields: Fieldsmanager
        """
        self.fields = fields
        self.par = mat_pars
        self.domain = self.fields.domain
        self.output_shape = self.outputShape()
        self.hist_array = self.initHistoryArray()
        self.hist_len = len(self.hist_array)
        self._initHistoriyVariables()

        cell = self.domain.basix_cell()
        quadrature_element = basix.ufl.quadrature_element(
            cell, degree=2, value_shape=(self.output_shape,)
        )
        self.function_space = fem.functionspace(self.domain, quadrature_element)
        self.function = fem.Function(self.function_space)
        self._dt = dt
        self.max_local_its = max_local
        self.dumm_hist = None
        self.computeGlobalValues()
        self.computeJacobian()

    def computeDrivingForce(self):
        """Define the primary external-operator driving force."""
        raise NotImplementedError("driving force must define in computeDrivingForce()")

    def buildFlux(self):

        self.flux = FEMExternalOperator(
            self.driving_force, function_space=self.function_space
        )
        self.flux.external_function = self.fluxExternal

    def outputShape(self):
        raise NotImplementedError("Subclasses must define outputShape()")

    def initHistoryArray(self):
        raise NotImplementedError("hist_array must be assign by initHistoryArray()")

    def _initHistoriyVariables(self):

        hist_e = basix.ufl.quadrature_element(
            self.domain.basix_cell(), degree=2, value_shape=(self.hist_len,)
        )  # For internal variables (at the quadrature point)

        G_func_hist = fem.functionspace(self.domain, hist_e)
        self.hist = fem.Function(G_func_hist)
        num_quadpoints = self.hist.x.array.size // self.hist_len
        init_values = np.tile(self.hist_array, num_quadpoints)
        self.hist.x.array[:] = init_values

    def computeQpFlux(self):
        self.computeFunctionDerivatives()
        self.qp_flux = self.flux_qp

    def computeFunctionDerivatives(self):
        """
        Register material-specific derivative functions.

        Concrete material subclasses must implement this method. Older material
        files may still provide the legacy misspelled method
        `computeFunctionDerivates`, which is accepted as a fallback.
        """
        legacy_impl = type(self).__dict__.get("computeFunctionDerivates")
        if legacy_impl is not None:
            return legacy_impl(self)

        raise NotImplementedError(
            "computeFunctionDerivatives() must be implemented in the concrete "
            "material subclass. For older files, the legacy method name "
            "computeFunctionDerivates() is also accepted."
        )

    def computeFunctionDerivates(self):
        """
        Backward-compatible alias for older material implementations.

        New material classes should implement computeFunctionDerivatives().
        """
        return self.computeFunctionDerivatives()

    def computeFlux(self):
        self.computeQpFlux()
        self._flux = jax.jit(
            jax.vmap(self.qp_flux, in_axes=(0, 0, None, None)), static_argnums=(3,)
        )

    def computeProperties(self):
        self.computeDrivingForce()
        self.computeFlux()
        self.buildFlux()

    def computeGlobalValues(self):

        self.global_vals = jnp.asarray([self._dt.value])

    def computeJacobian(self):
        self.computeProperties()
        _qp_jac = self.computeQpJacobian()
        self.jac = jax.jit(jax.vmap(_qp_jac, in_axes=(0, 0, None, None)))

    def computeQpJacobian(self):
        return jax.jacfwd(self.qp_flux, has_aux=True)

    def constitutiveUpdate(self, J_external_operators):
        """
         (Flux)  / (Tangent & History) 
         Jacobian 
        """

        def _constitutive_update():
            # ==========================================================

            # ==========================================================
            ops_F = [self.flux]
            eval_ops_F = evaluate_operands(ops_F)
            res_F = evaluate_external_operators(ops_F, eval_ops_F)




            fl_coeff = res_F[0]


            if isinstance(fl_coeff, (tuple, list)):
                fl_coeff = fl_coeff[0]


            fl_coeff_flat = np.asarray(fl_coeff).reshape(-1)

            # ===== FEM-level protection =====
            if not np.isfinite(fl_coeff_flat).all():
                raise RuntimeError(
                    "[FEMExternalOperator] Non-finite flux coefficient detected. "
                    "Aborting update before assembly."
                )

            self.flux.ref_coefficient.x.array[:] = fl_coeff_flat

            # ==========================================================

            # ==========================================================
            foo = None
            if J_external_operators:
                eval_ops_J = evaluate_operands(J_external_operators)
                res_J = evaluate_external_operators(J_external_operators, eval_ops_J)


                if not isinstance(res_J, (list, tuple)):
                    res_J = [res_J]

                for r in res_J:


                    if isinstance(r, (tuple, list)) and len(r) == 2:

                        foo = r[1]


            if foo is not None:

                foo_flat = np.asarray(foo).reshape(-1)
                if not np.all(np.isfinite(foo_flat)):
                    raise RuntimeError("[Material] Non-finite dummy history detected.")
                self.dumm_hist = foo_flat

        return _constitutive_update

    def postUpdate(self):
        self.hist.x.array[:] = self.dumm_hist

    def evaluateFlux(self, variables):
        vars_ = variables.reshape((-1, self.driving_force.ufl_shape[0]))
        hist_ = self.hist.x.array[:].reshape((-1, self.hist_len))
        self.computeGlobalValues()
        fl_, _ = self._flux(vars_, hist_, self.global_vals, self.par)
        return fl_.reshape(-1)

    def evaluateTangent(self, variables):
        vars_ = variables.reshape((-1, self.driving_force.ufl_shape[0]))
        old_hist = self.hist.x.array[:].reshape((-1, self.hist_len))
        self.computeGlobalValues()


        cijkl_, aux = self.jac(vars_, old_hist, self.global_vals, self.par)

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

            raise RuntimeError(
                "Local constitutive update did not converge on some rank"
            )


        cijkl_ = np.nan_to_num(np.array(cijkl_), nan=0.0, posinf=0.0, neginf=0.0)
        new_hist = np.nan_to_num(np.array(new_hist), nan=0.0, posinf=0.0, neginf=0.0)
        return cijkl_.reshape(-1), new_hist.reshape(-1)

    def fluxExternal(self, derivatives):
        if derivatives == (0,):
            return self.evaluateFlux
        elif derivatives == (1,):
            return self.evaluateTangent
        else:
            raise NotImplementedError()

    def dump_state(self):
        """hist  dumm_hist"""
        return {
            "hist": self.hist.x.array.copy(),
            "dumm_hist": (
                None if self.dumm_hist is None else np.array(self.dumm_hist, copy=True)
            ),
        }

    def load_state(self, snap):
        """"""
        self.hist.x.array[:] = snap["hist"]
        if snap["dumm_hist"] is None:
            self.dumm_hist = None
        else:
            self.dumm_hist = np.array(snap["dumm_hist"], copy=True)

    @staticmethod
    def printConvergeHistory(converged, niter, history):
        mean_history = jnp.mean(history, axis=0)
        history_trimmed = jnp.trim_zeros(mean_history, trim="b")

        conv_global = jnp.all(converged)  # all QPs converged?
        niter_max = jnp.max(niter)  # max local iterations

        # JAX scalar -> Python scalar (works for both CPU/GPU)
        conv_bool = bool(jnp.asarray(conv_global))
        niter_int = int(jnp.asarray(niter_max))

        print("      Solving internal varialbles...", flush=True)

        # Build the multiline "Local |R|" block without nested f-strings (Py<=3.11 safe)
        ht = jnp.asarray(history_trimmed)
        if ht.size == 0:
            block = "            Local |R| = : (empty)\n"
        else:
            first = float(ht[0])
            rest = [float(x) for x in ht[1:]]
            lines = [f"            Local |R| = : {first}\n"]
            lines.extend(f"                          {v}\n" for v in rest)
            block = "".join(lines)

        print(block, end="", flush=True)

        status = "converged" if conv_bool else "\033[31m not\033[0m converged"
        print(
            f"            The local residuum has {status} in {niter_int} steps",
            flush=True,
        )

    @staticmethod
    def internalVariableUpdate(L0, begin, end, func, *args):

        sol, converged, niter_total, history = ExoMaterial.localNewton(
            L0[begin:end], func, *args
        )

        def _accept():
            return L0.at[begin:end].set(sol)

        def _reject():
            return L0

        h1 = jax.lax.cond(converged, _accept, _reject)

        return h1, converged, niter_total, history

    @staticmethod
    def computeDerivates(func, *args):
        if args:
            for i in args:
                name = "d" + str(i)
                setattr(func, name, jax.grad(func, argnums=i))
        else:
            func.d = jax.grad(func)

    @staticmethod
    def computeTangentMatrix(func, *args):
        if args:
            for i in args:
                name = ("j" + str(i)) if i > 0 else "j"
                setattr(func, name, jax.jacfwd(func, argnums=i))
        else:
            func.j = jax.jacfwd(func)

    @staticmethod
    def localNewton(sol, func, *args):

        NITER_MAX = 10
        TOL_LOOP = 1.0e-7

        LS_MAX = 8
        LS_BETA = 0.5
        LM_DAMP = 1.0e-10

        j_func = func.j

        def all_finite(x):
            return jnp.all(jnp.isfinite(x))

        # initial residual
        res0 = func(sol, *args)
        norm0 = jnp.linalg.norm(res0)

        def bad_initial():
            hist = jnp.zeros(NITER_MAX + 1)
            hist = hist.at[0].set(jnp.inf)
            it0 = jnp.array(0, dtype=jnp.int32)
            conv0 = jnp.array(False)
            return sol, conv0, it0, hist

        def good_initial():
            history = jnp.zeros(NITER_MAX + 1)
            history = history.at[0].set(norm0)

            state0 = {
                "it": jnp.array(0, dtype=jnp.int32),
                "x": sol,
                "r": res0,
                "nr": norm0,
                "hist": history,
                "finite_ok": jnp.array(True),
            }

            def cond_fun(st):
                return (st["nr"] > TOL_LOOP) & (st["it"] < NITER_MAX) & st["finite_ok"]

            def body_fun(st):
                it = st["it"]
                x = st["x"]
                r = st["r"]
                nr = st["nr"]
                hist = st["hist"]
                finite_ok = st["finite_ok"]

                # Jacobian + LM regularization
                J = j_func(x, *args)


                J_ok = all_finite(J)
                r_ok = all_finite(r) & jnp.isfinite(nr) & all_finite(x)
                finite_ok = finite_ok & J_ok & r_ok

                def fail_path():
                    it_new = it + 1
                    hist_new = hist.at[it_new].set(jnp.inf)
                    return {
                        "it": it_new,
                        "x": x,
                        "r": r,
                        "nr": jnp.inf,
                        "hist": hist_new,
                        "finite_ok": jnp.array(False),
                    }

                def solve_path():
                    J_inf = jnp.max(jnp.abs(J))
                    lm = LM_DAMP * (1.0 + J_inf)
                    J_reg = J + lm * jnp.eye(J.shape[0], dtype=J.dtype)

                    dx = jnp.linalg.solve(J_reg, -r)
                    dx_ok = all_finite(dx)
                    finite_ok2 = finite_ok & dx_ok

                    def try_step(alpha):
                        x_try = x + alpha * dx
                        r_try = func(x_try, *args)
                        nr_try = jnp.linalg.norm(r_try)
                        ok = (
                            all_finite(x_try) & all_finite(r_try) & jnp.isfinite(nr_try)
                        )
                        return x_try, r_try, nr_try, ok

                    # full step
                    x_try, r_try, nr_try, ok = try_step(1.0)

                    def ls_cond(ls_st):
                        return (((ls_st["nr_try"] > nr) | (~ls_st["ok"]))) & (
                            ls_st["ls_it"] < LS_MAX
                        )

                    def ls_body(ls_st):
                        alpha = ls_st["alpha"] * LS_BETA
                        x_t, r_t, nr_t, ok_t = try_step(alpha)
                        return {
                            "ls_it": ls_st["ls_it"] + 1,
                            "alpha": alpha,
                            "x_try": x_t,
                            "r_try": r_t,
                            "nr_try": nr_t,
                            "ok": ok_t,
                        }

                    ls0 = {
                        "ls_it": jnp.array(0, dtype=jnp.int32),
                        "alpha": jnp.array(1.0, dtype=x.dtype),
                        "x_try": x_try,
                        "r_try": r_try,
                        "nr_try": nr_try,
                        "ok": ok,
                    }
                    lsF = jax.lax.while_loop(ls_cond, ls_body, ls0)


                    accept = lsF["ok"] & (lsF["nr_try"] <= nr)


                    finite_ok3 = finite_ok2 & lsF["ok"]

                    x_new = jnp.where(accept, lsF["x_try"], x)
                    r_new = jnp.where(accept, lsF["r_try"], r)

                    nr_acc = jnp.where(accept, lsF["nr_try"], nr)
                    nr_new = jnp.where(finite_ok3, nr_acc, jnp.inf)

                    it_new = it + 1
                    hist_new = hist.at[it_new].set(nr_new)

                    return {
                        "it": it_new,
                        "x": x_new,
                        "r": r_new,
                        "nr": nr_new,
                        "hist": hist_new,
                        "finite_ok": finite_ok3,
                    }

                return jax.lax.cond(finite_ok, solve_path, fail_path)

            stF = jax.lax.while_loop(cond_fun, body_fun, state0)

            converged = (
                stF["finite_ok"] & (stF["nr"] <= TOL_LOOP) & all_finite(stF["x"])
            )
            return stF["x"], converged, stF["it"], stF["hist"]

        return jax.lax.cond(
            all_finite(res0) & jnp.isfinite(norm0) & all_finite(sol),
            good_initial,
            bad_initial,
        )
