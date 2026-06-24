from Multiphysics.Materials import ExoMaterial
import jax.numpy as jnp
import jax
from fox.utils import (
    jax_nye_to_tensor as jaxn2t,
    jax_tensor_to_nye as jaxt2n,
    jax_nye_to_tensor_unsym as jax92t,
)
import ufl
from Multiphysics.tiliuTools.math import expm
from functools import partial
from Multiphysics.Functions.NonIsothermalEnergy import *

jax.config.update("jax_enable_x64", True)  # Set 64-bit arithmetic in JAX


class ADTVPkin(ExoMaterial):
    def outputShape(self):
        return 7

    def initHistoryArray(self):  # Initialize internal variables
        init_array = jnp.array(
            [
                1.0,  # Up
                1.0,
                1.0,
                0.0,
                0.0,
                0.0,
                0.0,  # kappa
                0.0,  # dlambda
                1.0,  # invUpi
                1.0,
                1.0,
                0.0,
                0.0,
                0.0,
                1.0,  # invUv
                1.0,
                1.0,
                0.0,
                0.0,
                0.0,
                1.0,  # C_n
                1.0,
                1.0,
                0.0,
                0.0,
                0.0,
            ]
        )

        return init_array

    def computeDrivingForce(self):
        Id = ufl.Identity(3)
        u_disp = self.fields.getVariable("displacement")
        temperature = self.fields.getVariable("temperature")
        F = Id + ufl.grad(u_disp)
        # GLS = 0.5 * (F.T * F - Id)
        self.driving_force = ufl.as_vector(
            [
                F[0, 0],
                F[1, 1],
                F[2, 2],
                F[0, 1],
                F[0, 2],
                F[1, 2],
                F[1, 0],
                F[2, 0],
                F[2, 1],
                temperature,
            ]
        )
        self.F = F

    def computeProperties(self):
        ExoMaterial.computeProperties(self)
        self.computeStress()
        self.computeHeatSource()

    def computeStress(self):
        self.stress = ufl.as_vector(
            [
                self.flux[0],
                self.flux[1],
                self.flux[2],
                self.flux[3],
                self.flux[4],
                self.flux[5],
            ]
        )

    def computeHeatSource(self):
        self.heat_source = self.flux[6]

    def computeQpFlux(self):
        self._FNS = self.computeFunctionDerivates(self.par)

        # First derivatives
        dPsi_plas_dRCGe = self._FNS["dpsi_plas_dRCGe"]  # ∂Ψᵖˡᵃˢ/∂RCGe
        dPsi_plas_dbpe = self._FNS["dpsi_plas_dbpe"]  # ∂Ψᵖˡᵃˢ/∂bpe
        dPsi_plas_dkappa = self._FNS["dpsi_plas_dkappa"]  # ∂Ψᵖˡᵃˢ/∂κ
        dPsi_AB_dRCGe = self._FNS["dpsi_AB_dRCGe"]  # ∂Ψᴬᴮ/∂RCGe

        # Dissipation-potential derivatives

        dg_vis_dY = self._FNS["dg_vis_dY"]  # ∂gᵛᶦˢ/∂Y
        dgkindTheta = self._FNS["dg_kin_dTheta"]  # ∂gᵏᶦⁿ/∂Θ

        # Yield function and derivatives
        Phi = self._FNS["Phi"]  # Φ(Y,R,T)
        dPhidY = self._FNS["dPhi_dY"]  # ∂Φ/∂Y
        dPhidR = self._FNS["dPhi_dR"]  # ∂Φ/∂R

        # Local residuum
        @jax.jit
        def local_residual_visco_elastic(L, hist_old, RCG, u_T, F, dt):
            invUv_n_nye = hist_old[0:6]
            invUv_nye = L[0:6]

            invUv = jaxn2t(invUv_nye, 2)
            invUv_n = jaxn2t(invUv_n_nye, 2)

            RCGe = invUv @ RCG @ invUv
            b = F @ jnp.transpose(F)

            dpsi_AB_dRCGe_ = dPsi_AB_dRCGe(RCGe, u_T)
            PK2 = 2.0 * invUv @ dpsi_AB_dRCGe_ @ invUv
            sigma = 1 / jnp.linalg.det(F) * F @ PK2 @ jnp.transpose(F)
            Y = 2.0 * RCGe @ dpsi_AB_dRCGe_
            Z = 2.0 * dt * dg_vis_dY(Y, u_T, b, sigma)

            res_Uv = invUv @ expm(Z) @ invUv - invUv_n @ invUv_n
            res_Uv_nye = jaxt2n(res_Uv, 1)
            res = res_Uv_nye

            return res

        @jax.jit
        def local_residual_elasto_plastic(L, hist_old, RCG, u_T):
            invUp_n_nye = hist_old[:6]
            invUpi_n_nye = hist_old[8:14]
            kappa_n = hist_old[6]
            invUp_nye = L[:6]
            invUpi_nye = L[8:14]
            kappa = L[6]
            dlambda = L[7]

            invUp = jaxn2t(invUp_nye, 2)
            invUpi = jaxn2t(invUpi_nye, 2)
            invUp_n = jaxn2t(invUp_n_nye, 2)
            invUpi_n = jaxn2t(invUpi_n_nye, 2)
            invCpi = invUpi @ invUpi
            RCGe = invUp @ RCG @ invUp
            Up = jnp.linalg.inv(invUp)
            bpe = Up @ invUpi @ invUpi @ Up

            dpsidRCGe_ = dPsi_plas_dRCGe(RCGe, bpe, kappa, u_T)
            dpsidbpe_ = dPsi_plas_dbpe(RCGe, bpe, kappa, u_T)

            Y = 2.0 * RCGe @ dpsidRCGe_
            X = 2.0 * bpe @ dpsidbpe_
            Sigma = Y - X
            R = dPsi_plas_dkappa(RCGe, bpe, kappa, u_T)
            Theta = invCpi @ Up @ dpsidbpe_ @ Up @ invCpi

            Z = 2.0 * dlambda * dPhidY(Sigma, R, u_T)
            Zpi = 2.0 * dlambda * dgkindTheta(Theta, u_T)

            res_Phi = Phi(Sigma, R, u_T)
            res_Up = invUp @ expm(Z) @ invUp - invUp_n @ invUp_n
            res_Upi = invUpi @ expm(Zpi) @ invUpi - invUpi_n @ invUpi_n
            res_Up_nye = jaxt2n(res_Up, 1)
            res_Upi_nye = jaxt2n(res_Upi, 1)
            res_kappa = kappa_n - kappa - dlambda * dPhidR(Sigma, R, u_T)
            res = jnp.concatenate(
                (res_Phi.reshape(1), res_Up_nye, res_kappa.reshape(1), res_Upi_nye)
            )

            return res

        self.computeTangentMatrix(local_residual_visco_elastic)
        self.computeTangentMatrix(local_residual_elasto_plastic)

        @jax.jit
        def flux_qp(driving_force, hist_old, glob_vals):
            """
            Core function. Performs the return-mapping
            algorithm on a Gauss-Point level. Takes the
            Green-Lagrange strain tensor and the history
            variables from the previos time step as inputs
            and calulates the 2nd Piola-Kirchoff stress
            tensors and the new history variables, alongside
            some additional control parameters.
            """
            F_nye = driving_force[:9]
            u_T = driving_force[9]
            invUp_n_nye = hist_old[:6]  # Plastic stretch from previous time step
            invUpi_n_nye = hist_old[8:14]  # Plastic stretch from previous time step
            kappa_n = hist_old[6]  # Accumulated plastic strain from previous time step
            F = jax92t(F_nye)
            RCG = jnp.transpose(F) @ F
            invUp_n = jaxn2t(invUp_n_nye, 2)
            invUpi_n = jaxn2t(invUpi_n_nye, 2)
            Up_n = jnp.linalg.inv(invUp_n)
            DT = glob_vals[0]
            # Trial step
            RCGe_trial = invUp_n @ RCG @ invUp_n
            bpe_trial = Up_n @ invUpi_n @ invUpi_n @ Up_n
            dpsidRCGe_trial = dPsi_plas_dRCGe(RCGe_trial, bpe_trial, kappa_n, u_T)
            dpsidbpe_trial = dPsi_plas_dbpe(RCGe_trial, bpe_trial, kappa_n, u_T)

            PK2_trial = 2.0 * invUp_n @ dpsidRCGe_trial @ invUp_n
            Y_trial = 2.0 * RCGe_trial @ dpsidRCGe_trial
            X_trial = 2.0 * bpe_trial @ dpsidbpe_trial
            Gam_trial = Y_trial - X_trial
            R_trial = dPsi_plas_dkappa(RCGe_trial, bpe_trial, kappa_n, u_T)

            Phi_trial = Phi(Gam_trial, R_trial, u_T)

            TOL_COND = 1.0e-10  # Tolerance for the return-mapping condition
            state_cond_initial = (
                PK2_trial,
                hist_old,
                RCG,
                F,
                DT,
                RCGe_trial,
                bpe_trial,
            )

            def elastic_step(state_cond):
                niter = 0
                converged = True
                history = jnp.zeros(50)
                return (
                    state_cond[0],
                    state_cond[1],
                    converged,
                    niter,
                    history,
                    0.0,
                    state_cond[5],
                    state_cond[6],
                    jaxn2t(state_cond[1][0:6], 2),
                )

            @jax.jit
            def viscoelastic_step(hist_old, RCG, F, dt):

                L0 = hist_old  # Initial local solution vector

                h1, converged, niter, history = self.internalVariableUpdate(
                    L0,
                    14,
                    20,
                    local_residual_visco_elastic,
                    hist_old[14:20],
                    RCG,
                    u_T,
                    F,
                    dt,
                )
                invUv_nye = h1[14:20]
                invUv = jaxn2t(invUv_nye, 2)

                RCGev = invUv @ RCG @ invUv
                b = F @ jnp.transpose(F)

                dpsi_AB_dRCGe_ = dPsi_AB_dRCGe(RCGev, u_T)
                PK2 = 2.0 * invUv @ dpsi_AB_dRCGe_ @ invUv
                sigma = 1 / jnp.linalg.det(F) * F @ PK2 @ jnp.transpose(F)
                Y = 2.0 * RCGev @ dpsi_AB_dRCGe_
                dv = dt * dg_vis_dY(Y, u_T, b, sigma)
                dYvdT = jax.jacfwd(lambda T: 2.0 * RCGev @ dPsi_AB_dRCGe(RCGev, T))(u_T)
                rv = jnp.sum((Y - u_T * dYvdT) * dv)

                return PK2, h1, converged, niter, history, rv, RCGev, invUv

            @jax.jit
            def plastic_step(state_cond_n):
                hist_old = state_cond_n[1]
                RCG = state_cond_n[2]

                L0 = hist_old.at[7].set(0.0)  # Initial local solution vector

                h1, converged, niter, history = self.internalVariableUpdate(
                    L0,
                    0,
                    14,
                    local_residual_elasto_plastic,
                    hist_old[0:14],
                    RCG,
                    u_T,
                )
                kappa = h1[6]
                dlambda = h1[7]
                invUp_nye = h1[:6]
                invUpi_nye = h1[8:14]

                invUp = jaxn2t(invUp_nye, 2)
                invUpi = jaxn2t(invUpi_nye, 2)
                invCpi = invUpi @ invUpi
                RCGep = invUp @ RCG @ invUp
                Up = jnp.linalg.inv(invUp)
                bpe = Up @ invUpi @ invUpi @ Up

                dpsidRCGe_ = dPsi_plas_dRCGe(RCGep, bpe, kappa, u_T)
                dpsidbpe_ = dPsi_plas_dbpe(RCGep, bpe, kappa, u_T)

                Yp = 2.0 * RCGep @ dpsidRCGe_
                X = 2.0 * bpe @ dpsidbpe_
                Sigma = Yp - X
                Rp = dPsi_plas_dkappa(RCGep, bpe, kappa, u_T)
                Theta = invCpi @ Up @ dpsidbpe_ @ Up @ invCpi

                dp = dlambda * dPhidY(Sigma, Rp, u_T)
                Cpidot = 2.0 * dlambda * dgkindTheta(Theta, u_T)

                PK2 = 2.0 * invUp @ dpsidRCGe_ @ invUp
                dYdT = jax.jacfwd(
                    lambda T: 2.0
                    * (
                        RCGep @ dPsi_plas_dRCGe(RCGep, bpe, kappa, T)
                        - bpe @ dPsi_plas_dbpe(RCGep, bpe, kappa, T)
                    )
                )(u_T)
                dThetadT = jax.jacfwd(
                    lambda T: 2.0 * bpe @ dPsi_plas_dbpe(RCGep, bpe, kappa, T)
                )(u_T)
                dRpdT = jax.grad(lambda T: dPsi_plas_dkappa(RCGep, bpe, kappa, T))(u_T)
                rp = (
                    jnp.sum((Yp - u_T * dYdT) * dp)
                    + dlambda * (Rp - u_T * dRpdT)
                    + jnp.sum((Theta - u_T * dThetadT) * Cpidot)
                )

                return PK2, h1, converged, niter, history, rp, RCGep, bpe, invUp

            re_v = 0
            re_p = 0
            rp = 0
            rv = 0
            # plastic return mapping
            PK2_p, h1, converged_p, niter_p, history_p, rp, RCGpe, bpe, invUp = (
                jax.lax.cond(
                    Phi_trial <= TOL_COND,
                    elastic_step,
                    plastic_step,
                    state_cond_initial,
                )
            )

            kappa = h1[6]

            dPK2_p_dT = jax.jacfwd(
                lambda T: 2.0 * invUp @ dPsi_plas_dRCGe(RCGpe, bpe, kappa, T) @ invUp
            )(u_T)
            RCG_n = jaxn2t(hist_old[-6:], 2)
            re_p = 0.5 * u_T * jnp.sum(dPK2_p_dT * (RCG - RCG_n))

            # viscoelastic internal variables update
            PK2_v, h1, converged_v, niter_v, history_v, rv, RCGve, invUv = (
                viscoelastic_step(
                    h1,
                    RCG,
                    F,
                    DT,
                )
            )
            dPK2dT_v = jax.jacfwd(
                lambda T: 2.0 * invUv @ dPsi_AB_dRCGe(RCGve, T) @ invUv
            )(u_T)
            re_v = 0.5 * u_T * jnp.sum(dPK2dT_v * (RCG - RCG_n))
            r_ht = re_p + rp + re_v + rv
            PK2 = PK2_p + PK2_v
            PK2_vec = jaxt2n(PK2, 1)  # shape (6,)

            h1 = h1.at[-6:].set(jaxt2n(RCG, 2))

            out = jnp.concatenate([PK2_vec, jnp.array([r_ht])])  # shape (13,)
            return out, (h1,)

        self.qp_flux = flux_qp

    @staticmethod
    def computeFunctionDerivates(pars):
        # Bind pars
        psi_plas_b = partial(psi_plas, pars=pars)
        psi_AB_b = partial(psi_AB, pars=pars)
        g_vis_b = partial(g_vis, pars=pars)
        g_kin_b = partial(g_kin, pars=pars)  # Added

        Phi_b = partial(Phi, pars=pars)

        fns = {
            # Free energy and gradients
            "psi_plas": psi_plas_b,
            "dpsi_plas_dRCGe": jax.jit(jax.grad(psi_plas_b, 0)),
            "dpsi_plas_dbpe": jax.jit(jax.grad(psi_plas_b, 1)),
            "dpsi_plas_dkappa": jax.jit(jax.grad(psi_plas_b, 2)),
            "psi_AB": psi_AB_b,
            "dpsi_AB_dRCGe": jax.grad(psi_AB_b, 0),
            # Viscoelastic dissipation potential
            "g_vis": g_vis_b,
            "dg_vis_dY": jax.jit(jax.grad(g_vis_b, 0)),
            # Kinematic-hardening dissipation potential
            "g_kin": g_kin_b,
            "dg_kin_dTheta": jax.jit(jax.grad(g_kin_b, 0)),
            # Yield function
            "Phi": Phi_b,
            "dPhi_dY": jax.jit(jax.grad(Phi_b, 0)),
            "dPhi_dR": jax.jit(jax.grad(Phi_b, 1)),
        }
        return fns
