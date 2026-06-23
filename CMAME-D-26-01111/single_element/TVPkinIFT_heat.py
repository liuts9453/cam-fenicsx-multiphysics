from functools import partial
from typing import NamedTuple

from Multiphysics.Materials import ExoMaterialIFT
import jax.numpy as jnp
import jax
import jax.scipy as jsp  # Use jax.scipy for small batched linear solves on GPU
from fox.utils import (
    jax_nye_to_tensor as jaxn2t,
    jax_tensor_to_nye as jaxt2n,
    jax_nye_to_tensor_unsym as jax92t,
)
import ufl
from Multiphysics.tiliuTools.math import expm, spd_norm2, sym_norm2, inv3x3, det3x3

jax.config.update("jax_enable_x64", True)


class PlasticState(NamedTuple):
    u_T: jnp.ndarray
    F: jnp.ndarray
    RCG: jnp.ndarray
    kappa: jnp.ndarray
    dlambda: jnp.ndarray
    invUp: jnp.ndarray
    invUpi: jnp.ndarray
    Up: jnp.ndarray
    invCpi: jnp.ndarray
    RCGep: jnp.ndarray
    bpe: jnp.ndarray
    Yp: jnp.ndarray
    Xp: jnp.ndarray
    Sigma: jnp.ndarray
    Rp: jnp.ndarray
    Theta: jnp.ndarray


class PlasticTrialState(NamedTuple):
    u_T: jnp.ndarray
    F: jnp.ndarray
    RCG: jnp.ndarray
    invUp_n: jnp.ndarray
    invUpi_n: jnp.ndarray
    Up_n: jnp.ndarray
    kappa_n: jnp.ndarray
    RCGe_trial: jnp.ndarray
    bpe_trial: jnp.ndarray
    Y_trial: jnp.ndarray
    X_trial: jnp.ndarray
    Gam_trial: jnp.ndarray
    R_trial: jnp.ndarray
    Phi_trial: jnp.ndarray


class ViscoState(NamedTuple):
    u_T: jnp.ndarray
    F: jnp.ndarray
    RCG: jnp.ndarray
    b: jnp.ndarray
    invUv: jnp.ndarray
    RCGev: jnp.ndarray
    PK2v: jnp.ndarray
    sigma: jnp.ndarray
    Yv: jnp.ndarray
    norm_b: jnp.ndarray
    norm_sigma: jnp.ndarray


class TVPkinIFT_heat(ExoMaterialIFT):
    """Thermomechanical PA6 material with IFT-based consistent tangent assembly."""

    IDX_INVUP = slice(0, 6)
    IDX_KAPPA = 6
    IDX_DLAM = 7
    IDX_INVUPI = slice(8, 14)
    IDX_INVUV = slice(14, 20)

    IDX_PHI_TRIAL = 20
    IDX_RE_P = 21
    IDX_RP = 22
    IDX_RE_V = 23
    IDX_RV = 24

    IDX_C_N = slice(25, 31)
    IDX_IS_PLASTIC = 31

    HISTORY_SIZE = 32

    @classmethod
    def localResidualBlockNames(cls):
        return ["Plastic", "Visco"]

    def outputShape(self):
        return 7

    def initHistoryArray(self):
        init_array = jnp.array(
            [
                # invUp (0:6)
                1.0,
                1.0,
                1.0,
                0.0,
                0.0,
                0.0,
                # kappa (6)
                0.0,
                # dlambda (7)
                0.0,
                # invUpi (8:14)
                1.0,
                1.0,
                1.0,
                0.0,
                0.0,
                0.0,
                # invUv (14:20)
                1.0,
                1.0,
                1.0,
                0.0,
                0.0,
                0.0,
                # Phi_trial (20)
                -1.0,
                # re_p (21)
                0.0,
                # rp (22)
                0.0,
                # re_v (23)
                0.0,
                # rv (24)
                0.0,
                # C_n (25:31)
                1.0,
                1.0,
                1.0,
                0.0,
                0.0,
                0.0,
                # is_plastic (31)
                0.0,
            ],
            dtype=jnp.float64,
        )
        return init_array

    def computeDrivingForce(self):
        Id = ufl.Identity(3)
        u_disp = self.fields.getVariable("displacement")
        temperature = self.fields.getVariable("temperature")
        F = Id + ufl.grad(u_disp)
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

    def computeFunctionDerivates(self):
        # -------------------------------------------------------------
        # Register only the local residual Jacobian required by ExoMaterial.localNewton (.j)
        # Other derivatives of the free energy (Psi), yield surface (Phi), and flux are
        # handled through jac_multi/grad_multi graph consolidation and need not be registered again.
        # -------------------------------------------------------------
        self.computeTangentMatrix(self.local_residual_elasto_plastic, 0)
        self.computeTangentMatrix(self.local_residual_visco_elastic, 0)

    def computeProperties(self):
        ExoMaterialIFT.computeProperties(self)
        self.computeStress()
        self.computeHeatSource()

    @staticmethod
    def buildPlasticTrialState(driving_force, hist_n, par):
        F_nye = driving_force[:9]
        u_T = driving_force[9]

        F = jax92t(F_nye)
        RCG = jnp.transpose(F) @ F

        invUp_n_nye = hist_n[TVPkinIFT_heat.IDX_INVUP]
        invUpi_n_nye = hist_n[TVPkinIFT_heat.IDX_INVUPI]
        kappa_n = hist_n[TVPkinIFT_heat.IDX_KAPPA]

        invUp_n = jaxn2t(invUp_n_nye, 2)
        invUpi_n = jaxn2t(invUpi_n_nye, 2)
        Up_n = inv3x3(invUp_n)

        RCGe_trial = invUp_n @ RCG @ invUp_n
        bpe_trial = Up_n @ invUpi_n @ invUpi_n @ Up_n

        # Graph consolidation: evaluate all trial-state free-energy derivatives with respect to internal variables in one call.
        dpsidRCGe_trial, dpsidbpe_trial, R_trial = jax.grad(
            Psi_plas, argnums=(0, 1, 2)
        )(RCGe_trial, bpe_trial, kappa_n, u_T, par)

        Y_trial = 2.0 * RCGe_trial @ dpsidRCGe_trial
        X_trial = 2.0 * bpe_trial @ dpsidbpe_trial
        Gam_trial = Y_trial - X_trial
        Phi_trial = Phi(Gam_trial, R_trial, u_T, par)

        return PlasticTrialState(
            u_T=u_T,
            F=F,
            RCG=RCG,
            invUp_n=invUp_n,
            invUpi_n=invUpi_n,
            Up_n=Up_n,
            kappa_n=kappa_n,
            RCGe_trial=RCGe_trial,
            bpe_trial=bpe_trial,
            Y_trial=Y_trial,
            X_trial=X_trial,
            Gam_trial=Gam_trial,
            R_trial=R_trial,
            Phi_trial=Phi_trial,
        )

    @staticmethod
    def computePlasticTrialFlag(driving_force, hist_n, par, tol=1.0e-10):
        trial = TVPkinIFT_heat.buildPlasticTrialState(driving_force, hist_n, par)
        is_plastic = trial.Phi_trial > tol
        return is_plastic, trial

    @staticmethod
    def flux_qp(driving_force, hist_old, glob_vals, par):
        cls = TVPkinIFT_heat

        DT = glob_vals[0]

        F_nye = driving_force[:9]
        F = jax92t(F_nye)
        RCG = jnp.transpose(F) @ F

        TOL_COND = 1.0e-10
        is_plastic, trial = cls.computePlasticTrialFlag(
            driving_force, hist_old, par, TOL_COND
        )

        L0 = hist_old.at[cls.IDX_DLAM].set(0.0)

        elastic_step = lambda: (
            hist_old,
            True,
            jnp.int32(0),
            jnp.zeros(11, dtype=jnp.float64),
        )

        plastic_step = lambda: cls.internalVariableUpdate(
            L0,
            0,
            14,
            cls.local_residual_elasto_plastic,
            hist_old,
            driving_force,
            par,
        )

        h1, converged_p, niter_p, history_p = jax.lax.cond(
            is_plastic,
            plastic_step,
            elastic_step,
        )

        h1, converged_v, niter_v, history_v = cls.internalVariableUpdate(
            h1,
            14,
            20,
            cls.local_residual_visco_elastic,
            hist_old[14:20],
            driving_force,
            DT,
            par,
        )

        Cn_vec = h1[cls.IDX_C_N]

        flux_p, re_p, rp, _ = cls.plasticResponse(driving_force, h1[:14], Cn_vec, par)
        flux_v, re_v, rv, _ = cls.viscoResponse(
            driving_force, h1[14:20], DT, Cn_vec, par
        )

        plastic_flag = jnp.where(is_plastic, 1.0, 0.0)
        plastic_flag = jax.lax.stop_gradient(plastic_flag)

        h1 = h1.at[cls.IDX_C_N].set(jaxt2n(RCG, 2))
        h1 = h1.at[cls.IDX_PHI_TRIAL].set(trial.Phi_trial)
        h1 = h1.at[cls.IDX_RE_P].set(re_p / DT)
        h1 = h1.at[cls.IDX_RP].set(rp / DT)
        h1 = h1.at[cls.IDX_RE_V].set(re_v / DT)
        h1 = h1.at[cls.IDX_RV].set(rv / DT)
        h1 = h1.at[cls.IDX_IS_PLASTIC].set(plastic_flag)

        out = flux_v + flux_p
        conv_flag = jnp.logical_and(converged_p, converged_v)
        return out, (h1, conv_flag, niter_p, history_p)

    @staticmethod
    def buildPlasticState(driving_force, int_var, par):
        F_nye = driving_force[:9]
        u_T = driving_force[9]

        F = jax92t(F_nye)
        RCG = jnp.transpose(F) @ F

        kappa = int_var[6]
        dlambda = int_var[7]
        invUp_nye = int_var[:6]
        invUpi_nye = int_var[8:14]

        invUp = jaxn2t(invUp_nye, 2)
        invUpi = jaxn2t(invUpi_nye, 2)
        Up = inv3x3(invUp)
        invCpi = invUpi @ invUpi

        RCGep = invUp @ RCG @ invUp
        bpe = Up @ invUpi @ invUpi @ Up

        # Graph consolidation: evaluate all updated-state free-energy derivatives with respect to internal variables in one call.
        dpsidRCGep, dpsidbpe, Rp = jax.grad(Psi_plas, argnums=(0, 1, 2))(
            RCGep, bpe, kappa, u_T, par
        )

        Yp = 2.0 * RCGep @ dpsidRCGep
        Xp = 2.0 * bpe @ dpsidbpe
        Sigma = Yp - Xp
        Theta = invCpi @ Up @ dpsidbpe @ Up @ invCpi

        return PlasticState(
            u_T=u_T,
            F=F,
            RCG=RCG,
            kappa=kappa,
            dlambda=dlambda,
            invUp=invUp,
            invUpi=invUpi,
            Up=Up,
            invCpi=invCpi,
            RCGep=RCGep,
            bpe=bpe,
            Yp=Yp,
            Xp=Xp,
            Sigma=Sigma,
            Rp=Rp,
            Theta=Theta,
        )

    @staticmethod
    def buildViscoState(driving_force, int_var, par):
        F_nye = driving_force[:9]
        u_T = driving_force[9]

        F = jax92t(F_nye)
        RCG = jnp.transpose(F) @ F
        b = F @ jnp.transpose(F)

        invUv_nye = int_var[0:6]
        invUv = jaxn2t(invUv_nye, 2)

        RCGev = invUv @ RCG @ invUv
        dpsi_AB_dRCGe = jax.grad(Psi_AB, argnums=0)(RCGev, u_T, par)

        PK2v = 2.0 * invUv @ dpsi_AB_dRCGe @ invUv
        sigma = 1.0 / det3x3(F) * F @ PK2v @ jnp.transpose(F)
        Yv = 2.0 * RCGev @ dpsi_AB_dRCGe

        norm_b = spd_norm2(b)
        norm_sigma = sym_norm2(sigma)

        return ViscoState(
            u_T=u_T,
            F=F,
            RCG=RCG,
            b=b,
            invUv=invUv,
            RCGev=RCGev,
            PK2v=PK2v,
            sigma=sigma,
            Yv=Yv,
            norm_b=norm_b,
            norm_sigma=norm_sigma,
        )

    @staticmethod
    def plasticResponse(driving_force, int_var, Cn_vec, par):
        state = TVPkinIFT_heat.buildPlasticState(driving_force, int_var, par)

        u_T = state.u_T
        RCG = state.RCG
        RCGep = state.RCGep
        bpe = state.bpe
        kappa = state.kappa
        dlambda = state.dlambda
        invUp = state.invUp
        Sigma = state.Sigma
        Rp = state.Rp
        Theta = state.Theta

        dpsidRCGep, dpsidbpe, _ = jax.grad(Psi_plas, argnums=(0, 1, 2))(
            RCGep, bpe, kappa, u_T, par
        )
        PK2 = 2.0 * invUp @ dpsidRCGep @ invUp
        PK2_vec = jaxt2n(PK2, 1)

        dp = dlambda * jax.grad(Phi, argnums=0)(Sigma, Rp, u_T, par)
        Cpidot = 2.0 * dlambda * jax.grad(g_kin, argnums=0)(Theta, u_T, par)

        # Temperature-dependent heat-source evaluation
        def thermo_funcs_plas(T):
            d0, d1, d2 = jax.grad(Psi_plas, argnums=(0, 1, 2))(
                RCGep, bpe, kappa, T, par
            )
            sig = 2.0 * (RCGep @ d0 - bpe @ d1)
            theta = 2.0 * bpe @ d1
            pk2_p = 2.0 * invUp @ d0 @ invUp
            return sig, theta, d2, pk2_p

        _, (dSigmadT, dThetadT, dRpdT, dPK2_p_dT) = jax.jvp(
            thermo_funcs_plas, (u_T,), (1.0,)
        )

        rp = (
            jnp.sum((Sigma - u_T * dSigmadT) * dp)
            + dlambda * (Rp - u_T * dRpdT)
            + jnp.sum((Theta - u_T * dThetadT) * Cpidot)
        )

        RCG_n = jaxn2t(Cn_vec, 2)
        re_p = 0.5 * u_T * jnp.sum(dPK2_p_dT * (RCG - RCG_n))

        heat = re_p + rp
        response = jnp.concatenate([PK2_vec, jnp.array([heat])])

        return response, re_p, rp, state

    @staticmethod
    def viscoResponse(driving_force, int_var, dt, Cn_vec, par):
        state = TVPkinIFT_heat.buildViscoState(driving_force, int_var, par)

        u_T = state.u_T
        RCG = state.RCG
        RCGev = state.RCGev
        invUv = state.invUv
        Yv = state.Yv

        PK2_vec = jaxt2n(state.PK2v, 1)
        dv = dt * jax.grad(g_vis, argnums=0)(
            Yv, u_T, state.norm_b, state.norm_sigma, par
        )

        def thermo_funcs_visco(T):
            d0 = jax.grad(Psi_AB, argnums=0)(RCGev, T, par)
            yv_val = 2.0 * RCGev @ d0
            pk2_v_val = 2.0 * invUv @ d0 @ invUv
            return yv_val, pk2_v_val

        _, (dYvdT, dPK2dT_v) = jax.jvp(thermo_funcs_visco, (u_T,), (1.0,))

        rv = jnp.sum((Yv - u_T * dYvdT) * dv)

        RCG_n = jaxn2t(Cn_vec, 2)
        re_v = 0.5 * u_T * jnp.sum(dPK2dT_v * (RCG - RCG_n))

        heat = re_v + rv
        response = jnp.concatenate([PK2_vec, jnp.array([heat])])

        return response, re_v, rv, state

    @staticmethod
    def local_residual_visco_elastic(L, hist_old, driving_force, dt, par):
        invUv_n_nye = hist_old[0:6]
        invUv_nye = L[0:6]

        invUv = jaxn2t(invUv_nye, 2)
        invUv_n = jaxn2t(invUv_n_nye, 2)

        state = TVPkinIFT_heat.buildViscoState(driving_force, L, par)
        Z = (
            2.0
            * dt
            * jax.grad(g_vis, argnums=0)(
                state.Yv, state.u_T, state.norm_b, state.norm_sigma, par
            )
        )

        res_Uv = invUv @ expm(Z) @ invUv - invUv_n @ invUv_n
        res_Uv_nye = jaxt2n(res_Uv, 1)
        return res_Uv_nye

    @staticmethod
    def local_residual_elasto_plastic(L, hist_old, driving_force, par):
        invUp_n_nye = hist_old[:6]
        invUpi_n_nye = hist_old[8:14]
        kappa_n = hist_old[6]

        invUp_n = jaxn2t(invUp_n_nye, 2)
        invUpi_n = jaxn2t(invUpi_n_nye, 2)

        state = TVPkinIFT_heat.buildPlasticState(driving_force, L, par)

        invUp = state.invUp
        invUpi = state.invUpi
        kappa = state.kappa
        dlambda = state.dlambda

        Sigma = state.Sigma
        Rp = state.Rp
        Theta = state.Theta

        Z = 2.0 * dlambda * jax.grad(Phi, argnums=0)(Sigma, Rp, state.u_T, par)
        Zpi = 2.0 * dlambda * jax.grad(g_kin, argnums=0)(Theta, state.u_T, par)

        res_Phi = Phi(Sigma, Rp, state.u_T, par)
        res_Up = invUp @ expm(Z) @ invUp - invUp_n @ invUp_n
        res_Upi = invUpi @ expm(Zpi) @ invUpi - invUpi_n @ invUpi_n

        res_Up_nye = jaxt2n(res_Up, 1)
        res_Upi_nye = jaxt2n(res_Upi, 1)
        res_kappa = (
            kappa_n
            - kappa
            - dlambda * jax.grad(Phi, argnums=1)(Sigma, Rp, state.u_T, par)
        )

        res = jnp.concatenate(
            (res_Phi.reshape(1), res_Up_nye, res_kappa.reshape(1), res_Upi_nye)
        )
        return res

    @staticmethod
    def heatPartsPlas(driving_force, int_var, Cn_vec, par):
        _, re_p, rp, _ = TVPkinIFT_heat.plasticResponse(
            driving_force, int_var, Cn_vec, par
        )
        return re_p, rp

    @staticmethod
    def heatPartsVisco(driving_force, int_var, dt, Cn_vec, par):
        _, re_v, rv, _ = TVPkinIFT_heat.viscoResponse(
            driving_force, int_var, dt, Cn_vec, par
        )
        return re_v, rv

    @staticmethod
    def fluxPlas(driving_force, int_var, Cn_vec, par):
        response, _, _, _ = TVPkinIFT_heat.plasticResponse(
            driving_force, int_var, Cn_vec, par
        )
        return response

    @staticmethod
    def fluxVisco(driving_force, int_var, dt, Cn_vec, par):
        response, _, _, _ = TVPkinIFT_heat.viscoResponse(
            driving_force, int_var, dt, Cn_vec, par
        )
        return response

    @staticmethod
    @partial(jax.jit, static_argnums=(3,))
    def qp_jacobian_elas(driving_force, hist, hist_n, par):
        Cn_vec = hist_n[TVPkinIFT_heat.IDX_C_N]
        return jax.jacfwd(TVPkinIFT_heat.fluxPlas, argnums=0)(
            driving_force, hist[0:14], Cn_vec, par
        )

    @staticmethod
    @partial(jax.jit, static_argnums=(3,))
    def qp_jacobian_plas(driving_force, hist, hist_n, par):
        Cn_vec = hist_n[TVPkinIFT_heat.IDX_C_N]
        L_current = hist[0:14]
        L_n = hist_n[0:14]

        # Graph consolidation and implicit-function tangent solve optimization
        drdL, drddf = jax.jacfwd(
            TVPkinIFT_heat.local_residual_elasto_plastic, argnums=(0, 2)
        )(L_current, L_n, driving_force, par)
        dLddf = jsp.linalg.solve(drdL, -drddf)

        df_ddf, df_dL = jax.jacfwd(TVPkinIFT_heat.fluxPlas, argnums=(0, 1))(
            driving_force, L_current, Cn_vec, par
        )

        return df_ddf + df_dL @ dLddf

    @staticmethod
    @partial(jax.jit, static_argnums=(4,))
    def qp_jacobian_visco(driving_force, hist, hist_n, dt, par):
        Cn_vec = hist_n[TVPkinIFT_heat.IDX_C_N]
        L_current = hist[14:20]
        L_n = hist_n[14:20]

        drdL, drddf = jax.jacfwd(
            TVPkinIFT_heat.local_residual_visco_elastic, argnums=(0, 2)
        )(L_current, L_n, driving_force, dt, par)
        dLddf = jsp.linalg.solve(drdL, -drddf)

        df_ddf, df_dL = jax.jacfwd(TVPkinIFT_heat.fluxVisco, argnums=(0, 1))(
            driving_force, L_current, dt, Cn_vec, par
        )

        return df_ddf + df_dL @ dLddf

    @staticmethod
    @partial(jax.jit, static_argnums=(4,))
    def computeQpJacobianIFT(driving_force, hist, hist_n, glob, par):
        dt = glob[0]

        elas_fn = partial(TVPkinIFT_heat.qp_jacobian_elas, par=par)
        plas_fn = partial(TVPkinIFT_heat.qp_jacobian_plas, par=par)

        plastic_flag = jax.lax.stop_gradient(hist[TVPkinIFT_heat.IDX_IS_PLASTIC])
        is_plastic = plastic_flag > 0.5

        plastic_tangent = jax.lax.cond(
            is_plastic,
            lambda args: plas_fn(*args),
            lambda args: elas_fn(*args),
            (driving_force, hist, hist_n),
        )

        viscoelastic_tangent = TVPkinIFT_heat.qp_jacobian_visco(
            driving_force, hist, hist_n, dt, par
        )

        return viscoelastic_tangent + plastic_tangent

    @staticmethod
    def getQpJacobianInAxes():
        return (0, 0, 0, None, None)

    @staticmethod
    def getQpJacobianStaticArgnums():
        return (4,)


def Psi_plas(RCGe, bpe, kappa, u_T, par):
    temp = u_T
    E = par.E(temp)
    sigY1 = par.sigY1(temp)
    sigY2 = par.sigY2(temp) * jnp.power(par.chi, par.alpha(temp))
    beta = par.beta(temp)
    H = par.H(temp)
    nu = par.nu
    c = par.c(temp)

    lmbda = E * nu / (1 - 2 * nu) / (1 + nu)
    mu = E / 2 / (1 + nu)

    I1 = jnp.trace(RCGe)
    I3 = det3x3(RCGe)

    psi_e = 0.5 * mu * (I1 - 3 - jnp.log(I3)) + 0.25 * lmbda * (I3 - 1 - jnp.log(I3))
    psi_iso = (sigY2 - sigY1) * (
        kappa + jnp.exp(-beta * kappa) / beta
    ) + 0.5 * H * kappa**2
    psi_kin = c / 2 * (jnp.trace(bpe) - 3 - jnp.log(det3x3(bpe)))

    return psi_e + psi_iso + psi_kin


def Psi_AB(Ce, u_T, par):
    temp = u_T
    E = par.E2(temp)
    nu = par.nu
    mu = E / 2 / (1 + nu)
    bulk = E / (3 * (1 - 2 * nu))
    lM = par.lM

    detCe = det3x3(Ce)
    Je = jnp.sqrt(detCe)
    psiVol = 0.5 * bulk * (jnp.log(Je) ** 2)

    C1, C2, C3, C4, C5 = 1 / 2, 1 / 20, 11 / 1050, 19 / 7000, 519 / 673750

    Cebar = Ce * (detCe ** (-1.0 / 3.0))
    trC = jnp.trace(Cebar)

    trC2 = trC * trC
    trC3 = trC2 * trC
    trC4 = trC3 * trC
    trC5 = trC4 * trC

    muStar = mu * (
        1
        + 3 / (5 * lM**2)
        + 99 / (175 * lM**4)
        + 513 / (875 * lM**6)
        + 42039 / (67375 * lM**8)
    ) ** (-1)

    psiIso = muStar * (
        C1 * (trC - 3)
        + C2 * (trC2 - 9) / lM**2
        + C3 * (trC3 - 27) / lM**4
        + C4 * (trC4 - 81) / lM**6
        + C5 * (trC5 - 243) / lM**8
    )

    return psiIso + psiVol


def g_vis(Sigma, u_T, norm_b, norm_sigma, par):
    temp = u_T
    tau0 = par.tau(temp)
    delta = par.delta(temp)
    phi = par.phi(temp)
    E = par.E2(temp)
    nu = par.nu
    mu = E / 2 / (1 + nu)
    bulk = E / (3 * (1 - 2 * nu))

    dev_Sigma = Sigma - jnp.trace(Sigma) / 3 * jnp.identity(3)
    tau = tau0 * (norm_b**phi) * jnp.exp(-delta * norm_sigma)

    return (1 / (4 * mu * tau)) * jnp.tensordot(dev_Sigma, dev_Sigma) + (
        1 / (18 * bulk * tau)
    ) * (jnp.trace(Sigma) ** 2)


def g_kin(Sigma, T, par):
    dev_Sigma = Sigma - jnp.trace(Sigma) / 3 * jnp.identity(3)
    c = par.c(T)
    b = par.b(T)
    return b / (2 * c) * jnp.tensordot(dev_Sigma, dev_Sigma)


def Phi(Y, R, u_T, par):
    temp = u_T
    sigY1 = par.sigY1(temp)
    devY = Y - jnp.trace(Y) / 3 * jnp.identity(3)
    return jnp.sqrt(1.5 * jnp.tensordot(devY, devY) + 1e-10) - (sigY1 + R)
