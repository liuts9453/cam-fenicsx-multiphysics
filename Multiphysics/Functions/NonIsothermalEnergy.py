import jax.numpy as jnp

from Multiphysics.tiliuTools.math import det3x3, spd_norm2, sym_norm2


def psi_plas(RCGe, bpe, kappa, u_T, pars):
    temp = u_T
    E = pars.E(temp)
    sigY1 = pars.sigY1(temp)
    sigY2 = pars.sigY2(temp) * jnp.power(pars.chi, pars.alpha(temp))
    beta = pars.beta(temp)
    H = pars.H(temp)
    nu = pars.nu
    c = pars.c(temp)

    lmbda = E * nu / (1 - 2 * nu) / (1 + nu)
    mu = E / 2 / (1 + nu)

    I1 = jnp.trace(RCGe)
    I3 = det3x3(RCGe)

    psi_e = 0.5 * mu * (I1 - 3 - jnp.log(I3)) + 0.25 * lmbda * (
        I3 - 1 - jnp.log(I3)
    )
    psi_iso = (sigY2 - sigY1) * (
        kappa + jnp.exp(-beta * kappa) / beta
    ) + 0.5 * H * kappa**2
    psi_kin = c / 2 * (jnp.trace(bpe) - 3 - jnp.log(det3x3(bpe)))

    return psi_e + psi_iso + psi_kin


def psi_AB(Ce, u_T, pars):
    temp = u_T
    E = pars.E2(temp)
    nu = pars.nu
    mu = E / 2 / (1 + nu)
    bulk = E / (3 * (1 - 2 * nu))
    lM = pars.lM

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


def g_vis(Y, u_T, b, sigma, pars):
    temp = u_T
    tau0 = pars.tau(temp)
    delta = pars.delta(temp)
    phi = pars.phi(temp)
    E = pars.E2(temp)
    nu = pars.nu
    mu = E / 2 / (1 + nu)
    bulk = E / (3 * (1 - 2 * nu))

    norm_b = spd_norm2(b)
    norm_sigma = sym_norm2(sigma)
    dev_Y = Y - jnp.trace(Y) / 3 * jnp.identity(3)
    tau = tau0 * (norm_b**phi) * jnp.exp(-delta * norm_sigma)

    return (1 / (4 * mu * tau)) * jnp.tensordot(dev_Y, dev_Y) + (
        1 / (18 * bulk * tau)
    ) * (jnp.trace(Y) ** 2)


def g_kin(Sigma, T, pars):
    dev_Sigma = Sigma - jnp.trace(Sigma) / 3 * jnp.identity(3)
    c = pars.c(T)
    b = pars.b(T)
    return b / (2 * c) * jnp.tensordot(dev_Sigma, dev_Sigma)


def Phi(Y, R, u_T, pars):
    temp = u_T
    sigY1 = pars.sigY1(temp)
    devY = Y - jnp.trace(Y) / 3 * jnp.identity(3)
    return jnp.sqrt(1.5 * jnp.tensordot(devY, devY) + 1e-10) - (sigY1 + R)
