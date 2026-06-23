from ufl import (
    Identity,
    det,
    tr,
    sqrt,
    conditional,
    gt,
    dot,
    exp,
    cos,
    acos,
    pi,
    lt,
    gt,
    And,
    variable,
    max_value,
    diff,
    as_vector,
    as_tensor,
)
import jax
import jax.numpy as jnp
import jax.scipy as jsp

@jax.custom_jvp
def expm(A):
    """Symmetric spectral matrix exponential used by CIC local residuals."""
    A = 0.5 * (A + A.T)
    vals, vecs = jnp.linalg.eigh(A)
    return vecs @ jnp.diag(jnp.exp(vals)) @ vecs.T


@expm.defjvp
def expm_jvp(primals, tangents):
    (A,) = primals
    (A_dot,) = tangents

    A = 0.5 * (A + A.T)
    A_dot = 0.5 * (A_dot + A_dot.T)

    vals, Q = jnp.linalg.eigh(A)
    exp_vals = jnp.exp(vals)
    F = Q @ jnp.diag(exp_vals) @ Q.T

    A_dot_prin = Q.T @ A_dot @ Q

    v_i = vals[:, None]
    v_j = vals[None, :]
    exp_i = exp_vals[:, None]
    exp_j = exp_vals[None, :]
    diff_v = v_i - v_j

    tol = 1e-6
    mask = jnp.abs(diff_v) > tol

    # Spectral Frechet derivative of exp(A) for symmetric residual blocks.
    div_diff = jnp.where(
        mask,
        (exp_i - exp_j) / jnp.where(mask, diff_v, 1.0),
        jnp.exp(0.5 * (v_i + v_j)),
    )

    F_dot_prin = div_diff * A_dot_prin
    F_dot = Q @ F_dot_prin @ Q.T

    return F, F_dot


@jax.custom_jvp
def sym_logm(A):
    # Enforce symmetry
    A = 0.5 * (A + A.T)
    vals, vecs = jnp.linalg.eigh(A)
    # Clip to avoid log(0)
    vals = jnp.clip(vals, 1e-14)
    return vecs @ jnp.diag(jnp.log(vals)) @ vecs.T


@sym_logm.defjvp
def sym_logm_jvp(primals, tangents):
    (A,) = primals
    (A_dot,) = tangents

    A = 0.5 * (A + A.T)
    A_dot = 0.5 * (A_dot + A_dot.T)

    vals, Q = jnp.linalg.eigh(A)
    vals = jnp.clip(vals, 1e-14)
    log_vals = jnp.log(vals)
    F = Q @ jnp.diag(log_vals) @ Q.T

    A_dot_prin = Q.T @ A_dot @ Q

    v_i = vals[:, None]
    v_j = vals[None, :]
    log_i = log_vals[:, None]
    log_j = log_vals[None, :]
    diff_v = v_i - v_j

    tol = 1e-6
    mask = jnp.abs(diff_v) > tol

    # Use first order Taylor expansion 1/x -> 2/(x+y) around x=y for the diagonal / small differences
    div_diff = jnp.where(
        mask,
        (log_i - log_j) / jnp.where(mask, diff_v, 1.0),
        2.0 / (v_i + v_j),
    )

    F_dot_prin = div_diff * A_dot_prin
    F_dot = Q @ F_dot_prin @ Q.T

    return F, F_dot


@jax.custom_jvp
def sym_mat_2norm(A):
    vals = jnp.linalg.eigvalsh(A)
    return jnp.max(jnp.abs(vals))


@sym_mat_2norm.defjvp
def sym_mat_2norm_jvp(primals, tangents):
    (A,) = primals
    (A_dot,) = tangents

    vals, vecs = jnp.linalg.eigh(A)
    abs_vals = jnp.abs(vals)

    idx_max = jnp.argmax(abs_vals)
    val_max = vals[idx_max]
    norm_val = abs_vals[idx_max]

    v_max = vecs[:, idx_max]
    val_dot = jnp.dot(v_max, A_dot @ v_max)
    norm_dot = jnp.sign(val_max + 1e-14) * val_dot

    return norm_val, norm_dot


@jax.custom_jvp
def spd_mat_2norm(A):
    vals = jnp.linalg.eigvalsh(A)
    return jnp.max(vals)


@spd_mat_2norm.defjvp
def spd_mat_2norm_jvp(primals, tangents):
    (A,) = primals
    (A_dot,) = tangents

    vals, vecs = jnp.linalg.eigh(A)
    idx_max = jnp.argmax(vals)
    norm_val = vals[idx_max]

    v_max = vecs[:, idx_max]
    norm_dot = jnp.dot(v_max, A_dot @ v_max)

    return norm_val, norm_dot


@jax.jit
def det3x3(A):
    return (
        A[0, 0] * (A[1, 1] * A[2, 2] - A[1, 2] * A[2, 1])
        - A[0, 1] * (A[1, 0] * A[2, 2] - A[1, 2] * A[2, 0])
        + A[0, 2] * (A[1, 0] * A[2, 1] - A[1, 1] * A[2, 0])
    )


@jax.jit
def inv3x3(A):
    det = det3x3(A)
    adj = jnp.array(
        [
            [
                A[1, 1] * A[2, 2] - A[1, 2] * A[2, 1],
                A[0, 2] * A[2, 1] - A[0, 1] * A[2, 2],
                A[0, 1] * A[1, 2] - A[0, 2] * A[1, 1],
            ],
            [
                A[1, 2] * A[2, 0] - A[1, 0] * A[2, 2],
                A[0, 0] * A[2, 2] - A[0, 2] * A[2, 0],
                A[0, 2] * A[1, 0] - A[0, 0] * A[1, 2],
            ],
            [
                A[1, 0] * A[2, 1] - A[1, 1] * A[2, 0],
                A[0, 1] * A[2, 0] - A[0, 0] * A[2, 1],
                A[0, 0] * A[1, 1] - A[0, 1] * A[1, 0],
            ],
        ]
    )
    return adj / det


expm = jax.jit(expm)
sym_logm = jax.jit(sym_logm)
spd_norm2 = jax.jit(spd_mat_2norm)
sym_norm2 = jax.jit(sym_mat_2norm)

class SplineInterpolator:
    def __init__(self, x_data: jnp.ndarray, y_data: jnp.ndarray):
        self.x_data = x_data
        self.y_data = y_data
        self.n = len(x_data)
        self.M = self._compute_moments()

    def _compute_moments(self):
        n = self.n
        h = self.x_data[1:] - self.x_data[:-1]

        A = jnp.zeros((n, n))
        b = jnp.zeros(n)

        def body(i, val):
            A, b = val
            A = A.at[i, i - 1].set(h[i - 1] / 6.0)
            A = A.at[i, i].set((h[i - 1] + h[i]) / 3.0)
            A = A.at[i, i + 1].set(h[i] / 6.0)
            b = b.at[i].set(
                (self.y_data[i + 1] - self.y_data[i]) / h[i]
                - (self.y_data[i] - self.y_data[i - 1]) / h[i - 1]
            )
            return A, b

        A, b = jax.lax.fori_loop(1, n - 1, body, (A, b))
        A = A.at[0, 0].set(1.0)
        A = A.at[-1, -1].set(1.0)
        b = b.at[0].set(0.0)
        b = b.at[-1].set(0.0)

        return jnp.linalg.solve(A, b)

    def evaluate(self, x_query):
        """
         x_query 
        """
        x_data = self.x_data
        y_data = self.y_data
        M = self.M
        n = self.n

        i = jnp.clip(jnp.searchsorted(x_data, x_query) - 1, 0, n - 2)

        xi, xi1 = x_data[i], x_data[i + 1]
        yi, yi1 = y_data[i], y_data[i + 1]
        Mi, Mi1 = M[i], M[i + 1]
        hi = xi1 - xi

        a = (xi1 - x_query) / hi
        b_ = (x_query - xi) / hi

        S = a * yi + b_ * yi1 + ((a**3 - a) * Mi + (b_**3 - b_) * Mi1) * hi**2 / 6.0
        return S


def matrix_exp_sym(A, eps=1e-12):
    """UFL """

    A = variable(A)
    devA = A - tr(A) / 3 * Identity(3)


    p = tr(dot(devA, devA))
    q = det(devA)


    eps_p = 0.02
    eps_r = 0.002


    Gbar_small_p = (
        3
        + q / 2
        + q**2 / 240
        + q**3 / 120960
        + q**4 / 159667200
        + (0.5 + q / 48 + q**2 / 10080 + q**3 / 7257600) * p
        + 0.5 * (1 / 24 + q / 1440 + q**2 / 483840) * p**2
        + (1 / 480 + q / 53760) / 6 * p**3
        + p**4 / 322560
        + p**5 / 58060800
    )


    r = 3 * sqrt(6.0) * q / (sqrt(p**3) + eps)
    t = sqrt(2.0 / 3) * sqrt(p)


    phi = acos(r) / 3
    lam1 = t * cos(phi)
    lam2 = t * cos(phi - 2 * pi / 3)
    lam3 = t * cos(phi + 2 * pi / 3)
    Gbar_tri = exp(lam1) + exp(lam2) + exp(lam3)


    sr = conditional(lt(r, 0), 1.0, -1.0)
    Gr0 = exp(-sr * t) + 2 * exp(sr * t / 2)
    Gr1 = (t / 18) * exp(-sr * t) * (exp(3 * sr * t / 2) * (3 * sr * t - 2) + 2)
    Gr2 = (
        (1 / 1944)
        * exp(-sr * t)
        * (
            8 * t * (3 * t + 8 * sr)
            + t * exp(3 * sr * t / 2) * (-64 * sr + 9 * t * (8 + t**2 - 4 * t * sr))
        )
    )
    Gr3 = (
        (1 / 349920)
        * exp(-sr * t)
        * (
            t
            * exp(3 * sr * t / 2)
            * (
                -10 * (896 + 480 * t**2 + 27 * t**4)
                + 3 * sr * t * (3200 + 480 * t**2 + 9 * t**4)
            )
            + 160 * t * (56 + 3 * t * (t + 8 * sr))
        )
    )
    Gbar_poly = (
        Gr0 + Gr1 * (r + sr) + 0.5 * Gr2 * (r + sr) ** 2 + (1 / 6) * Gr3 * (r + sr) ** 3
    )


    Gbar = conditional(
        lt(p, eps_p),
        Gbar_small_p,
        conditional(And(gt(r, -1 + eps_r), lt(r, 1 - eps_r)), Gbar_tri, Gbar_poly),
    )


    G = exp(tr(A) / 3) * Gbar
    expA = diff(G, A)
    return expA


def vec6(tensor):
    """
    Convert a symmetric 3x3 UFL tensor to a 6D Voigt notation vector.
    """
    assert tensor.ufl_shape == (3, 3), "Input must be a 3x3 tensor"
    return as_vector(
        [
            tensor[0, 0],
            tensor[1, 1],
            tensor[2, 2],  # Normal components
            tensor[0, 1],
            tensor[1, 2],
            tensor[0, 2],  # Shear components
        ]
    )


def ten(vector):
    """
    Convert a 6D or 9D UFL vector to a 3x3 tensor.
    """
    if vector.ufl_shape == (6,):
        return as_tensor(
            [
                [vector[0], vector[3], vector[5]],
                [vector[3], vector[1], vector[4]],
                [vector[5], vector[4], vector[2]],
            ]
        )
    elif vector.ufl_shape == (9,):
        return as_tensor(
            [
                [vector[0], vector[1], vector[2]],
                [vector[3], vector[4], vector[5]],
                [vector[6], vector[7], vector[8]],
            ]
        )
    else:
        raise ValueError(
            "Vector must have 6 or 9 components to be converted to a tensor"
        )


@jax.jit
def compute_moments(x_knots: jnp.ndarray, y_knots: jnp.ndarray) -> jnp.ndarray:
    """
    Compute natural cubic spline second derivatives (moments) for 1D knots.

    Parameters:
      x_knots: 1D array of length n_knots, strictly increasing
      y_knots: 1D array of length n_knots, corresponding values

    Returns:
      M: 1D array of length n_knots, the second derivatives at the knots
    """
    n = x_knots.shape[0]
    # intervals
    h = x_knots[1:] - x_knots[:-1]

    # build tridiagonal system A · M = b
    A = jnp.zeros((n, n))
    b = jnp.zeros((n,))

    def body(i, vals):
        A, b = vals
        A = A.at[i, i - 1].set(h[i - 1] / 6.0)
        A = A.at[i, i].set((h[i - 1] + h[i]) / 3.0)
        A = A.at[i, i + 1].set(h[i] / 6.0)
        b = b.at[i].set(
            (y_knots[i + 1] - y_knots[i]) / h[i]
            - (y_knots[i] - y_knots[i - 1]) / h[i - 1]
        )
        return A, b

    A, b = jax.lax.fori_loop(1, n - 1, body, (A, b))
    # natural boundary: second derivative zero at ends
    A = A.at[0, 0].set(1.0)
    A = A.at[-1, -1].set(1.0)
    b = b.at[0].set(0.0)
    b = b.at[-1].set(0.0)

    M = jnp.linalg.solve(A, b)
    return M


@jax.jit
def spline_interpolate(
    T: jnp.ndarray, x_knots: jnp.ndarray, y_knots: jnp.ndarray
) -> jnp.ndarray:
    """
    Natural cubic spline interpolation at a single query point.

    Parameters:
      T:        scalar or 0-d array (query point)
      x_knots:  1D array of length n_knots (knot positions)
      y_knots:  1D array of length n_knots (knot values)

    Returns:
      S: interpolated value at T (scalar)
    """
    # 1) compute moments
    M = compute_moments(x_knots, y_knots)
    n = x_knots.shape[0]

    # 2) find interval index
    i = jnp.clip(jnp.searchsorted(x_knots, T) - 1, 0, n - 2)

    xi, xi1 = x_knots[i], x_knots[i + 1]
    yi, yi1 = y_knots[i], y_knots[i + 1]
    Mi, Mi1 = M[i], M[i + 1]
    hi = xi1 - xi

    # 3) compute spline basis
    a = (xi1 - T) / hi
    b = (T - xi) / hi

    # 4) combine for final value
    S = a * yi + b * yi1 + ((a**3 - a) * Mi + (b**3 - b) * Mi1) * (hi**2) / 6.0
    return S
class SplineInterpolatorOpt:
    def __init__(self, x_data: jnp.ndarray, y_data: jnp.ndarray, clip: bool=False):
        self.x_data = jnp.asarray(x_data)
        self.y_data = jnp.asarray(y_data)
        self.n = len(x_data)

        self.M = jax.lax.stop_gradient(self._compute_moments())
        self.clip = clip


        @jax.custom_jvp
        def _f_scalar(xq):
            xq_use = jnp.clip(xq, self.x_data[0], self.x_data[-1]) if self.clip else xq
            yv, _ = self._value_and_deriv(xq_use)
            return yv

        @_f_scalar.defjvp
        def _f_scalar_jvp(primals, tangents):
            (xq,), (tx,) = primals, tangents
            xq_use = jnp.clip(xq, self.x_data[0], self.x_data[-1]) if self.clip else xq
            yv, dy = self._value_and_deriv(xq_use)
            return yv, dy * tx

        self._f = jax.jit(_f_scalar)
        self._f_batch = jax.jit(jax.vmap(_f_scalar))

    def _compute_moments(self):
        n = self.n
        h = self.x_data[1:] - self.x_data[:-1]
        A = jnp.zeros((n, n))
        b = jnp.zeros(n)

        def body(i, val):
            A, b = val
            A = A.at[i, i - 1].set(h[i - 1] / 6.0)
            A = A.at[i, i].set((h[i - 1] + h[i]) / 3.0)
            A = A.at[i, i + 1].set(h[i] / 6.0)
            b = b.at[i].set(
                (self.y_data[i + 1] - self.y_data[i]) / h[i]
                - (self.y_data[i] - self.y_data[i - 1]) / h[i - 1]
            )
            return A, b

        A, b = jax.lax.fori_loop(1, n - 1, body, (A, b))
        A = A.at[0, 0].set(1.0)
        A = A.at[-1, -1].set(1.0)
        b = b.at[0].set(0.0)
        b = b.at[-1].set(0.0)
        return jnp.linalg.solve(A, b)

    def _value_and_deriv(self, xq):
        x = self.x_data; y = self.y_data; M = self.M; n = self.n
        i = jnp.clip(jnp.searchsorted(x, xq) - 1, 0, n - 2)
        xi, xi1 = x[i], x[i + 1]
        yi, yi1 = y[i], y[i + 1]
        Mi, Mi1 = M[i], M[i + 1]
        h = xi1 - xi


        a = (xi1 - xq) / h
        b_ = (xq - xi) / h
        S = a * yi + b_ * yi1 + ((a**3 - a) * Mi + (b_**3 - b_) * Mi1) * h**2 / 6.0



        # S = Mi*(xi1-x)^3/(6h) + Mi1*(x-xi)^3/(6h) + (yi - Mi*h^2/6)*(xi1-x)/h + (yi1 - Mi1*h^2/6)*(x-xi)/h
        dS = (-Mi  * (xi1 - xq)**2) / (2*h) + (Mi1 * (xq - xi)**2) / (2*h) \
           + (-(yi  - Mi * (h**2)/6) / h) + ((yi1 - Mi1*(h**2)/6) / h)
        return S, dS


    def evaluate(self, x_query):
        return self._f(x_query)

    def evaluate_batch(self, x_query_vec):
        return self._f_batch(x_query_vec)

    __call__ = evaluate

class ConstitutiveBlock:
    """
     AD  (Graph Consolidation )
     @ConstitutiveBlock 
     XLA Primal 
    """
    def __init__(self, func):
        self.func = func

        self._grad_multi_cache = {}
        self._jac_multi_cache = {}

    def __call__(self, *args, **kwargs):
        return self.func(*args, **kwargs)

    # -------------------------------------------------------------

    # -------------------------------------------------------------
    def d0(self, *args, **kwargs): return jax.grad(self.func, argnums=0)(*args, **kwargs)
    def d1(self, *args, **kwargs): return jax.grad(self.func, argnums=1)(*args, **kwargs)
    def d2(self, *args, **kwargs): return jax.grad(self.func, argnums=2)(*args, **kwargs)

    def j(self,  *args, **kwargs): return jax.jacfwd(self.func, argnums=0)(*args, **kwargs)
    def j1(self, *args, **kwargs): return jax.jacfwd(self.func, argnums=1)(*args, **kwargs)
    def j2(self, *args, **kwargs): return jax.jacfwd(self.func, argnums=2)(*args, **kwargs)

    # -------------------------------------------------------------

    # -------------------------------------------------------------
    def grad_multi(self, argnums):
        """XLA """
        if argnums not in self._grad_multi_cache:
            self._grad_multi_cache[argnums] = jax.grad(self.func, argnums=argnums)
        return self._grad_multi_cache[argnums]

    def jac_multi(self, argnums):
        """ (IFT)"""
        if argnums not in self._jac_multi_cache:
            self._jac_multi_cache[argnums] = jax.jacfwd(self.func, argnums=argnums)
        return self._jac_multi_cache[argnums]

def compute_algorithmic_tangent(drdL, drddf, df_ddf, df_dL):
    """
     (Algorithmic Tangent)
     jax.scipy  jnp.linalg  GPU vmap 
    """

    dLddf = jsp.linalg.solve(drdL, -drddf)
    return df_ddf + df_dL @ dLddf
