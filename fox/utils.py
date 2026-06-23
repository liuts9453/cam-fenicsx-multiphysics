import jax
import ufl
from dolfinx.fem import Expression
import time
import functools

jax.config.update("jax_enable_x64", True)  # Set 64-bit arithmetic in JAX
from mpi4py import MPI


def jax_tensor_to_nye(A, x):
    return jax.numpy.array(
        [A[0, 0], A[1, 1], A[2, 2], x * A[0, 1], x * A[0, 2], x * A[1, 2]]
    )


def jax_nye_to_tensor(v, x):
    return jax.numpy.array(
        [
            [v[0], v[3] / x, v[4] / x],
            [v[3] / x, v[1], v[5] / x],
            [v[4] / x, v[5] / x, v[2]],
        ]
    )


def jax_nye_to_tensor_unsym(v):
    return jax.numpy.array(
        [
            [v[0], v[3], v[4]],
            [v[6], v[1], v[5]],
            [v[7], v[8], v[2]],
        ]
    )


def ufl_tensor_to_nye(A, x):
    return ufl.as_vector(
        [A[0, 0], A[1, 1], A[2, 2], x * A[0, 1], x * A[0, 2], x * A[1, 2]]
    )


def ufl_tensor_to_nye_unsym(A):
    return ufl.as_vector(
        [
            A[0, 0],
            A[1, 1],
            A[2, 2],
            A[0, 1],
            A[0, 2],
            A[1, 2],
            A[1, 0],
            A[2, 0],
            A[2, 1],
        ]
    )


def ufl_nye_to_tensor(v, x):
    return ufl.as_tensor(
        [
            [v[0], v[3] / x, v[4] / x],
            [v[3] / x, v[1], v[5] / x],
            [v[4] / x, v[5] / x, v[2]],
        ]
    )


def interpolate_expression(ufl_expr, func):
    func.interpolate(
        Expression(ufl_expr, func.function_space.element.interpolation_points)
    )


def timeit(name=None):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start = time.time()
            result = func(*args, **kwargs)
            end = time.time()
            elapsed = end - start
            label = name if name else func.__name__
            print(f"[{label}] executed in {elapsed:.6f} seconds")
            return result

        return wrapper

    return decorator
