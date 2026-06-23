import ufl
from Multiphysics.Materials import ExoMaterial
from dolfinx_external_operator import replace_external_operators


class Kernel:
    def __init__(
        self,
        test,
        material,
        dt,
        u,
        kernel_pararmeters={},
    ):
        self.u = u
        self.test = test
        self.material = material
        self.par = kernel_pararmeters
        self._dt = dt

    def computeQpResidual(self):

        raise NotImplementedError("computeQpResidual must be implemented in subclass.")


class ActionManager:
    def __init__(self, u_global, material, *actionPairs):
        self.u_global = u_global
        self.material = material
        self.test_global = ufl.TestFunction(self.u_global.function_space)
        self.du = ufl.TrialFunction(self.u_global.function_space)
        self.actions = list(actionPairs)  # [(u_i, kernel_class), ...]

        self.dx = ufl.Measure(
            "dx",
            domain=self.material.domain,
            metadata={"quadrature_degree": 2, "quadrature_rule": "default"},
        )  # Volume integration measur        self.u = material.u

    def getTestFunction(self, u_local):
        if len(ufl.shape(self.u_global)) == 0:
            return self.test_global
        else:

            shape_expr = ufl.shape(u_local)

            if len(shape_expr) == 0:
                _, idx = u_local.ufl_operands
                return self.test_global[int(idx.indices()[0])]

            elif len(shape_expr) == 1:
                foo = []
                for component in u_local:
                    _, idx = component.ufl_operands
                    foo.append(self.test_global[int(idx.indices()[0])])
                return ufl.as_vector(foo)

            else:
                raise NotImplementedError(f"Unsupported shape: {shape_expr}")

    def computeResidual(self):
        self.residual = 0
        for  act in self.actions:
            kernelClass=act[0]
            inputs = act[1:]
            u_local = inputs[0]
            v_local = self.getTestFunction(u_local)
            kernel = kernelClass(v_local, self.material,self._dt,*inputs)
            self.residual += kernel.computeQpResidual() * self.dx

    def computeJacobian(self):
        self.jacobian = ufl.derivative(self.residual, self.u_global, self.du)

        if isinstance(self.material, ExoMaterial):
            J_expanded = ufl.algorithms.expand_derivatives(self.jacobian)
            R_replaced, R_ex = replace_external_operators(self.residual)
            J_replaced, J_ex = replace_external_operators(J_expanded)
            self.residual = R_replaced
            self.jacobian = J_replaced
            self.residual_ex = R_ex
            self.jacobian_ex = J_ex
