from .Base import Kernel
import ufl


class TotalLagrangianStressDivergence(Kernel):

    def computeQpResidual(self):
        v = self.test
        F = self.material.F
        stress = self.material.stress

        v_disp = ufl.as_vector([v[0], v[1], v[2]])
        virt_GLS = ufl.sym(F.T * ufl.grad(v_disp))
        virt_GLS_nye = ufl.as_vector(
            [
                virt_GLS[0, 0],
                virt_GLS[1, 1],
                virt_GLS[2, 2],
                2.0 * virt_GLS[0, 1],
                2.0 * virt_GLS[0, 2],
                2.0 * virt_GLS[1, 2],
            ]
        )

        return ufl.inner(stress, virt_GLS_nye)

class TotalLagrangianStressDivergence2D(Kernel):

    def computeQpResidual(self):
        v = self.test
        F = self.material.F
        stress = self.material.stress

        v_disp = ufl.as_vector([v[0], v[1]])
        F2d = ufl.as_tensor([[F[0,0],F[0,1]],[F[1,0],F[1,1]]])
        virt_GLS = ufl.sym(F2d.T * ufl.grad(v_disp))
        virt_GLS_nye = ufl.as_vector(
            [
                virt_GLS[0, 0],
                virt_GLS[1, 1],
                0,
                2.0 * virt_GLS[0, 1],
                0,
                0,
            ]
        )

        return ufl.inner(stress, virt_GLS_nye)
