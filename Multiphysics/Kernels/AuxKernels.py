from .Base import Kernel
from fox.utils import (
    ufl_tensor_to_nye as vec,
    ufl_nye_to_tensor as ten,
    ufl_tensor_to_nye_unsym as vec9,
)
import ufl


class PushForwardCauchy(Kernel):

    def computeQpResidual(self):
        F = self.material.F
        CST = vec(
            1 / ufl.det(F) * F * ten(self.material.stress, 1) * ufl.transpose(F), 1
        )
        return ufl.dot(self.u - CST, self.test)


class PushForwardPK1(Kernel):

    def computeQpResidual(self):
        F = self.material.F
        CST = vec(F * ten(self.material.stress, 1), 1)
        return ufl.dot(self.u - CST, self.test)


class DeformationGradient(Kernel):

    def computeQpResidual(self):
        F = self.material.F
        F_vec = vec9(F)
        return ufl.dot(self.u - F_vec, self.test)


class HistVariables(Kernel):

    def computeQpResidual(self):
        hi = self.material.hist
        return (self.u - hi[self.par["index"]]) * self.test


class PushForwardCauchyPlas(Kernel):

    def computeQpResidual(self):
        F = self.material.F
        hi = self.material.hist
        plas_PK2_vec = ufl.as_vector(
            [
                hi[20],
                hi[21],
                hi[22],
                hi[23],
                hi[24],
                hi[25],
            ]
        )

        CST = vec(
            1 / ufl.det(F) * F * ten(plas_PK2_vec, 1) * ufl.transpose(F),
            1,
        )
        return ufl.dot(self.u - CST, self.test)

class PlasticStretch(Kernel):

    def computeQpResidual(self):

        hi = self.material.hist
        Up_vec = ufl.as_vector(
            [
                hi[0],
                hi[1],
                hi[2],
                hi[3],
                hi[4],
                hi[5],
            ]
        )


        return ufl.dot(self.u - Up_vec, self.test)
