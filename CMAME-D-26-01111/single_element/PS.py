from Multiphysics.Kernels import Kernel
from fox.utils import ufl_tensor_to_nye as t2n 
from fox.utils import ufl_nye_to_tensor as n2t 
import ufl
class PlasticStretch(Kernel):

    def computeQpResidual(self):

        hi = self.material.hist
        invUp_vec = ufl.as_vector(
            [
                hi[0],
                hi[1],
                hi[2],
                hi[3],
                hi[4],
                hi[5],
            ]
        )
        invUp_ten = n2t(invUp_vec,2)
        Up_ten = ufl.inv(invUp_ten)
        Up_vec = t2n(Up_ten,2)
        


        return ufl.dot(self.u - Up_vec, self.test)
