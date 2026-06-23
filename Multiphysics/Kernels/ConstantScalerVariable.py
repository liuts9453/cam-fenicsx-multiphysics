from .Base import Kernel
    
class ConstanScalerVariable(Kernel):

    def computeQpResidual(self):

        return (self.u - self.par["value"]) * self.test
