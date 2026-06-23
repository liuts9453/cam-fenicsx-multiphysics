from .Base import Kernel
import ufl


class HeatConduction(Kernel):

    def computeQpResidual(self):
        u = self.u
        v = self.test
        chi = 0.23
        unit = self.par["unit"]
        hc = ((u + unit) * 4.502 + 138.7) * (0.001195 * chi + 0.00109 * (1 - chi))

        dt = self._dt
        T_old = self.par["temperature_old"]
        conductivity = self.par["thermal_conductivity"]

        heat_source = self.material.heat_source * v
        heat_flux = -dt * conductivity * ufl.dot(ufl.grad(u), ufl.grad(v))
        time_derivative = -hc * (u - T_old) * v

        return heat_source + heat_flux + time_derivative


class PoissonEq(Kernel):

    def computeQpResidual(self):
        u = self.u
        v = self.test

        conductivity = self.par["thermal_conductivity"]

        heat_source = 2.0 * v
        heat_flux = 1 * conductivity * ufl.dot(ufl.grad(u), ufl.grad(v))

        return heat_source + heat_flux
