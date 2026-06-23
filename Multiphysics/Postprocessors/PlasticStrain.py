from .CustomPostprocessor import CustomPostprocessor
from fox.utils import (
    interpolate_expression as expr,
    ufl_nye_to_tensor as ten,
    ufl_tensor_to_nye as vec,
)
import ufl


class PlasticStrain(CustomPostprocessor):
    def run(self):
        o = self.fields.get("plastic_stretch")
        hi = self.material.hist
        v_inv = ufl.as_vector(
            [
                hi[0],
                hi[1],
                hi[2],
                hi[3],
                hi[4],
                hi[5],
            ]
        )
        t_inv = ten(v_inv, 1)
        t = ufl.inv(t_inv)
        plas = vec(t, 1)

        expr(plas, o)

        return

