from .CustomPostprocessor import CustomPostprocessor
from fox.utils import interpolate_expression as expr


class HistvarAtNode(CustomPostprocessor):
    def run(self):
        o = self.fields.get(self.par["field"])
        hi = self.material.hist

        expr(hi[self.par["index"]], o)

        return
