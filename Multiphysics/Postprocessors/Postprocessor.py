from dolfinx_external_operator import FEMExternalOperator
from fox.utils import interpolate_expression as expr2node
from ufl import as_vector,shape
from dolfinx.io import VTXWriter

def replaceWithRefCoefficient(expr):
    shape_expr = shape(expr)

    if len(shape_expr) == 0:
        if len(expr.ufl_operands) >0:
            base, idx = expr.ufl_operands
        else:
            base = expr
        if isinstance(base, FEMExternalOperator):
            return base.ref_coefficient[int(idx.indices()[0])]
        else:
            return expr

    elif len(shape_expr) == 1:
        replaced = []
        for component in expr:
            base, idx = component.ufl_operands
            if isinstance(base, FEMExternalOperator):
                replaced.append(base.ref_coefficient[int(idx.indices()[0])])
            else:
                replaced.append(component)
        return as_vector(replaced)

    else:
        raise NotImplementedError(f"Unsupported shape for replacement: {shape_expr}")


class Postprocessor:
    def __init__(self, domain, path="./result/output.bp"):
        self._registry = []
        self._function_set = set()
        self._functions = []
        self._writer = None
        self._domain = domain
        self._path = path

    def register(self, target_func,expr=None):
        if expr is not None:
            expr = replaceWithRefCoefficient(expr)
            self._registry.append((expr, target_func))
        if target_func not in self._function_set:
            self._function_set.add(target_func)
            self._functions.append(target_func)

    def setupWriter(self):
        self._writer = VTXWriter(self._domain.comm, self._path, self._functions, engine="BP5")
        self._writer.write(0.0)

    def write(self, time=None):
        for expr, func in self._registry:
            expr2node(expr, func)
        if self._writer and time is not None:
            self._writer.write(time)

    def close(self):
        if self._writer:
            self._writer.close()
