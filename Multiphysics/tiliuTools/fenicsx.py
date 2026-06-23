from __future__ import annotations
from typing import List
from dolfinx.fem import Expression
from dolfinx import mesh
import numpy as np
from dolfinx.fem import locate_dofs_topological, Constant, dirichletbc
import ufl
from dataclasses import dataclass
import numpy as np
from typing import Literal, Optional


def expr2qp(ufl_expr, func):
    func.interpolate(
        Expression(ufl_expr, func.function_space.element.interpolation_points())
    )


def view_qp_values(func, shape: tuple):
    func_vals = func.x.array.reshape(-1, *shape)
    for i, val in enumerate(func_vals):
        print(f"Qp {i}:\n {val}\n")
    return


def set_qp_vec(x, vec):
    def qp_vec(x):
        values = np.zeros((len(vec), x.shape[1]), dtype=np.float64)
        for i in range(len(vec)):
            values[i] = vec[i]
        return values

    x.interpolate(qp_vec)


class BoundaryCondiction:
    """Docstring for BoundaryCondiction."""

    def __init__(self, domain):
        self.domain = domain

    def _select(self, i, val):
        return lambda x: np.isclose(x[i], val)

    def _getSelectedDofs(self, ebene, var):
        dofs = []
        domain = self.domain
        fdim = domain.topology.dim - 1
        marked_facets = []
        marked_values = []

        for i in ebene:
            faceset = mesh.locate_entities_boundary(domain, fdim, self._select(*i))
            marked_facets.append(faceset)
        marked_values = np.hstack(
            [np.full_like(marked_facets[i], i) for i in range(len(marked_facets))]
        )
        marked_facets = np.hstack(marked_facets)
        sorted_facets = np.argsort(marked_facets)
        facet_tag = mesh.meshtags(
            domain, fdim, marked_facets[sorted_facets], marked_values[sorted_facets]
        )
        V = var.function_space
        for i in range(len(ebene)):
            dof = locate_dofs_topological(
                V.sub(ebene[i][0]), facet_tag.dim, facet_tag.find(i)
            )
            dofs.append(dof)
        return dofs

    def _updateDomainConst(self, domain_const, val):
        domain_const.value = val


from dolfinx import fem
import basix


class FieldManager:
    def __init__(self, domain):
        self.domain = domain
        self.cell = domain.basix_cell()
        self._element_cache = {}
        self._space_cache = {}
        self.std_fields = {}  # Fields on FE nodes
        self.qp_fields = {}  # Fields on Gauss points
        self.exprs = {}

    def _makeElement(self, shape, degree):
        key = ("std", shape, degree)
        if key not in self._element_cache:
            if shape == 1:
                el = basix.ufl.element("Lagrange", self.cell, degree)
            else:
                el = basix.ufl.element("Lagrange", self.cell, degree, shape=(shape,))
            self._element_cache[key] = el
        return self._element_cache[key]

    def _makeQpElement(self, shape, degree):
        key = ("qp", shape, degree)
        if key not in self._element_cache:
            el = basix.ufl.quadrature_element(
                self.cell, degree=degree, value_shape=(shape,)
            )
            self._element_cache[key] = el
        return self._element_cache[key]

    def _makeSpace(self, shape, degree, is_qp):
        key = ("qp" if is_qp else "std", shape, degree)
        if key not in self._space_cache:
            if is_qp:
                el = self._makeQpElement(shape, degree)
            else:
                el = self._makeElement(shape, degree)
            self._space_cache[key] = fem.functionspace(self.domain, el)
        return self._space_cache[key]

    def register(self, name, shape=1, degree=1):
        space = self._makeSpace(shape, degree, is_qp=False)
        func = fem.Function(space, name=name)
        self.std_fields[name] = func
        return func

    def registerVariable(self, name, expr):
        self.exprs[name] = expr
        return expr

    def registerQp(self, name, shape=1, degree=2):
        space = self._makeSpace(shape, degree, is_qp=True)
        func = fem.Function(space, name=name)
        self.qp_fields[name] = func
        return func

    def get(self, name):
        return self.std_fields.get(name, None) or self.qp_fields.get(name, None)

    def getStandard(self, name):
        return self.std_fields[name]

    def getVariable(self, name):
        return self.exprs.get(name, None)

    def getQp(self, name):
        return self.qp_fields[name]

    def allStandard(self):
        return list(self.std_fields.values())

    def allQp(self):
        return list(self.qp_fields.values())

    def scatterAll(self):
        for f in self.std_fields.values():
            f.x.scatter_forward()
        for f in self.qp_fields.values():
            f.x.scatter_forward()


def constantInitialValue(var, value):
    if len(var.ufl_operands) > 0:
        base, idx = var.ufl_operands

        component = int(idx.indices()[0])
        V = base.function_space
        n_comp = V.num_sub_spaces

        if len(ufl.shape(base)) == 0:
            # scalar function, like temperature alone
            base.x.array[:] = value
        elif len(ufl.shape(base)) == 1:
            # vector-valued function, only set the selected component
            base.x.array[component::n_comp] = value
        else:
            raise NotImplementedError(f"Unsupported function shape: {ufl.shape(base)}")
    else:
        var.x.array[:] = value


import gmsh
from dolfinx.io import gmsh as  gmshio


def read_msh_verbose(filename, comm):
    """
     .msh  Gmsh 
     (mesh, cell_tags, facet_tags)
    """
    if comm.rank == 0:
        gmsh.initialize()
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.open(filename)

        print(f"\n==== Mesh Summary for '{filename}' ====")
        dim_tags = gmsh.model.getEntities()
        summary = {}

        for dim, tag in dim_tags:
            try:
                element_types, element_tags, _ = gmsh.model.mesh.getElements(dim, tag)
            except:
                continue

            for etype, elems in zip(element_types, element_tags):
                props = gmsh.model.mesh.getElementProperties(etype)
                name = props[0]  # element name like "Triangle" or "Tetrahedron"
                summary[name] = summary.get(name, 0) + len(elems)

        for name, count in summary.items():
            print(f"  {name:<20} : {count:>6} elements")

        print("=======================================\n")
        gmsh.finalize()

    return gmshio.read_from_msh(filename, comm)


Mode = Literal[
    "total_load+n_steps", "total_time+n_steps", "total_load+dt", "total_time+dt"
]


@dataclass(frozen=True)
class LoadStepper:
    """
    Rate-controlled 1D loading schedule.

    State variables are abstract:
      - load(t): a scalar control quantity (could be displacement, strain, temperature, etc.)
      - rate: d(load)/dt (constant here)

    You can specify the schedule in several equivalent ways:
      1) total_load + n_steps  -> infer dt and total_time
      2) total_time + n_steps  -> infer dt and total_load
      3) total_load + dt       -> infer n_steps (rounded) and total_time
      4) total_time + dt       -> infer n_steps (rounded) and total_load
    """

    rate: float
    mode: Mode

    total_load: Optional[float] = None
    total_time: Optional[float] = None
    n_steps: Optional[int] = None
    dt: Optional[float] = None

    t0: float = 0.0
    load0: float = 0.0

    endpoint: bool = True  # include last point (t_end, load_end)
    dtype: type = np.float64

    def __post_init__(self):
        if not np.isfinite(self.rate) or self.rate <= 0.0:
            raise ValueError(f"rate must be positive and finite, got {self.rate}")

        # Basic presence checks per mode
        if self.mode == "total_load+n_steps":
            if self.total_load is None or self.n_steps is None:
                raise ValueError(
                    "mode 'total_load+n_steps' requires total_load and n_steps"
                )
            if self.total_load <= 0 or self.n_steps <= 0:
                raise ValueError("total_load > 0 and n_steps > 0 required")

        elif self.mode == "total_time+n_steps":
            if self.total_time is None or self.n_steps is None:
                raise ValueError(
                    "mode 'total_time+n_steps' requires total_time and n_steps"
                )
            if self.total_time <= 0 or self.n_steps <= 0:
                raise ValueError("total_time > 0 and n_steps > 0 required")

        elif self.mode == "total_load+dt":
            if self.total_load is None or self.dt is None:
                raise ValueError("mode 'total_load+dt' requires total_load and dt")
            if self.total_load <= 0 or self.dt <= 0:
                raise ValueError("total_load > 0 and dt > 0 required")

        elif self.mode == "total_time+dt":
            if self.total_time is None or self.dt is None:
                raise ValueError("mode 'total_time+dt' requires total_time and dt")
            if self.total_time <= 0 or self.dt <= 0:
                raise ValueError("total_time > 0 and dt > 0 required")

        else:
            raise ValueError(f"Unknown mode: {self.mode}")

    def build(self) -> dict[str, np.ndarray]:
        """
        Returns:
          {
            "dt": scalar float,
            "time_steps": (n_steps,) array,
            "time": (n_steps,) array of cumulative time (t0 excluded, like your original),
            "load": (n_steps,) array of load values at each step
          }
        """
        rate = float(self.rate)

        if self.mode == "total_load+n_steps":
            total_load = float(self.total_load)  # type: ignore[arg-type]
            n_steps = int(self.n_steps)  # type: ignore[arg-type]
            total_time = total_load / rate
            dt = total_time / n_steps

        elif self.mode == "total_time+n_steps":
            total_time = float(self.total_time)  # type: ignore[arg-type]
            n_steps = int(self.n_steps)  # type: ignore[arg-type]
            dt = total_time / n_steps
            total_load = rate * total_time

        elif self.mode == "total_load+dt":
            total_load = float(self.total_load)  # type: ignore[arg-type]
            dt = float(self.dt)  # type: ignore[arg-type]
            total_time = total_load / rate
            n_steps = int(np.round(total_time / dt))
            n_steps = max(n_steps, 1)
            # Recompute dt so the schedule hits total_load exactly at the end
            dt = total_time / n_steps

        elif self.mode == "total_time+dt":
            total_time = float(self.total_time)  # type: ignore[arg-type]
            dt = float(self.dt)  # type: ignore[arg-type]
            n_steps = int(np.round(total_time / dt))
            n_steps = max(n_steps, 1)
            dt = total_time / n_steps
            total_load = rate * total_time

        else:
            raise RuntimeError("unreachable")

        time_steps = np.full((n_steps,), dt, dtype=self.dtype)

        # Match your original convention:
        # time = cumsum(time_steps) starts at dt, ends at n_steps*dt
        time = self.t0 + np.cumsum(time_steps)
        load = self.load0 + rate * (time - self.t0)

        if not self.endpoint:
            # drop the last point (useful if you want "increments" only)
            time_steps = time_steps[:-1]
            time = time[:-1]
            load = load[:-1]

        return {
            "dt": np.asarray(dt, dtype=self.dtype),
            "time_steps": time_steps,
            "time": time,
            "load": load,
        }
