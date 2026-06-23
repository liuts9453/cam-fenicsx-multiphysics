import dolfinx
from dolfinx import fem
from ufl import inner, Measure
import numpy as np


class BcsBuilder:
    """
    A utility class for creating and updating Dirichlet boundary conditions.
    Supports three modes:
    - coordinate-based automatic facet detection
    - facet tag-based boundary condition construction
    - domain-based (all DOFs) boundary condition construction
    """

    def __init__(self, domain, function_space, test_global):
        self.domain = domain
        self.V = function_space
        self.tdim = domain.topology.dim
        self.conditions = []
        self._bc_values = []
        self._facet_indices = []
        self._facet_tags = []
        self.test_global = test_global
        self.residual = 0.0

    def addCondition(
        self,
        axis=None,
        value=None,
        dof=None,
        bc_value=0.0,
        bc_type=1,
        tags=False,
        tag_id=None,
        tag_facets=None,
        domain_all=False,
    ):
        """
        Add a boundary condition to the builder.

        :param axis: Coordinate axis for auto-location
        :param value: Coordinate value for auto-location
        :param dof: Degree of freedom index
        :param bc_value: Value to apply at the boundary
        :param bc_type: Type of BC (1 = Dirichlet, 2 = Neumann/Source)
        :param tags: Whether using facet tag (True) or auto-location (False)
        :param tag_id: Tag ID (required if tags=True)
        :param tag_facets: Facet indices corresponding to tag
        :param domain_all: If True, applies the condition to the entire domain
        """
        if bc_type == 1:
            val = fem.Constant(self.domain, bc_value)
            self._bc_values.append(val)
        elif bc_type == 2:
            #val = fem.Constant(self.domain, bc_value)
            self._bc_values.append(bc_value)
        else:
            raise NotImplementedError(f"Unsupported BC type {bc_type}")


        if domain_all:

            self.conditions.append((dof, "DOMAIN", bc_type))
        elif tags:
            if tag_id is None or tag_facets is None:
                raise ValueError("tags=True requires tag_id and tag_facets")
            self._facet_indices.append(tag_facets)
            self._facet_tags.append(np.full(len(tag_facets), tag_id, dtype=np.int32))
            self.conditions.append((dof, tag_id, bc_type))
        else:
            if axis is None or value is None or dof is None:
                raise ValueError("Missing axis/value/dof for tag-less condition")
            facets = dolfinx.mesh.locate_entities(
                self.domain,
                self.tdim - 1,
                lambda x, i=axis, v=value: np.isclose(x[i], v),
            )
            tag_id = 1000 + len(self._facet_tags)
            self._facet_indices.append(facets)
            self._facet_tags.append(np.full(len(facets), tag_id, dtype=np.int32))
            self.conditions.append((dof, tag_id, bc_type))

    def build(self):
        """
        Construct the full list of dolfinx DirichletBC objects.
        """

        if len(self._facet_indices) > 0:
            all_facets = np.concatenate(self._facet_indices)
            all_tags = np.concatenate(self._facet_tags)
            facet_tags = dolfinx.mesh.meshtags(
                self.domain, self.tdim - 1, all_facets, all_tags
            )
            ds = Measure("ds", domain=self.domain, subdomain_data=facet_tags)
        else:
            facet_tags = None
            ds = None

        bcs = []
        for i, (dof, tag_id, bc_type) in enumerate(self.conditions):
            val = self._bc_values[i]
            
            # ==========================================================

            if tag_id == "DOMAIN":
                if bc_type == 1:

                    def all_domain(x):
                        return np.full(x.shape[1], True, dtype=bool)

                    if dof == -1:
                        dofs = fem.locate_dofs_geometrical(self.V, all_domain)
                        bc = fem.dirichletbc(val, dofs, self.V)
                    else:
                        dofs = fem.locate_dofs_geometrical(self.V.sub(dof), all_domain)
                        bc = fem.dirichletbc(val, dofs, self.V.sub(dof))
                    bcs.append(bc)
                elif bc_type == 2:
                    dx = Measure("dx", domain=self.domain)
                    r = inner(
                        val, self.test_global if dof == -1 else self.test_global[dof]
                    ) * dx
                    self.residual += r
            # ==========================================================

            else:
                if bc_type == 1:
                    if dof == -1:
                        dofs = fem.locate_dofs_topological(
                            self.V, self.tdim - 1, facet_tags.find(tag_id)
                        )
                        bc = fem.dirichletbc(val, dofs, self.V)
                    else:
                        dofs = fem.locate_dofs_topological(
                            self.V.sub(dof), self.tdim - 1, facet_tags.find(tag_id)
                        )
                        bc = fem.dirichletbc(val, dofs, self.V.sub(dof))
                    bcs.append(bc)
                elif bc_type == 2:
                    r = inner(
                        val, self.test_global if dof == -1 else self.test_global[dof]
                    ) * (ds if tag_id == -1 else ds(tag_id))
                    self.residual += r
                else:
                    raise NotImplementedError(f"Unsupported BC type {bc_type}")

        return bcs

    def update(self, idx, val):
        bc_item = self._bc_values[idx]

        if hasattr(bc_item, "value"):
            bc_item.value = val

        else:
            pass
