# Kernels

This directory contains reusable weak-form kernels for the CAM-FEniCSx-Multiphysics framework. A kernel contributes one residual integrand to the monolithic finite-element problem or to an auxiliary projection solve.

Concrete material laws are not implemented here. Kernels consume fields exposed by a material class, such as `material.F`, `material.stress`, `material.heat_source`, and `material.hist`.

## Kernel Interface

All kernels inherit from `Kernel` in `Base.py` and implement:

```python
computeQpResidual(self)
```

The method returns a UFL expression for one quadrature-point residual contribution. `ActionManager` combines the selected kernels, attaches the correct test functions, applies the quadrature measure, differentiates the residual, and replaces external operators for material flux and tangent evaluation.

Simulation drivers register kernels as tuples:

```python
kernels=[
    (TotalLagrangianStressDivergence, var_displacement),
    (
        HeatConduction,
        var_temperature,
        {
            "temperature_old": T,
            "thermal_conductivity": 0.27,
            "unit": 0,
        },
    ),
]
```

The first entry is the kernel class. The second entry is the field variable the kernel acts on. An optional dictionary provides kernel parameters.

Auxiliary kernels use the same tuple pattern through `aux_kernels`.

## Current Kernels

### Mechanical Equilibrium

`TotalLagrangianStressDivergence`

Adds the total-Lagrangian mechanical residual for a 3D displacement field. It expects:

- `material.F`: deformation gradient
- `material.stress`: second Piola-Kirchhoff stress in six-component Nye ordering

`TotalLagrangianStressDivergence2D`

2D variant using the in-plane block of `material.F` while keeping the six-component stress-vector convention.

### Heat Equation

`HeatConduction`

Adds the thermal residual used by the thermomechanically coupled examples. It expects:

- `material.heat_source`
- `temperature_old`
- `thermal_conductivity`
- `unit`

The residual combines heat-source, conduction, and backward-Euler heat-capacity terms.

`PoissonEq`

Small Poisson-style reference kernel with a constant source term.

### Auxiliary Projection Kernels

`PushForwardCauchy`

Projects the Cauchy stress computed from `material.stress` and `material.F`.

`PushForwardPK1`

Projects the first Piola-Kirchhoff stress computed from `material.stress` and `material.F`.

`DeformationGradient`

Projects `material.F` into a nine-component field.

`HistVariables`

Projects one history-variable component selected by `{"index": i}`.

`PushForwardCauchyPlas`

Projects a plastic Cauchy-stress-like quantity from selected history entries. This kernel is tied to material history ordering and should be used only when that ordering matches.

`PlasticStretch`

Projects the first six history components as a plastic-stretch vector. This is also material-history-order dependent.

### Constraint/Reference Kernel

`ConstanScalerVariable`

Constrains a scalar field to a supplied value:

```python
(ConstanScalerVariable, var_temperature, {"value": init_temp})
```

The class name keeps the historical spelling used by existing scripts.

## Adding a Kernel

To add a new kernel:

1. Subclass `Kernel`.
2. Implement `computeQpResidual(self)` using UFL expressions only.
3. Read required fields from `self.u`, `self.test`, `self.material`, `self.par`, and `self._dt`.
4. Add the class to `Multiphysics/Kernels/__init__.py` if it should be public.
5. Register it in a simulation driver through `kernels` or `aux_kernels`.

Do not call JAX kernels inside a UFL kernel. JAX constitutive updates belong in material classes and are exposed to UFL through the external-operator material interface.
