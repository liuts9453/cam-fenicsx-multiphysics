# Material Interface

This directory contains the reusable material base classes for the external-operator material interface. Concrete material laws are implemented in subclasses outside the base classes. For the CMAME article, the reference implementation for the IFT tangent path is:

```text
CMAME-D-26-01111/single_element/TVPkinIFT_heat.py
```

## Conceptual Structure

A material subclass connects three layers:

1. A UFL driving force, such as deformation gradient components and temperature.
2. A JAX quadrature-point update, including the local constitutive update and history update.
3. The external-operator interface, including flux evaluation and tangent evaluation.

`TVPkinIFT_heat` follows this pattern. Its UFL layer builds a driving-force vector from nine deformation-gradient components plus temperature. Its JAX layer evaluates the quadrature-point constitutive update in `flux_qp`. Its external-operator layer exposes the flux vector and the consistent tangent to FEniCSx assembly.

## Required Methods

Concrete subclasses must implement the material-specific pieces below.

```python
outputShape(self)
```

Return the number of flux components exposed by the material. `TVPkinIFT_heat` returns `7`.

```python
initHistoryArray(self)
```

Return the initial quadrature-point history vector. The length and ordering must match every later read and write in the material update.

```python
computeDrivingForce(self)
```

Build the UFL vector passed to the external operator. In `TVPkinIFT_heat`, the driving force is:

```text
[F_11, F_22, F_33, F_12, F_13, F_23, F_21, F_31, F_32, temperature]
```

```python
computeFunctionDerivatives(self)
```

Register material-specific derivative functions needed by the quadrature-point update. For example, `TVPkinIFT_heat` registers tangent matrices for its local plastic and viscoelastic residuals.

This method is material-specific and must be implemented by concrete material subclasses. It is not provided automatically by `ExoMaterial` or `ExoMaterialIFT`.

Older files may still contain the misspelled legacy method name:

```python
computeFunctionDerivates
```

New code should use:

```python
computeFunctionDerivatives
```

The base classes accept the legacy name as a fallback for older material files.

```python
computeProperties(self)
```

Call the base material setup and expose any named material outputs used by kernels or postprocessors. In `TVPkinIFT_heat`, this calls `ExoMaterialIFT.computeProperties(self)` and then assigns stress and heat-source views.

```python
computeStress(self)
```

Map the stress portion of the flux vector to a UFL vector used by mechanics kernels.

```python
computeHeatSource(self)
```

Map the heat-source portion of the flux vector for thermomechanical coupling.

## Flux and Driving-Force Ordering

The ordering of driving-force, history, and flux vectors is part of the material interface. Keep it consistent across UFL construction, JAX quadrature-point functions, tangent evaluation, and postprocessing.

`TVPkinIFT_heat` returns seven flux components:

```text
[PK2_11, PK2_22, PK2_33, PK2_12, PK2_13, PK2_23, heat_source]
```

Its driving force consists of nine deformation-gradient components plus temperature, as listed above.

## IFT Tangent Path

Materials inheriting from `ExoMaterialIFT` use an explicit implicit-function-theorem tangent path. These subclasses must define:

```python
computeQpJacobianIFT(...)
getQpJacobianInAxes()
getQpJacobianStaticArgnums()
```

`TVPkinIFT_heat.computeQpJacobianIFT` computes the consistent quadrature-point tangent from the converged local state. `getQpJacobianInAxes()` returns the batching axes used by `jax.vmap`, and `getQpJacobianStaticArgnums()` identifies static arguments for `jax.jit`.

## Common Mistakes

- Do not put concrete material laws into `ExoMaterial` or `ExoMaterialIFT`.
- Do not assume `computeFunctionDerivatives` is provided automatically.
- Keep the order of driving-force, history, and flux vectors consistent.
- Keep quadrature-point functions JAX-compatible.
- Do not use FEniCSx/UFL objects inside `flux_qp`.
