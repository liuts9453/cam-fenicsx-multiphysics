# CAM-FEniCSx-Multiphysics

This repository is the official CAM-FEniCSx-Multiphysics package accompanying the CMAME article:

**Thermomechanically Coupled Visco-Hyperelastic-Plastic Model with Self-Heating for Thermoplastics**
*Computer Methods in Applied Mechanics and Engineering*, 2026.

The package combines reusable FEniCSx/JAX multiphysics components with the article-specific material implementations and representative simulations.

## Repository Structure

- `Multiphysics/` contains the reusable framework components, including kernels, simulation managers, postprocessors, and base material interfaces.
- `fox/` contains utility helpers used by the framework.
- `CMAME-D-26-01111/` contains the article-specific material implementations, parameter tables, meshes, and representative simulations.
- `postprocessing/` contains compact utilities for curve comparison and scalar error-metric evaluation.

Generated results, local logs, caches, and ADIOS output folders are intentionally not included.

## Environment

The simulations require a FEniCSx environment with the following Python packages available:

- `dolfinx`
- `dolfinx_external_operator`
- `ufl`
- `basix`
- `mpi4py`
- `petsc4py`
- `jax`
- `jaxlib`
- `numpy`
- `psutil`

The compact plotting utilities in `postprocessing/` additionally use `matplotlib`.

## Article Examples

Run the representative single-element thermomechanical PA6 example from the repository root:

```bash
python CMAME-D-26-01111/single_element/run_single_element.py
```

For point and quadrature-point extraction output, run:

```bash
python CMAME-D-26-01111/single_element/run_single_element_extract.py
```

The representative non-isothermal dogbone simulation is available at:

```bash
python CMAME-D-26-01111/noniso_dogbone/shen_dog.py
```

Archived CPU/GPU benchmark logs for the article runtime-scaling study are available in:

```text
CMAME-D-26-01111/gpu_benchmark/
```

## Material Implementations

Concrete material laws are not implemented inside the base material classes. The base classes in `Multiphysics/Materials/` define the external-operator interface and shared mechanics for quadrature-point evaluation, history storage, and tangent assembly. Material-specific constitutive equations, history updates, flux vectors, and derivative registrations belong in concrete subclasses.

Users who want to implement their own material should start with:

```text
Multiphysics/Materials/README.md
```

The thermomechanical PA6 implementation

```text
CMAME-D-26-01111/single_element/TVPkinIFT_heat.py
```

is the reference material implementation for the implicit-function-theorem tangent path used by the single-element example.
