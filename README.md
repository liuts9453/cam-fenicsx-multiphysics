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

## Framework Documentation

- `Multiphysics/Materials/README.md`: material-interface guide for implementing new material classes.
- `Multiphysics/Kernels/README.md`: weak-form kernel interface and current reusable kernels.
- `CMAME-D-26-01111/INPUT_FILES.md`: article-specific input tables, mesh files, and generated-file exclusions.

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

## GPU Benchmark Logs

The repository includes the CPU/GPU benchmark logs and related benchmark code used for the article runtime-scaling study:

```text
CMAME-D-26-01111/gpu_benchmark/
```

The benchmark measures the end-to-end thermomechanical simulation runtime for cubic hexahedral meshes from `1^3` to `30^3`, with 50 loading steps on both CPU and GPU. The logs include the total wall-clock time and a runtime decomposition into first-call/JIT cost, JAX constitutive compute time, host/device communication, and remaining global assembly and solve time.

JAX platform selection is controlled before the material kernels are initialized. In a script, use:

```python
import jax

jax.config.update("jax_platform_name", "cpu")  # CPU backend
jax.config.update("jax_platform_name", "gpu")  # CUDA-capable GPU backend
```

Only one platform should be selected in a run. The benchmark driver exposes the same switch through `--platform cpu` or `--platform gpu`. With CUDA-enabled `jaxlib`, the selected GPU device appears as a CUDA device such as `cuda:0`, even though the script-level JAX platform string is `"gpu"`.

The benchmark was run with `dolfinx 0.10.0` and JAX/CUDA on the workstation used for the article computations. The data are included to document the reported scaling behavior of the framework; they should be interpreted as benchmark records for this implementation and hardware configuration, not as hardware-independent performance claims.

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

## License

This repository is released under the MIT License; see `LICENSE`.

If you use this package, the article-specific material implementation, or the benchmark data in academic work, please cite the accompanying CMAME article listed at the top of this README.
