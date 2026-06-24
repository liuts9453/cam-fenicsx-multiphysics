# GPU Benchmark Code

This directory archives the scripts used to generate the CPU/GPU benchmark logs in `../logs/`.

## Files

- `run_benchmark.py`: batch runner for the benchmark matrix. It loops over mesh sizes and platforms, then launches `Felder_ele_bench.py`.
- `Felder_ele_bench.py`: benchmark simulation driver. It builds the thermomechanical single-element mesh family, runs 50 loading steps, and times flux-side host-to-device transfer, JAX constitutive compute, device-to-host transfer, and remaining FEM runtime.
- `TVPkinIFT_heat.py`: material implementation used by the benchmark driver. It uses the IFT tangent-construction path.
- `PA6_tvp.csv`: parameter table used by the benchmark material.
- `gen_table.py`: extracts the profiling block from `../logs/bench_*.log` and prints a Markdown summary table.
- `plot_bars.py`: parses benchmark logs and generates a runtime decomposition figure.
- `plot_composite.py`: generates the composite GPU benchmark figure from the logged summary values.
- `response_profiling_summary.md`: manually curated profiling table used during response preparation.

## Running the Benchmark

The original benchmark command pattern was:

```bash
conda run -n dof10 python Felder_ele_bench.py --mesh_size <N> --platform <cpu|gpu> --steps 50
```

`run_benchmark.py` executes the full CPU/GPU matrix:

```bash
python run_benchmark.py
```

The benchmark was run with `dolfinx 0.10.0`, JAX with CUDA support, and `JAX_PLATFORM_NAME` set to the selected platform.

Inside `Felder_ele_bench.py`, the selected backend is applied with:

```python
import jax

jax.config.update("jax_platform_name", args.platform)
```

For standalone scripts, use one of:

```python
jax.config.update("jax_platform_name", "cpu")
jax.config.update("jax_platform_name", "gpu")
```

With CUDA-enabled `jaxlib`, `"gpu"` selects the CUDA-capable GPU backend and arrays are placed on a CUDA device such as `cuda:0`. Set the platform before creating JAX arrays, JIT functions, or material instances.

Generated ADIOS `.bp` outputs are intentionally not included in this repository.
