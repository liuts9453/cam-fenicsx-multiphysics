# GPU Benchmark Code

This directory archives the scripts used to generate the CPU/GPU benchmark logs in `../logs/`.

## Files

- `run_benchmark.py`: batch runner for the benchmark matrix. It loops over mesh sizes and platforms, then launches `Felder_ele_bench.py`.
- `Felder_ele_bench.py`: benchmark simulation driver. It builds the thermomechanical single-element mesh family, runs 50 loading steps, and monkey-patches material flux/tangent evaluation to time host-to-device transfer, JAX compute, device-to-host transfer, and remaining FEM runtime.
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

Generated ADIOS `.bp` outputs are intentionally not included in this repository.
