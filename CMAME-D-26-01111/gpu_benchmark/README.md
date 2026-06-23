# GPU Benchmark Logs

This directory archives the CPU/GPU benchmark logs used to document the runtime scaling behavior of the thermomechanically coupled FEniCSx/JAX implementation.

Only log files are included. The generated ADIOS `.bp` result directories are intentionally omitted because they are large derived outputs.

## Benchmark Matrix

The benchmark was run for cubic hexahedral meshes with:

```text
1^3, 3^3, 5^3, 7^3, 10^3, 20^3, 30^3
```

For each mesh, two platforms were measured:

```text
cpu
gpu
```

Each run used 50 loading steps.

## Log Files

The logs are stored in:

```text
CMAME-D-26-01111/gpu_benchmark/logs/
```

The naming convention is:

```text
bench_<platform>_<mesh_size>.log
```

For example:

```text
bench_cpu_30.log
bench_gpu_30.log
```

## Recorded Quantities

Each log contains the full simulation console output and a profiling block at the end. The profiling block reports:

- total wall-clock time for 50 steps
- JAX first-call time, including JIT compilation and initial compute
- JAX pure constitutive compute time without first-call/JIT contribution
- average JAX constitutive compute time per call
- host-to-device transfer time
- device-to-host transfer time
- total host/device communication time
- remaining global assembly and solve time

The benchmark driver used the command pattern:

```bash
conda run -n dof10 python Felder_ele_bench.py --mesh_size <N> --platform <cpu|gpu> --steps 50
```

with `JAX_PLATFORM_NAME` set to the selected platform.

## Hardware and Software Context

The benchmark was run on the workstation used for the article computations:

```text
CPU: AMD Ryzen 5 5800X
GPU: NVIDIA GeForce RTX 5070 Ti
dolfinx: 0.10.0
JAX/CUDA: JAX with CUDA 13.2
```

These logs are intended to document the reported article benchmark data, not to provide hardware-independent performance claims.
