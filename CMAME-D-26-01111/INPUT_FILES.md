# Input Files

This document summarizes the article-specific input files stored under `CMAME-D-26-01111/`.

Generated simulation outputs, ADIOS `.bp` directories, cache files, and local logs are not considered input files and are not included, except for the archived GPU benchmark logs in `gpu_benchmark/logs/`.

## Parameter Tables

Material parameter tables are plain CSV files read with:

```python
Multiphysics.tiliuTools.data.load_tables_csv
```

The loader removes spaces from column names and converts each column to a `jax.numpy` array of floats.

### `single_element/PA6_tvp.csv`

Input table for the thermomechanically coupled single-element PA6 example:

```text
CMAME-D-26-01111/single_element/PA6_tvp.csv
```

Used by:

```text
CMAME-D-26-01111/single_element/run_single_element.py
CMAME-D-26-01111/single_element/run_single_element_extract.py
```

The table contains temperature-dependent material parameters with columns:

```text
T, E, sigY1, sigY2, beta, H, alpha, E2, tau, phi, delta, b, c
```

The simulation scripts convert `T` from Celsius to Kelvin before constructing spline interpolators.

### `noniso_dogbone/PA6_shen.csv`

Input table for the representative non-isothermal dogbone example:

```text
CMAME-D-26-01111/noniso_dogbone/PA6_shen.csv
```

Used by:

```text
CMAME-D-26-01111/noniso_dogbone/shen_dog.py
```

It uses the same column convention as the single-element table and supplies the Set 2 parameters used by the structural dogbone setup.

### `gpu_benchmark/code/PA6_tvp.csv`

Archived copy of the parameter table used by the GPU benchmark driver:

```text
CMAME-D-26-01111/gpu_benchmark/code/PA6_tvp.csv
```

It is kept with the benchmark code so the uploaded logs and benchmark scripts remain self-contained.

## Mesh Files

### `noniso_dogbone/dogbone_bak.msh`

Gmsh mesh for the non-isothermal dogbone example:

```text
CMAME-D-26-01111/noniso_dogbone/dogbone_bak.msh
```

Used by:

```text
CMAME-D-26-01111/noniso_dogbone/shen_dog.py
```

The driver reads this file with Gmsh/DOLFINx and expects the physical tags used by the boundary-condition setup in the script.

## Generated Files

The example scripts create result directories under their local `result/` folders. These generated files are intentionally excluded from the repository.

The GPU benchmark directory includes only log files and code. It does not include generated `.bp` result directories.
