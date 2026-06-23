# Writing Input Files

This guide explains how to prepare input files for simulations based on the article examples in `CMAME-D-26-01111/`.

The current examples use two kinds of user-provided input files:

- CSV material-parameter tables
- Gmsh `.msh` mesh files for structural examples

Python driver scripts then connect those files to material parameters, fields, kernels, boundary conditions, and output paths.

## Material Parameter CSV Files

Material tables are plain CSV files read with:

```python
from Multiphysics.tiliuTools.data import load_tables_csv

tables = load_tables_csv(str(data_table))
```

The loader:

- uses the first row as column names
- removes spaces from column names
- converts all values to floating-point arrays
- returns a dictionary of `jax.numpy` arrays

For example, a column named ` sigY1 ` is accessed as:

```python
tables["sigY1"]
```

### Required Table Shape

The thermomechanical PA6 material examples expect one row per temperature point and one column per temperature-dependent parameter:

```text
T, E, sigY1, sigY2, beta, H, alpha, E2, tau, phi, delta, b, c
```

`T` is stored in Celsius in the CSV files. The drivers convert it to Kelvin:

```python
T_table = tables["T"] + 273.15
```

The remaining columns are passed to spline interpolators:

```python
E_interp = SplineInterpolator(T_table, tables["E"])
sigY1_interp = SplineInterpolator(T_table, tables["sigY1"])
```

### Minimal CSV Pattern

Use this structure when creating a new parameter table:

```csv
T,E,sigY1,sigY2,beta,H,alpha,E2,tau,phi,delta,b,c
23,6610,139,67,1687,473,0.188,1210,156,0.8,0.211,26,25
50,6398,1.2,56,654,459,0.525,903,71,2,0.295,0.5,18
160,901,0.8,234,340,150,2.088,103,21,2.24,0.517,0.1,1
```

Keep the column names stable unless you also update the driver and material-parameter construction code. The material class does not infer missing parameters.

### Reference Tables

- `single_element/PA6_tvp.csv`: table used by the single-element PA6 examples.
- `noniso_dogbone/PA6_shen.csv`: table used by the non-isothermal dogbone example.
- `gpu_benchmark/code/PA6_tvp.csv`: archived table used by the GPU benchmark driver.

## Connecting a CSV Table in a Driver

Use paths relative to the driver file so scripts work from any current working directory:

```python
from pathlib import Path

CASE_DIR = Path(__file__).resolve().parent
data_table = CASE_DIR / "PA6_tvp.csv"
tables = load_tables_csv(str(data_table))
```

After loading the table, construct interpolators and wrap them in the material-parameter namedtuple expected by the material class:

```python
MaterialParameters = namedtuple(
    "MaterialParameters",
    [
        "E", "sigY1", "sigY2", "alpha", "beta", "H",
        "E2", "tau", "phi", "delta", "c", "b", "lM", "nu", "chi",
    ],
)
```

If you add a new material parameter, update all three places consistently:

- the CSV header
- the driver-side table loading and interpolation
- the concrete material class that reads `par.<name>`

## Mesh Input Files

Structural examples can read Gmsh `.msh` files with DOLFINx:

```python
from dolfinx.io import gmshio

domain, cell_tags, facet_tags = gmshio.read_from_msh(str(mesh_path), comm)
```

Use paths relative to the driver:

```python
CASE_DIR = Path(__file__).resolve().parent
mesh_path = CASE_DIR / "dogbone_bak.msh"
```

### Mesh Requirements

When preparing a new `.msh` file:

- Export a Gmsh `.msh` file readable by the DOLFINx/Gmsh bridge.
- Include physical groups for the regions or facets used by the driver.
- Keep the spatial dimension and element type compatible with the kernels and boundary conditions.
- Update boundary-condition tags in the driver if the physical group ids change.

The dogbone reference mesh is:

```text
noniso_dogbone/dogbone_bak.msh
```

It is used by:

```text
noniso_dogbone/shen_dog.py
```

## Driver-Level Inputs

The driver script is where scalar run inputs are currently defined:

- initial temperature
- crystallinity factor `chi`
- Poisson ratio `nu`
- maximum displacement
- strain rate or loading rate
- number of time steps
- thermal conductivity
- boundary-condition values and physical tags
- output directory and file name

For example, `single_element/run_single_element.py` defines a generated box mesh in Python and reads only the parameter CSV file. `noniso_dogbone/shen_dog.py` reads both a parameter CSV file and a Gmsh mesh file.

## Generated Files

Simulation outputs should go under local `result/` directories. Generated ADIOS `.bp` folders, cache files, and local run logs should not be committed as input files.

The exception is `gpu_benchmark/logs/`, which intentionally stores archived benchmark logs for traceability. The corresponding `.bp` benchmark outputs are not included.
