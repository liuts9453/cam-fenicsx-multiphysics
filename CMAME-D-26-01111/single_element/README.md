# Single-Element PA6 Example

This folder contains a representative thermomechanically coupled single-element PA6 simulation.

## Files

- `run_single_element.py`: main single-element thermomechanical simulation.
- `run_single_element_extract.py`: variant with point and quadrature-point extraction.
- `TVPkinIFT_heat.py`: local constitutive update for the thermomechanical PA6 model. The tangent uses AD derivatives of the local residual/flux together with the implicit-function theorem.
- `PS.py`: auxiliary output kernel for the plastic stretch.
- `PA6_tvp.csv`: tabulated temperature-dependent material parameters.

## Run

From the package root:

```bash
python CMAME-D-26-01111/single_element/run_single_element.py
```

The script writes its output under `CMAME-D-26-01111/single_element/result/`.
