# Non-Isothermal Dogbone Example

This folder contains a representative structural dogbone setup for the non-isothermal Set 2 case in the CMAME article. The generated output files are intentionally omitted.

## Files

- `shen_dog.py`: non-isothermal dogbone driver adapted to the current repository layout.
- `ADTVPkin.py`: AD-based thermoviscoplastic material used by the driver.
- `dogbone_bak.msh`: dogbone mesh with the prescribed localization imperfection.
- `PA6_shen.csv`: Set 2 PA6 parameter table used by the driver.

## Run

From the package root:

```bash
python CMAME-D-26-01111/noniso_dogbone/shen_dog.py
```

The run writes output under:

```text
CMAME-D-26-01111/noniso_dogbone/result/
```

## Notes

The active material parameters match the Set 2 columns in manuscript Table `tab:felder_params`.

- The CSV contains `alpha` values for Set 2, while the manuscript table uses `-`. In `shen_dog.py`, `chi=1.0`, so these alpha values do not affect `sigY2`.
- The mesh contains 1536 hexahedral elements and includes the small prescribed imperfection used to trigger off-center localization.
- The script uses `h = 0.025 mJ/(s mm^2 K)`, equivalent to `25 W/(m^2 K)`, matching the manuscript Robin boundary condition.
- The script uses thermal conductivity `k = 0.28` in the mm-mJ-s unit system. The manuscript weak form defines `k`, but the current text does not state this numerical value.
- The `ADTVPkin` name denotes the direct AD material path used by this structural setup. The single-element example uses the separate `IFT` tangent-construction path.
