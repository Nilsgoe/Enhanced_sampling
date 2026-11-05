# Malonaldehyde and F-malonaldehyde umbrella MD

Each molecule has the current UMD implementation for MACE, SO3LR, and
PET-SPICE-L.  `run_umbrella_mace.py` exposes MACE-OFF, MACE-MH-1, and
MACE-POLAR-1-M through `--model-family`/`--model-size`.  The SO3LR and
PET-SPICE-L variants are separate because they use their native calculators.

Run each window from a directory containing the appropriate optimized initial
XYZ file and an `outputs/` directory.  The Figure 3 plot script expects the
campaign CSV files at their original paths and reproduces the 1+9 ps analysis.
