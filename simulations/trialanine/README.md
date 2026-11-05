# Trialanine solution WT-MetaD

The `mace/`, `so3lr/`, `pet_spice/`, and `amber/` subdirectories contain the
current model-specific drivers.  All use the central-residue phi/psi CVs and
the 1 kJ mol^-1, 0.5 fs, 293 K protocol used for the Figure 5 comparison.
The PET-SPICE-L and SO3LR scripts are native-calculator variants; the MACE
driver supports OFF, OFF24, OMOL, MH-1, and POLAR selections.  The Amber
directory contains the ff19SB/OPC biASE driver plus the OpenMM comparator.

`plotting/plot_figure_05.py` is the current eight-model plotting script;
the two additional files reproduce the Amber replica and biASE/OpenMM SI plots.
