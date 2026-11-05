# Azobenzene 2D WT-MetaD

`run_2d_wtmetad.py` is the wall-constrained 333 K driver used for the symmetric
cis/trans Figure 4 comparison.  It accepts `mace_off`, `mace_mh1`,
`mace_polar_m`, `mace_scratch`, `mace_ft_mh1`, `pet_spice`, `sol3r`,
`ft_so3lr`, and `mace_les_off` via `--model`; select `cis` or `trans` with
`--system`.  It includes minimization, velocity initialization, restart-aware
chunks, the 10-degree two-CV bias, and the finite angular walls.

`plot_figure_04.py` regenerates the curated symmetric and individual
cis/trans figure products from the Figure 4-local raw bias archive.  Use
`--symmetric-only` or `--individual-only` to make either set.  The accompanying
analysis scripts are the native FES reconstruction code, not image overlays.
