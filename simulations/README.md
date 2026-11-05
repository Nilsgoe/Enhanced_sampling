# Simulation and plotting workflows

This directory collects the current workflows behind the manuscript figures.
They are organized by physical system and retain command-line arguments for
input structures, output locations, and restarts instead of embedding a
particular batch directory.

| System | Workflow represented in the manuscript figures |
| --- | --- |
| `malonaldehyde/` | 1D umbrella MD: 0.5 fs, 293 K, spring constant 50 eV Angstrom^-2; Figure 3 analysis discards the first 1 ps and uses the following 9 ps. |
| `azobenzene/` | 2D wall-constrained WT-MetaD at 333 K: 0.5 fs, friction 0.1 in ASE units, sigma (10 deg, 10 deg), bias factor 10, hills every 100 steps. The wall driver supports MACE, SO3LR, PET-SPICE-L, fine-tuned models, and the other models in the eight-model comparison. |
| `trialanine/` | Solution WT-MetaD at 293 K: 0.5 fs, 1 kJ mol^-1 hill height (0.01036426966 eV), sigma (15 deg, 15 deg), bias factor 10, hills every 100 steps. The MACE, SO3LR, PET-SPICE-L, and ff19SB/OPC drivers are staged separately. |

The matching figure plotting scripts are included in each system directory.
They write to the canonical `figure_XX_.../data` locations.  Large raw
trajectories and bias files remain in the original campaign directories; the
figure directories contain the curated inputs needed to reproduce the released
plots.

Important label provenance: historical trajectories whose filenames contain
`off_off24` are the legacy MACE-OFF23 campaign.  The published display label is
`MACE-OFF23(M)`; it is corrected in the Figure 3 and Figure 4 plotting drivers.
