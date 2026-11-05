# Are Molecular Foundation Models Ready for Free Energy Simulations?

This repository accompanies the paper **[Are Molecular Foundation Models Ready
for Free Energy Simulations?](https://chemrxiv.org/doi/full/10.26434/chemrxiv.15010593/v1)**.
It contains the figure inputs, analysis and plotting scripts, and submitted
figure products used in the manuscript and Supporting Information.

## What is included

The repository is organized by manuscript figure:

| Figure | Directory | Contents |
| --- | --- | --- |
| Fig. 1 | `figure_01_biase_workflow/` | Description of the submitted schematic. |
| Fig. 2 | `figure_02_collective_variables/` | Submitted collective-variable artwork. |
| Fig. 3 | `figure_03_malon_fmalon_umbrella/` | Malondialdehyde and fluoromalondialdehyde umbrella-sampling analyses, final curves, and raw MACE window data. |
| Fig. 4 | `figure_04_azobenzene_symmetric_fes/` | Azobenzene wall-MetaD bias histories, 3-ns reconstructed surfaces, and FES/pathway analyses. |
| Fig. 5 | `figure_05_trialanine_8model/` | Trialanine MetaD histories, eight-model comparison script, and numerical summaries. |
| SI Figs. 1–4 | `si_figure_*/` | Dataset-split, unsymmetrized azobenzene, and AMBER/OpenMM comparison inputs and scripts. |

`reference/manuscript.pdf` and `reference/SI.pdf` provide the submitted
manuscript and Supporting Information for figure ordering and captions.

## Reproducing packaged plots

Run commands from this directory. The following workflows use the packaged
local data and write their outputs to the relevant `data/` directory:

```bash
export MPLCONFIGDIR=/tmp/matplotlib

python figure_04_azobenzene_symmetric_fes/scripts/analyze_symmetric_azobenzene_mh1_label_2026-10-04.py
python figure_04_azobenzene_symmetric_fes/scripts/plot_azobenzene_3ns_native.py
python figure_05_trialanine_8model/scripts/plot_trialanine_solution_8model_kj_2026-10-09.py

python si_figure_01_azobenzene_dataset_split/scripts/plot_ft_mh1_train_test_validation_overlays_2026-09-14.py
python si_figure_02_azobenzene_unsymmetrized_fes/scripts/plot_azobenzene_8model_unsym_fes_pathways.py
python si_figure_03_amber_forcefield_replicas/scripts/plot_trialanine_amber_biase_4rep_kj_2026-10-09.py
python si_figure_04_amber_biase_openmm/scripts/plot_trialanine_amber14_biase_openmm_4rep_rows_kj_2026-10-09.py
```

The scripts require a Python environment containing at least NumPy,
Matplotlib, and ASE. The molecular-dynamics drivers in `simulations/` have
additional calculator and model-checkpoint requirements; they are provided as
workflows, not as a turnkey execution environment.

## Figure 3 raw-data note

The package includes final Figure-3 curves and diagnostics, raw legacy-MACE
windows, and explicit MACE-OFF24-M window data. Raw SO3LR and PET-SPICE-L
umbrella outputs remain in the original campaign locations because they are
large. Complete raw-data re-analysis of every Figure-3 model therefore
requires those campaign outputs.

## Provenance and terminology

Historical files named `off_off24` were generated with MACE-OFF23(M), despite
the older filename; the current figure scripts display the corrected
MACE-OFF23(M) label.

AMBER reference panels use four independent biASE replicas (replicas 1–4).

## Data scope

This is a figure-data and analysis package. Large trajectories, complete
simulation environments, and editable sources for schematic/vector-art
figures are not included. Where a submitted graphical asset has no numerical
generation step (Figures 1 and 2), the released artwork is retained directly.
