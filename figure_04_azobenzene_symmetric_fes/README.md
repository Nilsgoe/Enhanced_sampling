# Azobenzene 8-model FES figures

`scripts/export_nexus_20261009_off23.py` regenerates the figures directly
from the 16 raw wall-MetaD bias histories listed in `data/raw_bias_manifest.tsv`.
The exact successful histories are staged locally under `data/raw_bias/`, with
their original `runs/<model>/<cis-or-trans>/outputs/` layout preserved.
It invokes the archived native plotting scripts in the repository root and
uses the MACE-OFF23(M) model label.

Run from the repository root:

```bash
MPLCONFIGDIR=/tmp/matplotlib python figure_data_and_scripts/figure_04_azobenzene_symmetric_fes/scripts/export_nexus_20261009_off23.py
```

Authoritative outputs are:

- `data/azobenzene_symmetric_combined_fes.{png,pdf}` — symmetrised FES.
- `data/azobenzene_symmetric_combined_cnn_pathways_fes.{png,pdf}` — symmetric FES with the two CNN pathways.
- `data/azobenzene_symmetric_combined_cnn_pathways_profiles.{png,pdf}` — both pathway profiles.
- `data/azobenzene_cis_initialized_wall_fes_8model_pathways.{png,pdf}` — cis-initialised FES and both pathways.
- `data/azobenzene_trans_initialized_wall_fes_8model_pathways.{png,pdf}` — trans-initialised FES and both pathways.
