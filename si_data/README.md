# Supplementary data

## Azobenzene DFT validation and test sets

`azobenzene_dft/valid_azob.xyz` and `test_azob.xyz` are the 240-structure
validation and 240-structure test splits used for the azobenzene DFT benchmark.
Their extended-XYZ headers contain the reference total energy and Cartesian
forces (eV and eV/Angstrom, respectively).  The matching
`*_dft_reference.npz` files provide the same reference targets in arrays:
`e` (eV), `forces` (eV/Angstrom), and `n_atoms`.

The training set is deliberately not duplicated here because it is much larger;
the immutable source is
`/nexus/posix0/FHI-Theory/ngoen/Enhanced_sampling/MACE/azobenzene_dft_benchmark/train_azob.xyz`.
