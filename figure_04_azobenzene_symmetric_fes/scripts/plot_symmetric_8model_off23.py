#!/usr/bin/env python3
"""Build the paper's symmetric azobenzene panel from historical 8-model FESs."""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parents[1]
RAW = HERE.parents[1] / "azobenzene" / "analysis" / "model_comparison" / "sigma10_8model_current"
MODELS = [("mh1", "MACE-MH1"), ("off", "MACE-OFF23(M)"), ("polar_m", "MACE-Polar-M"),
          ("ft_mace_mh1", "Fine-tuned MACE-MH1"), ("so3lr", "SO3LR"),
          ("ft_so3lr", "Fine-tuned SO3LR"), ("scratch", "MACE from scratch"), ("pet_spice", "PET-SPICE")]

def main():
    records = []
    for key, label in MODELS:
        cis = np.load(RAW / f"azobenzene_cis_{key}_sigma10_current_fes.npz")
        trans = np.load(RAW / f"azobenzene_trans_{key}_sigma10_current_fes.npz")
        # Align the two arbitrary offsets before averaging, then impose phi symmetry.
        combined = 0.5 * ((cis["fes_kcal_mol"] - cis["fes_kcal_mol"].min()) +
                          (trans["fes_kcal_mol"] - trans["fes_kcal_mol"].min()))
        combined = 0.5 * (combined + combined[:, ::-1]); combined -= combined.min()
        records.append((label, cis["cv1_grid"], cis["cv2_grid"], combined))
    values = np.concatenate([fes[(cnn >= 60) & (cnn <= 180)].ravel() for _, _, cnn, fes in records])
    vmax = np.nanpercentile(values[np.isfinite(values)], 99.9); levels = np.linspace(0, vmax, 17)
    fig, axes = plt.subplots(2, 4, figsize=(15.2, 7.2), sharex=True, sharey=True)
    for ax, (label, phi, cnn, fes) in zip(axes.flat, records):
        image = ax.contourf(phi, cnn, np.clip(fes, 0, vmax), levels=levels, cmap="RdBu_r", extend="max")
        ax.set_title(label); ax.set(xlim=(-180,180), ylim=(60,180), xticks=(-180,-90,0,90,180), yticks=(60,90,120,150,180))
    for row in axes: row[0].set_ylabel("CNN angle (deg)")
    for ax in axes[-1]: ax.set_xlabel("CNNC dihedral (deg)")
    fig.colorbar(image, ax=axes, label="Relative free energy (kcal mol$^{-1}$)", shrink=.9)
    fig.suptitle("Symmetric cis/trans azobenzene FES, $\\sigma=(10^\\circ,10^\\circ)$", y=1.02)
    for ext in ("png", "pdf"): fig.savefig(HERE / "data" / f"azobenzene_symmetric_8model_off23.{ext}", dpi=450, bbox_inches="tight")

if __name__ == "__main__": main()
