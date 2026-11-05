#!/usr/bin/env python3
"""Regenerate the archived 8-model azobenzene figures from raw bias files.

The underlying MACE-OFF calculation is OFF23(M).  Earlier versions labelled
it OFF24(M); ``display_label`` fixes only that model name during native plot
generation.  No rendered image is edited by this script.
"""

from __future__ import annotations

import runpy
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
# Write to the canonical Figure 4 data directory, rather than creating a
# second, disconnected copy beside this archived driver.
OUT = ROOT / "figure_data_and_scripts" / "figure_04_azobenzene_symmetric_fes" / "data"
ANALYSIS_ROOT = ROOT / "azobenzene_angle_wall_test_20260812"
BIAS_ROOT = OUT / "raw_bias" / "azobenzene_angle_wall_test_20260812"


def display_label(label: str) -> str:
    if label == "MACE-OFF24(M)":
        return "MACE-OFF23(M)"
    if label == "MACE-MH1":
        return "MACE-MH-1"
    if label == "Fine-tuned MACE-MH1":
        return "Fine-tuned MACE-MH-1"
    if label == "PET-SPICE":
        return "PET-SPICE-L"
    return label


def run(source: Path) -> None:
    # The analysis helper is deliberately reloaded from ANALYSIS_ROOT, avoiding
    # other copies of the module elsewhere in the working directory.
    sys.modules.pop("analyze_two_cnn_pathways", None)
    sys.path.insert(0, str(ANALYSIS_ROOT))
    namespace = runpy.run_path(str(source))
    # Point the native reconstruction at the staged, figure-local bias files.
    sys.modules["analyze_two_cnn_pathways"].ROOT = BIAS_ROOT
    # ``run_path`` returns a copy, so update the function's live globals.
    globals_ = namespace["main"].__globals__
    globals_["OUTPUT_DIR"] = OUT
    globals_["display_label"] = display_label
    namespace["main"]()


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    if "--individual-only" not in sys.argv:
        run(ROOT / "analyze_symmetric_azobenzene_mh1_label_2026-10-04.py")
    if "--symmetric-only" not in sys.argv:
        run(ROOT / "plot_azobenzene_8model_unsym_fes_pathways.py")


if __name__ == "__main__":
    main()
