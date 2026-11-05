# Explicit MACE-OFF24-M umbrella data

These are the CSV observables from the explicit `MACE-OFF24_medium.model`
FHI-Raccoon campaign (`off24_raccoon_full_production_20261011`), not the
older MACE-OFF23 data that had been labelled as OFF24.

The plotting script applies the Figure-3 1+9 ps procedure in physical time.
Most windows are written every 5 fs, so it discards 200 frames and uses 1,800.
The five central reruns (shifts 12--16, marked `dense_center`) are written
every 0.5 fs, so it discards 2,000 frames and uses 18,000.  The retained
production data already begin after the separate 10 ps burn-in applied by the
production driver.

- `malonaldehyd`: 43 windows, shifts `-7` through `35`.
- `f-malonaldehyd`: 47 windows, shifts `-7` through `39`.
