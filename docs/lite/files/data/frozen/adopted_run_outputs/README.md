# Frozen Adopted Run Outputs

This folder contains the frozen first-stage, Polish output/spending and debt-accounting outputs for the adopted manuscript specification.

These files are copied into the public package as validation targets. The public runner rebuilds the source-level state variables from `data/frozen/adopted_sources/`, reruns estimation, and compares the recomputed outputs to these frozen benchmark targets. Manuscript-facing tables and figures must read recomputed outputs, not these files.

Hashes are written to `data/provenance/frozen_inputs_manifest_20260514.csv` by `code/run_public_repro.py`.
