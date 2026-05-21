# Frozen Source Inputs For Adopted Manuscript State Variables

This folder contains the small source-level CSV subset needed to rebuild the three manuscript state variables:

- investment import content from OECD TiVA 2025 GFCF domestic value-added shares;
- real PPP GDP per capita from Eurostat current PPS level and real GDP-per-capita index;
- household net financial worth from Eurostat financial-account assets, liabilities and nominal GDP.

The files are copied from the locally frozen source artifact after its source download and QA. They are not live downloads. The source-fetch manifest is in `../provenance/upstream_source_fetch_manifest_from_figaro_tiva_extension_20260514.csv`, and hashes for the copied subset are written by `code/rebuild_adopted_state_variables_from_sources.py` to `data/provenance/adopted_sources_manifest_20260514.csv`.

This folder deliberately excludes the large FIGARO matrices because FIGARO is diagnostic only and is not the adopted manuscript source.
