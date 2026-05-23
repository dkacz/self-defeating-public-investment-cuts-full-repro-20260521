# Public reproducibility package

Date: 2026-05-21

This package reproduces the manuscript-facing empirical results for the paper
`Self-Defeating Public Investment Cuts: Evidence from Poland under EU Fiscal
Surveillance`.

The state-variable system uses:

- investment import content from official OECD TiVA GFCF data;
- public debt;
- household net financial worth;
- real PPP GDP per capita.

The retained Polish evaluations are investment import content and household
net financial worth. The package ships frozen source extracts, frozen model
inputs, frozen run outputs, provenance manifests, table and figure builders,
the explanatory Jupyter notebook, a browser-executable JupyterLite site, and
local quality-assurance ledgers.

Appendix C source coverage is complete in the public package. The EU27
benchmark endpoint row and annual institutional debt-decomposition rows for
Appendix C.2 and C.3 are recomputed by `code/run_full_estimator_repro.py` under
`results/recomputed/eu27_benchmark/` and exported by the runner as
`tables/eu27_annual_debt_decomposition.csv`. The copied files under
`data/frozen/eu27_benchmark_debt/` are validation targets, not substitutes for
recomputation. This preserves the full content of the wide Appendix tables;
readability should be handled by splitting tables rather than dropping columns.

## Headline values

| Quantity | Value |
| --- | ---: |
| Horizon-8 EU27 output response | 2.11 |
| Horizon-8 investment-import-content Polish response | 1.84 |
| Horizon-8 household-net-financial-worth Polish response | 2.16 |
| Horizon-8 equal-weight Polish response | 2.00 |
| 2036 cut margin, equal-weight institutional debt equation | +5.7 pp |
| 2036 cut margin, equal-weight direct debt-to-GDP path | +3.0 pp |

## How to run

In the browser:

```text
https://dkacz.github.io/self-defeating-public-investment-cuts-full-repro-20260521/lite/lab/index.html?path=notebooks/self_defeating_public_investment_cuts_full_repro_20260521.ipynb
```

The browser run uses JupyterLite with the Python Pyodide kernel. The local
package includes frozen inputs and validation targets; the full local runner
recomputes the estimator before rebuilding the manuscript tables, figures and
QA ledgers.

From the package root:

```bash
python3 code/run_public_repro.py
```

A successful run prints:

```text
public reproducibility PASS
```

The main notebook is:

```text
notebooks/self_defeating_public_investment_cuts_full_repro_20260521.ipynb
```

The downloadable code and frozen-data bundle is:

```text
docs/downloads/full_repro_package_20260521.zip
```

Local quality-assurance results are recorded in
`qa/public_repro_qa_20260521.csv` and
`results/notebook_check_summary.csv`.

## Recompute And Public Package Status

The package is designed as a recompute-first replication of the manuscript
setting. Local QA passes from the package root and from a freshly extracted copy
of `docs/downloads/full_repro_package_20260521.zip`. JupyterLite metadata maps
notebook step logs to `text/plain`, so browser metadata does not depend on the
host operating system's MIME registry.

The notebook exposes the retained Polish output/spending paths and the Polish
debt-accounting paths through visible audit cells. It estimates retained
response paths, verifies cumulative K arithmetic, estimates direct debt kernels,
builds the three-year programme paths, reproduces the baseline, simulates debt
paths and writes the canonical output files. The canonical CSV outputs remain
schema-compatible with the frozen validation targets, and the full public runner
returns:

```text
public reproducibility PASS
```

Current QA:

- `qa/full_estimator_repro_validation.csv`: 16/16 PASS.
- `results/notebook_check_summary.csv`: 19/19 PASS.
- `qa/download_archives_qa_20260521.csv`: 8/8 PASS.
- `qa/jupyterlite_files_sync_qa_20260521.csv`: 6/6 PASS.

The package reproduces the current manuscript-facing specification. It does not
introduce an alternative empirical specification or revise the reported values;
its role is to make the data, estimation, accounting, tables and figures
auditable from the shipped inputs.

## Public URL Freshness

Public delivery is closed only by a no-cache freshness check against the live
GitHub Pages objects. The notebook object expected in the live JupyterLite
payload has SHA-256:

```text
4b17e280aa3e1e27952eb4228626fc5946afa9d8b162817e8e1177695edfbb76
```

The full-archive hash is intentionally recorded only in the external
live-public freshness report and in the adjacent `.sha256` file, not inside the
archive-bearing report itself. This avoids a self-referential archive hash.

The package includes the manuscript source bundle under `manuscript/`,
including the QMD file, the table and figure include targets, and the PDF
formatting header used by the manuscript source.
