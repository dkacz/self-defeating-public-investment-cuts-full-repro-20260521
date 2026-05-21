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

## GPT Pro R4 status and R5 repair

GPT Pro R4 returned `REVISE`, score `9/10`. Arithmetic, data flow,
recompute-first notebook structure, benchmark-file validation, and the absence
of inline hardcoded expected values passed. One mandatory package-stability
fix remains: `code/sync_jupyterlite_files.py` must set the MIME type for `.log`
files explicitly to `text/plain`, so JupyterLite API metadata does not depend on
the host operating system's MIME registry.

The R5 local repair implements that explicit `.log` mapping and reruns the
public runner. Local QA now passes from the package root and from a freshly
extracted copy of `docs/downloads/full_repro_package_20260521.zip`. The
JupyterLite API metadata reports both notebook step logs as `text/plain`, and a
double full-run hash check is stable.

Paper-truth impact: none. The R4/R5 issue concerns public-package determinism,
not accepted estimates, retained evaluations, the EU27 benchmark, debt paths,
equal-weight results, model hierarchy, or the operator's replacement decision.

Closure status: awaiting targeted GPT Pro R5 review of the MIME fix and local
QA evidence.
