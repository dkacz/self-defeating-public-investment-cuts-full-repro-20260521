# Self-Defeating Public Investment Cuts: Reproducibility Package

This package contains the public Jupyter reproducibility package for the
manuscript-facing empirical results.

Run the notebook in the browser:

```text
https://dkacz.github.io/self-defeating-public-investment-cuts-full-repro-20260521/lite/lab/index.html?path=notebooks/self_defeating_public_investment_cuts_full_repro_20260521.ipynb
```

The browser version is served through JupyterLite with the Python Pyodide
kernel. The local package includes frozen inputs and validation targets; the
full local runner recomputes the estimator before rebuilding the manuscript
tables, figures and QA ledgers.

Notebook source:

```text
notebooks/self_defeating_public_investment_cuts_full_repro_20260521.ipynb
```

Downloadable code and frozen-data bundle:

```text
docs/downloads/full_repro_package_20260521.zip
```

To rerun the full local reproduction from the package root:

```bash
python3 code/run_public_repro.py
```

Expected terminal result:

```text
public reproducibility PASS
```

The package includes the frozen data extracts, provenance manifests, scripts,
tables, figures, results and quality-assurance ledgers needed to reproduce the
reported values.

The Appendix C EU27 benchmark rows are recomputed by the local runner under
`results/recomputed/eu27_benchmark/` and then exported to
`tables/eu27_annual_debt_decomposition.csv`. The frozen EU27 files under
`data/frozen/eu27_benchmark_debt/` are validation targets only. Wide Appendix
tables should be split for reading rather than substantively reduced.
