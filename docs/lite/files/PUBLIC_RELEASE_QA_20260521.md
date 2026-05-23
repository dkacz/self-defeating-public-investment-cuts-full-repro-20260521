# Public Reproducibility Release QA

> **Historical public-release evidence.** This file records the 2026-05-21 public URL check from the earlier public-repro release. It is retained for provenance only. It is not current strict-packet public freshness evidence, and current strict packets must use the later local package QA or a fresh no-cache public URL verification.
Date: 2026-05-21

## Public object

- Repository: `dkacz/self-defeating-public-investment-cuts-full-repro-20260521`
- GitHub Pages source: `main:/docs`
- Browser-executable notebook: GitHub Pages JupyterLite build under `docs/lite/`
- Downloadable archive: `docs/downloads/full_repro_package_20260521.zip`

## Local release checks

The public package was rebuilt from the local release root with the full reproducibility runner. The runner rebuilds source-derived state variables, recomputes the estimator, rebuilds public tables and figures, executes the notebook, rebuilds the downloadable archive and refreshes the JupyterLite payload.

Current local status:

```text
public reproducibility PASS
```

The notebook executes the full estimator inside the notebook session. Frozen benchmark outputs are used only as validation targets; displayed tables, figures, debt paths and notebook QA ledgers are generated from recomputed objects in the current run.

## Reader-facing label QA

The public package now separates reader-facing labels from technical validation internals. Reader-facing prose, displayed notebook outputs, public tables, feature-screen reports, the executed preview, the downloadable archive and the JupyterLite payload use economic labels such as investment import content, household net financial worth, public debt and real PPP income.

Executable source code and machine-validation internals may retain compact variable names where they are required for reproducibility. Those internals are not used as display labels in the notebook, public reports or reader-facing tables.

## Download and JupyterLite checks

- The full archive includes code, frozen inputs, validation targets, public tables, figures, notebook, QA ledgers and documentation.
- The full archive excludes the static JupyterLite build and the archive directory itself, so the downloadable object does not contain stale copies of previous payloads.
- JupyterLite omits precomputed recomputed-output directories; the notebook recreates those outputs during execution.
- The archive and the JupyterLite payload are scanned for reader-facing label leaks before the runner can pass.

## Browser freshness check

The browser-executable notebook was first opened through a local JupyterLite build in both a fresh browser context and a persistent browser profile. Both checks opened the new notebook filename, and the old public notebook filename returned 404 from the payload. The local non-browser notebook execution also passed end to end through `code/execute_public_notebook.py`.

After publication, the GitHub Pages landing page and direct notebook URL both returned HTTP 200. A second browser QA pass opened the public URL in a fresh context and in a persistent profile after preloading the older 2026-05-14 notebook URL. Both checks saw `self_defeating_public_investment_cuts_full_repro_20260521.ipynb` and the new JupyterLite API did not list the old notebook filename.

## Public delivery

- Repository: `https://github.com/dkacz/self-defeating-public-investment-cuts-full-repro-20260521`
- Pages URL: `https://dkacz.github.io/self-defeating-public-investment-cuts-full-repro-20260521/`
- Notebook URL: `https://dkacz.github.io/self-defeating-public-investment-cuts-full-repro-20260521/lite/lab/index.html?path=notebooks/self_defeating_public_investment_cuts_full_repro_20260521.ipynb`
- Delivery QA: `delivery/PUBLIC_URL_BROWSER_QA_20260521.md`

## Result Impact

None. This QA concerns publication mechanics and reader-facing package hygiene only. It does not change the manuscript estimates, retained state-variable evaluations, debt paths, model hierarchy, or default replication design.
