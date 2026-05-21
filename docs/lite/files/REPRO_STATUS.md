# Reproduction Status

This package is recompute-first for the manuscript specification.

## Boundary

The default replication run uses frozen local inputs. It does not fetch live data during reproduction.

Frozen inputs:

- `data/frozen/adopted_sources/`: source-level CSV extracts used to rebuild the adopted state variables.
- `data/frozen/adopted_model_inputs/`: frozen local-projection panel, Commission baseline inputs, short-rate snapshots, and model-input scaffolding.
- `references/`: estimator code dependencies needed by the public reproduction scripts.

Frozen validation targets:

- `data/frozen/adopted_run_outputs/`: frozen feature-screen, output/spending, and debt-accounting outputs. These files are not used as substitutes for estimation. They are used only by `code/run_full_estimator_repro.py` to validate recomputed results.
- `data/frozen/eu27_benchmark_debt/`: frozen EU27 benchmark debt endpoint and annual decomposition targets. These files are not displayed by the manuscript-facing tables; they are used only to validate the recomputed EU27 debt benchmark.

Recomputed outputs:

- `results/recomputed/feature_screen/`: fresh 15-candidate feature screen.
- `results/recomputed/polish_output_spending/`: fresh retained Polish response paths.
- `results/recomputed/debt_accounting/`: fresh debt-accounting paths from the recomputed response paths.
- `results/recomputed/eu27_benchmark/`: fresh EU27 benchmark output/spending path, direct debt-to-GDP path, institutional debt recursion, 2036 endpoint and annual debt decomposition.
- `results/recomputed/estimation_output/`: coefficient-level regression output disclosure.

## Main Command

```bash
python3 code/run_public_repro.py
```

This command rebuilds source-level state variables, reruns the full estimator, rebuilds public tables and figures, builds the notebook, executes it, and writes QA ledgers.

## Validation

Latest local validation:

- `qa/full_estimator_model_input_rebuild_qa.csv`: 9/9 checks passed.
- `qa/full_estimator_repro_validation.csv`: 16/16 recomputed-versus-benchmark checks passed, including EU27 benchmark debt endpoint and annual decomposition.
- `qa/public_tables_figures_qa_20260514.csv`: 12/12 checks passed.
- `results/notebook_check_summary.csv`: 39/39 checks passed.
- `qa/public_repro_qa_20260521.csv`: package-level public reproduction PASS.

The maximum numeric differences in the full-estimator validation are within tolerance and arise only from CSV round-trip precision.

## Parameter Modes

The public notebook exposes:

- `PROFILE_YEAR`
- `SAMPLE_END_YEAR`
- `VALIDATION_MODE`

The default validation setting is `PROFILE_YEAR = 2022`, `SAMPLE_END_YEAR = 2022`, `VALIDATION_MODE = "benchmark"`. Changing the profile or sample requires `VALIDATION_MODE = "exploratory"`; those outputs are exploratory and are not compared to the frozen benchmark.

## Regression Output Disclosure

The package now writes compact and machine-readable regression-output tables:

- `tables/estimation_setup.csv`
- `tables/estimation_eu27_beta_by_horizon.csv`
- `tables/estimation_retained_beta_theta_by_horizon.csv`
- `tables/estimation_response_bridge_by_horizon.csv`
- `results/recomputed/estimation_output/retained_regression_coefficients.csv`
- `results/recomputed/estimation_output/eu27_benchmark_regression_coefficients.csv`

These tables report the visible regression output needed to address the annotation that the paper lacked estimation results.

## External Review Status

GPT Pro R4 returned `REVISE`, score `9/10`. Pro confirmed that the package is
recompute-first and accepted the arithmetic and data-flow evidence, but found
one deterministic-packaging defect: JupyterLite API metadata for `.log` files
could change across environments unless `.log` was explicitly mapped to
`text/plain` in `code/sync_jupyterlite_files.py`.

The MIME mapping was implemented and targeted GPT Pro R5 returned `PASS`, score
`10/10`, with no mandatory fixes. The full public runner passes from the
package root and from a freshly extracted public ZIP. JupyterLite API metadata
reports the notebook step logs as `text/plain`, and the full runner is
hash-stable across repeated runs.

## Public Delivery Status

The package is publicly delivered at:

```text
https://dkacz.github.io/self-defeating-public-investment-cuts-full-repro-20260521/lite/lab/index.html?path=notebooks/self_defeating_public_investment_cuts_full_repro_20260521.ipynb
```

GitHub Pages serves the repository
`dkacz/self-defeating-public-investment-cuts-full-repro-20260521` from
`main:/docs`. Public URL checks returned HTTP 200 for the landing page and the
browser-executable notebook. Browser QA passed in a fresh context and in a
persistent profile after preloading the older 2026-05-14 notebook URL.
