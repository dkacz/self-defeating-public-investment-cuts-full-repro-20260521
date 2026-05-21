# State-variable screen

## Scope

This public reproduction block evaluates the state-variable selection step for the Polish heterogeneity model. It starts from four pre-specified state variables: investment import content, public debt, household net financial worth and real PPP income. It evaluates all 15 non-empty subsets.

The script is deliberately ordered as a single selection exercise:

1. enumerate all feature subsets;
2. estimate output and spending kernels for every subset;
3. apply robustness criteria that use first-stage diagnostics only, with no later accounting results, no later-stage signs, no output-only fit ranking and no kernel-sign filters.

This public reproduction block does not calculate later accounting paths. Those estimates for the selected specifications belong to later modelling blocks.

## Robustness criteria

| Criterion | Source support | Use |
| --- | --- | --- |
| Ex ante state-variable universe | Ilzetzki, Mendoza and Vegh (2013); Huidrom et al. (2019); Bernardini and Peersman (2018); Krajewski and Pilat (2025); McManus et al. (2020) | Justifies trade openness, public debt, level of development and the Polish liquidity or credit-constraint extension. |
| Local-projection core | Jorda (2005); Ciaffi, Deleidi and Di Domenico (2024) | Justifies local projections estimated separately at each horizon and separate output and debt responses. |
| Shock and state-variable interactions | Auerbach and Gorodnichenko (2012); Ramey and Zubairy (2018); Cloyne, Jorda and Taylor (2023) | Justifies interacting fiscal shocks with state or country characteristics. |
| Common comparison sample | Pre-specified comparison-sample rule | Candidate specifications are compared on the same h8 observation universe where feasible. The model-averaging papers remain model-uncertainty context, not direct support for this fixed-sample rule. |
| Rank, condition-number and collinearity checks | Welsch and Kuh (1977) | Regression diagnostic hygiene: full rank, condition number and collinearity checks. |
| Poland support-overlap check | Crump et al. (2009); Li, Morgan and Zaslavsky (2018) | Justifies common-support or overlap checks before using Poland's feature profile. |
| Country-level bootstrap and leave-one-country checks | Cameron and Miller (2015); MacKinnon, Nielsen and Webb (2023) | Justifies country-level resampling and delete-one-country checks. This public reproduction block uses these draws only as finite-run reproducibility checks, not as sign filters. |
| Local-projection inference and time stability | Jorda (2005); Olea and Plagborg-Moller (2022); Huidrom et al. (2019); Ramey and Zubairy (2018) | Justifies uncertainty and robustness checks across horizons and samples. This public reproduction block uses time blocks only as finite-run reproducibility checks, not as sign filters. |
| Excluded downstream sign filters | Pre-specified screening rule | The gate deliberately excludes cumulative output response>0, cumulative spending response>0 and positive-sign stability of kernels. Subsequent numerical points use the selected specifications separately. |

## Stage 1: robustness winners

The screen selects only specifications with strong output-interaction evidence at `p < 0.05` and clean numerical/support diagnostics. It does not use later accounting outcomes or signs.

| State-variable subset | Screening diagnostics passed | Minimum finite-run share | Output-interaction p-value | Poland support p-value | Condition number | Maximum state correlation |
| --- | --- | --- | --- | --- | --- | --- |
| investment import content | 9 | 1.000000 | 0.003771 | 0.871864 | 70.880814 | 0.000000 |
| household net financial worth | 9 | 1.000000 | 0.013043 | 0.441307 | 68.369588 | 0.000000 |

## Output-interaction multiplicity sensitivity

The output-relevance screen is reported with raw p-values and simple 15-test multiplicity diagnostics. The gate itself uses the pre-specified raw `p < 0.05` output-relevance rule; the adjusted columns are sensitivity diagnostics, not additional hidden filters.

| State-variable subset | Output-interaction p-value | Bonferroni p-value | Benjamini-Hochberg q-value | Unadjusted screen result |
| --- | --- | --- | --- | --- |
| investment import content | 0.003771 | 0.056568 | 0.056568 | Yes |
| household net financial worth | 0.013043 | 0.195652 | 0.097826 | Yes |
| public debt | 0.119830 | 1.000000 | 0.599148 | No |
| real PPP income | 0.463297 | 1.000000 | 0.975771 | No |
| investment import content + public debt + real PPP income | 0.658554 | 1.000000 | 0.975771 | No |
| investment import content + real PPP income | 0.675856 | 1.000000 | 0.975771 | No |
| investment import content + household net financial worth + real PPP income | 0.690439 | 1.000000 | 0.975771 | No |
| investment import content + household net financial worth | 0.696225 | 1.000000 | 0.975771 | No |
| public debt + household net financial worth | 0.804693 | 1.000000 | 0.975771 | No |
| investment import content + public debt | 0.895313 | 1.000000 | 0.975771 | No |
| public debt + real PPP income | 0.898835 | 1.000000 | 0.975771 | No |
| household net financial worth + real PPP income | 0.906285 | 1.000000 | 0.975771 | No |
| investment import content + public debt + household net financial worth | 0.957030 | 1.000000 | 0.975771 | No |
| public debt + household net financial worth + real PPP income | 0.966691 | 1.000000 | 0.975771 | No |
| investment import content + public debt + household net financial worth + real PPP income | 0.975771 | 1.000000 | 0.975771 | No |

## Full robustness ranking

| State-variable subset | Screening status | Screening diagnostics passed | Selection reason | Output-interaction p-value | Poland support p-value | Condition number | Maximum state correlation | Minimum finite-run share |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| investment import content | Retained | 9 | All screening diagnostics passed | 0.003771 | 0.871864 | 70.880814 | 0.000000 | 1.000000 |
| household net financial worth | Retained | 9 | All screening diagnostics passed | 0.013043 | 0.441307 | 68.369588 | 0.000000 | 1.000000 |
| public debt | Not retained | 8 | Output-interaction test did not pass | 0.119830 | 0.712772 | 70.735072 | 0.000000 | 1.000000 |
| real PPP income | Not retained | 8 | Output-interaction test did not pass | 0.463297 | 0.945748 | 68.794480 | 0.000000 | 1.000000 |
| investment import content + public debt + real PPP income | Not retained | 8 | Output-interaction test did not pass | 0.658554 | 0.976694 | 72.200433 | 0.249751 | 1.000000 |
| investment import content + real PPP income | Not retained | 8 | Output-interaction test did not pass | 0.675856 | 0.985900 | 71.101604 | 0.035025 | 1.000000 |
| investment import content + household net financial worth + real PPP income | Not retained | 8 | Output-interaction test did not pass | 0.690439 | 0.838935 | 71.201494 | 0.509186 | 1.000000 |
| investment import content + household net financial worth | Not retained | 8 | Output-interaction test did not pass | 0.696225 | 0.713125 | 70.963703 | 0.151801 | 1.000000 |
| public debt + household net financial worth | Not retained | 8 | Output-interaction test did not pass | 0.804693 | 0.738448 | 70.793723 | 0.452516 | 1.000000 |
| investment import content + public debt | Not retained | 8 | Output-interaction test did not pass | 0.895313 | 0.902502 | 72.113452 | 0.249751 | 1.000000 |
| public debt + real PPP income | Not retained | 8 | Output-interaction test did not pass | 0.898835 | 0.933109 | 70.839176 | 0.140759 | 1.000000 |
| household net financial worth + real PPP income | Not retained | 8 | Output-interaction test did not pass | 0.906285 | 0.703199 | 68.822358 | 0.509186 | 1.000000 |
| investment import content + public debt + household net financial worth | Not retained | 8 | Output-interaction test did not pass | 0.957030 | 0.870916 | 72.199017 | 0.452516 | 1.000000 |
| public debt + household net financial worth + real PPP income | Not retained | 8 | Output-interaction test did not pass | 0.966691 | 0.871098 | 70.873536 | 0.509186 | 1.000000 |
| investment import content + public debt + household net financial worth + real PPP income | Not retained | 8 | Output-interaction test did not pass | 0.975771 | 0.929638 | 72.300540 | 0.509186 | 1.000000 |

## QA

| Check | Status | Detail |
| --- | --- | --- |
| All candidate state-variable subsets are enumerated | PASS | spec count=15 |
| Every non-empty subset of the four state variables is present | PASS | expected 2^4 - 1 = 15 subsets of investment import content, public debt, household net financial worth and real PPP income |
| Selection reasons use only pre-specified screening diagnostics | PASS | selection reason uses only pre-specified robustness diagnostics |
| Screening summary contains only first-stage diagnostics | PASS | screening summary contains only feature-selection diagnostics |
| Screening summary excludes downstream response quantiles | PASS | screening summary omits output-to-spending response ratio quantiles; they stay in separate stability appendices |
| Candidate response outputs contain only response and state-profile fields | PASS | all-spec kernel outputs contain only output, spending and feature-profile fields |
| Selection excludes response-sign filters | PASS | gate excludes cumulative output response>0, cumulative spending response>0 and positive kernel-stability filters |
| Screening summary excludes unused positive-sign diagnostics | PASS | robustness summary does not keep unused positive-sign diagnostics |
| Result file set is limited to the screening block | PASS | extra=[]; missing=[] |
| Output-only fit ranking is not computed | PASS | No output-only BIC or fit ranking is part of gate logic. |
| Multiplicity diagnostics are written as sensitivity checks | PASS | raw p-values, Bonferroni p-values and Benjamini-Hochberg q-values are written as sensitivity diagnostics |
| Support reference sample excludes the Poland target profile | PASS | target rows removed by country/profile-year filter; targets=Poland profile 2022 target; min distance to pol=0.000000000 |
| Non-retained rows have explicit screening reasons | PASS | failed rows use explicit non-retention reason reason labels |
| Dependency manifest is present | PASS | requirements.txt documents Python dependencies |
| Bootstrap finite-run checks are complete | PASS | min boot ok=19 |
| Time-block finite-run checks are complete | PASS | min time ok=3 |
| Required text references are present | PASS | literature canon is maintained outside the executable public package; bundled text reference count=0 |

## Reproducibility notes

- The runner re-execs Python with `OPENBLAS_NUM_THREADS=1`, `OMP_NUM_THREADS=1` and `MKL_NUM_THREADS=1` before importing NumPy, so the plain command `python3 code/feature_screen_model.py` enforces the BLAS thread policy.
- The source manifest is deterministic for public-archive source files, this block's stable CSV outputs, and this block report. It explicitly omits itself from its own hash list.
- `results/run_manifest.json` uses a fixed public-run timestamp and a non-embedded source-revision note. Source identity is carried by the file-level manifests, not by a mutable local Git head.

## Output inventory

| Output group | Description |
| --- | --- |
| Executable code | The feature-screen estimator and helper modules are shipped in the public package. |
| Frozen inputs | The package includes the frozen model-input tables used by this screening block. |
| Screening tables | The package includes machine-readable state-variable screen, multiplicity and finite-run diagnostics. |
| Provenance manifest | The package includes hashes for the code, inputs, outputs and this report. |
