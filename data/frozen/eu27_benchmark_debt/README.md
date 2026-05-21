# EU27 Benchmark Debt Validation Targets

This folder freezes the EU27 benchmark debt values used to validate
the public recomputation.

Source basis:

EU27 panel benchmark debt-accounting output copied here into packet-relative
frozen CSV files.

Frozen files:

- `eu27_benchmark_debt_2036.csv`: 2036 institutional and direct debt-to-GDP
  endpoint margins.
- `eu27_benchmark_annual_debt_decomposition.csv`: annual institutional
  debt-accounting rows for Appendix C.2 and C.3.

The target values evaluate three annual public-investment actions of 1
percentage point of GDP in 2028, 2029 and 2030. They are included here only as
validation targets. `code/run_full_estimator_repro.py` recomputes the EU27
output, spending, direct debt-to-GDP and institutional debt paths under
`results/recomputed/eu27_benchmark/`; manuscript-facing tables read those
recomputed files.
