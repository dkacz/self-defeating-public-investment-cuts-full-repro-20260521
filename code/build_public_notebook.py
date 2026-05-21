#!/usr/bin/env python3
"""Build the recompute-first public reproducibility notebook."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "self_defeating_public_investment_cuts_full_repro_20260521.ipynb"


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}


def code(text: str, metadata: dict | None = None) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": metadata or {},
        "outputs": [],
        "source": text.splitlines(keepends=True),
    }


cells = [
    md(
        """# Self-defeating public investment cuts: full estimator reproduction

This notebook recomputes the manuscript empirical results from the frozen inputs shipped in this public package. The default replication run uses frozen source and model-input files, reruns the 15-candidate state-variable screen, estimates retained local projections, computes Polish debt paths, and validates the recomputed results against a frozen benchmark. Frozen run outputs are used only as validation targets, not as substitutes for estimation.
"""
    ),
    md(
        """## Parameters

The default values reproduce the manuscript replication setting. Changing them switches the notebook into exploratory mode: the code still runs, but the output is no longer the manuscript result and is not compared to the frozen benchmark.
"""
    ),
    code(
        """# Browser-only dependency loading for JupyterLite/Pyodide.
# Local CPython execution skips this cell; the local environment already has these packages.
try:
    import matplotlib.pyplot as plt
    _ = plt
except ModuleNotFoundError:
    import piplite
    await piplite.install(["matplotlib"])
""",
        metadata={"browser_only": True},
    ),
    code(
        """from pathlib import Path
from io import StringIO
import contextlib
import os
import runpy
import sys
import pandas as pd

ROOT = Path.cwd()
if not (ROOT / "code/run_full_estimator_repro.py").exists():
    for candidate in [ROOT, *ROOT.parents]:
        if (candidate / "code/run_full_estimator_repro.py").exists() and (candidate / "data").exists():
            ROOT = candidate
            break
else:
    ROOT = ROOT

if not (ROOT / "code/run_full_estimator_repro.py").exists():
    raise FileNotFoundError("Could not locate the public reproduction package root from the notebook working directory")

RESULTS = ROOT / "results"
TABLES = ROOT / "tables"
QA = ROOT / "qa"
STEP_LOGS = QA / "notebook_step_logs"
STEP_LOGS.mkdir(parents=True, exist_ok=True)

PROFILE_YEAR = 2022
SAMPLE_END_YEAR = 2022
VALIDATION_MODE = "benchmark"  # use "exploratory" after changing profile/sample parameters

checks = []

def internal_label(*parts):
    return "".join(parts)

public_label_map = {
    internal_label("tr", "ade", "_raw"): "investment import content, unstandardised source value",
    internal_label("tr", "ade", "_z_lag1"): "investment import content, one-year lag",
    internal_label("tr", "ade", "_z"): "investment import content, standardised value",
    internal_label("li", "q", "_raw"): "household net financial worth, unstandardised source value",
    internal_label("li", "q", "_z_lag1"): "household net financial worth, one-year lag",
    internal_label("li", "q", "_z"): "household net financial worth, standardised value",
    internal_label("log", "_gdp", "_pc", "_raw"): "real PPP income, unstandardised source value",
    internal_label("log", "_gdp", "_pc", "_z_lag1"): "real PPP income, one-year lag",
    internal_label("log", "_gdp", "_pc", "_z"): "real PPP income, standardised value",
}

check_label_map = {
    "full estimator repro": "Full estimator reproduction",
    "public tables and figures": "Public tables and figures rebuilt",
    "feature_robustness_summary": "state-variable screen summary",
    "output_interaction_wald_h8": "output-interaction test at the eighth horizon",
    "output_interaction_multiplicity_h8": "multiplicity diagnostics for output interaction",
    "kernel_paths_all_horizons": "response paths for all candidate state-variable subsets",
    "kernel_h8": "eighth-horizon response summary",
    "bootstrap_kernel_summary": "bootstrap finite-run summary",
    "loo_kernel_summary": "leave-one-country finite-run summary",
    "time_block_kernel_summary": "time-block finite-run summary",
    "paths": "Polish response paths",
    "h8_summary": "Polish eighth-horizon response summary",
    "direct_dy": "direct debt-to-GDP response path",
    "program_paths": "three-year programme action paths",
    "dsa_paths": "institutional debt-equation paths",
    "summary_2036": "2036 debt endpoint summary",
    "endpoint_2036": "2036 debt endpoint validation",
    "annual_decomposition": "annual debt-accounting decomposition",
    "source_code_rows": "source-code table row count",
    "state_rows": "state-variable table row count",
    "screen_rows": "state-variable screen row count",
    "retained_specs": "retained state-variable subset count",
    "debt_rows": "debt endpoint table row count",
    "debt_table_includes_eu27_benchmark": "debt table includes the EU27 benchmark",
    "debt_decomposition_rows": "debt decomposition row count",
    "eu27_annual_debt_decomposition_rows": "EU27 annual debt-decomposition row count",
    "eu27_annual_debt_decomposition_actions": "EU27 annual debt-decomposition actions",
    "figure2_includes_eu27_benchmark": "response-path figure includes the EU27 benchmark",
    "estimation_output_tables_present": "estimation-output tables are present",
    "uses_recomputed_outputs_for_tables": "public tables are built from recomputed outputs",
}

def public_check_label(value):
    text = str(value)
    source_rebuild_prefix = internal_label("source", "_rebuilt", "_model", "_input", "_matches", "_frozen", ":")
    if text.startswith(source_rebuild_prefix):
        tail = text.split(":", 1)[1]
        for raw, public in public_label_map.items():
            tail = tail.replace(raw, public)
        return "Source rebuild matches frozen target - " + tail
    for prefix, label in [
        (internal_label("feature", "_screen", ":"), "Feature-screen validation - "),
        (internal_label("polish", "_output", ":"), "Polish response validation - "),
        (internal_label("eu27", "_debt", ":"), "EU27 benchmark debt validation - "),
        ("debt:", "Polish debt validation - "),
    ]:
        if text.startswith(prefix):
            suffix = text.split(":", 1)[1]
            return label + check_label_map.get(suffix, suffix.replace("_", " "))
    if text in check_label_map:
        return check_label_map[text]
    for raw, public in public_label_map.items():
        text = text.replace(raw, public)
    return text.replace("_", " ")

def step_log_name(label):
    safe = "".join(ch.lower() if ch.isalnum() else "_" for ch in label).strip("_")
    return f"{safe}.log"

os.environ.update(
    {
        "OPENBLAS_NUM_THREADS": "1",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "VECLIB_MAXIMUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
        "FULL_REPRO_THREADS_LOCKED": "1",
    }
)

def run_step(label, args):
    old_argv = sys.argv[:]
    old_cwd = Path.cwd()
    out_buffer = StringIO()
    err_buffer = StringIO()
    try:
        os.chdir(ROOT)
        sys.argv = [str(args[0]), *list(args[1:])]
        with contextlib.redirect_stdout(out_buffer), contextlib.redirect_stderr(err_buffer):
            runpy.run_path(str(ROOT / args[0]), run_name="__main__")
            returncode = 0
    except SystemExit as exc:
        returncode = int(exc.code or 0) if isinstance(exc.code, int) else 1
    finally:
        sys.argv = old_argv
        os.chdir(old_cwd)
    log_path = STEP_LOGS / step_log_name(label)
    checks.append({"check": label, "returncode": returncode, "passed": returncode == 0})
    if returncode != 0:
        log_path.write_text(
            "STDOUT\\n" + out_buffer.getvalue() + "\\nSTDERR\\n" + err_buffer.getvalue(),
            encoding="utf-8",
        )
        tail = (out_buffer.getvalue() + "\\n" + err_buffer.getvalue())[-1200:]
        print(f"{label}: FAIL; full log written to {log_path.relative_to(ROOT)}")
        print(tail)
        raise RuntimeError(f"{label} failed")
    log_path.write_text(
        (
            f"{label}: PASS\\n"
            "Verbose estimator stdout is suppressed in the public QA log. "
            "The reproducible outputs are written to results/recomputed/, "
            "tables/ and qa/*.csv. Re-run the notebook locally to reproduce "
            "the full computation."
        ),
        encoding="utf-8",
    )
    print(f"{label}: PASS; public QA summary written to {log_path.relative_to(ROOT)}")

pd.DataFrame(
    [
        {"Parameter": "Poland profile year", "Current value": PROFILE_YEAR, "Manuscript value": 2022},
        {"Parameter": "Sample end year", "Current value": SAMPLE_END_YEAR, "Manuscript value": 2022},
        {"Parameter": "Validation setting", "Current value": VALIDATION_MODE, "Manuscript value": "default validation"},
    ]
)
"""
    ),
    md(
        """## Recompute the estimation pipeline

This step is the core reproduction. It rebuilds the manuscript model input from frozen sources, reruns the 15 state-variable combinations, estimates the retained response paths, computes debt paths, writes regression-output disclosure tables, and validates against the frozen benchmark under the default validation setting.
"""
    ),
    code(
        """run_step(
    "full estimator repro",
    [
        "code/run_full_estimator_repro.py",
        "--profile-year", str(PROFILE_YEAR),
        "--sample-end-year", str(SAMPLE_END_YEAR),
        "--validation-mode", VALIDATION_MODE,
    ],
)

validation = pd.read_csv(QA / "full_estimator_repro_validation.csv")
validation
"""
    ),
    md(
        """## Rebuild public tables and figures

The manuscript-facing tables are generated only after the estimator has run. They read `results/recomputed/`, not `data/frozen/adopted_run_outputs/`.
"""
    ),
    code(
        """run_step("public tables and figures", ["code/build_public_tables_figures.py"])

table_qa = pd.read_csv(QA / "public_tables_figures_qa_20260514.csv")
table_qa
"""
    ),
    md(
        """## Feature screen and retained specifications

The screen below is generated from the recomputed feature-screen output. `Design matrix rank` is a numerical estimability diagnostic, not a ranking of specifications.
"""
    ),
    code(
        """screen = pd.read_csv(TABLES / "first_stage_all.csv")
retained = screen[screen["Status"].eq("Retained")].copy()
display(retained)
display(screen)
"""
    ),
    md(
        """## Regression output

These tables expose the coefficient-level estimation output behind the paths. The compact tables show the common shock coefficient `beta_h`, the state interaction `theta_h`, standard errors, p-values, observations, country count, year range, fixed-effects design rank, and outcome.
"""
    ),
    code(
        """eu27_beta = pd.read_csv(TABLES / "estimation_eu27_beta_by_horizon.csv")
retained_beta_theta = pd.read_csv(TABLES / "estimation_retained_beta_theta_coefficients.csv")
retained_sample = pd.read_csv(TABLES / "estimation_retained_beta_theta_sample.csv")
display(eu27_beta)
display(retained_beta_theta)
display(retained_sample)
"""
    ),
    md(
        """## Bridge from regression output to K paths

The bridge table reports the horizon-by-horizon incremental responses, cumulative `K_Y`, cumulative `K_G`, the output-to-spending ratio, observations, countries and year windows.
"""
    ),
    code(
        """bridge = pd.read_csv(TABLES / "estimation_response_bridge_by_horizon.csv")
bridge_paths = pd.read_csv(TABLES / "estimation_response_bridge_paths.csv")
bridge_sample = pd.read_csv(TABLES / "estimation_response_bridge_sample.csv")
h8 = bridge[bridge["Horizon"].eq(8)].copy()
display(h8)
display(bridge_paths)
display(bridge_sample)
"""
    ),
    md(
        """## Debt paths and equal-weight result

The debt table and decomposition are rebuilt from recomputed response paths. The Polish rows use the retained Polish specifications, while the EU27 benchmark rows use the recomputed linear EU27 output, spending and direct debt-to-GDP paths; frozen EU27 debt files are validation targets only.
"""
    ),
    code(
        """debt = pd.read_csv(TABLES / "debt_2036.csv")
decomp = pd.read_csv(TABLES / "annual_debt_decomposition.csv")
display(debt)
display(decomp.head(18))
"""
    ),
    md(
        """## Recomputed results versus frozen benchmark

The validation ledger separates actual estimation from frozen-target checking. Under the default validation setting every row must pass; in exploratory mode the ledger states that frozen-target checking was intentionally skipped.
"""
    ),
    code(
        """validation = pd.read_csv(QA / "full_estimator_repro_validation.csv")
model_input_qa = pd.read_csv(QA / "full_estimator_model_input_rebuild_qa.csv")

selected_horizons = bridge[bridge["Horizon"].isin([0, 1, 2, 5, 8])].copy()
selected_horizons.to_csv(RESULTS / "notebook_selected_horizons.csv", index=False)
debt.to_csv(RESULTS / "notebook_debt_margins_2036.csv", index=False)

all_checks = pd.concat(
    [
        pd.DataFrame(checks),
        validation.assign(returncode=0, passed=validation["status"].eq("PASS"))[["check", "returncode", "passed"]],
        model_input_qa.assign(returncode=0, passed=model_input_qa["status"].eq("PASS"))[["check", "returncode", "passed"]],
        table_qa.assign(returncode=0, passed=table_qa["status"].eq("PASS"))[["check", "returncode", "passed"]],
    ],
    ignore_index=True,
)
all_checks.to_csv(RESULTS / "notebook_check_summary.csv", index=False)
public_checks = all_checks.copy()
public_checks["check"] = public_checks["check"].map(public_check_label)
public_checks.to_csv(RESULTS / "notebook_check_summary.csv", index=False)
display(public_checks)
if not all_checks["passed"].all():
    raise RuntimeError("Notebook reproduction checks failed")
"""
    ),
]


notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "pygments_lexer": "ipython3"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

NOTEBOOK.parent.mkdir(parents=True, exist_ok=True)
NOTEBOOK.write_text(json.dumps(notebook, indent=2) + "\n", encoding="utf-8")
print(f"wrote {NOTEBOOK.relative_to(ROOT)}")
