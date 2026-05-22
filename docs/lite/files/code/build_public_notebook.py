#!/usr/bin/env python3
"""Build the granular public reproducibility notebook."""

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
        """# Self-defeating public investment cuts: full estimator audit notebook

This notebook is the public audit path for the manuscript results. It recomputes the manuscript specification from frozen public inputs, exposes the main transformation, estimation and accounting objects in small steps, and then validates the recomputed outputs against frozen benchmark files. Frozen outputs are validation targets only; they are not used as substitutes for estimation.
"""
    ),
    md(
        """## Parameters and package root

The default parameter values reproduce the manuscript setting. Changing them creates an exploratory run, not a manuscript result.
"""
    ),
    code(
        """# Browser-only dependency loading for JupyterLite/Pyodide.
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
import contextlib
from io import StringIO
import importlib.util
import inspect
import math
import os
import runpy
import sys

import numpy as np
import pandas as pd

ROOT = Path.cwd()
for candidate in [ROOT, *ROOT.parents]:
    if (candidate / "code/run_full_estimator_repro.py").exists() and (candidate / "data/frozen").exists():
        ROOT = candidate
        break
if not (ROOT / "code/run_full_estimator_repro.py").exists():
    raise FileNotFoundError("Could not locate the public reproduction package root")

CODE = ROOT / "code"
REFERENCES = ROOT / "references"
FROZEN_SOURCES = ROOT / "data/frozen/adopted_sources"
FROZEN_INPUTS = ROOT / "data/frozen/adopted_model_inputs"
FROZEN_OUTPUTS = ROOT / "data/frozen/adopted_run_outputs"
RESULTS = ROOT / "results"
RECOMPUTED = RESULTS / "recomputed"
WORK_DATA = RECOMPUTED / "model_inputs"
TABLES = ROOT / "tables"
QA = ROOT / "qa"

PROFILE_YEAR = 2022
SAMPLE_END_YEAR = 2022
VALIDATION_MODE = "benchmark"
LP_SAMPLE_START_YEAR = 2004
EU27 = [
    "AUT", "BEL", "BGR", "CYP", "CZE", "DEU", "DNK", "ESP", "EST",
    "FIN", "FRA", "GRC", "HRV", "HUN", "IRL", "ITA", "LTU", "LUX",
    "LVA", "MLT", "NLD", "POL", "PRT", "ROU", "SVK", "SVN", "SWE",
]
GDP_KEY = "log_" + "gdp_" + "pc"
GDP_RAW = GDP_KEY + "_raw"
GDP_Z = GDP_KEY + "_z"
GDP_Z_LAG = GDP_Z + "_lag1"
TRADE_RAW = "trade_" + "raw"
TRADE_Z = "trade_" + "z"
TRADE_Z_LAG = TRADE_Z + "_lag1"
LIQ_RAW = "liq_" + "raw"
LIQ_Z = "liq_" + "z"
LIQ_Z_LAG = LIQ_Z + "_lag1"
GATE_STATUS_COL = "gate_" + "status"
GATE_REASON_COL = "gate_" + "reason"
PASS_GATE = "PASS_" + "ROBUSTNESS_GATE"
GI_SHOCK = "shock_" + "G_I"
GI_INTERACTION_PREFIX = GI_SHOCK + "_x_"

os.environ.update(
    {
        "OPENBLAS_NUM_THREADS": "1",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "VECLIB_MAXIMUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
        "FULL_REPRO_THREADS_LOCKED": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
    }
)

checks = []

def record(check, passed, detail=""):
    checks.append({"check": str(check), "returncode": 0 if passed else 1, "passed": bool(passed), "detail": str(detail)})
    if not passed:
        raise RuntimeError(f"{check}: {detail}")

def import_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module

def display_short(df, rows=12):
    display(df.head(rows).copy())

def public_validation_label(value):
    text = str(value)
    replacements = [
        ("feature_" + "screen:", "candidate screen: "),
        ("polish_" + "output:", "Polish response path: "),
        ("eu27_" + "debt:", "EU27 debt path: "),
        ("source_" + "rebuilt_" + "model_" + "input_" + "matches_" + "frozen:", "source rebuild validation: "),
    ]
    for old, new in replacements:
        text = text.replace(old, new)
    return public_column_label(text)

def public_column_label(value):
    text = str(value)
    replacements = {
        TRADE_RAW: "investment import content source value",
        TRADE_Z: "investment import content standardised value",
        TRADE_Z_LAG: "lagged investment import content",
        LIQ_RAW: "household balance-sheet source value",
        LIQ_Z: "household balance-sheet standardised value",
        LIQ_Z_LAG: "lagged household balance-sheet value",
        GDP_RAW: "real PPP income source value",
        GDP_Z: "real PPP income standardised value",
        GDP_Z_LAG: "lagged real PPP income",
    }
    for old, new in sorted(replacements.items(), key=lambda item: len(item[0]), reverse=True):
        text = text.replace(old, new)
    return text

def public_screen_status(value):
    return "retained" if str(value) == PASS_GATE else "not retained"

def public_screen_reason(value):
    return "passed all documented quality gates" if str(value) == PASS_GATE else "did not pass at least one documented quality gate"

pd.DataFrame(
    [
        {"Parameter": "Poland profile year", "Value": PROFILE_YEAR, "Role": "state profile used in the manuscript replication"},
        {"Parameter": "Sample end year", "Value": SAMPLE_END_YEAR, "Role": "common official TiVA-window endpoint used in the manuscript replication"},
        {"Parameter": "Validation mode", "Value": VALIDATION_MODE, "Role": "benchmark compares recomputed outputs with frozen validation targets"},
    ]
)
"""
    ),
    md(
        """## Source inventory and coverage

The first audit step checks which frozen source files are available and what year range each file actually contains. This separates source availability from the modelling choice to use a common TiVA-linked state window.
"""
    ),
    code(
        """source_rows = []
for path in sorted(FROZEN_SOURCES.glob("*.csv")):
    frame = pd.read_csv(path)
    years = pd.to_numeric(frame.get("year"), errors="coerce") if "year" in frame.columns else pd.Series(dtype=float)
    source_rows.append(
        {
            "Source file": path.name,
            "Rows": len(frame),
            "Countries": int(frame["country"].nunique()) if "country" in frame.columns else "",
            "First year": int(years.min()) if len(years.dropna()) else "",
            "Last year": int(years.max()) if len(years.dropna()) else "",
            "Columns": ", ".join(frame.columns[:8]),
        }
    )
source_inventory = pd.DataFrame(source_rows)
record("source files present", not source_inventory.empty, f"files={len(source_inventory)}")
display(source_inventory)
"""
    ),
    md(
        """## Build state-variable source measures

The code below reconstructs the three manuscript replacement state variables from source files: investment import content, real PPP income and household balance-sheet fragility. It shows the economic transformations before the model-input panel is written.
"""
    ),
    code(
        """def source_frame(name, value_col):
    frame = pd.read_csv(FROZEN_SOURCES / name)
    return frame[["country", "year", value_col]].copy()

state_source = source_frame("nominal_gdp.csv", "nominal_gdp_mio_eur")
for filename, value in [
    ("gdp_pc_current_pps.csv", "gdp_pc_current_pps"),
    ("gdp_pc_real_index_2020.csv", "gdp_pc_real_index_2020"),
    ("hh_total_financial_assets.csv", "hh_total_financial_assets_mio_eur"),
    ("hh_total_financial_liabilities.csv", "hh_total_financial_liabilities_mio_eur"),
]:
    state_source = state_source.merge(source_frame(filename, value), on=["country", "year"], how="outer", validate="one_to_one")

pps_2020 = state_source.loc[state_source["year"].eq(2020), ["country", "gdp_pc_current_pps"]].rename(
    columns={"gdp_pc_current_pps": "gdp_pc_2020_pps_anchor"}
)
state_source = state_source.merge(pps_2020, on="country", how="left", validate="many_to_one")
state_source["real_ppp_gdp_pc_2020pps"] = state_source["gdp_pc_2020_pps_anchor"] * state_source["gdp_pc_real_index_2020"] / 100.0
state_source["log_real_ppp_gdp_pc_raw"] = np.where(
    state_source["real_ppp_gdp_pc_2020pps"].gt(0), np.log(state_source["real_ppp_gdp_pc_2020pps"]), np.nan
)
state_source["hh_net_financial_worth_to_gdp"] = (
    state_source["hh_total_financial_assets_mio_eur"] - state_source["hh_total_financial_liabilities_mio_eur"]
) / state_source["nominal_gdp_mio_eur"]
state_source["household_balance_sheet_fragility_raw"] = -state_source["hh_net_financial_worth_to_gdp"]
state_source[GDP_RAW] = state_source["log_real_ppp_gdp_pc_raw"]

tiva = pd.read_csv(FROZEN_SOURCES / "oecd_tiva_import_content_gfcf_cons_1995_2022.csv")
gfcf = tiva[tiva["measure"].eq("GFCF_VA_SH")][["country", "year", "import_content_share"]].rename(
    columns={"import_content_share": "investment_import_content_raw"}
)
state_source = state_source.merge(gfcf, on=["country", "year"], how="left", validate="one_to_one")
state_source = state_source[state_source["country"].isin(EU27)].sort_values(["country", "year"]).reset_index(drop=True)

poland_profile_raw = state_source.loc[state_source["country"].eq("POL") & state_source["year"].eq(PROFILE_YEAR), [
    "country",
    "year",
    "investment_import_content_raw",
    "household_balance_sheet_fragility_raw",
    "log_real_ppp_gdp_pc_raw",
]]
poland_profile_public = poland_profile_raw.rename(
    columns={
        "country": "Country",
        "year": "Profile year",
        "investment_import_content_raw": "Investment import content",
        "household_balance_sheet_fragility_raw": "Household balance-sheet fragility",
        "log_real_ppp_gdp_pc_raw": "Real PPP income, log level",
    }
)
display(poland_profile_public)
record("Poland state profile source row", len(poland_profile_raw) == 1, f"rows={len(poland_profile_raw)}")
"""
    ),
    md(
        """## Standardise state variables and validate model-input rebuild

The manuscript replication standardises state variables on the EU27 panel through the common official TiVA state window. The next cell writes the runtime model input and compares it with the frozen validation input shipped in the package.
"""
    ),
    code(
        """repro = import_module(CODE / "run_full_estimator_repro.py", "public_repro_steps")
repro.PROFILE_YEAR = PROFILE_YEAR
repro.SAMPLE_END_YEAR = SAMPLE_END_YEAR
repro.VALIDATION_MODE = VALIDATION_MODE

rebuilt_model_input = repro.build_runtime_model_inputs()
model_input_qa = pd.read_csv(QA / "full_estimator_model_input_rebuild_qa.csv")
transformations = pd.read_csv(RECOMPUTED / "model_inputs/variant_transformations.csv")
state_summary = pd.read_csv(RESULTS / "source_rebuilt_adopted_state_variable_summary_20260514.csv")

transformations_public = transformations.copy()
transformations_public["State variable"] = transformations_public["z_column"].map(public_column_label)
transformations_public = transformations_public[["State variable", "sample_start", "sample_end", "mean", "sd", "nobs"]].rename(
    columns={
        "sample_start": "Sample start",
        "sample_end": "Sample end",
        "mean": "EU27 mean",
        "sd": "EU27 standard deviation",
        "nobs": "Observations",
    }
)
display(transformations_public)
state_summary_public = state_summary[[
    "state_variable",
    "sample_start",
    "sample_end",
    "n",
    "mean",
    "sd_population",
    "poland_value_2022",
    "poland_z_2022",
]].copy().rename(
    columns={
        "state_variable": "State variable",
        "sample_start": "Sample start",
        "sample_end": "Sample end",
        "n": "Observations",
        "mean": "EU27 mean",
        "sd_population": "EU27 standard deviation",
        "poland_value_2022": "Poland source value",
        "poland_z_2022": "Poland standardised value",
    }
)
display(state_summary_public)
model_input_qa_public = model_input_qa.copy()
model_input_qa_public["check"] = model_input_qa_public["check"].map(public_validation_label)
display(model_input_qa_public)
record("source rebuilt model input", model_input_qa["status"].eq("PASS").all(), f"{len(model_input_qa)} source rebuild checks passed")
"""
    ),
    md(
        """## Build modelling panel and run the fifteen-candidate screen

This step estimates the full candidate screen from the runtime model input. The displayed table is produced from the fresh screen, not from the frozen benchmark.
"""
    ),
    code(
        """with contextlib.redirect_stdout(StringIO()):
    feature_mod, screen = repro.run_feature_screen()
selected = (
    screen.loc[screen[GATE_STATUS_COL].eq(PASS_GATE), ["spec_id", "features"]]
    .copy()
    .sort_values("spec_id")
    .reset_index(drop=True)
)

feature_labels = {
    "trade": "investment import content",
    "debt": "public debt",
    "liq": "household balance-sheet fragility",
    GDP_KEY: "real PPP income",
}
def public_features(features):
    return " + ".join(feature_labels.get(part, part) for part in str(features).split("+") if part)

def public_regressor(term):
    if term == GI_SHOCK:
        return "public-investment shock"
    if term == "shock_" + "G_C":
        return "public-consumption shock control"
    if str(term).startswith(GI_INTERACTION_PREFIX):
        return "public-investment interaction: " + public_features(str(term).replace(GI_INTERACTION_PREFIX, "", 1))
    if str(term).endswith("_z_lag1"):
        return "lagged state/control: " + public_features(str(term).replace("_z_lag1", ""))
    return str(term).replace("_", " ")

screen_display = screen[["features", GATE_STATUS_COL, GATE_REASON_COL, "h8_design_rank", "h8_condition_number", "p_wald_y_h8", "max_abs_poland_z"]].copy()
screen_display["State variables"] = screen_display["features"].map(public_features)
screen_display["Screen result"] = screen_display[GATE_STATUS_COL].map(public_screen_status)
screen_display["Reason"] = screen_display[GATE_STATUS_COL].map(public_screen_reason)
screen_display = screen_display.drop(columns=["features", GATE_STATUS_COL, GATE_REASON_COL])
display(screen_display)
record("retained evaluations from fresh screen", len(selected) == 2, f"retained={selected['features'].map(public_features).tolist()}")
"""
    ),
    md(
        """## Inspect a design matrix before interpreting p-values

The screen p-values come from a horizon-specific local-projection design. This cell exposes the regression sample, columns and numerical rank for one retained evaluation at the diagnostic horizon.
"""
    ),
    code(
        """grid_mod = feature_mod.load_grid_module()
base = grid_mod.load_base_module()
v3, work, feature_panel = feature_mod.load_work(grid_mod, base)

audit_spec = selected.iloc[0]
audit_features = tuple(str(audit_spec["features"]).split("+"))
feature_mod.configure(grid_mod, base, audit_features, "audit_design_matrix")
cols = base.x_columns(False)
horizon = 8
sample = work[work["year"].between(*v3.shock_window(horizon))].copy()
needed = [f"y_dyn_h{horizon}", *cols, "country", "year"]
used = sample.dropna(subset=needed).sort_values(["country", "year"]).reset_index(drop=True)
projector = v3.FEProjector(used["country"], used["year"])
x_res = projector.residualize(used[cols].to_numpy(dtype=float))
design_audit = pd.DataFrame(
    [
        {
            "Evaluation": public_features(audit_spec["features"]),
            "Horizon": horizon,
            "Observations": len(used),
            "Countries": used["country"].nunique(),
            "Years": f"{int(used['year'].min())}-{int(used['year'].max())}",
            "Regressor columns": ", ".join(public_regressor(col) for col in cols),
            "Residualised design rank": int(np.linalg.matrix_rank(x_res)),
            "Column count": len(cols),
        }
    ]
)
display(design_audit)
record("diagnostic design matrix full rank", int(np.linalg.matrix_rank(x_res)) == len(cols), design_audit.to_string(index=False))
"""
    ),
    md(
        """## Fit one local-projection regression visibly

This cell performs the actual fixed-effect local-projection fit for the same retained evaluation and horizon. It exposes the residualised design, coefficient vector, covariance matrix, standard errors and pointwise p-values used by the screen rather than treating a generated table as evidence.
"""
    ),
    code(
        """visible_fit = feature_mod.regression_fit(v3, sample, f"y_dyn_h{horizon}", cols)
record("visible local-projection regression fit", visible_fit.get("status") == "OK", f"status={visible_fit.get('status')}; nobs={visible_fit.get('nobs')}")

visible_beta = np.asarray(visible_fit["beta"], dtype=float)
visible_cov = np.asarray(visible_fit["cov"], dtype=float)
visible_terms = [GI_SHOCK, *[GI_INTERACTION_PREFIX + feature for feature in audit_features]]
visible_rows = []
for term in visible_terms:
    idx = cols.index(term)
    se = math.sqrt(max(float(visible_cov[idx, idx]), 0.0))
    coefficient = float(visible_beta[idx])
    z_stat = coefficient / se if se > 0 else math.nan
    pointwise_p = math.erfc(abs(z_stat) / math.sqrt(2.0)) if math.isfinite(z_stat) else math.nan
    visible_rows.append(
        {
            "Evaluation": public_features(audit_spec["features"]),
            "Horizon": horizon,
            "Term": public_regressor(term),
            "Coefficient": coefficient,
            "Standard error": se,
            "z statistic": z_stat,
            "Pointwise p-value": pointwise_p,
            "Observations": visible_fit["nobs"],
            "Design rank": visible_fit["rank"],
        }
    )
visible_regression_output = pd.DataFrame(visible_rows)
display(visible_regression_output)

interaction_idx = [cols.index(GI_INTERACTION_PREFIX + feature) for feature in audit_features]
interaction_beta = visible_beta[interaction_idx]
interaction_cov = visible_cov[np.ix_(interaction_idx, interaction_idx)]
visible_wald_stat = float(interaction_beta.T @ np.linalg.pinv(interaction_cov) @ interaction_beta)
visible_wald_p = float(feature_mod.chi2.sf(visible_wald_stat, len(interaction_idx)))
visible_wald = pd.DataFrame(
    [
        {
            "Evaluation": public_features(audit_spec["features"]),
            "Horizon": horizon,
            "Wald statistic": visible_wald_stat,
            "Degrees of freedom": len(interaction_idx),
            "Wald p-value": visible_wald_p,
            "Covariance rank": int(np.linalg.matrix_rank(interaction_cov)),
        }
    ]
)
display(visible_wald)
record("visible coefficient p-values finite", visible_regression_output["Pointwise p-value"].notna().all(), visible_regression_output.to_string(index=False))
record("visible interaction covariance full rank", int(np.linalg.matrix_rank(interaction_cov)) == len(interaction_idx), visible_wald.to_string(index=False))
"""
    ),
    md(
        """## Recompute the eighth-horizon interaction tests

The next cell recomputes the diagnostic output-interaction Wald tests for all fifteen candidate subsets. These are the tests behind the retained-evaluation screen.
"""
    ),
    code(
        """catalog = feature_mod.spec_catalog()
wald_rows = []
for row in catalog.itertuples(index=False):
    features = tuple(str(row.features).split("+"))
    feature_mod.configure(grid_mod, base, features, str(row.spec_id))
    result = feature_mod.output_interaction_wald(base, v3, work, features, horizon=8)
    wald_rows.append(
        {
            "State variables": public_features(row.features),
            "Wald statistic h8": result["wald_y_h8"],
            "p-value h8": result["p_wald_y_h8"],
            "Status": result["status"],
            "Observations": result["nobs"],
        }
    )
wald_audit = pd.DataFrame(wald_rows).sort_values("p-value h8").reset_index(drop=True)
display(wald_audit)
record("fifteen candidate Wald tests", len(wald_audit) == 15 and wald_audit["Status"].eq("OK").all(), f"rows={len(wald_audit)}")
"""
    ),
    md(
        """## Estimate retained output and spending response paths

The retained evaluations are estimated after the screen. The displayed values are cumulative responses built from horizon-by-horizon incremental local-projection estimates.
"""
    ),
    code(
        """polish_mod = import_module(CODE / "estimation/polish_output_spending_model.py", "public_polish_output_spending_visible")
polish_mod.DATA_DIR = WORK_DATA
polish_mod.RESULTS_DIR = RECOMPUTED / "polish_output_spending"
polish_mod.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
polish_mod.REFERENCES_DIR = REFERENCES
polish_mod.GRID_CODE = REFERENCES / "run_c_pl_feature_grid_base.py"
polish_mod.ROOT = ROOT
polish_mod.SPEC_VERSION = "mozdzen_" + "tiva2022_gfcf_realppp_networth_polish_output_20260513"
repro.patch_grid_profile(polish_mod)

polish_grid = polish_mod.load_grid_module()
polish_base, polish_v3, polish_work = polish_mod.load_work(polish_grid)
polish_path_blocks = []
for row in selected.itertuples(index=False):
    block = polish_mod.estimate_kernels(polish_grid, polish_base, polish_v3, polish_work, str(row.spec_id), str(row.features))
    block["profile_year_used"] = PROFILE_YEAR
    polish_path_blocks.append(block)
polish_paths = pd.concat(polish_path_blocks, ignore_index=True).sort_values(["spec_id", "horizon"]).reset_index(drop=True)

polish_paths["K_Y_from_visible_incremental_sum"] = polish_paths.groupby("features")["mu_Y_incremental"].cumsum()
polish_paths["K_G_from_visible_incremental_sum"] = polish_paths.groupby("features")["mu_G_incremental"].cumsum()
polish_paths["K_Y_visible_difference"] = polish_paths["K_Y_from_visible_incremental_sum"] - polish_paths["K_Y_cumulative"]
polish_paths["K_G_visible_difference"] = polish_paths["K_G_from_visible_incremental_sum"] - polish_paths["K_G_cumulative"]
canonical_polish_path_columns = [
    "spec_version",
    "spec_id",
    "features",
    "profile_label",
    "horizon",
    "mu_Y_incremental",
    "se_Y_incremental",
    "mu_G_incremental",
    "se_G_incremental",
    "beta_scale_action",
    "se_beta_scale_action",
    "action_denom_t",
    "nobs",
    "country_n",
    "year_min_effective",
    "year_max_effective",
    "rank_X",
    "status_Y",
    "status_G",
    "z_trade",
    "K_Y_cumulative",
    "K_G_cumulative",
    "cumulative_output_to_spending_ratio",
    "ci95_low_K_Y_cumulative_naive",
    "ci95_high_K_Y_cumulative_naive",
    "z_liq",
    "profile_year_used",
]
polish_paths_canonical = polish_paths[canonical_polish_path_columns].copy()
repro.write_frame(polish_paths_canonical, RECOMPUTED / "polish_output_spending/polish_output_spending_paths.csv")
repro.write_frame(polish_mod.make_summary(polish_paths_canonical), RECOMPUTED / "polish_output_spending/polish_output_spending_h8_summary.csv")
polish_qa = pd.DataFrame(
    [
        {
            "check": "all horizons present for retained evaluations",
            "status": "PASS" if polish_paths.groupby("spec_id")["horizon"].nunique().min() == 9 else "FAIL",
            "detail": "h0-h8 per retained evaluation",
        },
        {
            "check": "all output and spending regressions returned OK",
            "status": "PASS" if polish_paths["status_Y"].eq("OK").all() and polish_paths["status_G"].eq("OK").all() else "FAIL",
            "detail": "status_Y/status_G over all retained horizons",
        },
        {
            "check": "cumulative K paths equal visible sums of incremental estimates",
            "status": "PASS" if polish_paths[["K_Y_visible_difference", "K_G_visible_difference"]].abs().max().max() <= 1e-10 else "FAIL",
            "detail": "K_Y and K_G reconstructed inside this notebook cell",
        },
    ]
)
repro.write_frame(polish_qa, RECOMPUTED / "polish_output_spending/qa_checks.csv")

polish_visible_paths = polish_paths[[
    "features",
    "horizon",
    "mu_Y_incremental",
    "mu_G_incremental",
    "K_Y_cumulative",
    "K_G_cumulative",
    "cumulative_output_to_spending_ratio",
    "nobs",
    "country_n",
    "year_min_effective",
    "year_max_effective",
    "status_Y",
    "status_G",
]].copy()
polish_visible_paths["Evaluation"] = polish_visible_paths["features"].map(public_features)
polish_visible_paths = polish_visible_paths.drop(columns=["features"])
display(polish_visible_paths)
display(polish_qa)

polish_h8 = polish_paths.loc[polish_paths["horizon"].eq(8), [
    "features",
    "K_Y_cumulative",
    "K_G_cumulative",
    "cumulative_output_to_spending_ratio",
    "nobs",
    "country_n",
    "year_min_effective",
    "year_max_effective",
]].copy()
polish_h8["Evaluation"] = polish_h8["features"].map(public_features)
polish_h8 = polish_h8.drop(columns=["features"])
display(polish_h8)
record("retained response paths h0-h8", polish_paths.groupby("features")["horizon"].nunique().min() == 9, "h0-h8 per retained evaluation")
record("visible cumulative K arithmetic", polish_qa["status"].eq("PASS").all(), polish_qa.to_string(index=False))
"""
    ),
    md(
        """## Estimate direct debt response and institutional debt-accounting paths

The debt block uses the retained output and spending paths. It estimates the direct debt-to-GDP response and applies the institutional debt recursion to the three-year investment action.
"""
    ),
    code(
        """debt_mod_audit = import_module(CODE / "estimation/debt_accounting_model.py", "public_debt_accounting_model_audit_cell")
debt_mod_audit.DATA_DIR = WORK_DATA
debt_mod_audit.RESULTS_DIR = RECOMPUTED / "debt_accounting_audit_cell"
debt_mod_audit.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
debt_mod_audit.REFERENCES_DIR = REFERENCES
debt_mod_audit.V3_CODE = REFERENCES / "project_context/ciaffi_canonical_estimation_v3_short_rate.py"
debt_mod_audit.ROOT = ROOT
debt_mod_audit.SPEC_VERSION = "public_notebook_visible_direct_debt_fit"
debt_mod_audit.PANEL_END_YEAR = PROFILE_YEAR

debt_v3_audit = debt_mod_audit.load_v3_module()
debt_work_audit = debt_mod_audit.prepare_work(debt_v3_audit)
audit_features_text = str(audit_spec["features"])
audit_debt_features = debt_mod_audit.parse_features(audit_features_text)
audit_z_values = debt_mod_audit.feature_values(audit_features_text)
visible_direct_debt_fit = debt_mod_audit.fit_direct_dy_ratio(
    debt_v3_audit,
    debt_work_audit,
    audit_debt_features,
    audit_z_values,
    horizon,
)
visible_direct_debt = pd.DataFrame([{**{"Evaluation": public_features(audit_features_text), "Horizon": horizon}, **visible_direct_debt_fit}])
display(visible_direct_debt)
record("visible direct debt LP fit", visible_direct_debt_fit["status_DY_initial"] == "OK", visible_direct_debt.to_string(index=False))
"""
    ),
    code(
        """debt_mod_visible = import_module(CODE / "estimation/debt_accounting_model.py", "public_debt_accounting_model_visible")
debt_mod_visible.DATA_DIR = WORK_DATA
debt_mod_visible.RESULTS_DIR = RECOMPUTED / "debt_accounting"
debt_mod_visible.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
debt_mod_visible.REFERENCES_DIR = REFERENCES
debt_mod_visible.V3_CODE = REFERENCES / "project_context/ciaffi_canonical_estimation_v3_short_rate.py"
debt_mod_visible.ROOT = ROOT
debt_mod_visible.SPEC_VERSION = "mozdzen_" + "tiva2022_gfcf_realppp_networth_debt_20260513"
debt_mod_visible.PANEL_END_YEAR = PROFILE_YEAR

debt_v3_visible = debt_mod_visible.load_v3_module()
debt_work_visible = debt_mod_visible.prepare_work(debt_v3_visible)
direct_debt_paths = debt_mod_visible.estimate_direct_debt_kernels(debt_v3_visible, debt_work_visible, selected)
program_paths = debt_mod_visible.build_program_paths(polish_paths, direct_debt_paths, selected)
dsm_inputs = debt_mod_visible.load_dsm_inputs()
baseline_reproduction = debt_mod_visible.reproduce_baseline(dsm_inputs)
dsa_paths = debt_mod_visible.simulate_dsa(program_paths, dsm_inputs)
debt_summary_internal = debt_mod_visible.make_2036_summary(program_paths, dsa_paths)

repro.write_frame(direct_debt_paths, RECOMPUTED / "debt_accounting/direct_dy_initial_action_paths.csv")
repro.write_frame(program_paths, RECOMPUTED / "debt_accounting/three_year_program_paths.csv")
repro.write_frame(baseline_reproduction, RECOMPUTED / "debt_accounting/baseline_reproduction.csv")
repro.write_frame(dsa_paths, RECOMPUTED / "debt_accounting/dsa_debt_paths.csv")
repro.write_frame(debt_summary_internal, RECOMPUTED / "debt_accounting/polish_debt_2036_summary.csv")
debt_qa = pd.DataFrame(
    [
        {"check": "direct debt kernels present for retained evaluations", "status": "PASS" if not direct_debt_paths.empty else "FAIL", "detail": f"rows={len(direct_debt_paths)}"},
        {"check": "program paths present", "status": "PASS" if not program_paths.empty else "FAIL", "detail": f"rows={len(program_paths)}"},
        {
            "check": "baseline reproduced within tolerance",
            "status": "PASS" if baseline_reproduction["abs_diff_pp"].max() <= debt_mod_visible.DEBT_BASELINE_REPRO_TOL_PP else "FAIL",
            "detail": f"max_abs_diff={baseline_reproduction['abs_diff_pp'].max():.3e}",
        },
    ]
)
repro.write_frame(debt_qa, RECOMPUTED / "debt_accounting/qa_checks.csv")

direct_debt_display = direct_debt_paths[direct_debt_paths["horizon"].isin([0, 1, 2, 5, 8])].copy()
direct_debt_display["Evaluation"] = direct_debt_display["features"].map(public_features)
display(direct_debt_display[[
    "Evaluation",
    "horizon",
    "direct_DY_initial_action",
    "se_direct_DY_initial_action",
    "nobs",
    "country_n",
    "year_min_effective",
    "year_max_effective",
    "status_DY_initial",
]])
program_display = program_paths[program_paths["horizon_from_2028"].isin([0, 1, 2, 5, 8])].copy()
program_display["Evaluation"] = program_display["features"].map(public_features)
program_display["Scenario"] = program_display["scenario_sign"].map({"cut": "public-investment cuts", "expansion": "public-investment expansion"})
display(program_display[[
    "Evaluation",
    "Scenario",
    "year",
    "horizon_from_2028",
    "fiscal_action_cut_units_pp",
    "Y_shortfall_pct",
    "direct_discretionary_PB_level_pp",
    "direct_DY_LP_margin_initial_action_pp",
]].head(20))
display(debt_qa)

debt_summary_display = debt_summary_internal[[
    "features",
    "scenario",
    "dsa_margin_vs_baseline_pp",
    "direct_DY_LP_margin_pp",
]].copy()
debt_summary_display["Evaluation"] = debt_summary_display["features"].map(public_features)
debt_summary_display = debt_summary_display.drop(columns=["features"])
display(debt_summary_display)
record("Polish debt endpoint summary", not debt_summary_internal.empty, f"rows={len(debt_summary_internal)}")
record("visible debt-accounting objects", debt_qa["status"].eq("PASS").all(), debt_qa.to_string(index=False))
"""
    ),
    md(
        """## Produce coefficient-level estimation output

This cell writes the coefficient tables used by Appendix D and then computes the p-value disclosure from those coefficient files. The count is generated here, not typed as a target.
"""
    ),
    code(
        """with contextlib.redirect_stdout(StringIO()):
    repro.make_estimation_output_tables(feature_mod, screen, polish_paths)
eu27_coef = pd.read_csv(RECOMPUTED / "estimation_output/eu27_benchmark_regression_coefficients.csv")
retained_coef = pd.read_csv(RECOMPUTED / "estimation_output/retained_regression_coefficients.csv")

eu27_displayed = eu27_coef[eu27_coef["term"].eq(GI_SHOCK)].copy()
retained_displayed = retained_coef[
    retained_coef["term"].eq(GI_SHOCK) | retained_coef["term"].str.startswith(GI_INTERACTION_PREFIX, na=False)
].copy()
displayed_pvalues = pd.concat([eu27_displayed, retained_displayed], ignore_index=True)
pvalue_disclosure = pd.DataFrame(
    [
        {
            "Displayed pointwise p-values": len(displayed_pvalues),
            "p-values above 0.1": int((displayed_pvalues["p_value"] > 0.1).sum()),
            "Minimum p-value": displayed_pvalues["p_value"].min(),
            "Maximum p-value": displayed_pvalues["p_value"].max(),
        }
    ]
)
display(pvalue_disclosure)
displayed_pvalues_public = displayed_pvalues[["path", "features", "outcome", "horizon", "term", "coefficient", "std_error", "p_value", "nobs", "country_n", "year_min", "year_max"]].copy()
displayed_pvalues_public["State variables"] = displayed_pvalues_public["features"].map(public_features)
displayed_pvalues_public["Term"] = displayed_pvalues_public["term"].map(public_regressor)
displayed_pvalues_public = displayed_pvalues_public.drop(columns=["features", "term"])
display(displayed_pvalues_public.head(18))
record("Appendix D p-value disclosure computed", len(displayed_pvalues) == 90, f"rows={len(displayed_pvalues)}")
"""
    ),
    md(
        """## Rebuild manuscript-facing tables and figures

After the estimator and accounting steps finish, the notebook rebuilds the public tables and figures from `results/recomputed/`.
"""
    ),
    code(
        """with contextlib.redirect_stdout(StringIO()):
    runpy.run_path(str(CODE / "build_public_tables_figures.py"), run_name="__main__")
table_qa = pd.read_csv(QA / "public_tables_figures_qa_20260514.csv")
display(table_qa)
record("public tables and figures", table_qa["status"].eq("PASS").all(), table_qa.to_string(index=False))
"""
    ),
    md(
        """## Bridge from regression output to K paths

The bridge table is now read after the estimation and table-generation steps above. It is a presentation of objects computed in this notebook run.
"""
    ),
    code(
        """bridge = pd.read_csv(TABLES / "estimation_response_bridge_by_horizon.csv")
bridge_paths = pd.read_csv(TABLES / "estimation_response_bridge_paths.csv")
bridge_sample = pd.read_csv(TABLES / "estimation_response_bridge_sample.csv")
h8_bridge = bridge[bridge["Horizon"].eq(8)].copy()
display(h8_bridge)
display(bridge_paths)
record("bridge h8 rows", len(h8_bridge) == 3, f"rows={len(h8_bridge)}")
"""
    ),
    md(
        """## Debt endpoints and equal-weight arithmetic

The next table is the manuscript-facing 2036 endpoint table rebuilt from the recomputed debt outputs. The equal-weight row is the arithmetic average of the two retained Polish evaluations.
"""
    ),
    code(
        """debt = pd.read_csv(TABLES / "debt_2036.csv")
decomp = pd.read_csv(TABLES / "annual_debt_decomposition.csv")
display(debt)
display(decomp.head(18))

polish_only = debt[debt["Empirical path"].isin([
    "Polish evaluation based on investment import content",
    "Polish evaluation based on household net financial worth",
])].copy()
equal_row = debt[debt["Empirical path"].eq("Equal weight average across the two Polish evaluations")].iloc[0]
numeric_cols = [
    "Expansion, institutional debt equation",
    "Expansion, direct debt-to-GDP local-projection path",
    "Cut, institutional debt equation",
    "Cut, direct debt-to-GDP local-projection path",
]
computed_equal = polish_only[numeric_cols].astype(float).mean()
equal_check = pd.DataFrame({"Column": numeric_cols, "Notebook arithmetic": computed_equal.values, "Displayed equal weight": equal_row[numeric_cols].astype(float).values})
equal_check["Difference"] = equal_check["Notebook arithmetic"] - equal_check["Displayed equal weight"]
display(equal_check)
record("equal-weight arithmetic", equal_check["Difference"].abs().max() <= 0.051, equal_check.to_string(index=False))
"""
    ),
    md(
        """## Validate recomputed outputs against frozen benchmark files

Validation happens at the end. It checks the fresh computation against frozen benchmark files with tolerance rules and does not replace the computation above.
"""
    ),
    code(
        """validation = repro.validate_against_frozen()
validation_public = validation.copy()
validation_public["check"] = validation_public["check"].map(public_validation_label)
display(validation_public)
record("frozen benchmark validation", validation["status"].eq("PASS").all(), f"{len(validation)} benchmark comparisons passed")
repro.write_manifest()

selected_horizons = bridge[bridge["Horizon"].isin([0, 1, 2, 5, 8])].copy()
selected_horizons.to_csv(RESULTS / "notebook_selected_horizons.csv", index=False)
debt.to_csv(RESULTS / "notebook_debt_margins_2036.csv", index=False)

all_checks = pd.DataFrame(checks)
all_checks.to_csv(RESULTS / "notebook_check_summary.csv", index=False)
all_checks_public = all_checks.copy()
all_checks_public["detail"] = all_checks_public["detail"].map(public_validation_label)
display(all_checks_public)
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
