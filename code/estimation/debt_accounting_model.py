#!/usr/bin/env python3
from __future__ import annotations

import os
import subprocess
import sys

sys.dont_write_bytecode = True

REQUIRED_THREAD_ENV = {
    "OPENBLAS_NUM_THREADS": "1",
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
}

if any(os.environ.get(key) != value for key, value in REQUIRED_THREAD_ENV.items()):
    env = os.environ.copy()
    env.update(REQUIRED_THREAD_ENV)
    raise SystemExit(subprocess.call([sys.executable, *sys.argv], env=env))

import hashlib
import importlib.util
import json
import math
import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    from threadpoolctl import threadpool_info, threadpool_limits

    THREADPOOL_LIMITER = threadpool_limits(limits=1)
except Exception:
    threadpool_info = None
    THREADPOOL_LIMITER = None


SPEC_VERSION = "debt_accounting_model_20260510"
SOURCE_POLISH_RESPONSES_LABEL = "polish_output_spending_model_20260510"

START_YEAR = 2024
END_YEAR = 2036
ACTION_START_YEAR = 2028
ACTION_YEARS = (2028, 2029, 2030)
HORIZONS = tuple(range(9))
BUDGET_ELASTICITY = 0.48
HARD_TOL = 5e-9
DEBT_BASELINE_REPRO_TOL_PP = 0.02


def find_package_root(script_path: Path) -> Path:
    for path in script_path.parents:
        if (path / "code").is_dir() and (path / "data").is_dir() and (path / "references").is_dir():
            return path
    raise RuntimeError(f"Cannot locate package root from {script_path}")


ROOT = find_package_root(Path(__file__).resolve())
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results" / "debt_accounting"
REFERENCES_DIR = ROOT / "references"
FROZEN_TARGETS = ROOT / "data/frozen/adopted_run_outputs"
V3_CODE = REFERENCES_DIR / "project_context/ciaffi_canonical_estimation_v3_short_rate.py"

RAW_LAG_SUPPORT_START_YEAR = 2001
LP_SAMPLE_START_YEAR = 2004
PANEL_END_YEAR = 2024
LAG_DEPTH = 1
DENOMINATOR_T_THRESHOLD = 1.96
FEATURE_Z_COLUMNS = {
    "trade": "trade_z_lag1",
    "debt": "debt_z_lag1",
    "liq": "liq_z_lag1",
    "log_gdp_pc": "log_gdp_pc_z_lag1",
}


def rel(path: Path) -> str:
    return str(path.resolve().relative_to(ROOT))


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def clean_results_dir() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    for path in RESULTS_DIR.glob("*"):
        if path.is_file() and path.suffix in {".csv", ".json"}:
            path.unlink()


def validate_dependencies() -> None:
    required = [
        DATA_DIR / "country_feature_panel.csv",
        DATA_DIR / "eu27_lp_joint_panel_snapshot.csv",
        DATA_DIR / "ec_poland_dsm2025_baseline_table_20260308.csv",
        DATA_DIR / "commission_poland_exogenous_path_20260310.csv",
        V3_CODE,
        ROOT / "results" / "feature_screen" / "feature_robustness_summary.csv",
        ROOT / "results" / "polish_output_spending" / "polish_output_spending_paths.csv",
    ]
    missing = [str(path.relative_to(ROOT)) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required package inputs: " + "; ".join(missing))


def load_adopted_specification_targets() -> pd.DataFrame:
    target_path = FROZEN_TARGETS / "feature_screen/feature_robustness_summary.csv"
    target = pd.read_csv(target_path)
    selected = target.loc[
        target["gate_status"].eq("PASS_ROBUSTNESS_GATE"),
        ["spec_id", "features"],
    ].copy()
    selected["selection_source"] = "frozen_validation_feature_screen"
    selected["selection_status"] = "PASS_ROBUSTNESS_GATE"
    return selected.sort_values("spec_id").reset_index(drop=True)


def load_polish_response_path_targets() -> pd.DataFrame:
    target_path = FROZEN_TARGETS / "polish_output_spending/polish_output_spending_paths.csv"
    return pd.read_csv(target_path).sort_values(["spec_id", "horizon"]).reset_index(drop=True)


def load_direct_debt_path_targets() -> pd.DataFrame:
    target_path = FROZEN_TARGETS / "debt_accounting/direct_dy_initial_action_paths.csv"
    return pd.read_csv(target_path).sort_values(["spec_id", "horizon"]).reset_index(drop=True)


def load_debt_endpoint_targets() -> pd.DataFrame:
    target_path = FROZEN_TARGETS / "debt_accounting/polish_debt_2036_summary.csv"
    return pd.read_csv(target_path).sort_values(["spec_id", "scenario"]).reset_index(drop=True)


def max_abs_target_error(
    recomputed: pd.DataFrame,
    target: pd.DataFrame,
    key_cols: list[str],
    value_cols: list[str],
) -> tuple[float, str]:
    merged = recomputed.merge(target, on=key_cols, how="outer", suffixes=("_recomputed", "_target"), indicator=True)
    if not merged["_merge"].eq("both").all():
        missing = merged.loc[~merged["_merge"].eq("both"), [*key_cols, "_merge"]].head(3).to_dict("records")
        return math.inf, f"unmatched_rows={missing}"
    max_error = 0.0
    worst_col = "none"
    for col in value_cols:
        error = (merged[f"{col}_recomputed"].astype(float) - merged[f"{col}_target"].astype(float)).abs().max()
        if float(error) > max_error:
            max_error = float(error)
            worst_col = col
    return max_error, worst_col


def git_value(*args: str) -> str:
    try:
        proc = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=False)
    except Exception:
        return "unknown"
    return proc.stdout.strip() if proc.returncode == 0 else "unknown"


def threadpool_state() -> list[dict[str, Any]] | str:
    if threadpool_info is None:
        return "threadpoolctl_unavailable"
    sanitized: list[dict[str, Any]] = []
    for entry in threadpool_info():
        row = dict(entry)
        filepath = row.pop("filepath", None)
        if filepath:
            row["library_file"] = Path(filepath).name
        sanitized.append(row)
    return sanitized


def parse_features(value: str) -> tuple[str, ...]:
    return tuple(part for part in str(value).split("+") if part)


def load_v3_module() -> Any:
    spec = importlib.util.spec_from_file_location("debt_accounting_estimation_core", V3_CODE)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {V3_CODE}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    module.INPUT_PANEL = DATA_DIR / "eu27_lp_joint_panel_snapshot.csv"
    module.RESULTS_DIR = RESULTS_DIR
    module.REFERENCES_DIR = DATA_DIR
    module.SPEC_VERSION = SPEC_VERSION
    module.PANEL_START_YEAR = RAW_LAG_SUPPORT_START_YEAR
    module.SHOCK_START_YEAR = LP_SAMPLE_START_YEAR
    module.LAG_DEPTHS = (LAG_DEPTH,)

    def component_lp_controls_no_lagged_shocks(lag_depth: int) -> list[str]:
        return module.system_lag_controls(module.SYSTEM_COMPONENT, lag_depth)

    def aggregate_lp_controls_no_lagged_shocks(lag_depth: int) -> list[str]:
        return module.system_lag_controls(module.SYSTEM_AGGREGATE, lag_depth)

    module.component_lp_controls = component_lp_controls_no_lagged_shocks
    module.aggregate_lp_controls = aggregate_lp_controls_no_lagged_shocks
    return module


def feature_values(features_text: str) -> dict[str, float]:
    features = parse_features(features_text)
    feature_panel = pd.read_csv(DATA_DIR / "country_feature_panel.csv")
    row = feature_panel[(feature_panel["country"].eq("POL")) & (feature_panel["year"].eq(PANEL_END_YEAR))].iloc[0]
    return {feature: float(row[f"{feature}_z"]) for feature in features}


def prepare_work(v3: Any) -> pd.DataFrame:
    panel, _meta = v3.load_panel()
    feature_panel = pd.read_csv(DATA_DIR / "country_feature_panel.csv")
    work = v3.prepare_work(panel)
    work = work.sort_values(["country", "year"]).reset_index(drop=True)
    current_vars = ["dlog_gi", "dlog_gc", "dlog_g", "dlog_y", "i_rate"]
    work.loc[work["year"].lt(LP_SAMPLE_START_YEAR), current_vars] = np.nan
    work, _shock_meta = v3.attach_shocks(work, LAG_DEPTH)
    group = work.groupby("country", sort=False)
    for h in HORIZONS:
        gi_tph = group["gi_real"].shift(-h)
        gi_prev_incremental = group["gi_real"].shift(1) if h == 0 else group["gi_real"].shift(-(h - 1))
        work[f"gi_dyn_h{h}"] = (gi_tph - gi_prev_incremental) / work["y_level_lag1"]
        debt_current = group["debt_ratio"].shift(-h)
        debt_base = group["debt_ratio"].shift(1)
        work[f"debt_dyn_ratio_h{h}"] = debt_current - debt_base
    feature_cols = ["country", "year", *FEATURE_Z_COLUMNS.values()]
    work = work.merge(feature_panel[feature_cols], on=["country", "year"], how="left", validate="many_to_one")
    work = work.replace([np.inf, -np.inf], np.nan)
    for feature, z_col in FEATURE_Z_COLUMNS.items():
        work[f"shock_G_I_x_{feature}"] = work["shock_G_I"] * work[z_col]
    return work


def x_columns(features: tuple[str, ...]) -> list[str]:
    controls = [f"{var}_lag{LAG_DEPTH}" for var in ["dlog_gi", "dlog_gc", "dlog_y", "i_rate"]]
    cols = ["shock_G_I", "shock_G_C"]
    cols += [FEATURE_Z_COLUMNS[feature] for feature in features]
    cols += [f"shock_G_I_x_{feature}" for feature in features]
    cols += controls
    return cols


def contrast_vector(cols: list[str], z_values: dict[str, float]) -> np.ndarray:
    out = np.zeros(len(cols), dtype=float)
    out[cols.index("shock_G_I")] = 1.0
    for feature, value in z_values.items():
        name = f"shock_G_I_x_{feature}"
        if name in cols:
            out[cols.index(name)] = float(value)
    return out


def fit_direct_dy_ratio(
    v3: Any,
    work: pd.DataFrame,
    features: tuple[str, ...],
    z_values: dict[str, float],
    horizon: int,
) -> dict[str, Any]:
    dep_col = f"debt_dyn_ratio_h{horizon}"
    scale_col = "gi_dyn0"
    cols = x_columns(features)
    needed = [dep_col, scale_col, *cols, "country", "year"]
    window_start, window_end = v3.shock_window(horizon)
    sample = work[work["year"].between(window_start, window_end)].copy()
    sample = sample.dropna(subset=needed).sort_values(["country", "year"]).reset_index(drop=True)
    if len(sample) < 50:
        return {
            "direct_DY_initial_action": math.nan,
            "se_direct_DY_initial_action": math.nan,
            "status_DY_initial": "INSUFFICIENT_OBS",
            "nobs": int(len(sample)),
            "country_n": int(sample["country"].nunique()) if len(sample) else 0,
            "year_min_effective": math.nan,
            "year_max_effective": math.nan,
            "rank_X": 0,
        }
    projector = v3.FEProjector(sample["country"], sample["year"])
    x_res = projector.residualize(sample[cols].to_numpy(dtype=float))
    dep_res = projector.residualize(sample[dep_col].to_numpy(dtype=float))
    scale_res = projector.residualize(sample[scale_col].to_numpy(dtype=float))
    beta_dep, _fit_dep, resid_dep, xtx_inv, rank = v3.ols_fit(x_res, dep_res)
    beta_scale, _fit_scale, resid_scale, _xtx_scale, _rank_scale = v3.ols_fit(x_res, scale_res)
    years = sample["year"].to_numpy(dtype=int)
    bandwidth = max(int(horizon), 1)
    vcov_dep = v3.dk_covariance(x_res, resid_dep, years, xtx_inv, bandwidth)
    vcov_scale = v3.dk_covariance(x_res, resid_scale, years, xtx_inv, bandwidth)
    vcov_cross = v3.dk_cross_covariance(x_res, resid_dep, resid_scale, years, xtx_inv, bandwidth)
    c = contrast_vector(cols, z_values)
    beta_dep_c = float(c @ beta_dep)
    beta_scale_c = float(c @ beta_scale)
    var_dep = float(c @ vcov_dep @ c)
    var_scale = float(c @ vcov_scale @ c)
    cov_cross = float(c @ vcov_cross @ c)
    ratio, se = v3.ratio_and_se(beta_dep_c, beta_scale_c, var_dep, var_scale, cov_cross)
    se_scale = math.sqrt(max(var_scale, 0.0)) if math.isfinite(var_scale) else math.nan
    denom_t = abs(beta_scale_c / se_scale) if math.isfinite(se_scale) and se_scale > 0 else math.nan
    status = "OK" if math.isfinite(ratio) and math.isfinite(se) else "NONFINITE"
    if not math.isfinite(beta_scale_c) or abs(beta_scale_c) < 1e-12:
        status = "ZERO_SCALE_DENOMINATOR"
    elif not math.isfinite(denom_t) or denom_t < DENOMINATOR_T_THRESHOLD:
        status = "WEAK_SCALE_DENOMINATOR"
    return {
        "direct_DY_initial_action": float(ratio),
        "se_direct_DY_initial_action": float(se),
        "status_DY_initial": status,
        "nobs": int(len(sample)),
        "country_n": int(sample["country"].nunique()),
        "year_min_effective": int(sample["year"].min()),
        "year_max_effective": int(sample["year"].max()),
        "rank_X": int(rank),
    }


def load_selected_specs() -> pd.DataFrame:
    screen_path = ROOT / "results" / "feature_screen" / "feature_robustness_summary.csv"
    if not screen_path.exists():
        raise FileNotFoundError(
            "Run code/feature_screen_model.py before debt accounting; "
            "the admitted specifications must come from the freshly estimated screen."
        )
    screen = pd.read_csv(screen_path)
    selected = screen.loc[
        screen["gate_status"].eq("PASS_ROBUSTNESS_GATE"),
        ["spec_id", "features"],
    ].copy()
    selected["selection_source"] = "fresh_feature_screen"
    selected["selection_status"] = "PASS_ROBUSTNESS_GATE"
    selected = selected.sort_values("spec_id").reset_index(drop=True)
    target = load_adopted_specification_targets().copy()
    target["selection_source"] = "fresh_feature_screen"
    if not selected.equals(target):
        raise AssertionError("Fresh feature-screen winners do not match the frozen robustness-screen result")
    return selected


def load_polish_response_paths() -> pd.DataFrame:
    generated = ROOT / "results" / "polish_output_spending" / "polish_output_spending_paths.csv"
    if not generated.exists():
        raise FileNotFoundError(
            "Run code/polish_output_spending_model.py before debt accounting; "
            "the debt model must use freshly generated Polish response paths."
        )
    paths = pd.read_csv(generated)
    if "spec_version" in paths.columns and not paths["spec_version"].eq(SOURCE_POLISH_RESPONSES_LABEL).all():
        raise AssertionError("Polish response path input has an unexpected model label")
    target_pairs = set(zip(load_adopted_specification_targets()["spec_id"], load_adopted_specification_targets()["features"]))
    actual_pairs = set(zip(paths["spec_id"], paths["features"]))
    if actual_pairs != target_pairs:
        raise AssertionError(f"Polish response paths do not match selected specifications: {actual_pairs}")
    max_error, worst_col = max_abs_target_error(
        paths,
        load_polish_response_path_targets(),
        ["spec_id", "features", "horizon"],
        ["K_Y_cumulative", "K_G_cumulative"],
    )
    if max_error > HARD_TOL:
        raise AssertionError(f"Polish response path mismatch against frozen validation target: {worst_col}={max_error}")
    return paths.sort_values(["spec_id", "horizon"]).reset_index(drop=True)


def estimate_direct_debt_kernels(v3: Any, work: pd.DataFrame, selected: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for row in selected.itertuples(index=False):
        features = parse_features(str(row.features))
        z_values = feature_values(str(row.features))
        for h in HORIZONS:
            fit = fit_direct_dy_ratio(v3, work, features, z_values, h)
            rows.append(
                {
                    "spec_version": SPEC_VERSION,
                    "spec_id": row.spec_id,
                    "features": row.features,
                    "profile_label": "poland_2024",
                    "horizon": h,
                    **fit,
                    **{f"z_{feature}": float(z_values[feature]) for feature in features},
                }
            )
    return pd.DataFrame(rows).sort_values(["spec_id", "horizon"]).reset_index(drop=True)


def convolve_path(actions: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    out = np.zeros_like(actions, dtype=float)
    for h in range(len(actions)):
        out[h] = sum(actions[s] * kernel[h - s] for s in range(h + 1))
    return out


def scenario_definitions() -> list[dict[str, Any]]:
    cut_actions = np.zeros(len(HORIZONS), dtype=float)
    for year in ACTION_YEARS:
        cut_actions[year - ACTION_START_YEAR] = 1.0
    expansion_actions = -cut_actions
    return [
        {
            "scenario": "three_1pp_cut_2028_2030",
            "scenario_sign": "cut",
            "actions": cut_actions,
            "description": "Three 1 pp GDP public-investment cuts in 2028, 2029 and 2030.",
        },
        {
            "scenario": "three_1pp_expansion_2028_2030",
            "scenario_sign": "expansion",
            "actions": expansion_actions,
            "description": "Three 1 pp GDP public-investment expansions in 2028, 2029 and 2030.",
        },
    ]


def build_program_paths(
    polish_response_paths: pd.DataFrame,
    direct_kernels: pd.DataFrame,
    selected: pd.DataFrame | None = None,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    selected_specs = selected if selected is not None else load_selected_specs()
    for row in selected_specs.itertuples(index=False):
        spec_id = str(row.spec_id)
        features = str(row.features)
        outsp = polish_response_paths[polish_response_paths["spec_id"].eq(spec_id)].sort_values("horizon")
        direct = direct_kernels[direct_kernels["spec_id"].eq(spec_id)].sort_values("horizon")
        k_y = outsp["K_Y_cumulative"].to_numpy(dtype=float)
        k_g = outsp["K_G_cumulative"].to_numpy(dtype=float)
        dy_initial = direct["direct_DY_initial_action"].to_numpy(dtype=float)
        for scenario in scenario_definitions():
            actions = np.asarray(scenario["actions"], dtype=float)
            y_shortfall = convolve_path(actions, k_y)
            direct_pb = convolve_path(actions, k_g)
            direct_dy_margin = -convolve_path(actions, dy_initial)
            for h in HORIZONS:
                rows.append(
                    {
                        "spec_version": SPEC_VERSION,
                        "spec_id": spec_id,
                        "features": features,
                        "scenario": scenario["scenario"],
                        "scenario_sign": scenario["scenario_sign"],
                        "year": ACTION_START_YEAR + h,
                        "horizon_from_2028": h,
                        "fiscal_action_cut_units_pp": actions[h],
                        "Y_shortfall_pct": y_shortfall[h],
                        "direct_discretionary_PB_level_pp": direct_pb[h],
                        "direct_DY_LP_margin_initial_action_pp": direct_dy_margin[h],
                        "description": scenario["description"],
                    }
                )
    return pd.DataFrame(rows)


def load_dsm_inputs() -> pd.DataFrame:
    base = pd.read_csv(DATA_DIR / "ec_poland_dsm2025_baseline_table_20260308.csv")
    exog = pd.read_csv(DATA_DIR / "commission_poland_exogenous_path_20260310.csv")
    work = base[base["year"].between(START_YEAR, END_YEAR)].merge(
        exog[["year", "nominal_gdp_growth", "implicit_interest_rate"]],
        on="year",
        how="left",
        validate="one_to_one",
    )
    missing = work[work["year"].gt(START_YEAR)][["nominal_gdp_growth", "implicit_interest_rate"]].isna()
    if missing.any().any():
        raise RuntimeError("Missing DSM exogenous inputs")
    return work.sort_values("year").reset_index(drop=True)


def reproduce_baseline(dsm_inputs: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    prev_debt = math.nan
    for row in dsm_inputs.itertuples(index=False):
        year = int(row.year)
        if year == START_YEAR:
            debt = float(row.gross_debt_ratio) / 100.0
        else:
            debt = (
                prev_debt
                * (1.0 + float(row.implicit_interest_rate) / 100.0)
                / (1.0 + float(row.nominal_gdp_growth) / 100.0)
                - float(row.primary_balance) / 100.0
                + float(row.stock_flow_adjustments) / 100.0
            )
        rows.append(
            {
                "spec_version": SPEC_VERSION,
                "year": year,
                "baseline_reproduced_D_Y_pp": debt * 100.0,
                "source_D_Y_pp": float(row.gross_debt_ratio),
                "abs_diff_pp": abs(debt * 100.0 - float(row.gross_debt_ratio)),
            }
        )
        prev_debt = debt
    return pd.DataFrame(rows)


def simulate_dsa(program_paths: pd.DataFrame, dsm_inputs: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (spec_id, scenario), scenario_path in program_paths.groupby(["spec_id", "scenario"], sort=False):
        path_by_year = scenario_path.set_index("year")
        features = str(scenario_path["features"].iloc[0])
        sign = str(scenario_path["scenario_sign"].iloc[0])
        prev_debt = math.nan
        prev_gap_pct = 0.0
        for row in dsm_inputs.itertuples(index=False):
            year = int(row.year)
            baseline_debt = float(row.gross_debt_ratio)
            baseline_pb = float(row.primary_balance)
            if year < ACTION_START_YEAR:
                y_shortfall = 0.0
                direct_pb = 0.0
                direct_dy = 0.0
            else:
                path_row = path_by_year.loc[year] if year in path_by_year.index else path_by_year.iloc[-1]
                y_shortfall = float(path_row["Y_shortfall_pct"])
                direct_pb = float(path_row["direct_discretionary_PB_level_pp"])
                direct_dy = float(path_row["direct_DY_LP_margin_initial_action_pp"])
            if year < ACTION_START_YEAR:
                debt = baseline_debt
                pb_new = baseline_pb
                nominal_new = float(row.nominal_gdp_growth) if pd.notna(row.nominal_gdp_growth) else math.nan
                gap_pct = 0.0
                delta_cyclical = 0.0
                prev_debt = debt / 100.0
                prev_gap_pct = 0.0
            else:
                gap_pct = -y_shortfall
                delta_cyclical = -BUDGET_ELASTICITY * y_shortfall
                pb_new = baseline_pb + direct_pb + delta_cyclical
                nominal_base = float(row.nominal_gdp_growth)
                nominal_new_decimal = (
                    (1.0 + nominal_base / 100.0)
                    * (1.0 + gap_pct / 100.0)
                    / (1.0 + prev_gap_pct / 100.0)
                    - 1.0
                )
                debt_decimal = (
                    prev_debt
                    * (1.0 + float(row.implicit_interest_rate) / 100.0)
                    / (1.0 + nominal_new_decimal)
                    - pb_new / 100.0
                    + float(row.stock_flow_adjustments) / 100.0
                )
                debt = debt_decimal * 100.0
                nominal_new = nominal_new_decimal * 100.0
                prev_debt = debt_decimal
                prev_gap_pct = gap_pct
            rows.append(
                {
                    "spec_version": SPEC_VERSION,
                    "spec_id": spec_id,
                    "features": features,
                    "scenario": scenario,
                    "scenario_sign": sign,
                    "year": year,
                    "horizon_from_2028": year - ACTION_START_YEAR,
                    "Y_shortfall_pct": y_shortfall,
                    "direct_discretionary_PB_level_pp": direct_pb,
                    "delta_cyclical_PB_pp": delta_cyclical,
                    "baseline_PB_pp": baseline_pb,
                    "PB_new_pp": pb_new,
                    "nominal_gdp_growth_new_pct": nominal_new,
                    "baseline_D_Y_pp": baseline_debt,
                    "D_Y_new_pp": debt,
                    "dsa_margin_vs_baseline_pp": debt - baseline_debt,
                    "direct_DY_LP_margin_initial_action_pp": direct_dy,
                }
            )
    return pd.DataFrame(rows)


def make_2036_summary(program_paths: pd.DataFrame, dsa_paths: pd.DataFrame) -> pd.DataFrame:
    dsa_2036 = dsa_paths[dsa_paths["year"].eq(2036)].copy()
    program_2036 = program_paths[program_paths["year"].eq(2036)].copy()
    merged = dsa_2036.merge(
        program_2036[
            [
                "spec_id",
                "scenario",
                "direct_DY_LP_margin_initial_action_pp",
                "Y_shortfall_pct",
                "direct_discretionary_PB_level_pp",
            ]
        ],
        on=["spec_id", "scenario"],
        suffixes=("", "_program"),
        validate="one_to_one",
    )
    out = merged[
        [
            "spec_id",
            "features",
            "scenario",
            "scenario_sign",
            "dsa_margin_vs_baseline_pp",
            "direct_DY_LP_margin_initial_action_pp_program",
            "Y_shortfall_pct_program",
            "direct_discretionary_PB_level_pp_program",
        ]
    ].rename(
        columns={
            "direct_DY_LP_margin_initial_action_pp_program": "direct_DY_LP_margin_pp",
            "Y_shortfall_pct_program": "Y_shortfall_pct",
            "direct_discretionary_PB_level_pp_program": "direct_discretionary_PB_level_pp",
        }
    )
    return out.sort_values(["spec_id", "scenario"]).reset_index(drop=True)


def input_provenance() -> pd.DataFrame:
    rows = [
        {
            "file": "results/feature_screen/feature_robustness_summary.csv",
            "role": "admitted Polish model list",
            "upstream_source": "fresh feature-screen estimation within the public reproduction sequence",
            "provenance": "The debt arithmetic reads the specifications that pass the freshly re-estimated robustness screen.",
        },
        {
            "file": "results/polish_output_spending/polish_output_spending_paths.csv",
            "role": "K_Y and K_G paths used by the debt recursion",
            "upstream_source": "fresh Polish output and spending response estimation within the public reproduction sequence",
            "provenance": "The debt arithmetic reads output and spending response paths generated earlier in the same notebook run.",
        },
        {
            "file": "data/eu27_lp_joint_panel_snapshot.csv",
            "role": "panel used to estimate direct D/Y local projections",
            "upstream_source": "frozen EU27 estimation panel",
            "provenance": "Panel prepared from Eurostat and short-rate sources and bundled locally for reproduction.",
        },
        {
            "file": "data/country_feature_panel.csv",
            "role": "Poland feature profile and feature interactions",
            "upstream_source": "Eurostat raw JSON snapshots in data/raw",
            "provenance": "Feature panel with trade, debt, liquidity and log GDP per capita standardization.",
        },
        {
            "file": "data/ec_poland_dsm2025_baseline_table_20260308.csv",
            "role": "Poland debt baseline used in the debt recursion",
            "upstream_source": "European Commission DSM 2025 Poland table extraction",
            "provenance": "Poland baseline debt, primary balance and stock-flow-adjustment path.",
        },
        {
            "file": "data/commission_poland_exogenous_path_20260310.csv",
            "role": "nominal growth and implicit interest path",
            "upstream_source": "extraction from the Commission projection tables aligned with the debt-recursion shell",
            "provenance": "Exogenous nominal growth and implicit interest assumptions for the debt recursion.",
        },
    ]
    return pd.DataFrame(rows)


def qa_checks(
    selected: pd.DataFrame,
    polish_response_paths: pd.DataFrame,
    direct_kernels: pd.DataFrame,
    baseline_repro: pd.DataFrame,
    summary_2036: pd.DataFrame,
) -> pd.DataFrame:
    checks: list[dict[str, str]] = []
    target_specs = load_adopted_specification_targets()
    expected_pairs = set(zip(target_specs["spec_id"], target_specs["features"]))
    actual_pairs = set(zip(selected["spec_id"], selected["features"]))
    checks.append(
        {
            "check": "admitted_specifications_match_frozen_validation_screen",
            "status": "PASS" if actual_pairs == expected_pairs else "FAIL",
            "detail": "selected specifications=" + "; ".join(sorted(selected["features"].tolist())),
        }
    )
    horizon_ok = polish_response_paths.groupby("spec_id")["horizon"].apply(lambda s: tuple(sorted(s)) == HORIZONS).all()
    checks.append(
        {
            "check": "polish_response_h0_h8_paths_present",
            "status": "PASS" if bool(horizon_ok) else "FAIL",
            "detail": "K_Y and K_G h0..h8 are present for every selected specification",
        }
    )
    direct_ok = direct_kernels.groupby("spec_id")["horizon"].apply(lambda s: tuple(sorted(s)) == HORIZONS).all()
    checks.append(
        {
            "check": "direct_dy_h0_h8_paths_estimated",
            "status": "PASS" if bool(direct_ok) else "FAIL",
            "detail": "direct D/Y initial-action h0..h8 is estimated inside this block",
        }
    )
    status_ok = direct_kernels["status_DY_initial"].eq("OK").all()
    checks.append(
        {
            "check": "direct_dy_status_ok",
            "status": "PASS" if bool(status_ok) else "FAIL",
            "detail": "status_DY_initial is OK for every selected specification and horizon",
        }
    )
    max_direct_error, worst_direct_col = max_abs_target_error(
        direct_kernels,
        load_direct_debt_path_targets(),
        ["spec_id", "features", "horizon"],
        ["direct_DY_initial_action"],
    )
    checks.append(
        {
            "check": "direct_dy_path_matches_frozen_validation_target",
            "status": "PASS" if max_direct_error <= HARD_TOL else "FAIL",
            "detail": f"max_abs_error={max_direct_error:.12g} worst_column={worst_direct_col} tolerance={HARD_TOL:.1e}",
        }
    )
    max_baseline_diff = float(baseline_repro["abs_diff_pp"].max())
    checks.append(
        {
            "check": "dsm_baseline_reproduction",
            "status": "PASS" if max_baseline_diff <= DEBT_BASELINE_REPRO_TOL_PP else "FAIL",
            "detail": f"max_abs_diff_pp={max_baseline_diff:.9f} tolerance_pp={DEBT_BASELINE_REPRO_TOL_PP:.3f}",
        }
    )
    max_summary_error, worst_summary_col = max_abs_target_error(
        summary_2036,
        load_debt_endpoint_targets(),
        ["spec_id", "features", "scenario", "scenario_sign"],
        [
            "dsa_margin_vs_baseline_pp",
            "direct_DY_LP_margin_pp",
            "Y_shortfall_pct",
            "direct_discretionary_PB_level_pp",
        ],
    )
    checks.append(
        {
            "check": "debt_2036_summary_matches_frozen_validation_target",
            "status": "PASS" if max_summary_error <= HARD_TOL else "FAIL",
            "detail": f"max_abs_error={max_summary_error:.12g} worst_column={worst_summary_col} tolerance={HARD_TOL:.1e}",
        }
    )
    scenario_set = set(summary_2036["scenario"])
    checks.append(
        {
            "check": "only_three_year_program_scenarios",
            "status": "PASS" if scenario_set == {"three_1pp_cut_2028_2030", "three_1pp_expansion_2028_2030"} else "FAIL",
            "detail": ",".join(sorted(scenario_set)),
        }
    )
    source_note = DATA_DIR / "eurostat_short_rate_source_notes.md"
    source_note_text = source_note.read_text(encoding="utf-8") if source_note.exists() else ""
    expected_snapshot_paths = [f"data/eurostat_{dataset}_snapshot.csv" for dataset in v3_short_rate_datasets()]
    wrong_reference_paths = [line for line in source_note_text.splitlines() if "references/eurostat_" in line]
    missing_snapshot_paths = [
        path for path in expected_snapshot_paths if path not in source_note_text or not (ROOT / path).exists()
    ]
    source_note_ok = source_note.exists() and not wrong_reference_paths and not missing_snapshot_paths
    checks.append(
        {
            "check": "eurostat_short_rate_note_paths_exist",
            "status": "PASS" if source_note_ok else "FAIL",
            "detail": (
                "all snapshot paths point to bundled data/ files"
                if source_note_ok
                else f"wrong_reference_rows={len(wrong_reference_paths)} missing_data_paths={missing_snapshot_paths}"
            ),
        }
    )
    return pd.DataFrame(checks)


def v3_short_rate_datasets() -> tuple[str, ...]:
    return ("irt_st_a", "irt_h_mr3_a", "irt_st_m", "irt_h_mr3_m")


def md_table(df: pd.DataFrame, cols: list[str], float_cols: set[str]) -> str:
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in df[cols].iterrows():
        values = []
        for col in cols:
            val = row[col]
            if pd.isna(val):
                values.append("NA")
            elif col in float_cols:
                values.append(f"{float(val):.9f}")
            else:
                values.append(str(val))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def write_report(
    direct_kernels: pd.DataFrame,
    program_paths: pd.DataFrame,
    dsa_paths: pd.DataFrame,
    summary_2036: pd.DataFrame,
    provenance: pd.DataFrame,
    qa: pd.DataFrame,
) -> None:
    direct_h8 = direct_kernels[direct_kernels["horizon"].eq(8)].copy()
    path_2036 = program_paths[program_paths["year"].eq(2036)].copy()
    report = f"""# Polish debt-to-GDP effects

## Scope

This public reproduction block computes only debt-to-GDP effects for the two Polish models admitted by the state-variable screen and using the Polish output and spending response paths estimated earlier in the notebook.

It reports two debt paths:

1. Debt recursion with the Polish output and spending paths.
2. Direct debt-to-GDP local-projection path estimated within this public reproduction block for the same selected specifications.

It uses only the three-year program: 1 pp GDP in 2028, 2029 and 2030. Both directions are reported. No one-year 3 pp scenario is computed.

## Inputs

{md_table(provenance, ["file", "role", "upstream_source", "provenance"], set())}

## Direct debt-to-GDP h8 kernels

{md_table(direct_h8, ["features", "direct_DY_initial_action", "se_direct_DY_initial_action", "nobs", "country_n", "year_min_effective", "year_max_effective"], {"direct_DY_initial_action", "se_direct_DY_initial_action"})}

## 2036 debt results

{md_table(summary_2036, ["features", "scenario", "dsa_margin_vs_baseline_pp", "direct_DY_LP_margin_pp", "Y_shortfall_pct", "direct_discretionary_PB_level_pp"], {"dsa_margin_vs_baseline_pp", "direct_DY_LP_margin_pp", "Y_shortfall_pct", "direct_discretionary_PB_level_pp"})}

## 2036 program inputs

{md_table(path_2036, ["features", "scenario", "Y_shortfall_pct", "direct_discretionary_PB_level_pp", "direct_DY_LP_margin_initial_action_pp"], {"Y_shortfall_pct", "direct_discretionary_PB_level_pp", "direct_DY_LP_margin_initial_action_pp"})}

## QA

{md_table(qa, ["check", "status", "detail"], set())}

## Reproducibility notes

- The plain command is `python3 code/debt_accounting_model.py`.
- The runner re-execs Python with one BLAS/OpenMP thread before importing NumPy.
- `results/source_manifest.csv` is deterministic for public-archive source files, this block's stable CSV outputs, and this block report.
- `results/run_manifest.json` is dynamic run metadata and is not expected to be byte-identical across machines.
"""
    (RESULTS_DIR / "REPORT.md").write_text(report)


def write_manifest() -> None:
    manifest_path = RESULTS_DIR / "source_manifest.csv"
    paths = (
        list((ROOT / "code").glob("*.py"))
        + list((ROOT / "data").glob("*.csv"))
        + list((ROOT / "data").glob("*.md"))
        + list((ROOT / "data/raw").glob("*"))
        + list((ROOT / "references").glob("*.py"))
        + list((ROOT / "references").glob("*.md"))
        + list((ROOT / "references/commission").glob("*"))
        + list((ROOT / "references/project_context").glob("*"))
        + list((ROOT / "references/papers_txt").glob("*"))
        + list((ROOT / "results").glob("*.csv"))
        + list(RESULTS_DIR.glob("*.csv"))
        + [RESULTS_DIR / "REPORT.md"]
        + [ROOT / "REPORT.md", ROOT / "requirements.txt"]
    )
    if Path(DATA_DIR).resolve() == (ROOT / "data/current_model_inputs").resolve() or "current_data_branch" in RESULTS_DIR.parts:
        paths = paths + (
            list((ROOT / "data/current_model_inputs").glob("*.csv"))
            + list((ROOT / "data/current_downloads").glob("*.csv"))
            + list((ROOT / "data/current_downloads").glob("*.json"))
        )
    rows = []
    for path in sorted(paths):
        if path.resolve() == manifest_path.resolve():
            continue
        if path.is_file():
            rows.append({"path": rel(path), "sha256": sha256(path), "bytes": path.stat().st_size})
    pd.DataFrame(rows).to_csv(manifest_path, index=False)
    run_manifest = {
        "spec_version": SPEC_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git_head": git_value("rev-parse", "HEAD"),
        "git_status_note": "not embedded; source_manifest.csv carries public-archive file hashes for standalone review",
        "python": sys.version,
        "platform": platform.platform(),
        "selected_specifications": load_adopted_specification_targets()[["features"]]
        .rename(columns={"features": "specification"})
        .to_dict("records"),
        "action_years": list(ACTION_YEARS),
        "horizons": list(HORIZONS),
        "budget_elasticity": BUDGET_ELASTICITY,
        "blas_thread_env": {key: os.environ.get(key, "") for key in REQUIRED_THREAD_ENV},
        "threadpoolctl_info": threadpool_state(),
        "excluded_from_scope": [
            "state_variable_screen",
            "one_year_3pp_scenario",
            "cash-basis fiscal sensitivity",
            "model_average",
            "compiled manuscript render",
        ],
    }
    (RESULTS_DIR / "run_manifest.json").write_text(json.dumps(run_manifest, indent=2, sort_keys=True) + "\n")


def main() -> None:
    validate_dependencies()
    selected = load_selected_specs()
    polish_response_paths = load_polish_response_paths()
    v3 = load_v3_module()
    clean_results_dir()
    work = prepare_work(v3)

    direct_kernels = estimate_direct_debt_kernels(v3, work, selected)
    direct_kernels.to_csv(RESULTS_DIR / "direct_dy_initial_action_paths.csv", index=False, float_format="%.9f")

    program_paths = build_program_paths(polish_response_paths, direct_kernels, selected)
    program_paths.to_csv(RESULTS_DIR / "three_year_program_paths.csv", index=False, float_format="%.9f")

    dsm_inputs = load_dsm_inputs()
    baseline_repro = reproduce_baseline(dsm_inputs)
    baseline_repro.to_csv(RESULTS_DIR / "baseline_reproduction.csv", index=False, float_format="%.9f")

    dsa_paths = simulate_dsa(program_paths, dsm_inputs)
    dsa_paths.to_csv(RESULTS_DIR / "dsa_debt_paths.csv", index=False, float_format="%.9f")

    summary_2036 = make_2036_summary(program_paths, dsa_paths)
    summary_2036.to_csv(RESULTS_DIR / "polish_debt_2036_summary.csv", index=False, float_format="%.9f")

    provenance = input_provenance()
    provenance.to_csv(RESULTS_DIR / "input_provenance.csv", index=False)

    qa = qa_checks(selected, polish_response_paths, direct_kernels, baseline_repro, summary_2036)
    qa.to_csv(RESULTS_DIR / "qa_checks.csv", index=False)

    write_report(direct_kernels, program_paths, dsa_paths, summary_2036, provenance, qa)
    write_manifest()

    if not qa["status"].eq("PASS").all():
        raise SystemExit("QA failed")
    trade_cut = summary_2036[
        summary_2036["spec_id"].eq("investment_import_content") & summary_2036["scenario"].eq("three_1pp_cut_2028_2030")
    ].iloc[0]
    liq_cut = summary_2036[
        summary_2036["spec_id"].eq("household_net_financial_worth") & summary_2036["scenario"].eq("three_1pp_cut_2028_2030")
    ].iloc[0]
    print(
        f"OK debt_accounting_model specs={len(selected)} "
        f"trade_dsa_cut_2036={float(trade_cut.dsa_margin_vs_baseline_pp):.9f} "
        f"liq_direct_cut_2036={float(liq_cut.direct_DY_LP_margin_pp):.9f}"
    )


if __name__ == "__main__":
    main()
