#!/usr/bin/env python3
"""Lean output/spending estimator used by the state-variable screen.

This module intentionally contains only output and spending response logic.
The state-variable screen uses it before later debt-accounting calculations.
"""

from __future__ import annotations

import importlib.util
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from statistics import NormalDist
from typing import Any

import numpy as np
import pandas as pd


def find_package_root(script_path: Path) -> Path:
    for path in script_path.parents:
        if (path / "code").is_dir() and (path / "data").is_dir() and (path / "references").is_dir():
            return path
    raise RuntimeError(f"Cannot locate package root from {script_path}")


SCRIPT_PATH = Path(__file__).resolve()
ROOT = find_package_root(SCRIPT_PATH)
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
REFERENCES_DIR = ROOT / "references"
V3_CODE = REFERENCES_DIR / "project_context/ciaffi_canonical_estimation_v3_short_rate.py"

SPEC_VERSION = "state_variable_screen_reference_estimator_20260510"
MODEL_ID = "state_variable_screen_reference_estimator"
RAW_LAG_SUPPORT_START_YEAR = 2001
LP_SAMPLE_START_YEAR = 2004
END_YEAR = 2024
LAG_DEPTH = 1
HORIZONS = tuple(range(9))
FEATURE_NAMES: tuple[str, ...] = ("trade", "debt", "liq", "log_gdp_pc")
FEATURE_Z_COLUMNS = {feature: f"{feature}_z_lag1" for feature in FEATURE_NAMES}
DENOMINATOR_T_THRESHOLD = 1.96
Z95 = NormalDist().inv_cdf(0.975)


@dataclass(frozen=True)
class Fit:
    ratio: float
    se: float
    beta_dep: float
    beta_scale: float
    se_beta_scale: float
    denom_t: float
    nobs: int
    country_n: int
    year_min: int | None
    year_max: int | None
    rank: int
    status: str


def load_v3_module() -> Any:
    spec = importlib.util.spec_from_file_location("ciaffi_v3_for_state_variable_screen", V3_CODE)
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


def x_columns(include_gc_interactions: bool = False) -> list[str]:
    controls = [f"{var}_lag{LAG_DEPTH}" for var in ["dlog_gi", "dlog_gc", "dlog_y", "i_rate"]]
    cols = ["shock_G_I", "shock_G_C"]
    cols += [FEATURE_Z_COLUMNS[feature] for feature in FEATURE_NAMES]
    cols += [f"shock_G_I_x_{feature}" for feature in FEATURE_NAMES]
    if include_gc_interactions:
        cols += [f"shock_G_C_x_{feature}" for feature in FEATURE_NAMES]
    cols += controls
    return cols


def contrast_vector(cols: list[str], z_values: dict[str, float]) -> np.ndarray:
    out = np.zeros(len(cols), dtype=float)
    out[cols.index("shock_G_I")] = 1.0
    for feature in FEATURE_NAMES:
        name = f"shock_G_I_x_{feature}"
        if name in cols:
            out[cols.index(name)] = float(z_values[feature])
    return out


def fit_conditional_ratio(
    v3: Any,
    work: pd.DataFrame,
    dep_col: str,
    scale_col: str,
    cols: list[str],
    z_values: dict[str, float],
    horizon: int,
    exclude_country: str | None = None,
) -> Fit:
    needed = [dep_col, scale_col, *cols, "country", "year"]
    window_start, window_end = v3.shock_window(horizon)
    sample = work[work["year"].between(window_start, window_end)].copy()
    if exclude_country is not None:
        sample = sample[~sample["country"].eq(exclude_country)].copy()
    sample = sample.dropna(subset=needed).sort_values(["country", "year"]).reset_index(drop=True)
    if len(sample) < 50:
        return Fit(math.nan, math.nan, math.nan, math.nan, math.nan, math.nan, len(sample), 0, None, None, 0, "INSUFFICIENT_OBS")
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
    return Fit(
        ratio=float(ratio),
        se=float(se),
        beta_dep=beta_dep_c,
        beta_scale=beta_scale_c,
        se_beta_scale=se_scale,
        denom_t=float(denom_t) if math.isfinite(denom_t) else math.nan,
        nobs=int(len(sample)),
        country_n=int(sample["country"].nunique()),
        year_min=int(sample["year"].min()),
        year_max=int(sample["year"].max()),
        rank=int(rank),
        status=status,
    )
