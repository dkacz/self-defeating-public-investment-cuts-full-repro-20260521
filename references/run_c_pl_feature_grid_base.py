#!/usr/bin/env python3
"""Lean feature-interaction helpers used by the state-variable screen.

This module contains the data preparation and diagnostic pieces needed for the
feature-selection screen. It intentionally does not contain later fiscal-outcome
simulation routines.
"""

from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def find_package_root(script_path: Path) -> Path:
    for path in script_path.parents:
        if (path / "code").is_dir() and (path / "data").is_dir() and (path / "references").is_dir():
            return path
    raise RuntimeError(f"Cannot locate package root from {script_path}")


SCRIPT_PATH = Path(__file__).resolve()
TASK_DIR = find_package_root(SCRIPT_PATH)
ROOT = TASK_DIR
DATA_DIR = TASK_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
RESULTS_DIR = TASK_DIR / "results"
REFERENCES_DIR = TASK_DIR / "references"
BASE_RUNNER = REFERENCES_DIR / "run_c_pl_full3_r3_base.py"
V3_CODE = REFERENCES_DIR / "project_context/ciaffi_canonical_estimation_v3_short_rate.py"

SPEC_VERSION = "state_variable_screen_reference_grid_20260510"
LP_SAMPLE_START_YEAR = 2004
LAG_DEPTH = 1
HORIZONS = tuple(range(9))
PRIMARY_PROFILE_YEAR = 2024
FEATURES = ("trade", "debt", "liq", "log_gdp_pc")
FEATURE_Z_COLUMNS = {feature: f"{feature}_z_lag1" for feature in FEATURES}


def load_base_module() -> Any:
    spec = importlib.util.spec_from_file_location("state_variable_output_spending_base", BASE_RUNNER)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {BASE_RUNNER}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    module.ROOT = ROOT
    module.DATA_DIR = DATA_DIR
    module.RESULTS_DIR = RESULTS_DIR
    module.REFERENCES_DIR = REFERENCES_DIR
    module.V3_CODE = V3_CODE
    module.SPEC_VERSION = SPEC_VERSION
    module.LP_SAMPLE_START_YEAR = LP_SAMPLE_START_YEAR
    module.LAG_DEPTH = LAG_DEPTH
    module.HORIZONS = HORIZONS
    return module


def load_v3(base: Any) -> Any:
    return base.load_v3_module()


def prepare_work(v3: Any, panel: pd.DataFrame, feature_panel: pd.DataFrame) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    work = v3.prepare_work(panel)
    work = work.sort_values(["country", "year"]).reset_index(drop=True)
    current_vars = ["dlog_gi", "dlog_gc", "dlog_g", "dlog_y", "i_rate"]
    work.loc[work["year"].lt(LP_SAMPLE_START_YEAR), current_vars] = np.nan
    work, shock_meta = v3.attach_shocks(work, LAG_DEPTH)
    group = work.groupby("country", sort=False)
    for h in HORIZONS:
        gi_tph = group["gi_real"].shift(-h)
        gi_prev_incremental = group["gi_real"].shift(1) if h == 0 else group["gi_real"].shift(-(h - 1))
        work[f"gi_dyn_h{h}"] = (gi_tph - gi_prev_incremental) / work["y_level_lag1"]
    feature_cols = ["country", "year", *[FEATURE_Z_COLUMNS[feature] for feature in FEATURES]]
    work = work.merge(feature_panel[feature_cols], on=["country", "year"], how="left", validate="many_to_one")
    work = work.replace([np.inf, -np.inf], np.nan)
    for feature in FEATURES:
        work[f"shock_G_I_x_{feature}"] = work["shock_G_I"] * work[FEATURE_Z_COLUMNS[feature]]
        work[f"shock_G_C_x_{feature}"] = work["shock_G_C"] * work[FEATURE_Z_COLUMNS[feature]]
    return work, shock_meta


def configure_base_for_spec(base: Any, features: tuple[str, ...], model_id: str) -> None:
    base.FEATURE_NAMES = features
    base.FEATURE_Z_COLUMNS = {feature: f"{feature}_z_lag1" for feature in features}
    base.MODEL_ID = model_id
    base.SPEC_VERSION = SPEC_VERSION


def feature_values(feature_panel: pd.DataFrame, features: tuple[str, ...], profile: str = "poland_2024") -> dict[str, float]:
    if profile != "poland_2024":
        raise ValueError(profile)
    row = feature_panel[(feature_panel["country"].eq("POL")) & (feature_panel["year"].eq(PRIMARY_PROFILE_YEAR))].iloc[0]
    return {feature: float(row[f"{feature}_z"]) for feature in features}


def design_diagnostics_for_spec(base: Any, v3: Any, work: pd.DataFrame, features: tuple[str, ...]) -> dict[str, Any]:
    cols = base.x_columns(False)
    sample = work[work["year"].between(*v3.shock_window(8))].dropna(subset=[*cols, "country", "year"]).copy()
    projector = v3.FEProjector(sample["country"], sample["year"])
    x_res = projector.residualize(sample[cols].to_numpy(dtype=float))
    tol = float(getattr(v3, "LINALG_RANK_TOL", 1e-10))
    svals = np.linalg.svd(x_res, compute_uv=False)
    nonzero = svals[svals > tol]
    condition_number = float(nonzero.max() / nonzero.min()) if len(nonzero) else math.inf
    rank = int(np.linalg.matrix_rank(x_res, tol=tol))
    active_cols = [f"{feature}_z_lag1" for feature in features]
    corr_max = 0.0
    if len(active_cols) > 1:
        corr = sample[active_cols].corr().abs().to_numpy(dtype=float)
        corr_max = float(np.nanmax(corr[np.triu_indices_from(corr, k=1)]))
    return {
        "h8_design_nobs": int(len(sample)),
        "h8_design_rank": rank,
        "h8_regressor_count": len(cols),
        "h8_design_full_rank": rank == len(cols),
        "h8_condition_number": condition_number,
        "max_abs_feature_corr_h8": corr_max,
    }
