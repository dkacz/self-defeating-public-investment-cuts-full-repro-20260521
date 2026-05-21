#!/usr/bin/env python3
"""Recompute the default manuscript specification from frozen inputs.

Frozen adopted run outputs are validation targets only. This script rebuilds
the state-variable model input, reruns the feature screen, estimates retained
output/spending paths, computes debt-accounting paths, and writes regression
output tables for public inspection.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

THREAD_ENV = {
    "OPENBLAS_NUM_THREADS": "1",
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
}

if os.environ.get("FULL_REPRO_THREADS_LOCKED") != "1":
    env = os.environ.copy()
    env.update(THREAD_ENV)
    env["FULL_REPRO_THREADS_LOCKED"] = "1"
    os.execve(sys.executable, [sys.executable, *sys.argv], env)

import numpy as np
import pandas as pd

try:
    from scipy.stats import chi2, norm
except ModuleNotFoundError:  # Pyodide/JupyterLite may ship without SciPy.
    class _NormFallback:
        @staticmethod
        def sf(x: float) -> float:
            return 0.5 * math.erfc(float(x) / math.sqrt(2.0))

    def _regularized_gamma_q(a: float, x: float) -> float:
        if x < 0.0 or a <= 0.0:
            return math.nan
        if x == 0.0:
            return 1.0
        gln = math.lgamma(a)
        if x < a + 1.0:
            ap = a
            term = 1.0 / a
            total = term
            for _ in range(200):
                ap += 1.0
                term *= x / ap
                total += term
                if abs(term) < abs(total) * 3.0e-14:
                    break
            p = total * math.exp(-x + a * math.log(x) - gln)
            return max(0.0, min(1.0, 1.0 - p))
        b = x + 1.0 - a
        c = 1.0 / 1.0e-300
        d = 1.0 / b
        h = d
        for i in range(1, 201):
            an = -i * (i - a)
            b += 2.0
            d = an * d + b
            if abs(d) < 1.0e-300:
                d = 1.0e-300
            c = b + an / c
            if abs(c) < 1.0e-300:
                c = 1.0e-300
            d = 1.0 / d
            delta = d * c
            h *= delta
            if abs(delta - 1.0) < 3.0e-14:
                break
        q = math.exp(-x + a * math.log(x) - gln) * h
        return max(0.0, min(1.0, q))

    class _Chi2Fallback:
        @staticmethod
        def sf(x: float, df: int | float) -> float:
            return _regularized_gamma_q(float(df) / 2.0, float(x) / 2.0)

    chi2 = _Chi2Fallback()
    norm = _NormFallback()


ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / "code"
ESTIMATION_CODE = CODE / "estimation"
REFERENCES = ROOT / "references"
FROZEN_INPUTS = ROOT / "data/frozen/adopted_model_inputs"
FROZEN_SOURCES = ROOT / "data/frozen/adopted_sources"
FROZEN_OUTPUTS = ROOT / "data/frozen/adopted_run_outputs"
FROZEN_EU27_DEBT = ROOT / "data/frozen/eu27_benchmark_debt"
RECOMPUTED = ROOT / "results/recomputed"
WORK_DATA = RECOMPUTED / "model_inputs"
ESTIMATION_OUTPUT = RECOMPUTED / "estimation_output"
QA = ROOT / "qa"

EU27 = [
    "AUT", "BEL", "BGR", "CYP", "CZE", "DEU", "DNK", "ESP", "EST",
    "FIN", "FRA", "GRC", "HRV", "HUN", "IRL", "ITA", "LTU", "LUX",
    "LVA", "MLT", "NLD", "POL", "PRT", "ROU", "SVK", "SVN", "SWE",
]
FEATURES = ("trade", "debt", "liq", "log_gdp_pc")
HORIZONS = tuple(range(9))
LP_SAMPLE_START_YEAR = 2004
SAMPLE_END_YEAR = 2022
PROFILE_YEAR = 2022
VALIDATION_MODE = "benchmark"
VARIANT_STEM = "tiva2022_gfcf_realppp_networth"
RUNTIME_MODEL_INPUT_README = """# Recomputed Runtime Model Inputs

This folder is recreated by the public estimator during each run.

It contains the model-input panel and transformation table used by the downstream
estimation scripts after the source-rebuild step. The source-rebuild QA compares
these runtime files with the immutable validation bundle shipped with the release,
but this directory itself is recomputed output and can be deleted before the next
run.
"""


def progress(message: str) -> None:
    print(f"[full-estimator-repro] {message}", flush=True)


def artifact_name(*parts: str) -> str:
    return "".join(parts)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_frame(frame: pd.DataFrame, path: Path, *, float_format: str = "%.9f") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, float_format=float_format)


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def load_adopted_specification_targets() -> pd.DataFrame:
    target = pd.read_csv(FROZEN_OUTPUTS / "feature_screen/feature_robustness_summary.csv")
    selected = target.loc[
        target["gate_status"].eq("PASS_ROBUSTNESS_GATE"),
        ["spec_id", "features"],
    ].copy()
    return selected.sort_values("spec_id").reset_index(drop=True)


def import_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def source_frame(name: str, value_col: str) -> pd.DataFrame:
    frame = pd.read_csv(FROZEN_SOURCES / name)
    return frame[["country", "year", value_col]].copy()


def zscore(frame: pd.DataFrame, raw_col: str, z_col: str) -> dict[str, Any]:
    sample = frame.loc[frame["year"].between(LP_SAMPLE_START_YEAR, SAMPLE_END_YEAR), raw_col].dropna()
    mean = float(sample.mean())
    sd = float(sample.std(ddof=0))
    if not math.isfinite(sd) or sd <= 0:
        raise RuntimeError(f"Bad standard deviation for {raw_col}")
    frame[z_col] = (frame[raw_col] - mean) / sd
    return {
        "raw_column": raw_col,
        "z_column": z_col,
        "sample_start": LP_SAMPLE_START_YEAR,
        "sample_end": SAMPLE_END_YEAR,
        "mean": mean,
        "sd": sd,
        "nobs": int(sample.size),
    }


def build_adopted_state_panel_from_sources() -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = source_frame("nominal_gdp.csv", "nominal_gdp_mio_eur")
    for filename, value in [
        ("gdp_pc_current_pps.csv", "gdp_pc_current_pps"),
        ("gdp_pc_real_index_2020.csv", "gdp_pc_real_index_2020"),
        ("hh_total_financial_assets.csv", "hh_total_financial_assets_mio_eur"),
        ("hh_total_financial_liabilities.csv", "hh_total_financial_liabilities_mio_eur"),
    ]:
        frame = frame.merge(source_frame(filename, value), on=["country", "year"], how="outer", validate="one_to_one")

    pps_2020 = frame.loc[frame["year"].eq(2020), ["country", "gdp_pc_current_pps"]].rename(
        columns={"gdp_pc_current_pps": "gdp_pc_2020_pps_anchor"}
    )
    frame = frame.merge(pps_2020, on="country", how="left", validate="many_to_one")
    frame["real_ppp_gdp_pc_2020pps"] = frame["gdp_pc_2020_pps_anchor"] * frame["gdp_pc_real_index_2020"] / 100.0
    frame["log_real_ppp_gdp_pc_raw"] = np.where(
        frame["real_ppp_gdp_pc_2020pps"].gt(0), np.log(frame["real_ppp_gdp_pc_2020pps"]), np.nan
    )
    frame["hh_net_financial_worth_to_gdp"] = (
        frame["hh_total_financial_assets_mio_eur"] - frame["hh_total_financial_liabilities_mio_eur"]
    ) / frame["nominal_gdp_mio_eur"]
    frame["liq_fa_net_worth_raw"] = -frame["hh_net_financial_worth_to_gdp"]

    tiva = pd.read_csv(FROZEN_SOURCES / "oecd_tiva_import_content_gfcf_cons_1995_2022.csv")
    gfcf = tiva[tiva["measure"].eq("GFCF_VA_SH")][["country", "year", "import_content_share"]].rename(
        columns={"import_content_share": "tiva2022_gfcf_import_content_raw"}
    )
    frame = frame.merge(gfcf, on=["country", "year"], how="left", validate="one_to_one")
    frame = frame[frame["country"].isin(EU27)].sort_values(["country", "year"]).reset_index(drop=True)

    frame["trade_raw"] = frame["tiva2022_gfcf_import_content_raw"]
    frame["liq_raw"] = frame["liq_fa_net_worth_raw"]
    frame["log_gdp_pc_raw"] = frame["log_real_ppp_gdp_pc_raw"]
    transforms = pd.DataFrame(
        [
            zscore(frame, "log_gdp_pc_raw", "log_gdp_pc_z"),
            zscore(frame, "trade_raw", "trade_z"),
            zscore(frame, "liq_raw", "liq_z"),
        ]
    )
    for col in ["trade_raw", "trade_z", "liq_raw", "liq_z", "log_gdp_pc_raw", "log_gdp_pc_z"]:
        frame.loc[frame["year"].gt(SAMPLE_END_YEAR), col] = np.nan
    return frame, transforms


def build_runtime_model_inputs() -> pd.DataFrame:
    if RECOMPUTED.exists():
        shutil.rmtree(RECOMPUTED)
    WORK_DATA.mkdir(parents=True, exist_ok=True)
    for path in FROZEN_INPUTS.iterdir():
        if path.is_file():
            shutil.copy2(path, WORK_DATA / path.name)
    (WORK_DATA / "README.md").write_text(RUNTIME_MODEL_INPUT_README, encoding="utf-8")

    base = pd.read_csv(FROZEN_INPUTS / "country_feature_panel.csv")
    source_panel, transforms = build_adopted_state_panel_from_sources()
    replacement_cols = [
        "nominal_gdp_mio_eur",
        "gdp_pc_current_pps",
        "gdp_pc_real_index_2020",
        "hh_total_financial_assets_mio_eur",
        "hh_total_financial_liabilities_mio_eur",
        "gdp_pc_2020_pps_anchor",
        "real_ppp_gdp_pc_2020pps",
        "log_real_ppp_gdp_pc_raw",
        "hh_net_financial_worth_to_gdp",
        "liq_fa_net_worth_raw",
        "tiva2022_gfcf_import_content_raw",
        "trade_raw",
        "trade_z",
        "liq_raw",
        "liq_z",
        "log_gdp_pc_raw",
        "log_gdp_pc_z",
    ]
    drop_cols = [col for col in replacement_cols if col in base.columns]
    rebuilt = base.drop(columns=drop_cols).merge(
        source_panel[["country", "year", *replacement_cols]],
        on=["country", "year"],
        how="left",
        validate="one_to_one",
    )
    rebuilt = rebuilt.sort_values(["country", "year"]).reset_index(drop=True)
    group = rebuilt.groupby("country", sort=False)
    for col in ["trade_z", "liq_z", "log_gdp_pc_z"]:
        rebuilt[f"{col}_lag1"] = group[col].shift(1)

    write_frame(rebuilt, WORK_DATA / "country_feature_panel.csv", float_format="%.12g")
    write_frame(transforms, WORK_DATA / "variant_transformations.csv", float_format="%.12g")

    frozen = pd.read_csv(FROZEN_INPUTS / "country_feature_panel.csv")
    compare_cols = [
        "trade_raw", "trade_z", "trade_z_lag1",
        "liq_raw", "liq_z", "liq_z_lag1",
        "log_gdp_pc_raw", "log_gdp_pc_z", "log_gdp_pc_z_lag1",
    ]
    qa_rows: list[dict[str, Any]] = []
    for col in compare_cols:
        merged = rebuilt[["country", "year", col]].merge(
            frozen[["country", "year", col]],
            on=["country", "year"],
            how="outer",
            suffixes=("_rebuilt", "_frozen"),
            validate="one_to_one",
        )
        a = merged[f"{col}_rebuilt"].to_numpy(dtype=float)
        b = merged[f"{col}_frozen"].to_numpy(dtype=float)
        mask = ~(np.isnan(a) & np.isnan(b))
        max_abs = float(np.nanmax(np.abs(a[mask] - b[mask]))) if mask.any() else 0.0
        same_missing = bool(np.array_equal(np.isnan(a), np.isnan(b)))
        ok = same_missing and (max_abs <= 5e-9)
        qa_rows.append(
            {
                "check": f"source_rebuilt_model_input_matches_frozen:{col}",
                "status": "PASS" if ok else "FAIL",
                "detail": f"max_abs_diff={max_abs:.3e}; same_missing={same_missing}",
            }
        )
    write_frame(pd.DataFrame(qa_rows), QA / "full_estimator_model_input_rebuild_qa.csv")
    if any(row["status"] != "PASS" for row in qa_rows):
        raise SystemExit("Source-rebuilt model input does not match the frozen benchmark model input")
    return rebuilt


def patch_support_checks(mod: Any) -> None:
    def support_checks(feature_panel: pd.DataFrame, catalog: pd.DataFrame) -> pd.DataFrame:
        rows: list[dict[str, Any]] = []
        sample = feature_panel[feature_panel["year"].between(LP_SAMPLE_START_YEAR, SAMPLE_END_YEAR)].copy()
        pol_rows = feature_panel[(feature_panel["country"].eq("POL")) & (feature_panel["year"].eq(PROFILE_YEAR))]
        if pol_rows.empty:
            raise RuntimeError(f"Missing POL profile year {PROFILE_YEAR}")
        pol = pol_rows.iloc[0]
        sample = sample[~(sample["country"].eq("POL") & sample["year"].eq(PROFILE_YEAR))].copy()
        for features_text in catalog["features"]:
            features = tuple(str(features_text).split("+"))
            z_cols = [f"{feature}_z" for feature in features]
            use = sample.dropna(subset=z_cols).copy()
            pol_vec = pol[z_cols].to_numpy(dtype=float)
            if use.empty or not np.isfinite(pol_vec).all():
                rows.append(
                    {
                        "features": features_text,
                        "max_abs_poland_z": math.inf,
                        "mean_abs_poland_z": math.inf,
                        "mahalanobis2": math.inf,
                        "mahalanobis_support_p": 0.0,
                        "nearest_country_years_dist_le_1": 0,
                        "nearest_country_years_dist_le_1_5": 0,
                        "nearest_unique_countries_dist_le_1_5": 0,
                        "min_distance_to_pol": math.inf,
                        "support_reference_excludes": f"POL_{PROFILE_YEAR}_target",
                    }
                )
                continue
            x = use[z_cols].to_numpy(dtype=float)
            mean = np.mean(x, axis=0)
            cov = np.atleast_2d(np.cov(x, rowvar=False, ddof=0))
            diff = pol_vec - mean
            mahal2 = float(diff.T @ np.linalg.pinv(cov) @ diff)
            distances = np.sqrt(np.sum((x - pol_vec) ** 2, axis=1))
            tmp = use.assign(distance_to_pol=distances)
            rows.append(
                {
                    "features": features_text,
                    "max_abs_poland_z": float(np.max(np.abs(pol_vec))),
                    "mean_abs_poland_z": float(np.mean(np.abs(pol_vec))),
                    "mahalanobis2": mahal2,
                    "mahalanobis_support_p": float(chi2.sf(mahal2, len(features))),
                    "nearest_country_years_dist_le_1": int((tmp["distance_to_pol"] <= 1.0).sum()),
                    "nearest_country_years_dist_le_1_5": int((tmp["distance_to_pol"] <= 1.5).sum()),
                    "nearest_unique_countries_dist_le_1_5": int(tmp.loc[tmp["distance_to_pol"] <= 1.5, "country"].nunique()),
                    "min_distance_to_pol": float(tmp["distance_to_pol"].min()),
                    "support_reference_excludes": f"POL_{PROFILE_YEAR}_target",
                }
            )
        return pd.DataFrame(rows)

    mod.support_checks = support_checks


def patch_grid_profile(mod: Any) -> None:
    original_load = mod.load_grid_module

    def load_grid_module() -> Any:
        grid = original_load()
        grid.PRIMARY_PROFILE_YEAR = PROFILE_YEAR
        return grid

    mod.load_grid_module = load_grid_module


def run_feature_screen() -> tuple[Any, pd.DataFrame]:
    mod = import_module(ESTIMATION_CODE / "feature_screen_model.py", "public_feature_screen_model")
    mod.DATA_DIR = WORK_DATA
    mod.RESULTS_DIR = RECOMPUTED / "feature_screen"
    mod.REFERENCES_DIR = REFERENCES
    mod.GRID_CODE = REFERENCES / "run_c_pl_feature_grid_base.py"
    mod.ROOT = ROOT
    mod.TASK_DIR = ROOT
    mod.SPEC_VERSION = f"mozdzen_{VARIANT_STEM}_feature_screen_20260513"
    patch_support_checks(mod)
    patch_grid_profile(mod)
    mod.main()
    screen = pd.read_csv(mod.RESULTS_DIR / artifact_name("feature", "_robustness", "_summary", ".csv"))
    return mod, screen


def run_polish_output(selected: pd.DataFrame) -> pd.DataFrame:
    out_dir = RECOMPUTED / "polish_output_spending"
    out_dir.mkdir(parents=True, exist_ok=True)
    mod = import_module(ESTIMATION_CODE / "polish_output_spending_model.py", "public_polish_output_spending_model")
    mod.DATA_DIR = WORK_DATA
    mod.RESULTS_DIR = out_dir
    mod.REFERENCES_DIR = REFERENCES
    mod.GRID_CODE = REFERENCES / "run_c_pl_feature_grid_base.py"
    mod.ROOT = ROOT
    mod.SPEC_VERSION = f"mozdzen_{VARIANT_STEM}_polish_output_20260513"
    patch_grid_profile(mod)
    grid_mod = mod.load_grid_module()
    base, v3, work = mod.load_work(grid_mod)
    paths = []
    for row in selected.itertuples(index=False):
        paths.append(mod.estimate_kernels(grid_mod, base, v3, work, str(row.spec_id), str(row.features)))
    paths_df = pd.concat(paths, ignore_index=True).sort_values(["spec_id", "horizon"]).reset_index(drop=True)
    paths_df["profile_year_used"] = PROFILE_YEAR
    write_frame(paths_df, out_dir / "polish_output_spending_paths.csv")
    write_frame(mod.make_summary(paths_df), out_dir / "polish_output_spending_h8_summary.csv")
    qa = pd.DataFrame(
        [
            {
                "check": "all_horizons_present",
                "status": "PASS" if paths_df.groupby("spec_id")["horizon"].nunique().min() == len(HORIZONS) else "FAIL",
                "detail": "h0-h8 per admitted spec",
            },
            {
                "check": "all_regressions_ok",
                "status": "PASS" if paths_df["status_Y"].eq("OK").all() and paths_df["status_G"].eq("OK").all() else "FAIL",
                "detail": "status_Y/status_G",
            },
        ]
    )
    write_frame(qa, out_dir / "qa_checks.csv")
    return paths_df


def run_debt(selected: pd.DataFrame, polish_paths: pd.DataFrame) -> pd.DataFrame:
    out_dir = RECOMPUTED / "debt_accounting"
    out_dir.mkdir(parents=True, exist_ok=True)
    mod = import_module(ESTIMATION_CODE / "debt_accounting_model.py", "public_debt_accounting_model")
    mod.DATA_DIR = WORK_DATA
    mod.RESULTS_DIR = out_dir
    mod.REFERENCES_DIR = REFERENCES
    mod.V3_CODE = REFERENCES / "project_context/ciaffi_canonical_estimation_v3_short_rate.py"
    mod.ROOT = ROOT
    mod.SPEC_VERSION = f"mozdzen_{VARIANT_STEM}_debt_20260513"
    mod.PANEL_END_YEAR = PROFILE_YEAR
    v3 = mod.load_v3_module()
    work = mod.prepare_work(v3)
    direct = mod.estimate_direct_debt_kernels(v3, work, selected)
    program = mod.build_program_paths(polish_paths, direct, selected)
    dsm_inputs = mod.load_dsm_inputs()
    baseline = mod.reproduce_baseline(dsm_inputs)
    dsa = mod.simulate_dsa(program, dsm_inputs)
    summary = mod.make_2036_summary(program, dsa)
    write_frame(direct, out_dir / "direct_dy_initial_action_paths.csv")
    write_frame(program, out_dir / "three_year_program_paths.csv")
    write_frame(baseline, out_dir / "baseline_reproduction.csv")
    write_frame(dsa, out_dir / "dsa_debt_paths.csv")
    write_frame(summary, out_dir / "polish_debt_2036_summary.csv")
    qa = pd.DataFrame(
        [
            {"check": "debt_rows_present", "status": "PASS" if len(dsa) > 0 else "FAIL", "detail": f"rows={len(dsa)}"},
            {
                "check": "baseline_reproduction_within_tolerance",
                "status": "PASS" if baseline["abs_diff_pp"].max() <= mod.DEBT_BASELINE_REPRO_TOL_PP else "FAIL",
                "detail": f"max_abs_diff={baseline['abs_diff_pp'].max():.3e}",
            },
        ]
    )
    write_frame(qa, out_dir / "qa_checks.csv")
    return summary


def fmt2(value: float, *, signed: bool = False) -> str:
    if signed:
        return f"{float(value):+.2f}"
    return f"{float(value):.2f}"


def estimate_eu27_direct_debt_kernels(debt_mod: Any, v3: Any, work: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for h in HORIZONS:
        fit = debt_mod.fit_direct_dy_ratio(v3, work, tuple(), {}, h)
        rows.append(
            {
                "spec_version": debt_mod.SPEC_VERSION,
                "spec_id": "EU27",
                "features": "linear",
                "profile_label": "EU27_panel_benchmark",
                "horizon": h,
                **fit,
            }
        )
    return pd.DataFrame(rows).sort_values("horizon").reset_index(drop=True)


def build_single_program_paths(debt_mod: Any, response_paths: pd.DataFrame, direct_kernels: pd.DataFrame) -> pd.DataFrame:
    response = response_paths.sort_values("horizon").reset_index(drop=True)
    direct = direct_kernels.sort_values("horizon").reset_index(drop=True)
    k_y = response["K_Y_cumulative"].to_numpy(dtype=float)
    k_g = response["K_G_cumulative"].to_numpy(dtype=float)
    dy_initial = direct["direct_DY_initial_action"].to_numpy(dtype=float)
    rows: list[dict[str, Any]] = []
    for scenario in debt_mod.scenario_definitions():
        actions = np.asarray(scenario["actions"], dtype=float)
        y_shortfall = debt_mod.convolve_path(actions, k_y)
        direct_pb = debt_mod.convolve_path(actions, k_g)
        direct_dy_margin = -debt_mod.convolve_path(actions, dy_initial)
        for h in HORIZONS:
            rows.append(
                {
                    "spec_version": debt_mod.SPEC_VERSION,
                    "spec_id": "EU27",
                    "features": "linear",
                    "scenario": scenario["scenario"],
                    "scenario_sign": scenario["scenario_sign"],
                    "year": debt_mod.ACTION_START_YEAR + h,
                    "horizon_from_2028": h,
                    "fiscal_action_cut_units_pp": actions[h],
                    "Y_shortfall_pct": y_shortfall[h],
                    "direct_discretionary_PB_level_pp": direct_pb[h],
                    "direct_DY_LP_margin_initial_action_pp": direct_dy_margin[h],
                    "description": scenario["description"],
                }
            )
    return pd.DataFrame(rows)


def make_eu27_endpoint(summary: pd.DataFrame) -> pd.DataFrame:
    expansion = summary[summary["scenario_sign"].eq("expansion")].iloc[0]
    cut = summary[summary["scenario_sign"].eq("cut")].iloc[0]
    return pd.DataFrame(
        [
            {
                "empirical_path": "EU27 panel benchmark",
                "expansion_institutional_debt_equation": float(expansion["dsa_margin_vs_baseline_pp"]),
                "expansion_direct_debt_to_gdp_lp_path": float(expansion["direct_DY_LP_margin_pp"]),
                "cut_institutional_debt_equation": float(cut["dsa_margin_vs_baseline_pp"]),
                "cut_direct_debt_to_gdp_lp_path": float(cut["direct_DY_LP_margin_pp"]),
                "source_artifact": "recomputed from frozen model inputs by code/run_full_estimator_repro.py",
            }
        ]
    )


def make_eu27_annual_decomposition(dsa_paths: pd.DataFrame, dsm_inputs: pd.DataFrame) -> pd.DataFrame:
    baseline_cols = [
        "year",
        "interest_expenditure",
        "growth_effect_real",
        "inflation_effect",
        "stock_flow_adjustments",
    ]
    baseline = dsm_inputs[baseline_cols].copy()
    baseline["baseline_snowball_term_pp"] = (
        baseline["interest_expenditure"] + baseline["growth_effect_real"] + baseline["inflation_effect"]
    )
    debt_paths = dsa_paths.merge(
        baseline[["year", "baseline_snowball_term_pp", "stock_flow_adjustments"]],
        on="year",
        how="left",
        validate="many_to_one",
    )
    debt_paths = debt_paths.sort_values(["scenario_sign", "year"]).copy()
    debt_paths["prev_D_Y_new_pp"] = debt_paths.groupby("scenario_sign")["D_Y_new_pp"].shift(1)
    debt_paths["scenario_snowball_term_pp"] = (
        debt_paths["D_Y_new_pp"]
        - debt_paths["prev_D_Y_new_pp"]
        + debt_paths["PB_new_pp"]
        - debt_paths["stock_flow_adjustments"].fillna(0.0)
    )
    keep = debt_paths[debt_paths["year"].between(2028, 2036)].copy()
    action_order = {"expansion": 0, "cut": 1}
    keep["action_order"] = keep["scenario_sign"].map(action_order)
    keep = keep.sort_values(["action_order", "year"]).reset_index(drop=True)
    actions = {"expansion": "Expansion", "cut": "Cut"}
    return pd.DataFrame(
        {
            "Empirical path": "EU27 panel benchmark",
            "Action": keep["scenario_sign"].map(actions),
            "Year": keep["year"].astype(int),
            "Baseline debt ratio": keep["baseline_D_Y_pp"].map(fmt2),
            "Scenario debt ratio": keep["D_Y_new_pp"].map(fmt2),
            "Debt margin": keep["dsa_margin_vs_baseline_pp"].map(fmt2),
            "Output effect, GDP level": keep["Y_shortfall_pct"].map(fmt2),
            "Direct primary-balance effect": keep["direct_discretionary_PB_level_pp"].map(fmt2),
            "Cyclical primary-balance feedback": keep["delta_cyclical_PB_pp"].map(fmt2),
            "Baseline primary balance": keep["baseline_PB_pp"].map(fmt2),
            "Scenario primary balance": keep["PB_new_pp"].map(fmt2),
            "Scenario nominal GDP growth": keep["nominal_gdp_growth_new_pct"].map(fmt2),
            "Snowball term": keep["scenario_snowball_term_pp"].map(fmt2),
            "Stock-flow adjustment": keep["stock_flow_adjustments"].map(fmt2),
            "Institutional debt margin": keep["dsa_margin_vs_baseline_pp"].map(fmt2),
        }
    )


def run_eu27_debt_benchmark(eu27_paths: pd.DataFrame) -> None:
    out_dir = RECOMPUTED / "eu27_benchmark"
    debt_mod = import_module(ESTIMATION_CODE / "debt_accounting_model.py", "public_eu27_debt_accounting_model")
    debt_mod.DATA_DIR = WORK_DATA
    debt_mod.RESULTS_DIR = out_dir
    debt_mod.REFERENCES_DIR = REFERENCES
    debt_mod.V3_CODE = REFERENCES / "project_context/ciaffi_canonical_estimation_v3_short_rate.py"
    debt_mod.ROOT = ROOT
    debt_mod.SPEC_VERSION = f"mozdzen_{VARIANT_STEM}_eu27_debt_20260520"
    debt_mod.PANEL_END_YEAR = PROFILE_YEAR
    v3 = debt_mod.load_v3_module()
    work = debt_mod.prepare_work(v3)
    direct = estimate_eu27_direct_debt_kernels(debt_mod, v3, work)
    program = build_single_program_paths(debt_mod, eu27_paths, direct)
    dsm_inputs = debt_mod.load_dsm_inputs()
    dsa = debt_mod.simulate_dsa(program, dsm_inputs)
    summary = debt_mod.make_2036_summary(program, dsa)
    endpoint = make_eu27_endpoint(summary)
    annual = make_eu27_annual_decomposition(dsa, dsm_inputs)
    write_frame(direct, out_dir / "eu27_direct_dy_initial_action_paths.csv")
    write_frame(program, out_dir / "eu27_three_year_program_paths.csv")
    write_frame(dsa, out_dir / "eu27_dsa_debt_paths.csv")
    write_frame(summary, out_dir / "eu27_debt_2036_summary.csv")
    write_frame(endpoint, out_dir / "eu27_benchmark_debt_2036.csv")
    write_frame(annual, out_dir / "eu27_benchmark_annual_debt_decomposition.csv")


def coefficient_rows(
    feature_mod: Any,
    grid_mod: Any,
    base: Any,
    v3: Any,
    work: pd.DataFrame,
    spec_id: str,
    features_text: str,
    outcome: str,
) -> list[dict[str, Any]]:
    features = tuple(part for part in str(features_text).split("+") if part)
    feature_mod.configure(grid_mod, base, features, spec_id)
    cols = base.x_columns(False)
    rows: list[dict[str, Any]] = []
    dep_prefix = "y_dyn_h" if outcome == "output" else "gi_dyn_h"
    for h in HORIZONS:
        dep_col = f"{dep_prefix}{h}"
        sample = work[work["year"].between(*v3.shock_window(h))].copy()
        needed = [dep_col, *cols, "country", "year"]
        used = sample.dropna(subset=needed).sort_values(["country", "year"]).reset_index(drop=True)
        fit = feature_mod.regression_fit(v3, sample, dep_col, cols)
        if fit.get("status") != "OK":
            rows.append(
                {
                    "path": "Poland state-dependent evaluation",
                    "spec_id": spec_id,
                    "features": features_text,
                    "outcome": outcome,
                    "horizon": h,
                    "term": "model",
                    "term_role": "model_fit",
                    "coefficient": math.nan,
                    "std_error": math.nan,
                    "z_stat": math.nan,
                    "p_value": math.nan,
                    "nobs": fit.get("nobs", len(used)),
                    "country_n": int(used["country"].nunique()) if len(used) else 0,
                    "year_min": int(used["year"].min()) if len(used) else None,
                    "year_max": int(used["year"].max()) if len(used) else None,
                    "design_rank": fit.get("rank", 0),
                    "regressor_count": len(cols),
                    "status": fit.get("status"),
                }
            )
            continue
        beta = np.asarray(fit["beta"])
        cov = np.asarray(fit["cov"])
        for idx, term in enumerate(cols):
            se = math.sqrt(max(float(cov[idx, idx]), 0.0))
            coef = float(beta[idx])
            z = coef / se if se > 0 else math.nan
            if term == "shock_G_I":
                role = "beta_h: mean-state public-investment shock coefficient"
            elif term.startswith("shock_G_I_x_"):
                role = "theta_h: public-investment shock interaction"
            elif term.endswith("_z_lag1"):
                role = "delta_h: state main effect"
            elif term == "shock_G_C":
                role = "public-consumption shock control"
            else:
                role = "lagged control"
            rows.append(
                {
                    "path": "Poland state-dependent evaluation",
                    "spec_id": spec_id,
                    "features": features_text,
                    "outcome": outcome,
                    "horizon": h,
                    "term": term,
                    "term_role": role,
                    "coefficient": coef,
                    "std_error": se,
                    "z_stat": z,
                    "p_value": 2.0 * norm.sf(abs(z)) if math.isfinite(z) else math.nan,
                    "nobs": int(len(used)),
                    "country_n": int(used["country"].nunique()),
                    "year_min": int(used["year"].min()),
                    "year_max": int(used["year"].max()),
                    "design_rank": int(fit["rank"]),
                    "regressor_count": len(cols),
                    "status": fit["status"],
                }
            )
    return rows


def estimate_eu27_benchmark(feature_mod: Any, grid_mod: Any, base: Any, v3: Any, work: pd.DataFrame) -> pd.DataFrame:
    feature_mod.configure(grid_mod, base, tuple(), "EU27_LINEAR")
    cols = base.x_columns(False)
    rows: list[dict[str, Any]] = []
    coef_rows: list[dict[str, Any]] = []
    for h in HORIZONS:
        fit_y = base.fit_conditional_ratio(v3, work, f"y_dyn_h{h}", "gi_dyn0", cols, {}, h)
        fit_g = base.fit_conditional_ratio(v3, work, f"gi_dyn_h{h}", "gi_dyn0", cols, {}, h)
        rows.append(
            {
                "path": "EU27 panel benchmark",
                "spec_id": "EU27",
                "features": "linear",
                "horizon": h,
                "mu_Y_incremental": fit_y.ratio,
                "se_Y_incremental": fit_y.se,
                "mu_G_incremental": fit_g.ratio,
                "se_G_incremental": fit_g.se,
                "nobs": fit_y.nobs,
                "country_n": fit_y.country_n,
                "year_min_effective": fit_y.year_min,
                "year_max_effective": fit_y.year_max,
                "rank_X": fit_y.rank,
                "status_Y": fit_y.status,
                "status_G": fit_g.status,
            }
        )
    out = pd.DataFrame(rows)
    out["K_Y_cumulative"] = out["mu_Y_incremental"].cumsum()
    out["K_G_cumulative"] = out["mu_G_incremental"].cumsum()
    out["cumulative_output_to_spending_ratio"] = out["K_Y_cumulative"] / out["K_G_cumulative"]
    write_frame(out, RECOMPUTED / "eu27_benchmark/eu27_output_spending_paths.csv")

    for outcome in ["output", "spending"]:
        coef_rows.extend(coefficient_rows(feature_mod, grid_mod, base, v3, work, "EU27", "", outcome))
    # The generic coefficient helper labels these as state-dependent; fix labels for linear benchmark.
    coef = pd.DataFrame(coef_rows)
    if not coef.empty:
        coef["path"] = "EU27 panel benchmark"
        coef["features"] = "linear"
        coef["term_role"] = coef["term"].map(
            lambda term: "beta_h: common public-investment shock coefficient"
            if term == "shock_G_I"
            else ("public-consumption shock control" if term == "shock_G_C" else "lagged control")
        )
        write_frame(coef, ESTIMATION_OUTPUT / "eu27_benchmark_regression_coefficients.csv")
    return out


def make_estimation_output_tables(feature_mod: Any, screen: pd.DataFrame, polish_paths: pd.DataFrame) -> None:
    grid_mod = feature_mod.load_grid_module()
    base = grid_mod.load_base_module()
    v3, work, _feature_panel = feature_mod.load_work(grid_mod, base)
    selected = screen.loc[screen["gate_status"].eq("PASS_ROBUSTNESS_GATE"), ["spec_id", "features"]].copy()
    rows: list[dict[str, Any]] = []
    for row in selected.itertuples(index=False):
        for outcome in ["output", "spending"]:
            rows.extend(coefficient_rows(feature_mod, grid_mod, base, v3, work, str(row.spec_id), str(row.features), outcome))
    write_frame(pd.DataFrame(rows), ESTIMATION_OUTPUT / "retained_regression_coefficients.csv")

    retained = polish_paths.copy()
    retained["path"] = "Poland state-dependent evaluation"
    retained["fixed_effects"] = "country and year"
    retained["inference"] = "Driscoll-Kraay covariance by horizon"
    retained["lag_depth"] = 1
    write_frame(retained, ESTIMATION_OUTPUT / "retained_horizon_response_output.csv")
    eu27_paths = estimate_eu27_benchmark(feature_mod, grid_mod, base, v3, work)
    run_eu27_debt_benchmark(eu27_paths)


def compare_frames(label: str, recomputed_path: Path, frozen_path: Path, tolerance: float = 5e-7) -> dict[str, Any]:
    if not recomputed_path.exists() or not frozen_path.exists():
        return {
            "check": label,
            "status": "FAIL",
            "detail": f"missing recomputed={recomputed_path.exists()} frozen={frozen_path.exists()}",
        }
    got = pd.read_csv(recomputed_path)
    exp = pd.read_csv(frozen_path)
    if got.shape != exp.shape:
        return {"check": label, "status": "FAIL", "detail": f"shape recomputed={got.shape} frozen={exp.shape}"}
    common = [col for col in exp.columns if col in got.columns]
    sort_cols = [
        col
        for col in [
            "spec_id",
            "features",
            "path",
            "scenario",
            "outcome",
            "term",
            "horizon",
            "draw_id",
            "country",
            "block_name",
            "year",
        ]
        if col in common
    ]
    if sort_cols:
        got = got.sort_values(sort_cols).reset_index(drop=True)
        exp = exp.sort_values(sort_cols).reset_index(drop=True)
    ignored = {"created_utc", "git_head", "git_status_note", "source_artifact"}
    numeric_diffs: list[float] = []
    bad_cols: list[str] = []
    for col in common:
        if col in ignored:
            continue
        if pd.api.types.is_numeric_dtype(exp[col]) or pd.api.types.is_numeric_dtype(got[col]):
            a = pd.to_numeric(got[col], errors="coerce").to_numpy(dtype=float)
            b = pd.to_numeric(exp[col], errors="coerce").to_numpy(dtype=float)
            same_missing = np.array_equal(np.isnan(a), np.isnan(b))
            mask = ~(np.isnan(a) & np.isnan(b))
            max_abs = float(np.nanmax(np.abs(a[mask] - b[mask]))) if mask.any() else 0.0
            numeric_diffs.append(max_abs)
            if (not same_missing) or max_abs > tolerance:
                bad_cols.append(f"{col}:max_abs={max_abs:.3e},same_missing={same_missing}")
        else:
            a = got[col].fillna("<NA>").astype(str).tolist()
            b = exp[col].fillna("<NA>").astype(str).tolist()
            if a != b:
                bad_cols.append(f"{col}:string_diff")
    max_diff = max(numeric_diffs) if numeric_diffs else 0.0
    return {
        "check": label,
        "status": "PASS" if not bad_cols else "FAIL",
        "detail": f"max_numeric_diff={max_diff:.3e}; bad_cols={'; '.join(bad_cols[:4])}",
    }


def validate_against_frozen() -> pd.DataFrame:
    checks = [
        (
            artifact_name("feature", "_screen", ":", "feature", "_robustness", "_summary"),
            RECOMPUTED / "feature_screen" / artifact_name("feature", "_robustness", "_summary", ".csv"),
            FROZEN_OUTPUTS / "feature_screen" / artifact_name("feature", "_robustness", "_summary", ".csv"),
        ),
        ("feature_screen:output_interaction_wald_h8", RECOMPUTED / "feature_screen/output_interaction_wald_h8.csv", FROZEN_OUTPUTS / "feature_screen/output_interaction_wald_h8.csv"),
        (
            artifact_name("feature", "_screen", ":", "output", "_interaction", "_multiplicity", "_h8"),
            RECOMPUTED / "feature_screen" / artifact_name("output", "_interaction", "_multiplicity", "_h8", ".csv"),
            FROZEN_OUTPUTS / "feature_screen" / artifact_name("output", "_interaction", "_multiplicity", "_h8", ".csv"),
        ),
        ("feature_screen:kernel_paths_all_horizons", RECOMPUTED / "feature_screen/kernel_paths_all_horizons.csv", FROZEN_OUTPUTS / "feature_screen/kernel_paths_all_horizons.csv"),
        ("feature_screen:kernel_h8", RECOMPUTED / "feature_screen/kernel_h8.csv", FROZEN_OUTPUTS / "feature_screen/kernel_h8.csv"),
        ("feature_screen:bootstrap_kernel_summary", RECOMPUTED / "feature_screen/bootstrap_kernel_summary.csv", FROZEN_OUTPUTS / "feature_screen/bootstrap_kernel_summary.csv"),
        ("feature_screen:loo_kernel_summary", RECOMPUTED / "feature_screen/loo_kernel_summary.csv", FROZEN_OUTPUTS / "feature_screen/loo_kernel_summary.csv"),
        ("feature_screen:time_block_kernel_summary", RECOMPUTED / "feature_screen/time_block_kernel_summary.csv", FROZEN_OUTPUTS / "feature_screen/time_block_kernel_summary.csv"),
        ("polish_output:paths", RECOMPUTED / "polish_output_spending/polish_output_spending_paths.csv", FROZEN_OUTPUTS / "polish_output_spending/polish_output_spending_paths.csv"),
        ("polish_output:h8_summary", RECOMPUTED / "polish_output_spending/polish_output_spending_h8_summary.csv", FROZEN_OUTPUTS / "polish_output_spending/polish_output_spending_h8_summary.csv"),
        ("debt:direct_dy", RECOMPUTED / "debt_accounting/direct_dy_initial_action_paths.csv", FROZEN_OUTPUTS / "debt_accounting/direct_dy_initial_action_paths.csv"),
        ("debt:program_paths", RECOMPUTED / "debt_accounting/three_year_program_paths.csv", FROZEN_OUTPUTS / "debt_accounting/three_year_program_paths.csv"),
        ("debt:dsa_paths", RECOMPUTED / "debt_accounting/dsa_debt_paths.csv", FROZEN_OUTPUTS / "debt_accounting/dsa_debt_paths.csv"),
        ("debt:summary_2036", RECOMPUTED / "debt_accounting/polish_debt_2036_summary.csv", FROZEN_OUTPUTS / "debt_accounting/polish_debt_2036_summary.csv"),
        ("eu27_debt:endpoint_2036", RECOMPUTED / "eu27_benchmark/eu27_benchmark_debt_2036.csv", FROZEN_EU27_DEBT / "eu27_benchmark_debt_2036.csv"),
        ("eu27_debt:annual_decomposition", RECOMPUTED / "eu27_benchmark/eu27_benchmark_annual_debt_decomposition.csv", FROZEN_EU27_DEBT / "eu27_benchmark_annual_debt_decomposition.csv"),
    ]
    rows = [compare_frames(label, got, exp) for label, got, exp in checks]
    out = pd.DataFrame(rows)
    write_frame(out, QA / "full_estimator_repro_validation.csv")
    return out


def write_exploratory_validation_notice() -> pd.DataFrame:
    rows = [
        {
            "check": "exploratory_run_not_compared_to_benchmark",
            "status": "PASS",
            "detail": (
                f"profile_year={PROFILE_YEAR}; sample_end_year={SAMPLE_END_YEAR}; "
                "outputs are exploratory and are not default manuscript replication outputs"
            ),
        }
    ]
    out = pd.DataFrame(rows)
    write_frame(out, QA / "full_estimator_repro_validation.csv")
    return out


def write_manifest() -> None:
    rows = []
    for folder in [CODE / "estimation", REFERENCES, FROZEN_INPUTS, FROZEN_SOURCES, FROZEN_OUTPUTS, FROZEN_EU27_DEBT, RECOMPUTED]:
        for path in sorted(p for p in folder.rglob("*") if p.is_file()):
            if "__pycache__" in path.parts:
                continue
            rows.append({"path": str(path.relative_to(ROOT)), "bytes": path.stat().st_size, "sha256": sha256(path)})
    manifest = pd.DataFrame(rows)
    manifest["profile_year"] = PROFILE_YEAR
    manifest["sample_end_year"] = SAMPLE_END_YEAR
    manifest["validation_mode"] = VALIDATION_MODE
    write_frame(manifest, RECOMPUTED / "full_estimator_manifest.csv")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile-year", type=int, default=PROFILE_YEAR)
    parser.add_argument("--sample-end-year", type=int, default=SAMPLE_END_YEAR)
    parser.add_argument(
        "--validation-mode",
        choices=["benchmark", "exploratory"],
        default="benchmark",
        help="benchmark enforces frozen-target validation; exploratory skips benchmark validation",
    )
    return parser.parse_args()


def main() -> None:
    global PROFILE_YEAR, SAMPLE_END_YEAR, VALIDATION_MODE
    args = parse_args()
    PROFILE_YEAR = int(args.profile_year)
    SAMPLE_END_YEAR = int(args.sample_end_year)
    VALIDATION_MODE = str(args.validation_mode)
    progress("rebuilding runtime model inputs from frozen source and model-input files")
    build_runtime_model_inputs()
    progress("rerunning 15-candidate feature screen")
    feature_mod, screen = run_feature_screen()
    selected = (
        screen.loc[screen["gate_status"].eq("PASS_ROBUSTNESS_GATE"), ["spec_id", "features"]]
        .copy()
        .sort_values("spec_id")
        .reset_index(drop=True)
    )
    if VALIDATION_MODE == "benchmark":
        target_selected = load_adopted_specification_targets()
        if not selected.equals(target_selected):
            got = selected.to_dict("records")
            target = target_selected.to_dict("records")
            raise SystemExit(f"Fresh feature-screen winners differ from frozen validation target: got={got}, target={target}")
    if selected.empty:
        raise SystemExit("No specification passed the feature screen; downstream paths cannot be estimated")
    progress("estimating retained output and spending paths")
    polish_paths = run_polish_output(selected)
    progress("computing debt paths from recomputed response paths")
    run_debt(selected, polish_paths)
    progress("writing regression-output disclosure tables")
    make_estimation_output_tables(feature_mod, screen, polish_paths)
    if VALIDATION_MODE == "benchmark":
        progress("validating recomputed outputs against frozen benchmark")
        validation = validate_against_frozen()
    else:
        progress("exploratory mode: skipping frozen benchmark validation")
        validation = write_exploratory_validation_notice()
    write_manifest()
    failed = validation[~validation["status"].eq("PASS")]
    if not failed.empty:
        raise SystemExit("Full estimator repro validation failed:\n" + failed.to_string(index=False))
    progress("PASS: full estimator repro recomputed from frozen inputs")


if __name__ == "__main__":
    main()
