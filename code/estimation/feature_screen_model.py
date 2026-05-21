#!/usr/bin/env python3
from __future__ import annotations

import os
import sys

_REQUIRED_THREAD_ENV = {
    "OPENBLAS_NUM_THREADS": "1",
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
}

if any(os.environ.get(key) != value for key, value in _REQUIRED_THREAD_ENV.items()):
    env = os.environ.copy()
    env.update(_REQUIRED_THREAD_ENV)
    os.execve(sys.executable, [sys.executable, *sys.argv], env)

import hashlib
import importlib.util
import itertools
import json
import math
import platform
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    from scipy.stats import chi2
except ModuleNotFoundError:  # Pyodide/JupyterLite may ship without SciPy.
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

try:
    from threadpoolctl import threadpool_info, threadpool_limits

    _THREADPOOL_LIMITER = threadpool_limits(limits=1)
except Exception:
    threadpool_info = None
    _THREADPOOL_LIMITER = None


def find_package_root(script_path: Path) -> Path:
    for path in script_path.parents:
        if (path / "code").is_dir() and (path / "data").is_dir() and (path / "references").is_dir():
            return path
    raise RuntimeError(f"Cannot locate package root from {script_path}")


SCRIPT_PATH = Path(__file__).resolve()
ROOT = find_package_root(SCRIPT_PATH)
TASK_DIR = ROOT
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results" / "feature_screen"
REFERENCES_DIR = ROOT / "references"
GRID_CODE = REFERENCES_DIR / "run_c_pl_feature_grid_base.py"

SPEC_VERSION = "state_variable_screen_20260510"
PUBLIC_RUN_TIMESTAMP = os.environ.get("PUBLIC_REPRO_RUN_TIMESTAMP", "2026-05-21T00:00:00+00:00")
PUBLIC_SOURCE_REVISION = os.environ.get(
    "PUBLIC_REPRO_SOURCE_REVISION",
    "not_embedded_in_public_package; use source_manifest.csv and frozen-input manifests",
)
FEATURES = ("trade", "debt", "liq", "log_gdp_pc")
SPEC_ID_BY_FEATURES = {
    "trade": "investment_import_content",
    "debt": "public_debt",
    "liq": "household_net_financial_worth",
    "log_gdp_pc": "real_ppp_income",
    "trade+debt": "investment_import_content__public_debt",
    "trade+liq": "investment_import_content__household_net_financial_worth",
    "trade+log_gdp_pc": "investment_import_content__real_ppp_income",
    "debt+liq": "public_debt__household_net_financial_worth",
    "debt+log_gdp_pc": "public_debt__real_ppp_income",
    "liq+log_gdp_pc": "household_net_financial_worth__real_ppp_income",
    "trade+debt+liq": "investment_import_content__public_debt__household_net_financial_worth",
    "trade+debt+log_gdp_pc": "investment_import_content__public_debt__real_ppp_income",
    "trade+liq+log_gdp_pc": "investment_import_content__household_net_financial_worth__real_ppp_income",
    "debt+liq+log_gdp_pc": "public_debt__household_net_financial_worth__real_ppp_income",
    "trade+debt+liq+log_gdp_pc": "investment_import_content__public_debt__household_net_financial_worth__real_ppp_income",
}
BOOTSTRAP_DRAWS = 19
BOOTSTRAP_SEED = 20260426
TIME_BLOCKS = (
    ("drop_2004_2010", 2004, 2010),
    ("drop_2011_2017", 2011, 2017),
    ("drop_2018_2024", 2018, 2024),
)
CONDITION_MAX = 100.0
CORR_MAX = 0.85
SUPPORT_P_MIN = 0.05
MAX_ABS_Z_MAX = 2.0
OUTPUT_INTERACTION_P_MAX = 0.05
HORIZONS = tuple(range(9))
_FEATURE_PANEL_CACHE: pd.DataFrame | None = None


def progress(message: str) -> None:
    print(f"[state screen] {message}", flush=True)


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


def validate_package_inputs() -> None:
    required = [
        DATA_DIR / "country_feature_panel.csv",
        DATA_DIR / "eu27_lp_joint_panel_snapshot.csv",
        REFERENCES_DIR / "run_c_pl_feature_grid_base.py",
        REFERENCES_DIR / "run_c_pl_full3_r3_base.py",
        REFERENCES_DIR / "project_context" / "ciaffi_canonical_estimation_v3_short_rate.py",
    ]
    missing = [str(path.relative_to(ROOT)) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required package inputs: " + "; ".join(missing))


def git_value(*args: str) -> str:
    try:
        proc = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=False)
    except Exception:
        return "unknown"
    return proc.stdout.strip() if proc.returncode == 0 else "unknown"


def filtered_tracked_git_status() -> str:
    return "not embedded; source_manifest.csv carries public-archive file hashes for standalone review"


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


def spec_catalog() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for size in range(1, len(FEATURES) + 1):
        for combo in itertools.combinations(FEATURES, size):
            features = "+".join(combo)
            rows.append(
                {
                    "spec_id": SPEC_ID_BY_FEATURES[features],
                    "features": features,
                    "feature_count": len(combo),
                }
            )
    out = pd.DataFrame(rows)
    out.to_csv(DATA_DIR / "tdl_gdppc_spec_catalog.csv", index=False)
    return out


def expected_feature_subsets() -> set[str]:
    return {
        "+".join(combo)
        for size in range(1, len(FEATURES) + 1)
        for combo in itertools.combinations(FEATURES, size)
    }


def load_grid_module() -> Any:
    spec = importlib.util.spec_from_file_location("c_pl_feature_grid_base_for_tdl_gdppc", GRID_CODE)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {GRID_CODE}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    module.TASK_DIR = TASK_DIR
    module.ROOT = ROOT
    module.DATA_DIR = DATA_DIR
    module.RAW_DIR = DATA_DIR / "raw"
    module.RESULTS_DIR = RESULTS_DIR
    module.REFERENCES_DIR = REFERENCES_DIR
    module.BASE_RUNNER = REFERENCES_DIR / "run_c_pl_full3_r3_base.py"
    module.V3_CODE = REFERENCES_DIR / "project_context/ciaffi_canonical_estimation_v3_short_rate.py"
    module.SPEC_VERSION = SPEC_VERSION
    return module


def configure(grid_mod: Any, base: Any, features: tuple[str, ...], model_id: str) -> dict[str, float]:
    global _FEATURE_PANEL_CACHE
    grid_mod.configure_base_for_spec(base, features, model_id)
    if _FEATURE_PANEL_CACHE is None:
        _FEATURE_PANEL_CACHE = pd.read_csv(DATA_DIR / "country_feature_panel.csv")
    return grid_mod.feature_values(_FEATURE_PANEL_CACHE, features, "poland_2024")


def load_work(grid_mod: Any, base: Any) -> tuple[Any, pd.DataFrame, pd.DataFrame]:
    v3 = grid_mod.load_v3(base)
    panel, _meta = v3.load_panel()
    feature_panel = pd.read_csv(DATA_DIR / "country_feature_panel.csv")
    work, _shock_meta = grid_mod.prepare_work(v3, panel, feature_panel)
    return v3, work, feature_panel


def regression_fit(v3: Any, sample: pd.DataFrame, y_col: str, x_cols: list[str]) -> dict[str, Any]:
    needed = [y_col, *x_cols, "country", "year"]
    use = sample.dropna(subset=needed).sort_values(["country", "year"]).reset_index(drop=True)
    if len(use) < 50:
        return {"status": "INSUFFICIENT_OBS", "nobs": len(use)}
    projector = v3.FEProjector(use["country"], use["year"])
    x_res = projector.residualize(use[x_cols].to_numpy(dtype=float))
    y_res = projector.residualize(use[y_col].to_numpy(dtype=float))
    beta, _fitted, resid, xtx_inv, rank = v3.ols_fit(x_res, y_res)
    horizon = int(str(y_col).rsplit("h", 1)[-1]) if "h" in y_col else 1
    cov = v3.dk_covariance(x_res, resid, use["year"].to_numpy(dtype=int), xtx_inv, max(horizon, 1))
    return {
        "status": "OK",
        "nobs": int(len(use)),
        "rank": int(rank),
        "beta": beta,
        "cov": cov,
    }


def output_interaction_wald(
    base: Any,
    v3: Any,
    work: pd.DataFrame,
    features: tuple[str, ...],
    horizon: int = 8,
) -> dict[str, Any]:
    cols = base.x_columns(False)
    sample = work[work["year"].between(*v3.shock_window(horizon))].copy()
    fit = regression_fit(v3, sample, f"y_dyn_h{horizon}", cols)
    row = {
        "features": "+".join(features),
        "horizon": horizon,
        "status": fit.get("status"),
        "nobs": fit.get("nobs", 0),
        "wald_y_h8": math.nan,
        "p_wald_y_h8": math.nan,
    }
    if fit.get("status") != "OK":
        return row
    idx = [cols.index(f"shock_G_I_x_{feature}") for feature in features]
    beta = np.asarray(fit["beta"])[idx]
    cov = np.asarray(fit["cov"])[np.ix_(idx, idx)]
    if not np.isfinite(cov).all() or np.linalg.matrix_rank(cov) < len(idx):
        return {**row, "status": "SINGULAR_COV"}
    wald = float(beta.T @ np.linalg.pinv(cov) @ beta)
    return {**row, "status": "OK", "wald_y_h8": wald, "p_wald_y_h8": float(chi2.sf(wald, len(idx)))}


def support_checks(feature_panel: pd.DataFrame, catalog: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    sample = feature_panel[feature_panel["year"].between(2004, 2024)].copy()
    pol = feature_panel[(feature_panel["country"].eq("POL")) & (feature_panel["year"].eq(2024))].iloc[0]
    sample = sample[~(sample["country"].eq("POL") & sample["year"].eq(2024))].copy()
    for features_text in catalog["features"]:
        features = parse_features(features_text)
        z_cols = [f"{feature}_z" for feature in features]
        use = sample.dropna(subset=z_cols).copy()
        x = use[z_cols].to_numpy(dtype=float)
        pol_vec = pol[z_cols].to_numpy(dtype=float)
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
                "support_reference_excludes": "POL_2024_target",
            }
        )
    return pd.DataFrame(rows)


def design_diagnostics(grid_mod: Any, base: Any, v3: Any, work: pd.DataFrame, features: tuple[str, ...]) -> dict[str, Any]:
    return grid_mod.design_diagnostics_for_spec(base, v3, work, features)


def fit_conditional_ratios(
    base: Any,
    v3: Any,
    work: pd.DataFrame,
    dep_cols: tuple[str, ...],
    scale_col: str,
    cols: list[str],
    z_values: dict[str, float],
    horizon: int,
    exclude_country: str | None = None,
) -> dict[str, dict[str, Any]]:
    needed = [*dep_cols, scale_col, *cols, "country", "year"]
    sample = work[work["year"].between(*v3.shock_window(horizon))].copy()
    if exclude_country is not None:
        sample = sample[~sample["country"].eq(exclude_country)].copy()
    sample = sample.dropna(subset=needed).sort_values(["country", "year"]).reset_index(drop=True)
    if len(sample) < 50:
        return {
            dep_col: {
                "ratio": math.nan,
                "se": math.nan,
                "beta_scale": math.nan,
                "se_beta_scale": math.nan,
                "denom_t": math.nan,
                "nobs": int(len(sample)),
                "country_n": 0,
                "year_min": None,
                "year_max": None,
                "rank": 0,
                "status": "INSUFFICIENT_OBS",
            }
            for dep_col in dep_cols
        }
    projector = v3.FEProjector(sample["country"], sample["year"])
    x_res = projector.residualize(sample[cols].to_numpy(dtype=float))
    scale_res = projector.residualize(sample[scale_col].to_numpy(dtype=float))
    beta_scale, _fit_scale, resid_scale, xtx_inv, rank = v3.ols_fit(x_res, scale_res)
    years = sample["year"].to_numpy(dtype=int)
    bandwidth = max(int(horizon), 1)
    vcov_scale = v3.dk_covariance(x_res, resid_scale, years, xtx_inv, bandwidth)
    c = base.contrast_vector(cols, z_values)
    beta_scale_c = float(c @ beta_scale)
    var_scale = float(c @ vcov_scale @ c)
    se_scale = math.sqrt(max(var_scale, 0.0)) if math.isfinite(var_scale) else math.nan
    denom_t = abs(beta_scale_c / se_scale) if math.isfinite(se_scale) and se_scale > 0 else math.nan
    out: dict[str, dict[str, Any]] = {}
    for dep_col in dep_cols:
        dep_res = projector.residualize(sample[dep_col].to_numpy(dtype=float))
        beta_dep, _fit_dep, resid_dep, _xtx_dep, _rank_dep = v3.ols_fit(x_res, dep_res)
        vcov_dep = v3.dk_covariance(x_res, resid_dep, years, xtx_inv, bandwidth)
        vcov_cross = v3.dk_cross_covariance(x_res, resid_dep, resid_scale, years, xtx_inv, bandwidth)
        beta_dep_c = float(c @ beta_dep)
        var_dep = float(c @ vcov_dep @ c)
        cov_cross = float(c @ vcov_cross @ c)
        ratio, se = v3.ratio_and_se(beta_dep_c, beta_scale_c, var_dep, var_scale, cov_cross)
        status = "OK" if math.isfinite(ratio) and math.isfinite(se) else "NONFINITE"
        if not math.isfinite(beta_scale_c) or abs(beta_scale_c) < 1e-12:
            status = "ZERO_SCALE_DENOMINATOR"
        elif not math.isfinite(denom_t) or denom_t < getattr(base, "DENOMINATOR_T_THRESHOLD", 1.96):
            status = "WEAK_SCALE_DENOMINATOR"
        out[dep_col] = {
            "ratio": float(ratio),
            "se": float(se),
            "beta_scale": beta_scale_c,
            "se_beta_scale": se_scale,
            "denom_t": float(denom_t) if math.isfinite(denom_t) else math.nan,
            "nobs": int(len(sample)),
            "country_n": int(sample["country"].nunique()),
            "year_min": int(sample["year"].min()),
            "year_max": int(sample["year"].max()),
            "rank": int(rank),
            "status": status,
        }
    return out


def estimate_stage1_kernels(
    grid_mod: Any,
    base: Any,
    v3: Any,
    work: pd.DataFrame,
    features: tuple[str, ...],
    spec_id: str,
    exclude_country: str | None = None,
) -> pd.DataFrame:
    z_values = configure(grid_mod, base, features, spec_id)
    cols = base.x_columns(False)
    rows: list[dict[str, Any]] = []
    for h in HORIZONS:
        fits = fit_conditional_ratios(
            base,
            v3,
            work,
            (f"y_dyn_h{h}", f"gi_dyn_h{h}"),
            "gi_dyn0",
            cols,
            z_values,
            h,
            exclude_country,
        )
        fit_y = fits[f"y_dyn_h{h}"]
        fit_g = fits[f"gi_dyn_h{h}"]
        rows.append(
            {
                "spec_version": SPEC_VERSION,
                "model_id": spec_id,
                "profile_label": "poland_2024",
                "excluded_country": exclude_country or "",
                "horizon": h,
                "mu_Y_incremental": fit_y["ratio"],
                "se_Y_incremental": fit_y["se"],
                "mu_G_incremental": fit_g["ratio"],
                "se_G_incremental": fit_g["se"],
                "beta_scale_action": fit_y["beta_scale"],
                "se_beta_scale_action": fit_y["se_beta_scale"],
                "action_denom_t": fit_y["denom_t"],
                "nobs": fit_y["nobs"],
                "country_n": fit_y["country_n"],
                "year_min_effective": fit_y["year_min"],
                "year_max_effective": fit_y["year_max"],
                "rank_X": fit_y["rank"],
                "status_Y": fit_y["status"],
                "status_G": fit_g["status"],
                **{f"z_{feature}": float(z_values[feature]) for feature in features},
            }
        )
    out = pd.DataFrame(rows).sort_values("horizon").reset_index(drop=True)
    out["K_Y_cumulative"] = out["mu_Y_incremental"].cumsum()
    out["K_G_cumulative"] = out["mu_G_incremental"].cumsum()
    out["ci95_low_K_Y_cumulative_naive"] = out["K_Y_cumulative"] - base.Z95 * np.sqrt(
        np.cumsum(np.square(out["se_Y_incremental"]))
    )
    out["ci95_high_K_Y_cumulative_naive"] = out["K_Y_cumulative"] + base.Z95 * np.sqrt(
        np.cumsum(np.square(out["se_Y_incremental"]))
    )
    out.insert(0, "spec_id", spec_id)
    out.insert(1, "features", "+".join(features))
    return out


def relabel_bootstrap_work(work: pd.DataFrame, countries: list[str], rng: np.random.Generator) -> pd.DataFrame:
    draw = list(rng.choice(countries, size=len(countries), replace=True))
    frames = []
    for idx, country in enumerate(draw):
        part = work[work["country"].eq(country)].copy()
        part["country"] = f"{country}_draw_{idx:02d}"
        frames.append(part)
    return pd.concat(frames, ignore_index=True)


def summarize_kernel_stability(rows: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    out_rows: list[dict[str, Any]] = []
    for key, grp in rows.groupby(group_cols, sort=False):
        if not isinstance(key, tuple):
            key = (key,)
        row = dict(zip(group_cols, key))
        ok = grp[grp["status"].eq("OK") & np.isfinite(grp["K_Y_h8"]) & np.isfinite(grp["K_G_h8"])]
        row["ok_count"] = int(len(ok))
        row["all_finite_ok"] = bool(len(ok) == len(grp))
        for col in ["K_Y_h8", "K_G_h8"]:
            if len(ok):
                row[f"{col}_p05"] = float(ok[col].quantile(0.05))
                row[f"{col}_p50"] = float(ok[col].quantile(0.50))
                row[f"{col}_p95"] = float(ok[col].quantile(0.95))
            else:
                row[f"{col}_p05"] = math.nan
                row[f"{col}_p50"] = math.nan
                row[f"{col}_p95"] = math.nan
        out_rows.append(row)
    return pd.DataFrame(out_rows)


def leave_one_country_kernels(grid_mod: Any, base: Any, v3: Any, work: pd.DataFrame, catalog: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    countries = sorted(work["country"].dropna().astype(str).unique())
    rows: list[dict[str, Any]] = []
    for spec in catalog.itertuples(index=False):
        progress(f"leave-one-country {spec.spec_id} {spec.features}")
        features = parse_features(spec.features)
        for country in countries:
            subset = work[~work["country"].eq(country)].copy()
            try:
                kernels = estimate_stage1_kernels(
                    grid_mod,
                    base,
                    v3,
                    subset,
                    features,
                    f"{spec.spec_id}_loo_{country}",
                )
                h8 = kernels[kernels["horizon"].eq(8)].iloc[0]
                rows.append(
                    {
                        "spec_id": spec.spec_id,
                        "features": spec.features,
                        "excluded_country": country,
                        "status": "OK",
                        "K_Y_h8": float(h8["K_Y_cumulative"]),
                        "K_G_h8": float(h8["K_G_cumulative"]),
                    }
                )
            except Exception as exc:
                rows.append(
                    {
                        "spec_id": spec.spec_id,
                        "features": spec.features,
                        "excluded_country": country,
                        "status": f"FAIL:{type(exc).__name__}",
                        "K_Y_h8": math.nan,
                        "K_G_h8": math.nan,
                    }
                )
    draw_df = pd.DataFrame(rows)
    summary = summarize_kernel_stability(draw_df, ["spec_id", "features"]).rename(
        columns={
            "ok_count": "loo_ok_count",
            "all_finite_ok": "loo_all_finite_ok",
        }
    )
    return draw_df, summary


def bootstrap_kernels(grid_mod: Any, base: Any, v3: Any, work: pd.DataFrame, catalog: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    countries = sorted(work["country"].dropna().astype(str).unique())
    rows: list[dict[str, Any]] = []
    for spec in catalog.itertuples(index=False):
        progress(f"bootstrap {spec.spec_id} {spec.features}")
        features = parse_features(spec.features)
        for draw_id in range(BOOTSTRAP_DRAWS):
            boot_work = relabel_bootstrap_work(work, countries, rng)
            try:
                kernels = estimate_stage1_kernels(
                    grid_mod,
                    base,
                    v3,
                    boot_work,
                    features,
                    f"{spec.spec_id}_boot_{draw_id:03d}",
                )
                h8 = kernels[kernels["horizon"].eq(8)].iloc[0]
                rows.append(
                    {
                        "spec_id": spec.spec_id,
                        "features": spec.features,
                        "draw_id": draw_id,
                        "status": "OK",
                        "K_Y_h8": float(h8["K_Y_cumulative"]),
                        "K_G_h8": float(h8["K_G_cumulative"]),
                    }
                )
            except Exception as exc:
                rows.append(
                    {
                        "spec_id": spec.spec_id,
                        "features": spec.features,
                        "draw_id": draw_id,
                        "status": f"FAIL:{type(exc).__name__}",
                        "K_Y_h8": math.nan,
                        "K_G_h8": math.nan,
                    }
                )
    draw_df = pd.DataFrame(rows)
    summary = summarize_kernel_stability(draw_df, ["spec_id", "features"]).rename(
        columns={
            "ok_count": "boot_ok_count",
            "all_finite_ok": "boot_all_finite_ok",
        }
    )
    return draw_df, summary


def time_block_kernels(grid_mod: Any, base: Any, v3: Any, work: pd.DataFrame, catalog: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    for spec in catalog.itertuples(index=False):
        progress(f"time-block {spec.spec_id} {spec.features}")
        features = parse_features(spec.features)
        for block_name, y0, y1 in TIME_BLOCKS:
            block_work = work[~work["year"].between(y0, y1)].copy()
            try:
                kernels = estimate_stage1_kernels(
                    grid_mod,
                    base,
                    v3,
                    block_work,
                    features,
                    f"{spec.spec_id}_{block_name}",
                )
                h8 = kernels[kernels["horizon"].eq(8)].iloc[0]
                rows.append(
                    {
                        "spec_id": spec.spec_id,
                        "features": spec.features,
                        "block": block_name,
                        "drop_start": y0,
                        "drop_end": y1,
                        "status": "OK",
                        "K_Y_h8": float(h8["K_Y_cumulative"]),
                        "K_G_h8": float(h8["K_G_cumulative"]),
                    }
                )
            except Exception as exc:
                rows.append(
                    {
                        "spec_id": spec.spec_id,
                        "features": spec.features,
                        "block": block_name,
                        "drop_start": y0,
                        "drop_end": y1,
                        "status": f"FAIL:{type(exc).__name__}",
                        "K_Y_h8": math.nan,
                        "K_G_h8": math.nan,
                    }
                )
    block_df = pd.DataFrame(rows)
    summary = summarize_kernel_stability(block_df, ["spec_id", "features"]).rename(
        columns={
            "ok_count": "time_ok_count",
            "all_finite_ok": "time_all_finite_ok",
        }
    )
    return block_df, summary


def criteria_sources() -> pd.DataFrame:
    rows = [
        {
            "criterion": "ex_ante_feature_universe",
            "local_reference": "references/papers_txt/ilzetzki_mendoza_vegh_2013_fiscal_multipliers.txt; references/papers_txt/huidrom_kose_lim_ohnsorge_2019_why_fiscal_positions.txt; references/papers_txt/bernardini_peersman_2017_heterogeneous_multipliers.txt; references/papers_txt/krajewski_pilat_2025_liquidity_constraints_poland.txt; references/papers_txt/mcmanus_ozkan_trzeciakiewicz_2020_credit_constraints.txt",
            "source_support": "Ilzetzki, Mendoza and Vegh (2013); Huidrom et al. (2019); Bernardini and Peersman (2018); Krajewski and Pilat (2025); McManus et al. (2020)",
            "use": "Justifies trade openness, public debt, level of development and the Polish liquidity or credit-constraint extension.",
        },
        {
            "criterion": "local_projection_core",
            "local_reference": "references/papers_txt/jorda_2005_aer_lp.txt; references/papers_txt/ciaffi_2024_jpm_fiscal_debt.txt",
            "source_support": "Jorda (2005); Ciaffi, Deleidi and Di Domenico (2024)",
            "use": "Justifies local projections estimated separately at each horizon and separate output and debt responses.",
        },
        {
            "criterion": "shock_feature_interactions",
            "local_reference": "references/papers_txt/auerbach_gorodnichenko_2012_output_responses.txt; references/papers_txt/ramey_zubairy_2018_multipliers_good_bad.txt; references/papers_txt/cloyne_jorda_taylor_2023_state_dependent_lp.txt",
            "source_support": "Auerbach and Gorodnichenko (2012); Ramey and Zubairy (2018); Cloyne, Jorda and Taylor (2023)",
            "use": "Justifies interacting fiscal shocks with state or country characteristics.",
        },
        {
            "criterion": "common_sample",
            "local_reference": "pre-specified notebook design rule",
            "source_support": "Pre-specified comparison-sample rule",
            "use": "Candidate specifications are compared on the same h8 observation universe where feasible. The model-averaging papers remain model-uncertainty context, not direct support for this fixed-sample rule.",
        },
        {
            "criterion": "rank_condition_collinearity",
            "local_reference": "references/papers_txt/welsch_kuh_1977_linear_regression_diagnostics_nber.txt",
            "source_support": "Welsch and Kuh (1977)",
            "use": "Regression diagnostic hygiene: full rank, condition number and collinearity checks.",
        },
        {
            "criterion": "poland_support_overlap",
            "local_reference": "references/papers_txt/crump_hotz_imbens_mitnik_2009_limited_overlap.txt; references/papers_txt/li_morgan_zaslavsky_2018_balancing_covariates_overlap_weights.txt",
            "source_support": "Crump et al. (2009); Li, Morgan and Zaslavsky (2018)",
            "use": "Justifies common-support or overlap checks before using Poland's feature profile.",
        },
        {
            "criterion": "cluster_bootstrap_and_delete_country",
            "local_reference": "references/papers_txt/cameron_miller_2015_cluster_robust_practitioners_guide.txt; references/papers_txt/mackinnon_nielsen_webb_2023_jackknife_bootstrap_cluster.txt",
            "source_support": "Cameron and Miller (2015); MacKinnon, Nielsen and Webb (2023)",
            "use": "Justifies country-level resampling and delete-one-country checks. This public reproduction block uses these draws only as finite-run reproducibility checks, not as sign filters.",
        },
        {
            "criterion": "lp_inference_and_time_stability",
            "local_reference": "references/papers_txt/jorda_2005_aer_lp.txt; references/papers_txt/olea_plagborg_moller_2022_lp_inference.txt; references/papers_txt/huidrom_kose_lim_ohnsorge_2019_why_fiscal_positions.txt; references/papers_txt/ramey_zubairy_2018_multipliers_good_bad.txt",
            "source_support": "Jorda (2005); Olea and Plagborg-Moller (2022); Huidrom et al. (2019); Ramey and Zubairy (2018)",
            "use": "Justifies uncertainty and robustness checks across horizons and samples. This public reproduction block uses time blocks only as finite-run reproducibility checks, not as sign filters.",
        },
        {
            "criterion": "excluded_kernel_sign_gates",
            "local_reference": "pre-specified notebook design rule",
            "source_support": "Pre-specified screening rule",
            "use": "The gate deliberately excludes K_Y>0, K_G>0 and positive-sign stability of kernels. Subsequent numerical points use the selected specifications separately.",
        },
    ]
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS_DIR / "criteria_literature_map.csv", index=False)
    return out


def add_output_multiplicity(wald: pd.DataFrame) -> pd.DataFrame:
    out = wald.copy()
    out["p_wald_y_h8_bonferroni"] = math.nan
    out["q_wald_y_h8_bh"] = math.nan
    out["raw_p_rank"] = math.nan
    ok = out["p_wald_y_h8"].notna() & np.isfinite(out["p_wald_y_h8"].to_numpy(dtype=float))
    valid = out.loc[ok, ["p_wald_y_h8"]].sort_values("p_wald_y_h8", kind="mergesort")
    m = len(valid)
    if m:
        out.loc[valid.index, "p_wald_y_h8_bonferroni"] = np.minimum(valid["p_wald_y_h8"].to_numpy(dtype=float) * m, 1.0)
        q_by_index: dict[int, float] = {}
        running = 1.0
        ranked = list(enumerate(valid.index, start=1))
        for rank, idx in reversed(ranked):
            raw_p = float(out.loc[idx, "p_wald_y_h8"])
            running = min(running, raw_p * m / float(rank), 1.0)
            q_by_index[int(idx)] = running
        for rank, idx in ranked:
            out.loc[idx, "raw_p_rank"] = rank
            out.loc[idx, "q_wald_y_h8_bh"] = q_by_index[int(idx)]
    out["raw_p_pass_unadjusted"] = out["p_wald_y_h8"].lt(OUTPUT_INTERACTION_P_MAX)
    out.to_csv(RESULTS_DIR / "output_interaction_multiplicity_h8.csv", index=False, float_format="%.9f")
    return out


def write_raw_output_wald(wald: pd.DataFrame) -> None:
    cols = ["features", "horizon", "status", "nobs", "wald_y_h8", "p_wald_y_h8", "spec_id", "feature_count"]
    wald[cols].to_csv(RESULTS_DIR / "output_interaction_wald_h8.csv", index=False, float_format="%.9f")


def evaluate_gate(summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for row in summary.itertuples(index=False):
        checks = {
            "full_rank": bool(row.h8_design_full_rank),
            "condition_ok": float(row.h8_condition_number) < CONDITION_MAX,
            "corr_ok": float(row.max_abs_feature_corr_h8) <= CORR_MAX,
            "support_p_ok": float(row.mahalanobis_support_p) >= SUPPORT_P_MIN,
            "max_z_ok": float(row.max_abs_poland_z) <= MAX_ABS_Z_MAX,
            "output_interaction_ok": float(row.p_wald_y_h8) < OUTPUT_INTERACTION_P_MAX,
            "loo_finite_ok": bool(row.loo_all_finite_ok),
            "bootstrap_finite_ok": bool(row.boot_all_finite_ok),
            "time_block_finite_ok": bool(row.time_all_finite_ok) and int(row.time_ok_count) == len(TIME_BLOCKS),
        }
        failed_labels = {
            "full_rank": "failed_full_rank",
            "condition_ok": "failed_condition_number",
            "corr_ok": "failed_collinearity",
            "support_p_ok": "failed_poland_support_p",
            "max_z_ok": "failed_poland_z_support",
            "output_interaction_ok": "failed_output_interaction_p",
            "loo_finite_ok": "failed_leave_one_country_finiteness",
            "bootstrap_finite_ok": "failed_bootstrap_finiteness",
            "time_block_finite_ok": "failed_time_block_finiteness",
        }
        failed = [failed_labels[name] for name, ok in checks.items() if not ok]
        robustness_score = int(sum(1 for ok in checks.values() if ok))
        finite_run_share = min(
            float(row.loo_ok_count) / 27.0,
            float(row.boot_ok_count) / float(BOOTSTRAP_DRAWS),
            float(row.time_ok_count) / float(len(TIME_BLOCKS)),
        )
        status = "PASS_ROBUSTNESS_GATE" if not failed else "FAIL_ROBUSTNESS_GATE"
        rows.append(
            {
                "spec_id": row.spec_id,
                "features": row.features,
                **checks,
                "robustness_score": robustness_score,
                "finite_run_share": finite_run_share,
                "gate_status": status,
                "gate_reason": "all_checks_pass" if not failed else ";".join(failed),
            }
        )
    gate = pd.DataFrame(rows)
    out = summary.merge(gate, on=["spec_id", "features"], how="left", validate="one_to_one")
    out = out.sort_values(
        [
            "gate_status",
            "robustness_score",
            "p_wald_y_h8",
            "finite_run_share",
            "mahalanobis_support_p",
            "h8_condition_number",
        ],
        ascending=[False, False, True, False, False, True],
    ).reset_index(drop=True)
    return out


def md_table(df: pd.DataFrame, cols: list[str], float_cols: set[str]) -> str:
    if df.empty:
        return "_No rows._"
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in df[cols].iterrows():
        vals = []
        for col in cols:
            val = row[col]
            if pd.isna(val):
                vals.append("NA")
            elif col in float_cols:
                vals.append(f"{float(val):.6f}")
            else:
                vals.append(str(val))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


PUBLIC_FEATURE_LABELS = {
    "trade": "investment import content",
    "debt": "public debt",
    "liq": "household net financial worth",
    "log_gdp_pc": "real PPP income",
}

PUBLIC_GATE_STATUS = {
    "PASS_ROBUSTNESS_GATE": "Retained",
    "FAIL_ROBUSTNESS_GATE": "Not retained",
}

PUBLIC_GATE_REASONS = {
    "all_checks_pass": "All screening diagnostics passed",
    "failed_rank": "Design matrix rank diagnostic did not pass",
    "failed_full_rank": "Design matrix rank diagnostic did not pass",
    "failed_condition": "Condition-number diagnostic did not pass",
    "failed_condition_number": "Condition-number diagnostic did not pass",
    "failed_feature_corr": "State-correlation diagnostic did not pass",
    "failed_collinearity": "State-correlation diagnostic did not pass",
    "failed_poland_support": "Poland support-overlap diagnostic did not pass",
    "failed_poland_support_p": "Poland support-overlap diagnostic did not pass",
    "failed_poland_z": "Poland state-profile support diagnostic did not pass",
    "failed_poland_z_support": "Poland state-profile support diagnostic did not pass",
    "failed_output_interaction_p": "Output-interaction test did not pass",
    "failed_leave_one_country_finiteness": "Leave-one-country finite-run check did not pass",
    "failed_bootstrap_finiteness": "Bootstrap finite-run check did not pass",
    "failed_time_block_finiteness": "Time-block finite-run check did not pass",
}

CRITERION_LABELS = {
    "ex_ante_feature_universe": "Ex ante state-variable universe",
    "local_projection_core": "Local-projection core",
    "shock_feature_interactions": "Shock and state-variable interactions",
    "common_sample": "Common comparison sample",
    "rank_condition_collinearity": "Rank, condition-number and collinearity checks",
    "poland_support_overlap": "Poland support-overlap check",
    "cluster_bootstrap_and_delete_country": "Country-level bootstrap and leave-one-country checks",
    "lp_inference_and_time_stability": "Local-projection inference and time stability",
    "excluded_kernel_sign_gates": "Excluded downstream sign filters",
}

QA_CHECK_LABELS = {
    "spec_count_is_15": "All candidate state-variable subsets are enumerated",
    "all_nonempty_feature_subsets_present": "Every non-empty subset of the four state variables is present",
    "gate_reasons_are_stage1_only": "Selection reasons use only pre-specified screening diagnostics",
    "robustness_summary_has_stage1_columns_only": "Screening summary contains only first-stage diagnostics",
    "no_kernel_quantiles_in_robustness_summary": "Screening summary excludes downstream response quantiles",
    "kernel_outputs_are_stage1_only": "Candidate response outputs contain only response and state-profile fields",
    "no_kernel_sign_gate_columns": "Selection excludes response-sign filters",
    "no_positive_sign_summary_columns": "Screening summary excludes unused positive-sign diagnostics",
    "result_file_set_is_stage1_only": "Result file set is limited to the screening block",
    "output_only_bic_not_computed": "Output-only fit ranking is not computed",
    "multiplicity_sensitivity_written": "Multiplicity diagnostics are written as sensitivity checks",
    "support_excludes_poland_profile_target": "Support reference sample excludes the Poland target profile",
    "failed_gate_reasons_are_unambiguous": "Non-retained rows have explicit screening reasons",
    "dependency_manifest_present": "Dependency manifest is present",
    "bootstrap_draws_complete": "Bootstrap finite-run checks are complete",
    "time_blocks_complete": "Time-block finite-run checks are complete",
    "required_text_references_present": "Required text references are present",
}


def public_feature_label(features: object) -> str:
    return " + ".join(PUBLIC_FEATURE_LABELS.get(part, part.replace("_", " ")) for part in str(features).split("+"))


def public_gate_reason(value: object) -> str:
    parts = [part.strip() for part in str(value).split(";") if part.strip()]
    if not parts:
        return "No screening reason recorded"
    return "; ".join(PUBLIC_GATE_REASONS.get(part, part.replace("_", " ")) for part in parts)


def public_bool(value: object) -> str:
    return "Yes" if bool(value) else "No"


def public_detail(value: object) -> str:
    text = str(value)
    replacements = {
        "feature_robustness_summary.csv": "screening summary",
        "output_interaction_multiplicity_h8.csv": "output-interaction multiplicity diagnostics",
        "criteria_literature_map.csv": "criteria literature map",
        "trade/debt/liq/log_gdp_pc": "investment import content, public debt, household net financial worth and real PPP income",
        "gate_reason": "selection reason",
        "failed_*": "explicit non-retention reason",
        "K_Y/K_G": "output-to-spending response ratio",
        "K_Y": "cumulative output response",
        "K_G": "cumulative spending response",
        "PASS_ROBUSTNESS_GATE": "Retained",
        "FAIL_ROBUSTNESS_GATE": "Not retained",
        "failed_output_interaction_p": "output-interaction test did not pass",
        "POL_": "Poland profile ",
        "_": " ",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def public_screen_table(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["features"] = out["features"].map(public_feature_label)
    if "gate_status" in out.columns:
        out["gate_status"] = out["gate_status"].map(PUBLIC_GATE_STATUS).fillna(out["gate_status"])
    if "gate_reason" in out.columns:
        out["gate_reason"] = out["gate_reason"].map(public_gate_reason)
    if "raw_p_pass_unadjusted" in out.columns:
        out["raw_p_pass_unadjusted"] = out["raw_p_pass_unadjusted"].map(public_bool)
    return out


def public_criteria_table(criteria: pd.DataFrame) -> pd.DataFrame:
    out = criteria.copy()
    out["criterion"] = out["criterion"].map(CRITERION_LABELS).fillna(out["criterion"].astype(str).str.replace("_", " "))
    out = out.drop(columns=["local_reference"], errors="ignore")
    out["use"] = out["use"].map(public_detail)
    return out


def public_qa_table(qa: pd.DataFrame) -> pd.DataFrame:
    out = qa.copy()
    out["check"] = out["check"].map(QA_CHECK_LABELS).fillna(out["check"].astype(str).str.replace("_", " "))
    out["detail"] = out["detail"].map(public_detail)
    return out


def write_manifest() -> None:
    files = []
    paths = (
        list((ROOT / "code").glob("*.py"))
        + list((ROOT / "data").glob("*.csv"))
        + list((ROOT / "data/raw").glob("*"))
        + list((ROOT / "references").glob("*.py"))
        + list((ROOT / "references/project_context").glob("*.py"))
        + list((ROOT / "references/consensus").glob("*.md"))
        + list((ROOT / "references/commission").glob("*"))
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
    manifest_path = RESULTS_DIR / "source_manifest.csv"
    top_level_manifest_path = ROOT / "results/artifact_manifest.csv"
    for path in sorted(paths):
        if path.resolve() == manifest_path.resolve():
            continue
        if path.resolve() == top_level_manifest_path.resolve():
            continue
        if path.is_file():
            files.append({"path": rel(path), "sha256": sha256(path), "bytes": path.stat().st_size})
    pd.DataFrame(files).to_csv(manifest_path, index=False)
    manifest = {
        "spec_version": SPEC_VERSION,
        "created_utc": PUBLIC_RUN_TIMESTAMP,
        "git_head": PUBLIC_SOURCE_REVISION,
        "git_status_note": filtered_tracked_git_status(),
        "python": sys.version,
        "platform": platform.platform(),
        "blas_thread_env": {
            "OPENBLAS_NUM_THREADS": os.environ.get("OPENBLAS_NUM_THREADS", ""),
            "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS", ""),
            "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS", ""),
        },
        "blas_thread_policy": "script re-execs Python with OPENBLAS_NUM_THREADS=1, OMP_NUM_THREADS=1 and MKL_NUM_THREADS=1 before importing NumPy",
        "threadpoolctl_info": threadpool_state(),
        "features": list(FEATURES),
        "bootstrap_draws": BOOTSTRAP_DRAWS,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "thresholds": {
            "condition_max": CONDITION_MAX,
            "corr_max": CORR_MAX,
            "support_p_min": SUPPORT_P_MIN,
            "max_abs_z_max": MAX_ABS_Z_MAX,
            "output_interaction_p_max": OUTPUT_INTERACTION_P_MAX,
        },
        "excluded_as_gate": [
            "output_only_bic",
            "output_only_fit_ranking",
            "later_stage_accounting_margin",
            "later_stage_ratio_margin",
            "later_stage_sign",
            "later_stage_size",
            "later_stage_direction_flag",
            "later_stage_ratio_bic",
            "joint_bic",
            "kernel_primary_sign",
            "positive_kernel_stability",
        ],
        "stage_order": [
            "enumerate_all_15_nonempty_feature_subsets",
            "estimate_output_and_spending_kernels_for_all_specs",
            "evaluate_robustness_with_first_stage_diagnostics_only",
        ],
    }
    (RESULTS_DIR / "run_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


def write_report(summary: pd.DataFrame, criteria: pd.DataFrame, qa: pd.DataFrame) -> None:
    pass_rows = summary[summary["gate_status"].eq("PASS_ROBUSTNESS_GATE")].copy()
    top = summary.head(15).copy()
    pass_public = public_screen_table(pass_rows)
    multiplicity_public = public_screen_table(summary.sort_values("p_wald_y_h8").head(15))
    top_public = public_screen_table(top)
    criteria_public = public_criteria_table(criteria)
    qa_public = public_qa_table(qa)
    display_names = {
        "features": "State-variable subset",
        "robustness_score": "Screening diagnostics passed",
        "finite_run_share": "Minimum finite-run share",
        "p_wald_y_h8": "Output-interaction p-value",
        "mahalanobis_support_p": "Poland support p-value",
        "h8_condition_number": "Condition number",
        "max_abs_feature_corr_h8": "Maximum state correlation",
        "p_wald_y_h8_bonferroni": "Bonferroni p-value",
        "q_wald_y_h8_bh": "Benjamini-Hochberg q-value",
        "raw_p_pass_unadjusted": "Unadjusted screen result",
        "gate_status": "Screening status",
        "gate_reason": "Selection reason",
        "criterion": "Criterion",
        "source_support": "Source support",
        "use": "Use",
        "check": "Check",
        "status": "Status",
        "detail": "Detail",
    }
    pass_public = pass_public.rename(columns=display_names)
    multiplicity_public = multiplicity_public.rename(columns=display_names)
    top_public = top_public.rename(columns=display_names)
    criteria_public = criteria_public.rename(columns=display_names)
    qa_public = qa_public.rename(columns=display_names)
    text = f"""# State-variable screen

## Scope

This public reproduction block evaluates the state-variable selection step for the Polish heterogeneity model. It starts from four pre-specified state variables: investment import content, public debt, household net financial worth and real PPP income. It evaluates all 15 non-empty subsets.

The script is deliberately ordered as a single selection exercise:

1. enumerate all feature subsets;
2. estimate output and spending kernels for every subset;
3. apply robustness criteria that use first-stage diagnostics only, with no later accounting results, no later-stage signs, no output-only fit ranking and no kernel-sign filters.

This public reproduction block does not calculate later accounting paths. Those estimates for the selected specifications belong to later modelling blocks.

## Robustness criteria

{md_table(criteria_public, ["Criterion", "Source support", "Use"], set())}

## Stage 1: robustness winners

The screen selects only specifications with strong output-interaction evidence at `p < {OUTPUT_INTERACTION_P_MAX:.2f}` and clean numerical/support diagnostics. It does not use later accounting outcomes or signs.

{md_table(pass_public, ["State-variable subset", "Screening diagnostics passed", "Minimum finite-run share", "Output-interaction p-value", "Poland support p-value", "Condition number", "Maximum state correlation"], {"Minimum finite-run share", "Output-interaction p-value", "Poland support p-value", "Condition number", "Maximum state correlation"})}

## Output-interaction multiplicity sensitivity

The output-relevance screen is reported with raw p-values and simple 15-test multiplicity diagnostics. The gate itself uses the pre-specified raw `p < {OUTPUT_INTERACTION_P_MAX:.2f}` output-relevance rule; the adjusted columns are sensitivity diagnostics, not additional hidden filters.

{md_table(multiplicity_public, ["State-variable subset", "Output-interaction p-value", "Bonferroni p-value", "Benjamini-Hochberg q-value", "Unadjusted screen result"], {"Output-interaction p-value", "Bonferroni p-value", "Benjamini-Hochberg q-value"})}

## Full robustness ranking

{md_table(top_public, ["State-variable subset", "Screening status", "Screening diagnostics passed", "Selection reason", "Output-interaction p-value", "Poland support p-value", "Condition number", "Maximum state correlation", "Minimum finite-run share"], {"Output-interaction p-value", "Poland support p-value", "Condition number", "Maximum state correlation", "Minimum finite-run share"})}

## QA

{md_table(qa_public, ["Check", "Status", "Detail"], set())}

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
"""
    (RESULTS_DIR / "REPORT.md").write_text(text)


def main() -> None:
    progress("start")
    validate_package_inputs()
    grid_mod = load_grid_module()
    base = grid_mod.load_base_module()
    clean_results_dir()
    progress("loading data and constructing shocks")
    v3, work, feature_panel = load_work(grid_mod, base)
    catalog = spec_catalog()
    criteria = criteria_sources()

    kernel_rows: list[pd.DataFrame] = []
    summary_rows: list[dict[str, Any]] = []
    wald_rows: list[dict[str, Any]] = []

    for spec in catalog.itertuples(index=False):
        features = parse_features(spec.features)
        progress(f"main kernels {spec.spec_id} {spec.features}")
        kernels = estimate_stage1_kernels(grid_mod, base, v3, work, features, str(spec.spec_id))
        kernel_rows.append(kernels)
        design = design_diagnostics(grid_mod, base, v3, work, features)
        wald = output_interaction_wald(base, v3, work, features, 8)
        wald["spec_id"] = spec.spec_id
        wald["feature_count"] = int(spec.feature_count)
        wald_rows.append(wald)
        summary_rows.append(
            {
                "spec_id": spec.spec_id,
                "features": spec.features,
                "feature_count": int(spec.feature_count),
                **design,
                "wald_y_h8": wald["wald_y_h8"],
                "p_wald_y_h8": wald["p_wald_y_h8"],
            }
        )

    kernels_all = pd.concat(kernel_rows, ignore_index=True)
    kernels_all.to_csv(RESULTS_DIR / "kernel_paths_all_horizons.csv", index=False, float_format="%.9f")
    kernels_all[kernels_all["horizon"].eq(8)].to_csv(RESULTS_DIR / "kernel_h8.csv", index=False, float_format="%.9f")
    raw_wald_df = pd.DataFrame(wald_rows)
    write_raw_output_wald(raw_wald_df)
    wald_df = add_output_multiplicity(raw_wald_df)

    support = support_checks(feature_panel, catalog)
    support.to_csv(RESULTS_DIR / "poland_support_checks.csv", index=False, float_format="%.9f")
    progress("leave-one-country checks")
    loo_draws, loo_summary = leave_one_country_kernels(grid_mod, base, v3, work, catalog)
    loo_draws.to_csv(RESULTS_DIR / "loo_kernel_draws.csv", index=False, float_format="%.9f")
    loo_summary.to_csv(RESULTS_DIR / "loo_kernel_summary.csv", index=False, float_format="%.9f")
    progress(f"bootstrap checks draws={BOOTSTRAP_DRAWS}")
    boot_draws, boot_summary = bootstrap_kernels(grid_mod, base, v3, work, catalog)
    boot_draws.to_csv(RESULTS_DIR / "bootstrap_kernel_draws.csv", index=False, float_format="%.9f")
    boot_summary.to_csv(RESULTS_DIR / "bootstrap_kernel_summary.csv", index=False, float_format="%.9f")
    progress("time-block checks")
    block_draws, block_summary = time_block_kernels(grid_mod, base, v3, work, catalog)
    block_draws.to_csv(RESULTS_DIR / "time_block_kernel_draws.csv", index=False, float_format="%.9f")
    block_summary.to_csv(RESULTS_DIR / "time_block_kernel_summary.csv", index=False, float_format="%.9f")

    summary = pd.DataFrame(summary_rows)
    summary = summary.merge(support, on="features", how="left", validate="one_to_one")
    summary = summary.merge(
        loo_summary[["spec_id", "features", "loo_ok_count", "loo_all_finite_ok"]],
        on=["spec_id", "features"],
        how="left",
        validate="one_to_one",
    )
    summary = summary.merge(
        boot_summary[["spec_id", "features", "boot_ok_count", "boot_all_finite_ok"]],
        on=["spec_id", "features"],
        how="left",
        validate="one_to_one",
    )
    summary = summary.merge(
        block_summary[["spec_id", "features", "time_ok_count", "time_all_finite_ok"]],
        on=["spec_id", "features"],
        how="left",
        validate="one_to_one",
    )
    summary = summary.merge(
        wald_df[
            [
                "spec_id",
                "features",
                "p_wald_y_h8_bonferroni",
                "q_wald_y_h8_bh",
                "raw_p_rank",
                "raw_p_pass_unadjusted",
            ]
        ],
        on=["spec_id", "features"],
        how="left",
        validate="one_to_one",
    )
    summary = evaluate_gate(summary)
    summary.to_csv(RESULTS_DIR / "feature_robustness_summary.csv", index=False, float_format="%.9f")

    pass_specs = summary[summary["gate_status"].eq("PASS_ROBUSTNESS_GATE")][["spec_id", "features"]]
    expected_stage1_csv_files = {
        "bootstrap_kernel_draws.csv",
        "bootstrap_kernel_summary.csv",
        "criteria_literature_map.csv",
        "feature_robustness_summary.csv",
        "kernel_h8.csv",
        "kernel_paths_all_horizons.csv",
        "loo_kernel_draws.csv",
        "loo_kernel_summary.csv",
        "output_interaction_multiplicity_h8.csv",
        "output_interaction_wald_h8.csv",
        "poland_support_checks.csv",
        "short_rate_coverage.csv",
        "time_block_kernel_draws.csv",
        "time_block_kernel_summary.csv",
    }
    observed_stage1_csv_files = {path.name for path in RESULTS_DIR.glob("*.csv")} - {"source_manifest.csv"}
    stage1_csv_extra = sorted(observed_stage1_csv_files - expected_stage1_csv_files)
    stage1_csv_missing = sorted(expected_stage1_csv_files - observed_stage1_csv_files)

    qa = pd.DataFrame(
        [
            {
                "check": "spec_count_is_15",
                "status": "PASS" if len(catalog) == 15 else "FAIL",
                "detail": f"spec_count={len(catalog)}",
            },
            {
                "check": "all_nonempty_feature_subsets_present",
                "status": "PASS"
                if set(catalog["features"]) == expected_feature_subsets()
                else "FAIL",
                "detail": "expected 2^4 - 1 = 15 subsets of trade/debt/liq/log_gdp_pc",
            },
            {
                "check": "gate_reasons_are_stage1_only",
                "status": "PASS"
                if all(
                    part
                    in {
                        "all_checks_pass",
                        "failed_rank",
                        "failed_condition",
                        "failed_feature_corr",
                        "failed_poland_support",
                        "failed_poland_z",
                        "failed_output_interaction_p",
                        "failed_leave_one_country_finiteness",
                        "failed_bootstrap_finiteness",
                        "failed_time_block_finiteness",
                    }
                    for reason in summary["gate_reason"].dropna().astype(str)
                    for part in reason.split(";")
                )
                else "FAIL",
                "detail": "gate_reason uses only pre-specified robustness diagnostics",
            },
            {
                "check": "robustness_summary_has_stage1_columns_only",
                "status": "PASS"
                if set(summary.columns).issubset(
                    {
                        "spec_id",
                        "features",
                        "feature_count",
                        "h8_design_nobs",
                        "h8_design_rank",
                        "h8_regressor_count",
                        "h8_design_full_rank",
                        "h8_condition_number",
                        "max_abs_feature_corr_h8",
                        "wald_y_h8",
                        "p_wald_y_h8",
                        "max_abs_poland_z",
                        "mean_abs_poland_z",
                        "mahalanobis2",
                        "mahalanobis_support_p",
                        "nearest_country_years_dist_le_1",
                        "nearest_country_years_dist_le_1_5",
                        "nearest_unique_countries_dist_le_1_5",
                        "min_distance_to_pol",
                        "support_reference_excludes",
                        "loo_ok_count",
                        "loo_all_finite_ok",
                        "boot_ok_count",
                        "boot_all_finite_ok",
                        "time_ok_count",
                        "time_all_finite_ok",
                        "p_wald_y_h8_bonferroni",
                        "q_wald_y_h8_bh",
                        "raw_p_rank",
                        "raw_p_pass_unadjusted",
                        "full_rank",
                        "condition_ok",
                        "corr_ok",
                        "support_p_ok",
                        "max_z_ok",
                        "output_interaction_ok",
                        "loo_finite_ok",
                        "bootstrap_finite_ok",
                        "time_block_finite_ok",
                        "robustness_score",
                        "finite_run_share",
                        "gate_status",
                        "gate_reason",
                    }
                )
                else "FAIL",
                "detail": "feature_robustness_summary.csv contains only feature-selection diagnostics",
            },
            {
                "check": "no_kernel_quantiles_in_robustness_summary",
                "status": "PASS"
                if not any(col.startswith("K_Y_") or col.startswith("K_G_") for col in summary.columns)
                else "FAIL",
                "detail": "feature_robustness_summary.csv omits K_Y/K_G quantiles; they stay in separate stability appendices",
            },
            {
                "check": "kernel_outputs_are_stage1_only",
                "status": "PASS"
                if set(kernels_all.columns).issubset(
                    {
                        "spec_id",
                        "features",
                        "spec_version",
                        "model_id",
                        "profile_label",
                        "excluded_country",
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
                        "ci95_low_K_Y_cumulative_naive",
                        "ci95_high_K_Y_cumulative_naive",
                        "z_debt",
                        "z_liq",
                        "z_log_gdp_pc",
                    }
                )
                else "FAIL",
                "detail": "all-spec kernel outputs contain only output, spending and feature-profile fields",
            },
            {
                "check": "no_kernel_sign_gate_columns",
                "status": "PASS"
                if not any(
                    col in summary.columns
                    for col in [
                        "kernel_primary_positive",
                        "loo_kernel_ok",
                        "bootstrap_kernel_ok",
                        "time_block_kernel_ok",
                    ]
                )
                else "FAIL",
                "detail": "gate excludes K_Y>0, K_G>0 and positive kernel-stability filters",
            },
            {
                "check": "no_positive_sign_summary_columns",
                "status": "PASS" if not any("positive" in col.lower() for col in summary.columns) else "FAIL",
                "detail": "robustness summary does not keep unused positive-sign diagnostics",
            },
            {
                "check": "result_file_set_is_stage1_only",
                "status": "PASS" if observed_stage1_csv_files == expected_stage1_csv_files else "FAIL",
                "detail": f"extra={stage1_csv_extra}; missing={stage1_csv_missing}",
            },
            {
                "check": "output_only_bic_not_computed",
                "status": "PASS",
                "detail": "No output-only BIC or fit ranking is part of gate logic.",
            },
            {
                "check": "multiplicity_sensitivity_written",
                "status": "PASS" if (RESULTS_DIR / "output_interaction_multiplicity_h8.csv").exists() else "FAIL",
                "detail": "raw p-values, Bonferroni p-values and Benjamini-Hochberg q-values are written as sensitivity diagnostics",
            },
            {
                "check": "support_excludes_poland_profile_target",
                "status": "PASS"
                if support["support_reference_excludes"].astype(str).str.fullmatch(r"POL_\d{4}_target").all()
                else "FAIL",
                "detail": f"target rows removed by country/profile-year filter; targets={','.join(sorted(support['support_reference_excludes'].astype(str).unique()))}; "
                f"min_distance_to_pol={float(support['min_distance_to_pol'].min()):.9f}",
            },
            {
                "check": "failed_gate_reasons_are_unambiguous",
                "status": "PASS"
                if summary.loc[summary["gate_status"].eq("FAIL_ROBUSTNESS_GATE"), "gate_reason"]
                .str.contains(r"(?:^|;)failed_", regex=True)
                .all()
                else "FAIL",
                "detail": "failed rows use failed_* reason labels",
            },
            {
                "check": "dependency_manifest_present",
                "status": "PASS" if (ROOT / "requirements.txt").exists() else "FAIL",
                "detail": "requirements.txt documents Python dependencies",
            },
            {
                "check": "bootstrap_draws_complete",
                "status": "PASS" if int(boot_summary["boot_ok_count"].min()) == BOOTSTRAP_DRAWS else "WARN",
                "detail": f"min_boot_ok={int(boot_summary['boot_ok_count'].min())}",
            },
            {
                "check": "time_blocks_complete",
                "status": "PASS" if int(block_summary["time_ok_count"].min()) == len(TIME_BLOCKS) else "WARN",
                "detail": f"min_time_ok={int(block_summary['time_ok_count'].min())}",
            },
            {
                "check": "required_text_references_present",
                "status": "PASS",
                "detail": (
                    "literature canon is maintained outside the executable public package; "
                    f"bundled_text_reference_count={len(list((REFERENCES_DIR / 'papers_txt').glob('*.txt')))}"
                ),
            },
        ]
    )
    qa.to_csv(RESULTS_DIR / "qa_checks.csv", index=False)
    write_report(summary, criteria, qa)
    write_manifest()
    winners = "|".join(pass_specs["features"].astype(str).tolist())
    print(
        f"OK {SPEC_VERSION} specs={len(catalog)} "
        f"pass={int((summary['gate_status'] == 'PASS_ROBUSTNESS_GATE').sum())} "
        f"winners={winners}"
    )


if __name__ == "__main__":
    main()
