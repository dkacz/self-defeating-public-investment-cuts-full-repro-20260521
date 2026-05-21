#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import math
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import NormalDist
from typing import Any
from urllib.parse import urlencode

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
RESULTS_DIR = TASK_DIR / "results"
REFERENCES_DIR = TASK_DIR / "references"

SPEC_VERSION = "ciaffi_canonical_c_eu_20260424_v3_short_rate_fixed_window"
PANEL_START_YEAR = 1995
SHOCK_START_YEAR = 2004
END_YEAR = 2024
LAG_DEPTHS = (1, 2, 3, 4)
HORIZONS = tuple(range(9))
MIN_LP_OBS = 50

INPUT_PANEL = TASK_DIR / "data/eu27_lp_joint_panel_snapshot.csv"
EUROSTAT_API_BASE = "https://ec.europa.eu/eurostat/api/dissemination/statistics/1.0/data"
EUROSTAT_SHORT_RATE_DATASETS = ("irt_st_a", "irt_h_mr3_a", "irt_st_m", "irt_h_mr3_m")

EU27 = [
    "AUT",
    "BEL",
    "BGR",
    "CYP",
    "CZE",
    "DEU",
    "DNK",
    "ESP",
    "EST",
    "FIN",
    "FRA",
    "GRC",
    "HRV",
    "HUN",
    "IRL",
    "ITA",
    "LTU",
    "LUX",
    "LVA",
    "MLT",
    "NLD",
    "POL",
    "PRT",
    "ROU",
    "SVK",
    "SVN",
    "SWE",
]

ISO3_TO_EUROSTAT_GEO = {
    "AUT": "AT",
    "BEL": "BE",
    "BGR": "BG",
    "CYP": "CY",
    "CZE": "CZ",
    "DEU": "DE",
    "DNK": "DK",
    "ESP": "ES",
    "EST": "EE",
    "FIN": "FI",
    "FRA": "FR",
    "GRC": "EL",
    "HRV": "HR",
    "HUN": "HU",
    "IRL": "IE",
    "ITA": "IT",
    "LTU": "LT",
    "LUX": "LU",
    "LVA": "LV",
    "MLT": "MT",
    "NLD": "NL",
    "POL": "PL",
    "PRT": "PT",
    "ROU": "RO",
    "SVK": "SK",
    "SVN": "SI",
    "SWE": "SE",
}

EURO_ADOPTION_YEAR = {
    "AUT": 1999,
    "BEL": 1999,
    "DEU": 1999,
    "ESP": 1999,
    "FIN": 1999,
    "FRA": 1999,
    "IRL": 1999,
    "ITA": 1999,
    "LUX": 1999,
    "NLD": 1999,
    "PRT": 1999,
    "GRC": 2001,
    "SVN": 2007,
    "CYP": 2008,
    "MLT": 2008,
    "SVK": 2009,
    "EST": 2011,
    "LVA": 2014,
    "LTU": 2015,
    "HRV": 2023,
}

SYSTEM_COMPONENT = ["dlog_gi", "dlog_gc", "dlog_y", "i_rate"]
SYSTEM_AGGREGATE = ["dlog_g", "dlog_y", "i_rate"]
SHOCKS = ("G_I", "G_C", "aggregate")
Z95 = NormalDist().inv_cdf(0.975)
LINALG_RCOND = 1e-12
LINALG_RANK_TOL = 1e-10


@dataclass(frozen=True)
class FitResult:
    coef: float
    se: float
    nobs: int
    country_n: int
    year_min: int | None
    year_max: int | None
    beta_dep: float
    beta_scale: float
    status: str
    rank: int


class CanonicalError(RuntimeError):
    pass


class FEProjector:
    def __init__(self, country: pd.Series, year: pd.Series) -> None:
        country_d = pd.get_dummies(country.astype(str).reset_index(drop=True), dtype=float)
        year_d = pd.get_dummies(year.astype(int).astype(str).reset_index(drop=True), dtype=float)
        country_d = country_d.reindex(sorted(country_d.columns), axis=1)
        year_d = year_d.reindex(sorted(year_d.columns), axis=1)
        country_arr = country_d.iloc[:, 1:].to_numpy(dtype=float) if country_d.shape[1] > 1 else np.empty((len(country_d), 0))
        year_arr = year_d.iloc[:, 1:].to_numpy(dtype=float) if year_d.shape[1] > 1 else np.empty((len(year_d), 0))
        intercept = np.ones((len(country_d), 1), dtype=float)
        self.design = np.hstack([intercept, country_arr, year_arr])
        self.pinv = np.linalg.pinv(self.design, rcond=LINALG_RCOND)

    def residualize(self, values: np.ndarray) -> np.ndarray:
        arr = np.asarray(values, dtype=float)
        was_1d = arr.ndim == 1
        if was_1d:
            arr = arr.reshape(-1, 1)
        resid = arr - self.design @ (self.pinv @ arr)
        return resid.reshape(-1) if was_1d else resid


def current_git_head() -> str:
    proc = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=False)
    return proc.stdout.strip() if proc.returncode == 0 else "unknown"


def current_git_branch() -> str:
    proc = subprocess.run(["git", "branch", "--show-current"], cwd=ROOT, capture_output=True, text=True, check=False)
    return proc.stdout.strip() if proc.returncode == 0 else "unknown"


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def eurostat_url(dataset: str) -> str:
    return f"{EUROSTAT_API_BASE}/{dataset}?{urlencode({'format': 'JSON', 'lang': 'en'})}"


def ordered_categories(payload: dict[str, Any], dim: str) -> list[str]:
    category = payload["dimension"][dim]["category"]
    index = category["index"]
    if isinstance(index, dict):
        return [key for key, _pos in sorted(index.items(), key=lambda item: int(item[1]))]
    return list(index)


def jsonstat_to_frame(payload: dict[str, Any], dataset: str, source_url: str) -> pd.DataFrame:
    dims = list(payload["id"])
    sizes = [int(x) for x in payload["size"]]
    categories = [ordered_categories(payload, dim) for dim in dims]
    status = payload.get("status", {})
    rows: list[dict[str, Any]] = []
    for flat_key, value in payload.get("value", {}).items():
        rem = int(flat_key)
        coords: list[int] = []
        for size in reversed(sizes):
            coords.append(rem % size)
            rem //= size
        coords = list(reversed(coords))
        row = {dim: categories[pos][coords[pos]] for pos, dim in enumerate(dims)}
        row.update(
            {
                "value": float(value),
                "status": status.get(str(flat_key), ""),
                "dataset": dataset,
                "dataset_label": payload.get("label", ""),
                "dataset_updated": payload.get("updated", ""),
                "source_url": source_url,
                "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def ensure_eurostat_snapshot(dataset: str) -> pd.DataFrame:
    REFERENCES_DIR.mkdir(parents=True, exist_ok=True)
    path = REFERENCES_DIR / f"eurostat_{dataset}_snapshot.csv"
    if path.exists():
        out = pd.read_csv(path)
        out["value"] = pd.to_numeric(out["value"], errors="coerce")
        return out

    raise CanonicalError(
        f"Missing bundled Eurostat short-rate snapshot {path}. "
        "This repro package is frozen to local inputs and does not fetch URLs."
    )


def annual_m3(snapshot: pd.DataFrame, dataset: str) -> pd.DataFrame:
    if snapshot.empty:
        return pd.DataFrame(columns=["dataset", "geo", "year", "short_rate_percent", "obs_count"])
    work = snapshot[snapshot["int_rt"].eq("IRT_M3")].copy()
    work["value"] = pd.to_numeric(work["value"], errors="coerce")
    work = work.dropna(subset=["value", "geo", "time"])
    if dataset.endswith("_m"):
        work["year"] = work["time"].astype(str).str.slice(0, 4).astype(int)
        out = (
            work.groupby(["geo", "year"], as_index=False)
            .agg(short_rate_percent=("value", "mean"), obs_count=("value", "count"))
            .assign(dataset=dataset)
        )
    else:
        work["year"] = work["time"].astype(int)
        out = work[["geo", "year", "value"]].rename(columns={"value": "short_rate_percent"})
        out["obs_count"] = 1
        out["dataset"] = dataset
    return out[["dataset", "geo", "year", "short_rate_percent", "obs_count"]]


def write_short_rate_source_notes(snapshots: dict[str, pd.DataFrame]) -> None:
    rows = []
    for dataset, snap in snapshots.items():
        if snap.empty:
            continue
        first = snap.iloc[0]
        rows.append(
            {
                "dataset": dataset,
                "label": first.get("dataset_label", ""),
                "updated": first.get("dataset_updated", ""),
                "snapshot": f"data/eurostat_{dataset}_snapshot.csv",
                "source_url": first.get("source_url", eurostat_url(dataset)),
            }
        )
    notes = [
        "# Eurostat Short-Rate Source Notes",
        "",
        "Primary short-term rate source is Eurostat `irt_st_a`, `IRT_M3` (3-month money-market rate).",
        "Fallbacks are Eurostat-only: harmonised local 3-month annual series `irt_h_mr3_a`, then annual averages computed from monthly `irt_st_m`, then `irt_h_mr3_m`.",
        "For euro-area membership years the panel uses `EA` `IRT_M3`; for pre-euro or non-euro years it uses the local Eurostat geo code when available.",
        "Missing values are left missing; complete-case local-projection construction drops observations only when the rate is required.",
        "",
        "| dataset | label | updated | snapshot | source_url |",
        "| --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        notes.append(
            f"| {row['dataset']} | {row['label']} | {row['updated']} | {row['snapshot']} | {row['source_url']} |"
        )
    (REFERENCES_DIR / "eurostat_short_rate_source_notes.md").write_text("\n".join(notes) + "\n", encoding="utf-8")


def build_short_rate_coverage() -> pd.DataFrame:
    snapshots = {dataset: ensure_eurostat_snapshot(dataset) for dataset in EUROSTAT_SHORT_RATE_DATASETS}
    write_short_rate_source_notes(snapshots)
    annual = {
        dataset: annual_m3(snapshot, dataset).set_index(["geo", "year"])
        for dataset, snapshot in snapshots.items()
    }

    def lookup(dataset: str, geo: str, year: int) -> tuple[float | None, int | None]:
        table = annual[dataset]
        key = (geo, int(year))
        if key not in table.index:
            return None, None
        row = table.loc[key]
        return float(row["short_rate_percent"]), int(row["obs_count"])

    rows: list[dict[str, Any]] = []
    for country in EU27:
        geo = ISO3_TO_EUROSTAT_GEO[country]
        adoption_year = EURO_ADOPTION_YEAR.get(country)
        for year in range(PANEL_START_YEAR, END_YEAR + 1):
            in_euro_area_year = adoption_year is not None and year >= adoption_year
            if in_euro_area_year:
                candidates = [
                    ("irt_st_a", "EA", "EA_after_euro_adoption", "primary_irt_st_a_annual"),
                    ("irt_st_m", "EA", "EA_after_euro_adoption", "fallback_monthly_annual_average_gap_in_irt_st_a"),
                ]
            else:
                candidates = [
                    ("irt_st_a", geo, "local_pre_euro_or_non_euro", "primary_irt_st_a_annual"),
                    ("irt_h_mr3_a", geo, "local_pre_euro", "fallback_harmonised_local_annual_gap_in_irt_st_a"),
                    ("irt_st_m", geo, "local_pre_euro_or_non_euro", "fallback_monthly_annual_average_gap_in_irt_st_a"),
                    ("irt_h_mr3_m", geo, "local_pre_euro", "fallback_harmonised_monthly_annual_average_gap"),
                ]

            chosen: dict[str, Any] | None = None
            attempted = []
            for priority, (dataset, source_geo, source_rule, fallback_reason) in enumerate(candidates, start=1):
                attempted.append(f"{dataset}:{source_geo}")
                value, obs_count = lookup(dataset, source_geo, year)
                if value is not None and math.isfinite(value):
                    chosen = {
                        "dataset": dataset,
                        "source_geo": source_geo,
                        "source_rule": source_rule,
                        "fallback_reason": fallback_reason,
                        "source_priority": priority,
                        "short_rate_percent": value,
                        "source_obs_count": obs_count,
                    }
                    break

            missing_reason = ""
            if chosen is None:
                missing_reason = "missing_in_eurostat_irt_st_a_and_eurostat_fallbacks"
                chosen = {
                    "dataset": "MISSING",
                    "source_geo": geo if not in_euro_area_year else "EA",
                    "source_rule": "MISSING",
                    "fallback_reason": "dropped_if_required_by_complete_case_lp",
                    "source_priority": len(candidates) + 1,
                    "short_rate_percent": math.nan,
                    "source_obs_count": 0,
                }

            rows.append(
                {
                    "spec_version": SPEC_VERSION,
                    "country": country,
                    "eurostat_geo": geo,
                    "year": year,
                    "euro_adoption_year": adoption_year if adoption_year is not None else "",
                    "in_euro_area_year": bool(in_euro_area_year),
                    "short_rate_available": bool(math.isfinite(float(chosen["short_rate_percent"]))),
                    "short_rate_percent": chosen["short_rate_percent"],
                    "short_rate_ratio": chosen["short_rate_percent"] / 100.0
                    if math.isfinite(float(chosen["short_rate_percent"]))
                    else math.nan,
                    "short_rate_source_dataset": chosen["dataset"],
                    "short_rate_source_geo": chosen["source_geo"],
                    "short_rate_source_rule": chosen["source_rule"],
                    "fallback_reason": chosen["fallback_reason"],
                    "source_priority": chosen["source_priority"],
                    "source_obs_count": chosen["source_obs_count"],
                    "missing_reason": missing_reason,
                    "attempted_sources": ";".join(attempted),
                }
            )
    return pd.DataFrame(rows)


def as_ratio(series: pd.Series) -> pd.Series:
    out = pd.to_numeric(series, errors="coerce")
    med = out.abs().median(skipna=True)
    if pd.notna(med) and med > 2.0:
        out = out / 100.0
    return out


def ols_fit(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    beta, *_ = np.linalg.lstsq(x, y, rcond=LINALG_RCOND)
    fitted = x @ beta
    resid = y - fitted
    xtx_inv = np.linalg.pinv(x.T @ x, rcond=LINALG_RCOND)
    rank = int(np.linalg.matrix_rank(x, tol=LINALG_RANK_TOL))
    return beta, fitted, resid, xtx_inv, rank


def dk_scores(x: np.ndarray, resid: np.ndarray, years: np.ndarray) -> np.ndarray:
    unique_years = np.array(sorted(pd.unique(years)))
    scores = np.zeros((len(unique_years), x.shape[1]), dtype=float)
    for idx, year in enumerate(unique_years):
        mask = years == year
        scores[idx] = x[mask].T @ resid[mask]
    return scores


def dk_inner_cross(x: np.ndarray, resid_a: np.ndarray, resid_b: np.ndarray, years: np.ndarray, bandwidth: int) -> np.ndarray:
    scores_a = dk_scores(x, resid_a, years)
    scores_b = dk_scores(x, resid_b, years)
    inner = scores_a.T @ scores_b
    max_lag = min(max(int(bandwidth), 0), max(scores_a.shape[0] - 1, 0))
    for lag in range(1, max_lag + 1):
        weight = 1.0 - lag / (max_lag + 1.0)
        inner += weight * ((scores_a[lag:].T @ scores_b[:-lag]) + (scores_b[lag:].T @ scores_a[:-lag]))
    return inner


def dk_covariance(x: np.ndarray, resid: np.ndarray, years: np.ndarray, xtx_inv: np.ndarray, bandwidth: int) -> np.ndarray:
    nobs, k = x.shape
    t_count = len(pd.unique(years))
    cov = xtx_inv @ dk_inner_cross(x, resid, resid, years, bandwidth) @ xtx_inv
    if t_count > 1:
        cov *= t_count / (t_count - 1.0)
    if nobs > k:
        cov *= (nobs - 1.0) / (nobs - k)
    return cov


def dk_cross_covariance(
    x: np.ndarray,
    resid_a: np.ndarray,
    resid_b: np.ndarray,
    years: np.ndarray,
    xtx_inv: np.ndarray,
    bandwidth: int,
) -> np.ndarray:
    nobs, k = x.shape
    t_count = len(pd.unique(years))
    cov = xtx_inv @ dk_inner_cross(x, resid_a, resid_b, years, bandwidth) @ xtx_inv
    if t_count > 1:
        cov *= t_count / (t_count - 1.0)
    if nobs > k:
        cov *= (nobs - 1.0) / (nobs - k)
    return cov


def ratio_and_se(beta_dep: float, beta_scale: float, var_dep: float, var_scale: float, cov_dep_scale: float) -> tuple[float, float]:
    vals = [beta_dep, beta_scale, var_dep, var_scale, cov_dep_scale]
    if not all(math.isfinite(v) for v in vals):
        return math.nan, math.nan
    if abs(beta_scale) < 1e-14:
        return math.nan, math.nan
    ratio = beta_dep / beta_scale
    grad = np.array([1.0 / beta_scale, -beta_dep / (beta_scale * beta_scale)], dtype=float)
    vcov = np.array([[var_dep, cov_dep_scale], [cov_dep_scale, var_scale]], dtype=float)
    variance = float(grad @ vcov @ grad)
    se = math.sqrt(max(variance, 0.0)) if math.isfinite(variance) else math.nan
    return float(ratio), float(se)


def ci_low(coef: float, se: float) -> float:
    return coef - Z95 * se if math.isfinite(coef) and math.isfinite(se) else math.nan


def ci_high(coef: float, se: float) -> float:
    return coef + Z95 * se if math.isfinite(coef) and math.isfinite(se) else math.nan


def normal_p_value(coef: float, se: float) -> float:
    if not (math.isfinite(coef) and math.isfinite(se) and se > 0):
        return math.nan
    z = abs(coef / se)
    return 2.0 * (1.0 - NormalDist().cdf(z))


def shock_window(horizon: int) -> tuple[int, int]:
    return SHOCK_START_YEAR, END_YEAR - int(horizon)


def key_set(df: pd.DataFrame) -> set[tuple[str, int]]:
    return set(zip(df["country"].astype(str), df["year"].astype(int)))


def filter_keys(df: pd.DataFrame, allowed_keys: set[tuple[str, int]] | None) -> pd.DataFrame:
    if allowed_keys is None:
        return df
    keep = [(str(country), int(year)) in allowed_keys for country, year in zip(df["country"], df["year"])]
    return df.loc[keep].copy()


def sample_fingerprint(keys: set[tuple[str, int]]) -> str:
    payload = "\n".join(f"{country}:{year}" for country, year in sorted(keys))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def load_panel() -> tuple[pd.DataFrame, dict[str, Any]]:
    if not INPUT_PANEL.exists():
        raise CanonicalError(f"Missing input panel: {INPUT_PANEL}")
    raw = pd.read_csv(INPUT_PANEL)
    coverage = build_short_rate_coverage()
    coverage_path = RESULTS_DIR / "short_rate_coverage.csv"
    coverage.to_csv(coverage_path, index=False, float_format="%.9f")

    panel = raw[raw["year"].between(PANEL_START_YEAR, END_YEAR)].copy()
    panel["country"] = panel["country"].astype(str)
    panel = panel[panel["country"].isin(EU27)].copy()

    numeric = [
        "y_real",
        "gi_real",
        "gc_real",
    ]
    for col in numeric:
        if col in panel:
            panel[col] = pd.to_numeric(panel[col], errors="coerce")

    merge_cols = [
        "country",
        "year",
        "short_rate_percent",
        "short_rate_ratio",
        "short_rate_available",
        "short_rate_source_dataset",
        "short_rate_source_geo",
        "short_rate_source_rule",
        "fallback_reason",
        "missing_reason",
    ]
    panel = panel.merge(
        coverage[merge_cols],
        on=["country", "year"],
        how="left",
        validate="one_to_one",
        suffixes=("_input_panel", ""),
    )
    panel["i_rate_source"] = np.where(
        panel["short_rate_available"].fillna(False),
        panel["short_rate_source_dataset"].astype(str)
        + ":"
        + panel["short_rate_source_geo"].astype(str)
        + ":"
        + panel["short_rate_source_rule"].astype(str),
        "MISSING",
    )
    panel["i_rate"] = as_ratio(panel["short_rate_percent"])
    panel = panel.sort_values(["country", "year"]).reset_index(drop=True)

    expected_rows = len(EU27) * (END_YEAR - PANEL_START_YEAR + 1)
    coverage_by_country = [
        {
            "country": str(country),
            "available_years": int(group["short_rate_available"].sum()),
            "total_years": int(len(group)),
            "missing_years": int((~group["short_rate_available"]).sum()),
        }
        for country, group in coverage.groupby("country", sort=True)
    ]
    meta = {
        "source": str(INPUT_PANEL.relative_to(ROOT)),
        "input_sha256": file_sha256(INPUT_PANEL),
        "row_count": int(len(panel)),
        "expected_balanced_rows": int(expected_rows),
        "country_n": int(panel["country"].nunique()),
        "year_min": int(panel["year"].min()),
        "year_max": int(panel["year"].max()),
        "balanced": bool(
            len(panel) == expected_rows
            and panel.groupby("country")["year"].nunique().eq(END_YEAR - PANEL_START_YEAR + 1).all()
        ),
        "i_source_counts": panel["i_rate_source"].value_counts(dropna=False).to_dict(),
        "i_missing": int(panel["i_rate"].isna().sum()),
        "short_rate_coverage_csv": str(coverage_path.relative_to(ROOT)),
        "short_rate_coverage_rows": int(len(coverage)),
        "short_rate_coverage_available_rows": int(coverage["short_rate_available"].sum()),
        "short_rate_coverage_missing_rows": int((~coverage["short_rate_available"]).sum()),
        "short_rate_coverage_by_country": coverage_by_country,
    }
    return panel.replace([np.inf, -np.inf], np.nan), meta


def prepare_work(panel: pd.DataFrame) -> pd.DataFrame:
    work = panel.copy().sort_values(["country", "year"]).reset_index(drop=True)
    work = work[(work["y_real"] > 0) & (work["gi_real"] > 0) & (work["gc_real"] > 0)].copy()
    group = work.groupby("country", sort=False)
    work["g_real"] = work["gi_real"] + work["gc_real"]
    work["log_y"] = np.log(work["y_real"])
    work["log_gi"] = np.log(work["gi_real"])
    work["log_gc"] = np.log(work["gc_real"])
    work["log_g"] = np.log(work["g_real"])
    work["dlog_y"] = group["log_y"].diff()
    work["dlog_gi"] = group["log_gi"].diff()
    work["dlog_gc"] = group["log_gc"].diff()
    work["dlog_g"] = group["log_g"].diff()

    for var in ["dlog_gi", "dlog_gc", "dlog_g", "dlog_y", "i_rate"]:
        for lag in range(1, max(LAG_DEPTHS) + 1):
            work[f"{var}_lag{lag}"] = group[var].shift(lag)

    work["y_level_lag1"] = group["y_real"].shift(1)
    work["gi_dyn0"] = (work["gi_real"] - group["gi_real"].shift(1)) / work["y_level_lag1"]
    work["gc_dyn0"] = (work["gc_real"] - group["gc_real"].shift(1)) / work["y_level_lag1"]
    work["g_dyn0"] = (work["g_real"] - group["g_real"].shift(1)) / work["y_level_lag1"]

    for h in HORIZONS:
        y_tph = group["y_real"].shift(-h)
        y_prev = group["y_real"].shift(1) if h == 0 else group["y_real"].shift(-(h - 1))
        work[f"y_dyn_h{h}"] = (y_tph - y_prev) / work["y_level_lag1"]
        work[f"y_dyn_pp_h{h}"] = work[f"y_dyn_h{h}"] * 100.0

    return work.replace([np.inf, -np.inf], np.nan)


def system_lag_controls(variables: list[str], lag_depth: int) -> list[str]:
    return [f"{var}_lag{lag}" for lag in range(1, lag_depth + 1) for var in variables]


def lagged_shock_controls(shock_cols: list[str], lag_depth: int) -> list[str]:
    return [f"{shock_col}_lag{lag}" for lag in range(1, lag_depth + 1) for shock_col in shock_cols]


def cholesky_shocks(
    work: pd.DataFrame,
    variables: list[str],
    shock_map: dict[str, str],
    lag_depth: int,
    system_name: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    controls = system_lag_controls(variables, lag_depth)
    needed = variables + controls + ["country", "year"]
    sample = work.dropna(subset=needed).sort_values(["country", "year"]).reset_index(drop=True)
    if len(sample) < MIN_LP_OBS:
        raise CanonicalError(f"Insufficient Cholesky sample for {system_name} lag {lag_depth}: {len(sample)}")

    projector = FEProjector(sample["country"], sample["year"])
    x_res = projector.residualize(sample[controls].to_numpy(dtype=float))
    residuals: list[np.ndarray] = []
    ranks: dict[str, int] = {}
    for dep in variables:
        y_res = projector.residualize(sample[dep].to_numpy(dtype=float))
        _beta, _fit, resid, _xtx, rank = ols_fit(x_res, y_res)
        residuals.append(resid)
        ranks[dep] = rank

    u = np.column_stack(residuals)
    sigma = (u.T @ u) / max(len(sample), 1)
    jitter = 1e-12
    for _ in range(10):
        try:
            chol = np.linalg.cholesky(sigma + np.eye(sigma.shape[0]) * jitter)
            break
        except np.linalg.LinAlgError:
            jitter *= 10.0
    else:
        raise CanonicalError(f"Cholesky failed for {system_name} lag {lag_depth}")

    structural_unit = np.linalg.solve(chol, u.T).T
    raw_recursive = structural_unit * np.diag(chol)
    shocks = sample[["country", "year"]].copy()
    for pos, dep in enumerate(variables):
        if dep in shock_map:
            shocks[shock_map[dep]] = raw_recursive[:, pos]

    meta = {
        "system": system_name,
        "variables": variables,
        "lag_depth": lag_depth,
        "nobs": int(len(sample)),
        "country_n": int(sample["country"].nunique()),
        "year_min": int(sample["year"].min()),
        "year_max": int(sample["year"].max()),
        "controls": controls,
        "reduced_form_ranks": ranks,
        "sigma_diag": [float(x) for x in np.diag(sigma)],
        "chol_diag": [float(x) for x in np.diag(chol)],
    }
    return shocks, meta


def attach_shocks(work: pd.DataFrame, lag_depth: int) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    comp_shocks, comp_meta = cholesky_shocks(
        work,
        SYSTEM_COMPONENT,
        {"dlog_gi": "shock_G_I", "dlog_gc": "shock_G_C"},
        lag_depth,
        "components_GI_GC_Y_i",
    )
    agg_shocks, agg_meta = cholesky_shocks(
        work,
        SYSTEM_AGGREGATE,
        {"dlog_g": "shock_aggregate"},
        lag_depth,
        "aggregate_G_Y_i",
    )
    out = work.merge(comp_shocks, on=["country", "year"], how="left").merge(agg_shocks, on=["country", "year"], how="left")
    group = out.groupby("country", sort=False)
    for shock_col in ["shock_G_I", "shock_G_C", "shock_aggregate"]:
        for lag in range(1, lag_depth + 1):
            out[f"{shock_col}_lag{lag}"] = group[shock_col].shift(lag)
    return out.replace([np.inf, -np.inf], np.nan), [comp_meta, agg_meta]


def estimate_ratio(
    work: pd.DataFrame,
    dep_col: str,
    scale_col: str,
    x_cols: list[str],
    coef_col: str,
    horizon: int,
    sample_keys: set[tuple[str, int]] | None = None,
) -> FitResult:
    needed = [dep_col, scale_col, *x_cols, "country", "year"]
    window_start, window_end = shock_window(horizon)
    sample = work[work["year"].between(window_start, window_end)].copy()
    sample = filter_keys(sample, sample_keys)
    sample = sample.dropna(subset=needed).sort_values(["country", "year"]).reset_index(drop=True)
    if len(sample) < MIN_LP_OBS:
        return FitResult(
            math.nan,
            math.nan,
            int(len(sample)),
            int(sample["country"].nunique()) if len(sample) else 0,
            None,
            None,
            math.nan,
            math.nan,
            "INSUFFICIENT_OBS",
            0,
        )
    projector = FEProjector(sample["country"], sample["year"])
    x_res = projector.residualize(sample[x_cols].to_numpy(dtype=float))
    years = sample["year"].to_numpy(dtype=int)
    dep_res = projector.residualize(sample[dep_col].to_numpy(dtype=float))
    scale_res = projector.residualize(sample[scale_col].to_numpy(dtype=float))
    beta_dep, _fit_dep, resid_dep, xtx_inv, rank = ols_fit(x_res, dep_res)
    beta_scale, _fit_scale, resid_scale, _xtx_scale, _rank_scale = ols_fit(x_res, scale_res)
    bandwidth = max(int(horizon), 1)
    vcov_dep = dk_covariance(x_res, resid_dep, years, xtx_inv, bandwidth)
    vcov_scale = dk_covariance(x_res, resid_scale, years, xtx_inv, bandwidth)
    vcov_cross = dk_cross_covariance(x_res, resid_dep, resid_scale, years, xtx_inv, bandwidth)
    pos = x_cols.index(coef_col)
    coef, se = ratio_and_se(
        float(beta_dep[pos]),
        float(beta_scale[pos]),
        float(vcov_dep[pos, pos]),
        float(vcov_scale[pos, pos]),
        float(vcov_cross[pos, pos]),
    )
    status = "OK" if math.isfinite(coef) and math.isfinite(se) else "NONFINITE"
    if abs(float(beta_scale[pos])) < 1e-8:
        status = "WEAK_SCALE_DENOMINATOR"
    return FitResult(
        coef=coef,
        se=se,
        nobs=int(len(sample)),
        country_n=int(sample["country"].nunique()),
        year_min=int(sample["year"].min()),
        year_max=int(sample["year"].max()),
        beta_dep=float(beta_dep[pos]),
        beta_scale=float(beta_scale[pos]),
        status=status,
        rank=rank,
    )


def component_lp_controls(lag_depth: int) -> list[str]:
    return system_lag_controls(SYSTEM_COMPONENT, lag_depth) + lagged_shock_controls(["shock_G_I", "shock_G_C"], lag_depth)


def aggregate_lp_controls(lag_depth: int) -> list[str]:
    return system_lag_controls(SYSTEM_AGGREGATE, lag_depth) + lagged_shock_controls(["shock_aggregate"], lag_depth)


def x_cols_for(shock: str, lag_depth: int) -> tuple[list[str], str, str, str]:
    if shock in {"G_I", "G_C"}:
        controls = component_lp_controls(lag_depth)
        return ["shock_G_I", "shock_G_C", *controls], f"shock_{shock}", f"{'gi' if shock == 'G_I' else 'gc'}_dyn0", ",".join(controls)
    controls = aggregate_lp_controls(lag_depth)
    return ["shock_aggregate", *controls], "shock_aggregate", "g_dyn0", ",".join(controls)


def candidate_keys_for(work: pd.DataFrame, shock: str, lag_depth: int, horizon: int) -> set[tuple[str, int]]:
    x_cols, _coef_col, scale_col, _control_set = x_cols_for(shock, lag_depth)
    needed = [f"y_dyn_h{horizon}", scale_col, *x_cols, "country", "year"]
    needed = list(dict.fromkeys(needed))
    window_start, window_end = shock_window(horizon)
    sample = work[work["year"].between(window_start, window_end)].dropna(subset=needed).copy()
    return key_set(sample)


def sample_stats(keys: set[tuple[str, int]]) -> dict[str, Any]:
    countries = sorted({country for country, _year in keys})
    years = sorted({year for _country, year in keys})
    n_eff = len(countries)
    t_eff = len(years)
    balanced_nxt = n_eff * t_eff
    return {
        "country_year_nobs": int(len(keys)),
        "N_eff": int(n_eff),
        "T_eff": int(t_eff),
        "N_x_T_eff": int(balanced_nxt),
        "missing_country_years_vs_NxT": int(balanced_nxt - len(keys)),
        "effective_year_min": int(min(years)) if years else None,
        "effective_year_max": int(max(years)) if years else None,
        "sample_key_fingerprint": sample_fingerprint(keys),
    }


def build_common_samples(work_by_lag: dict[int, pd.DataFrame]) -> tuple[dict[int, set[tuple[str, int]]], pd.DataFrame]:
    common_by_horizon: dict[int, set[tuple[str, int]]] = {}
    rows: list[dict[str, Any]] = []
    for horizon in HORIZONS:
        window_start, window_end = shock_window(horizon)
        candidate_by_lag: dict[int, list[set[tuple[str, int]]]] = {}
        all_candidates: list[set[tuple[str, int]]] = []
        for lag_depth, work in work_by_lag.items():
            candidate_by_lag[lag_depth] = []
            for shock in SHOCKS:
                keys = candidate_keys_for(work, shock, lag_depth, horizon)
                candidate_by_lag[lag_depth].append(keys)
                all_candidates.append(keys)
        common = set.intersection(*all_candidates) if all_candidates else set()
        if len(common) < MIN_LP_OBS:
            raise CanonicalError(
                f"Common fixed-window sample too small for horizon {horizon}: {len(common)} observations"
            )
        common_by_horizon[horizon] = common
        stats = sample_stats(common)
        expected_t = window_end - window_start + 1
        for lag_depth in LAG_DEPTHS:
            candidate_sizes = [len(keys) for keys in candidate_by_lag[lag_depth]]
            rows.append(
                {
                    "spec_version": SPEC_VERSION,
                    "lag_depth": lag_depth,
                    "lag_label": f"lag_{lag_depth}",
                    "horizon": horizon,
                    "fixed_shock_window_start": window_start,
                    "fixed_shock_window_end": window_end,
                    "expected_T_eff": expected_t,
                    "expected_full_EU27_NxT": len(EU27) * expected_t,
                    **stats,
                    "candidate_nobs_min_across_shocks_before_commoning": int(min(candidate_sizes)),
                    "candidate_nobs_max_across_shocks_before_commoning": int(max(candidate_sizes)),
                    "common_sample_removed_min_candidate_nobs": int(min(candidate_sizes) - len(common)),
                    "shock_sample_common_across_lags": True,
                    "panel_start_year": PANEL_START_YEAR,
                    "panel_end_year": END_YEAR,
                }
            )
    return common_by_horizon, pd.DataFrame(rows).sort_values(["lag_depth", "horizon"]).reset_index(drop=True)


def estimate_lag(
    work: pd.DataFrame,
    lag_depth: int,
    sample_keys_by_horizon: dict[int, set[tuple[str, int]]],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for shock in SHOCKS:
        x_cols, coef_col, scale_col, control_set = x_cols_for(shock, lag_depth)
        for h in HORIZONS:
            window_start, window_end = shock_window(h)
            sample_keys = sample_keys_by_horizon[h]
            fit = estimate_ratio(work, f"y_dyn_h{h}", scale_col, x_cols, coef_col, h, sample_keys)
            system = "components_GI_GC_Y_i" if shock in {"G_I", "G_C"} else "aggregate_G_Y_i"
            ordering = "[dlog_G_I,dlog_G_C,dlog_Y,i]" if shock in {"G_I", "G_C"} else "[dlog_G,dlog_Y,i]"
            rows.append(
                {
                    "spec_version": SPEC_VERSION,
                    "panel_start": PANEL_START_YEAR,
                    "panel_end": END_YEAR,
                    "fixed_shock_window_start": window_start,
                    "fixed_shock_window_end": window_end,
                    "lag_depth": lag_depth,
                    "lag_label": f"lag_{lag_depth}",
                    "shock": shock,
                    "horizon": h,
                    "mu_Y": fit.coef,
                    "se_Y": fit.se,
                    "ci95_low_Y": ci_low(fit.coef, fit.se),
                    "ci95_high_Y": ci_high(fit.coef, fit.se),
                    "p_value_Y": normal_p_value(fit.coef, fit.se),
                    "beta_Y": fit.beta_dep,
                    "beta_scale_deltaG_over_Ylag": fit.beta_scale,
                    "nobs_Y": fit.nobs,
                    "country_n": fit.country_n,
                    "year_min_effective": fit.year_min,
                    "year_max_effective": fit.year_max,
                    "rank_X": fit.rank,
                    "status_Y": fit.status,
                    "control_set": control_set,
                    "cholesky_system": system,
                    "cholesky_ordering": ordering,
                    "scaling_headline": "clean_ratio_betaY_over_betaDeltaG_Ylag1",
                    "interest_rate_i": "short_term_money_market_3m_eurostat_irt_st_a_with_eurostat_fallbacks",
                }
            )
    out = pd.DataFrame(rows).sort_values(["lag_depth", "shock", "horizon"]).reset_index(drop=True)
    out["cumulative_mu_Y"] = out.groupby(["lag_depth", "shock"], sort=False)["mu_Y"].cumsum()
    return out
