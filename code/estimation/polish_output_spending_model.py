#!/usr/bin/env python3
from __future__ import annotations

import os
import sys

REQUIRED_THREAD_ENV = {
    "OPENBLAS_NUM_THREADS": "1",
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
}

if any(os.environ.get(key) != value for key, value in REQUIRED_THREAD_ENV.items()):
    env = os.environ.copy()
    env.update(REQUIRED_THREAD_ENV)
    os.execve(sys.executable, [sys.executable, *sys.argv], env)

import hashlib
import importlib.util
import json
import math
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    from threadpoolctl import threadpool_info, threadpool_limits
except Exception:
    threadpool_info = None
    threadpool_limits = None


SPEC_VERSION = "polish_output_spending_model_20260510"
HORIZONS = tuple(range(9))
HARD_TOL = 5e-9


def find_package_root(script_path: Path) -> Path:
    for path in script_path.parents:
        if (path / "code").is_dir() and (path / "data").is_dir() and (path / "references").is_dir():
            return path
    raise RuntimeError(f"Cannot locate package root from {script_path}")


SCRIPT_PATH = Path(__file__).resolve()
ROOT = find_package_root(SCRIPT_PATH)
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results" / "polish_output_spending"
REFERENCES_DIR = ROOT / "references"
GRID_CODE = REFERENCES_DIR / "run_c_pl_feature_grid_base.py"
FROZEN_TARGETS = ROOT / "data/frozen/adopted_run_outputs"


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


def load_adopted_specification_targets() -> pd.DataFrame:
    path = FROZEN_TARGETS / "feature_screen" / "feature_robustness_summary.csv"
    screen = pd.read_csv(path)
    selected = screen.loc[
        screen["gate_status"].eq("PASS_ROBUSTNESS_GATE"),
        ["spec_id", "features"],
    ].copy()
    selected["selection_source"] = "frozen_validation_target"
    selected["selection_status"] = "PASS_ROBUSTNESS_GATE"
    return selected.sort_values("spec_id").reset_index(drop=True)


def load_output_spending_path_targets() -> pd.DataFrame:
    path = FROZEN_TARGETS / "polish_output_spending" / "polish_output_spending_paths.csv"
    return pd.read_csv(path).sort_values(["spec_id", "horizon"]).reset_index(drop=True)


def load_grid_module() -> Any:
    spec = importlib.util.spec_from_file_location("polish_response_feature_grid_base", GRID_CODE)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {GRID_CODE}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    module.TASK_DIR = ROOT
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
    grid_mod.configure_base_for_spec(base, features, model_id)
    feature_panel = pd.read_csv(DATA_DIR / "country_feature_panel.csv")
    return grid_mod.feature_values(feature_panel, features, "poland_2024")


def load_work(grid_mod: Any) -> tuple[Any, Any, pd.DataFrame]:
    base = grid_mod.load_base_module()
    v3 = grid_mod.load_v3(base)
    panel, _meta = v3.load_panel()
    ancillary_coverage = RESULTS_DIR / "short_rate_coverage.csv"
    if ancillary_coverage.exists():
        ancillary_coverage.unlink()
    feature_panel = pd.read_csv(DATA_DIR / "country_feature_panel.csv")
    work, _shock_meta = grid_mod.prepare_work(v3, panel, feature_panel)
    return base, v3, work


def estimate_kernels(grid_mod: Any, base: Any, v3: Any, work: pd.DataFrame, spec_id: str, features_text: str) -> pd.DataFrame:
    features = parse_features(features_text)
    z_values = configure(grid_mod, base, features, spec_id)
    cols = base.x_columns(False)
    rows: list[dict[str, Any]] = []
    for h in HORIZONS:
        fit_y = base.fit_conditional_ratio(v3, work, f"y_dyn_h{h}", "gi_dyn0", cols, z_values, h)
        fit_g = base.fit_conditional_ratio(v3, work, f"gi_dyn_h{h}", "gi_dyn0", cols, z_values, h)
        rows.append(
            {
                "spec_version": SPEC_VERSION,
                "spec_id": spec_id,
                "features": features_text,
                "profile_label": "poland_2024",
                "horizon": h,
                "mu_Y_incremental": fit_y.ratio,
                "se_Y_incremental": fit_y.se,
                "mu_G_incremental": fit_g.ratio,
                "se_G_incremental": fit_g.se,
                "beta_scale_action": fit_y.beta_scale,
                "se_beta_scale_action": fit_y.se_beta_scale,
                "action_denom_t": fit_y.denom_t,
                "nobs": fit_y.nobs,
                "country_n": fit_y.country_n,
                "year_min_effective": fit_y.year_min,
                "year_max_effective": fit_y.year_max,
                "rank_X": fit_y.rank,
                "status_Y": fit_y.status,
                "status_G": fit_g.status,
                **{f"z_{feature}": float(z_values[feature]) for feature in features},
            }
        )
    out = pd.DataFrame(rows).sort_values("horizon").reset_index(drop=True)
    out["K_Y_cumulative"] = out["mu_Y_incremental"].cumsum()
    out["K_G_cumulative"] = out["mu_G_incremental"].cumsum()
    out["cumulative_output_to_spending_ratio"] = out["K_Y_cumulative"] / out["K_G_cumulative"]
    out["ci95_low_K_Y_cumulative_naive"] = out["K_Y_cumulative"] - base.Z95 * np.sqrt(
        np.cumsum(np.square(out["se_Y_incremental"]))
    )
    out["ci95_high_K_Y_cumulative_naive"] = out["K_Y_cumulative"] + base.Z95 * np.sqrt(
        np.cumsum(np.square(out["se_Y_incremental"]))
    )
    return out


def load_selected_specs() -> pd.DataFrame:
    screen_path = ROOT / "results" / "feature_screen" / "feature_robustness_summary.csv"
    if not screen_path.exists():
        raise FileNotFoundError(
            "Run code/feature_screen_model.py before Polish response estimation; "
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


def make_summary(paths: pd.DataFrame) -> pd.DataFrame:
    h8 = paths[paths["horizon"].eq(8)].copy()
    return h8[
        [
            "spec_id",
            "features",
            "K_Y_cumulative",
            "K_G_cumulative",
            "cumulative_output_to_spending_ratio",
            "mu_Y_incremental",
            "mu_G_incremental",
            "nobs",
            "country_n",
            "year_min_effective",
            "year_max_effective",
        ]
    ].rename(
        columns={
            "K_Y_cumulative": "K_Y_h8",
            "K_G_cumulative": "K_G_h8",
            "cumulative_output_to_spending_ratio": "cumulative_output_to_spending_ratio_h8",
        }
    )


def qa_checks(paths: pd.DataFrame, summary: pd.DataFrame) -> pd.DataFrame:
    checks: list[dict[str, str]] = []
    selected_pairs = set(zip(paths["spec_id"], paths["features"]))
    target_specs = load_adopted_specification_targets()
    target_pairs = set(zip(target_specs["spec_id"], target_specs["features"]))
    checks.append(
        {
            "check": "admitted_specifications_match_fresh_screen_winners",
            "status": "PASS" if selected_pairs == target_pairs else "FAIL",
            "detail": "selected specifications=" + "; ".join(sorted(paths["features"].unique())),
        }
    )
    checks.append(
        {
            "check": "all_horizons_present",
            "status": "PASS"
            if paths.groupby("spec_id")["horizon"].apply(lambda s: tuple(sorted(s)) == HORIZONS).all()
            else "FAIL",
            "detail": "required h0..h8 for each selected specification",
        }
    )
    checks.append(
        {
            "check": "all_regressions_ok",
            "status": "PASS" if paths["status_Y"].eq("OK").all() and paths["status_G"].eq("OK").all() else "FAIL",
            "detail": "status_Y and status_G are OK for every selected specification and horizon",
        }
    )
    target_paths = load_output_spending_path_targets()
    compare = paths[
        ["spec_id", "features", "horizon", "K_Y_cumulative", "K_G_cumulative"]
    ].merge(
        target_paths[["spec_id", "features", "horizon", "K_Y_cumulative", "K_G_cumulative"]],
        on=["spec_id", "features", "horizon"],
        how="outer",
        suffixes=("_recomputed", "_target"),
        indicator=True,
        validate="one_to_one",
    )
    if compare["_merge"].eq("both").all():
        max_abs_error = float(
            np.nanmax(
                np.abs(
                    compare[
                        ["K_Y_cumulative_recomputed", "K_G_cumulative_recomputed"]
                    ].to_numpy(dtype=float)
                    - compare[
                        ["K_Y_cumulative_target", "K_G_cumulative_target"]
                    ].to_numpy(dtype=float)
                )
            )
        )
    else:
        max_abs_error = math.inf
    checks.append(
        {
            "check": "output_spending_paths_match_frozen_validation_target",
            "status": "PASS" if max_abs_error <= HARD_TOL else "FAIL",
            "detail": f"max_abs_error={max_abs_error:.3e} tolerance={HARD_TOL:.1e}",
        }
    )
    checks.append(
        {
            "check": "summary_has_two_rows",
            "status": "PASS" if len(summary) == 2 else "FAIL",
            "detail": f"rows={len(summary)}",
        }
    )
    bad_result_names = [path.name for path in RESULTS_DIR.glob("*.csv") if "debt" in path.name.lower()]
    checks.append(
        {
            "check": "no_debt_path_outputs_written",
            "status": "PASS" if not bad_result_names else "FAIL",
            "detail": ",".join(bad_result_names) if bad_result_names else "this block writes only output and spending responses",
        }
    )
    checks.append(
        {
            "check": "no_short_rate_coverage_result_output",
            "status": "PASS" if not (RESULTS_DIR / "short_rate_coverage.csv").exists() else "FAIL",
            "detail": "ancillary source coverage metadata is not written as a Polish response result",
        }
    )
    source_note = DATA_DIR / "eurostat_short_rate_source_notes.md"
    source_note_text = source_note.read_text(encoding="utf-8") if source_note.exists() else ""
    expected_snapshot_paths = [f"data/eurostat_{dataset}_snapshot.csv" for dataset in v3_short_rate_datasets()]
    wrong_reference_paths = [line for line in source_note_text.splitlines() if "references/eurostat_" in line]
    forbidden_coverage_ref = "results/short_rate_coverage.csv" in source_note_text
    missing_snapshot_paths = [
        path for path in expected_snapshot_paths if path not in source_note_text or not (ROOT / path).exists()
    ]
    source_note_ok = (
        source_note.exists()
        and not wrong_reference_paths
        and not forbidden_coverage_ref
        and not missing_snapshot_paths
    )
    checks.append(
        {
            "check": "eurostat_short_rate_note_paths_exist",
            "status": "PASS" if source_note_ok else "FAIL",
            "detail": (
                "all snapshot paths point to bundled data/ files and no removed coverage CSV is referenced"
                if source_note_ok
                else (
                    f"wrong_reference_rows={len(wrong_reference_paths)} "
                    f"forbidden_coverage_ref={forbidden_coverage_ref} "
                    f"missing_data_paths={missing_snapshot_paths}"
                )
            ),
        }
    )
    allowed_debt_feature_cols = {"z_debt"}
    bad_cols = [
        col
        for col in paths.columns
        if (
            "debt" in col.lower() and col not in allowed_debt_feature_cols
        )
    ]
    checks.append(
        {
            "check": "no_debt_path_columns",
            "status": "PASS" if not bad_cols else "FAIL",
            "detail": ",".join(bad_cols) if bad_cols else "kernel tables exclude debt path columns; z_debt is an admitted feature",
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


def write_report(paths: pd.DataFrame, summary: pd.DataFrame, qa: pd.DataFrame) -> None:
    report = f"""# Polish output and public-investment responses

## Scope

This public reproduction block computes only output and public-investment spending responses for the Polish specifications selected by the state-variable screen. It does not select features and it does not compute debt paths.

Selected specifications:

{md_table(summary, ["features", "K_Y_h8", "K_G_h8", "cumulative_output_to_spending_ratio_h8", "nobs", "country_n", "year_min_effective", "year_max_effective"], {"K_Y_h8", "K_G_h8", "cumulative_output_to_spending_ratio_h8"})}

## Full h0..h8 paths

{md_table(paths, ["features", "horizon", "K_Y_cumulative", "K_G_cumulative", "cumulative_output_to_spending_ratio", "mu_Y_incremental", "mu_G_incremental", "nobs"], {"K_Y_cumulative", "K_G_cumulative", "cumulative_output_to_spending_ratio", "mu_Y_incremental", "mu_G_incremental"})}

## QA

{md_table(qa, ["check", "status", "detail"], set())}

## Reproducibility notes

- The plain command is `python3 code/polish_output_spending_model.py`.
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
        "selected_specifications": [
            {"specification": features} for features in sorted(load_adopted_specification_targets()["features"])
        ],
        "horizons": list(HORIZONS),
        "blas_thread_env": {key: os.environ.get(key, "") for key in REQUIRED_THREAD_ENV},
        "threadpoolctl_info": threadpool_state(),
        "excluded_from_scope": [
            "state variable screen",
            "debt-to-GDP paths",
            "equal-weight average",
        ],
    }
    (RESULTS_DIR / "run_manifest.json").write_text(json.dumps(run_manifest, indent=2, sort_keys=True) + "\n")


def main() -> None:
    selected = load_selected_specs()
    grid_mod = load_grid_module()
    base, v3, work = load_work(grid_mod)
    clean_results_dir()

    paths = []
    for row in selected.itertuples(index=False):
        paths.append(estimate_kernels(grid_mod, base, v3, work, row.spec_id, row.features))
    paths_df = pd.concat(paths, ignore_index=True)
    paths_df.to_csv(RESULTS_DIR / "polish_output_spending_paths.csv", index=False, float_format="%.9f")
    summary = make_summary(paths_df)
    summary.to_csv(RESULTS_DIR / "polish_output_spending_h8_summary.csv", index=False, float_format="%.9f")
    qa = qa_checks(paths_df, summary)
    qa.to_csv(RESULTS_DIR / "qa_checks.csv", index=False)
    write_report(paths_df, summary, qa)
    write_manifest()
    if not qa["status"].eq("PASS").all():
        raise SystemExit("QA failed")
    print(
        f"OK polish_output_spending_model specs={len(selected)} "
        f"trade_KY_h8={summary.loc[summary['spec_id'].eq('investment_import_content'), 'K_Y_h8'].iloc[0]:.9f} "
        f"liq_KY_h8={summary.loc[summary['spec_id'].eq('household_net_financial_worth'), 'K_Y_h8'].iloc[0]:.9f}"
    )


if __name__ == "__main__":
    if threadpool_limits is None:
        main()
    else:
        with threadpool_limits(limits=1):
            main()
