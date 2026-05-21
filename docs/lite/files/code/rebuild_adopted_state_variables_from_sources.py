#!/usr/bin/env python3
"""Rebuild adopted state variables from frozen source CSVs."""

from __future__ import annotations

import csv
import hashlib
import math
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "data/frozen/adopted_sources"
PROV = ROOT / "data/provenance"
RESULTS = ROOT / "results"
QA = ROOT / "qa"
STATE_TABLE = ROOT / "tables/state_variables.csv"

EU27 = [
    "AUT", "BEL", "BGR", "CYP", "CZE", "DEU", "DNK", "ESP", "EST",
    "FIN", "FRA", "GRC", "HRV", "HUN", "IRL", "ITA", "LTU", "LUX",
    "LVA", "MLT", "NLD", "POL", "PRT", "ROU", "SVK", "SVN", "SWE",
]
SAMPLE_START = 2004
SAMPLE_END = 2022

SOURCE_FILES = [
    "nominal_gdp.csv",
    "gdp_pc_current_pps.csv",
    "gdp_pc_real_index_2020.csv",
    "hh_total_financial_assets.csv",
    "hh_total_financial_liabilities.csv",
    "oecd_tiva_import_content_gfcf_cons_1995_2022.csv",
    "oecd_tiva_mainsh_domestic_va_shares_gfcf_cons_requested_1995_2024_returned_1995_2022.csv",
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def source_manifest() -> None:
    rows = []
    for name in SOURCE_FILES:
        path = SRC / name
        rows.append(
            {
                "path": str(path.relative_to(ROOT)),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )
    write_csv(PROV / "adopted_sources_manifest_20260514.csv", rows, ["path", "bytes", "sha256"])


def source_frame(name: str, value_col: str) -> pd.DataFrame:
    frame = pd.read_csv(SRC / name)
    return frame[["country", "year", value_col]].copy()


def build_panel() -> pd.DataFrame:
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

    tiva = pd.read_csv(SRC / "oecd_tiva_import_content_gfcf_cons_1995_2022.csv")
    gfcf = tiva[tiva["measure"].eq("GFCF_VA_SH")][["country", "year", "import_content_share"]].rename(
        columns={"import_content_share": "tiva2022_gfcf_import_content_raw"}
    )
    frame = frame.merge(gfcf, on=["country", "year"], how="left", validate="one_to_one")
    frame = frame[frame["country"].isin(EU27)].sort_values(["country", "year"]).reset_index(drop=True)

    sample = frame[frame["year"].between(SAMPLE_START, SAMPLE_END)].copy()
    for raw, z_col in [
        ("tiva2022_gfcf_import_content_raw", "trade_z"),
        ("liq_fa_net_worth_raw", "liq_z"),
        ("log_real_ppp_gdp_pc_raw", "log_gdp_pc_z"),
    ]:
        values = sample[raw].dropna()
        mean = values.mean()
        sd = values.std(ddof=0)
        frame[z_col] = (frame[raw] - mean) / sd

    out = frame[
        [
            "country",
            "year",
            "tiva2022_gfcf_import_content_raw",
            "trade_z",
            "liq_fa_net_worth_raw",
            "liq_z",
            "real_ppp_gdp_pc_2020pps",
            "log_real_ppp_gdp_pc_raw",
            "log_gdp_pc_z",
        ]
    ].copy()
    out.to_csv(RESULTS / "source_rebuilt_adopted_state_variables_20260514.csv", index=False, float_format="%.12g")
    return out


def summary(panel: pd.DataFrame) -> pd.DataFrame:
    specs = [
        ("Investment import content", "tiva2022_gfcf_import_content_raw", "trade_z"),
        ("Household net financial worth", "liq_fa_net_worth_raw", "liq_z"),
        ("Real PPP income level", "log_real_ppp_gdp_pc_raw", "log_gdp_pc_z"),
    ]
    rows = []
    sample = panel[panel["year"].between(SAMPLE_START, SAMPLE_END)].copy()
    for label, raw, z_col in specs:
        values = sample[raw].dropna()
        pol = panel[(panel["country"].eq("POL")) & (panel["year"].eq(SAMPLE_END))].iloc[0]
        rows.append(
            {
                "state_variable": label,
                "raw_column": raw,
                "z_column": z_col,
                "sample_start": SAMPLE_START,
                "sample_end": SAMPLE_END,
                "n": int(values.shape[0]),
                "mean": values.mean(),
                "sd_population": values.std(ddof=0),
                "poland_value_2022": pol[raw],
                "poland_z_2022": pol[z_col],
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS / "source_rebuilt_adopted_state_variable_summary_20260514.csv", index=False, float_format="%.12g")
    return out


def parse_float(value: object) -> float:
    text = str(value).replace("+", "").strip()
    return float(text)


def qa_summary(rebuilt: pd.DataFrame) -> None:
    table = pd.read_csv(STATE_TABLE)
    target = table[table["State variable"].isin(rebuilt["state_variable"])].copy()
    target_by_name = target.set_index("State variable").to_dict("index")
    rows = []
    for row in rebuilt.to_dict("records"):
        name = row["state_variable"]
        target_row = target_by_name[name]
        checks = [
            ("n", int(row["n"]), int(target_row["N"]), 0.0),
            ("mean_rounded_3", round(float(row["mean"]), 3), parse_float(target_row["Mean"]), 0.0005),
            ("sd_rounded_3", round(float(row["sd_population"]), 3), parse_float(target_row["SD"]), 0.0005),
            ("poland_value_rounded_3", round(float(row["poland_value_2022"]), 3), parse_float(target_row["Poland value"]), 0.0005),
            ("poland_z_rounded_3", round(float(row["poland_z_2022"]), 3), parse_float(target_row["Poland z"]), 0.0005),
        ]
        for check, got, expected, tol in checks:
            if isinstance(got, int) and isinstance(expected, int):
                ok = got == expected
            else:
                ok = math.isclose(float(got), float(expected), abs_tol=tol)
            rows.append(
                {
                    "state_variable": name,
                    "check": check,
                    "status": "PASS" if ok else "FAIL",
                    "rebuilt": got,
                    "table_value": expected,
                    "tolerance": tol,
                }
            )
    write_csv(
        QA / "source_rebuild_adopted_state_variables_qa_20260514.csv",
        rows,
        ["state_variable", "check", "status", "rebuilt", "table_value", "tolerance"],
    )
    failures = [row for row in rows if row["status"] != "PASS"]
    if failures:
        raise SystemExit(f"source rebuild QA failed: {failures}")


def main() -> None:
    source_manifest()
    panel = build_panel()
    rebuilt = summary(panel)
    qa_summary(rebuilt)
    print("source rebuild QA PASS for adopted state variables")


if __name__ == "__main__":
    main()
