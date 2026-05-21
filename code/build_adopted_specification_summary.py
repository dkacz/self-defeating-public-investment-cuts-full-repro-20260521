#!/usr/bin/env python3
"""Build compact summary for the adopted empirical specification."""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
GRID = ROOT / "references/source_results/specification_grid_summary_from_figaro_tiva_extension_20260514.csv"
CASH_GRID = ROOT / "references/source_results/specification_grid_summary_from_cash_transferable_trial_20260514.csv"
OUT_CSV = ROOT / "results/adopted_specification_summary_20260514.csv"
OUT_MD = ROOT / "results/adopted_specification_summary_20260514.md"
OUT_QA = ROOT / "qa/adopted_specification_summary_qa_20260514.csv"
OUT_MANIFEST = ROOT / "results/adopted_specification_summary_manifest_20260514.csv"


SOURCE_SELECTIONS = [
    {
        "source_key": "baseline_frozen_control",
        "category": "Reference baseline",
        "specification": "Reference baseline specification",
        "data_basis": "Baseline input set",
        "income_measure": "Original income construction",
        "liquidity_measure": "Household credit-access proxy",
        "note": "Comparison row for the baseline specification.",
    },
    {
        "source_key": "baseline_current_control",
        "category": "Reference calculation",
        "specification": "Current control calculation",
        "data_basis": "Control inputs retained for comparison",
        "income_measure": "Original income construction",
        "liquidity_measure": "Household credit-access proxy",
        "note": "Control row used to check that the public package reproduces the comparison family.",
    },
    {
        "source_key": "tiva2022_gfcf_realppp_networth",
        "category": "Manuscript specification",
        "specification": "Investment import content, real PPP income and household net financial worth",
        "data_basis": "Official OECD TiVA GFCF import-content source with Eurostat financial accounts",
        "income_measure": "Real GDP per capita in PPP terms",
        "liquidity_measure": "Household net financial worth",
        "note": "Manuscript-facing specification.",
    },
    {
        "source_key": "tiva2022_gfcf_paper_networth",
        "category": "Diagnostic check",
        "specification": "Investment import content and household net financial worth with the earlier income construction",
        "data_basis": "Official OECD TiVA GFCF import-content source with Eurostat financial accounts",
        "income_measure": "Earlier income construction",
        "liquidity_measure": "Household net financial worth",
        "note": "Shows sensitivity to the income-state definition while keeping the import-content and net-worth measures.",
    },
    {
        "source_key": "tiva2022_gfcf_realppp_credit",
        "category": "Diagnostic check",
        "specification": "Investment import content and real PPP income with the earlier credit-access proxy",
        "data_basis": "Official OECD TiVA GFCF import-content source",
        "income_measure": "Real GDP per capita in PPP terms",
        "liquidity_measure": "Household credit-access proxy",
        "note": "Shows sensitivity to the earlier household credit-access proxy.",
    },
    {
        "source_key": "figaro2023_gfcf_realppp_networth",
        "category": "Diagnostic check",
        "specification": "FIGARO-based import-content approximation with real PPP income and net financial worth",
        "data_basis": "FIGARO 2023 import-content approximation",
        "income_measure": "Real GDP per capita in PPP terms",
        "liquidity_measure": "Household net financial worth",
        "note": "Uses a newer FIGARO-based source for the import-content state; it is not the adopted manuscript source.",
    },
    {
        "source_key": "tiva_cf2024_gfcf_realppp_networth",
        "category": "Diagnostic check",
        "specification": "Mechanical 2024 carry-forward with real PPP income and net financial worth",
        "data_basis": "Official OECD TiVA carried forward mechanically to 2024",
        "income_measure": "Real GDP per capita in PPP terms",
        "liquidity_measure": "Household net financial worth",
        "note": "Mechanical carry-forward beyond official OECD TiVA coverage; it is not the adopted manuscript source.",
    },
]

CASH_SOURCE_SELECTIONS = [
    {
        "source_key": "tiva2022_gfcf_realppp_cash_transferable",
        "category": "Diagnostic check",
        "specification": "Investment import content and real PPP income with cash plus transferable deposits",
        "data_basis": "Official OECD TiVA GFCF import-content source with Eurostat financial accounts",
        "income_measure": "Real GDP per capita in PPP terms",
        "liquidity_measure": "Cash plus transferable household deposits",
        "note": "Uses a narrower liquidity measure; it is not the adopted manuscript specification.",
    }
]

NUMERIC_FIELDS = [
    "K_Y_h8",
    "K_G_h8",
    "dsa_margin_2036_cut_pp",
    "direct_dy_margin_2036_cut_pp",
    "dsa_margin_2036_expansion_pp",
    "direct_dy_margin_2036_expansion_pp",
]

SPEC_LABELS = {
    "investment_import_content": "investment import content",
    "household_net_financial_worth": "household net financial worth",
    "investment_import_content__public_debt__household_net_financial_worth": "baseline comparison evaluation",
    "public_debt__household_net_financial_worth__real_ppp_income": "earlier credit-access proxy evaluation",
}

OUTPUT_FIELDS = [
    "category",
    "specification",
    "data_basis",
    "income_measure",
    "liquidity_measure",
    "profile_year",
    "sample_end_year",
    "pass_count",
    "passing_evaluations",
    *NUMERIC_FIELDS,
    "note",
]


def read_rows(path: Path) -> dict[str, dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return {row["variant"]: row for row in csv.DictReader(handle)}


def public_row(row: dict[str, str], meta: dict[str, str]) -> dict[str, str]:
    out = {field: "" for field in OUTPUT_FIELDS}
    for field in ["category", "specification", "data_basis", "income_measure", "liquidity_measure", "note"]:
        out[field] = meta[field]
    for field in ["profile_year", "sample_end_year", "pass_count", *NUMERIC_FIELDS]:
        out[field] = row.get(field, "")
    passing_specs = row.get("passing_specs", "")
    out["_internal_passing_specs"] = passing_specs
    out["passing_evaluations"] = " + ".join(
        SPEC_LABELS.get(part.strip(), part.strip()) for part in passing_specs.split("+") if part.strip()
    )
    return out


def write_csv(path: Path, rows: list[dict[str, str]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def md_escape(value: str) -> str:
    return str(value).replace("|", "\\|")


def write_md(path: Path, rows: list[dict[str, str]]) -> None:
    cols = [
        "category",
        "specification",
        "passing_evaluations",
        "K_Y_h8",
        "K_G_h8",
        "dsa_margin_2036_cut_pp",
        "direct_dy_margin_2036_cut_pp",
        "dsa_margin_2036_expansion_pp",
        "direct_dy_margin_2036_expansion_pp",
        "note",
    ]
    display_cols = {
        "category": "Category",
        "specification": "Specification",
        "passing_evaluations": "Passing evaluations",
        "K_Y_h8": "Cumulative output response, eighth year",
        "K_G_h8": "Cumulative spending response, eighth year",
        "dsa_margin_2036_cut_pp": "Institutional debt margin in 2036, cut scenario, percentage points",
        "direct_dy_margin_2036_cut_pp": "Direct debt-to-GDP margin in 2036, cut scenario, percentage points",
        "dsa_margin_2036_expansion_pp": "Institutional debt margin in 2036, expansion scenario, percentage points",
        "direct_dy_margin_2036_expansion_pp": "Direct debt-to-GDP margin in 2036, expansion scenario, percentage points",
        "note": "Note",
    }
    lines = [
        "# Manuscript Specification Summary",
        "",
        "This table separates the manuscript-facing specification from diagnostic checks. Values are reproduced from frozen source tables shipped in this public package, with the manuscript-facing specification identified explicitly.",
        "",
        "|" + "|".join(display_cols[col] for col in cols) + "|",
        "|" + "|".join(["---"] * len(cols)) + "|",
    ]
    for row in rows:
        lines.append("|" + "|".join(md_escape(row.get(col, "")) for col in cols) + "|")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    grid = read_rows(GRID)
    cash_grid = read_rows(CASH_GRID)
    rows: list[dict[str, str]] = []
    qa_rows: list[dict[str, str]] = []

    for meta in SOURCE_SELECTIONS:
        source_key = meta["source_key"]
        found = source_key in grid
        qa_rows.append(
            {
                "check": f"source_row_present:{meta['category']} - {meta['specification']}",
                "status": "PASS" if found else "FAIL",
                "detail": str(GRID.relative_to(ROOT)),
            }
        )
        if found:
            rows.append(public_row(grid[source_key], meta))

    for meta in CASH_SOURCE_SELECTIONS:
        source_key = meta["source_key"]
        found = source_key in cash_grid
        qa_rows.append(
            {
                "check": f"source_row_present:{meta['category']} - {meta['specification']}",
                "status": "PASS" if found else "FAIL",
                "detail": str(CASH_GRID.relative_to(ROOT)),
            }
        )
        if found:
            rows.append(public_row(cash_grid[source_key], meta))

    adopted = next((row for row in rows if row["category"] == "Manuscript specification"), None)
    qa_rows.append(
        {
            "check": "adopted_specification_passes_two_specs",
            "status": "PASS" if adopted and adopted["_internal_passing_specs"] == "investment_import_content+household_net_financial_worth" else "FAIL",
            "detail": adopted["passing_evaluations"] if adopted else "missing",
        }
    )
    qa_rows.append(
        {
            "check": "adopted_specification_uses_official_tiva_realppp_networth",
            "status": "PASS"
            if adopted
            and adopted["data_basis"].startswith("Official OECD TiVA GFCF")
            and adopted["income_measure"] == "Real GDP per capita in PPP terms"
            and adopted["liquidity_measure"] == "Household net financial worth"
            else "FAIL",
            "detail": "" if adopted else "missing",
        }
    )

    write_csv(OUT_CSV, rows, OUTPUT_FIELDS)
    write_md(OUT_MD, rows)
    write_csv(OUT_QA, qa_rows, ["check", "status", "detail"])

    manifest_rows = []
    for path in [GRID, CASH_GRID, OUT_CSV, OUT_MD, OUT_QA]:
        manifest_rows.append(
            {
                "path": str(path.relative_to(ROOT)),
                "bytes": str(path.stat().st_size),
                "sha256": sha256(path),
            }
        )
    write_csv(OUT_MANIFEST, manifest_rows, ["path", "bytes", "sha256"])

    failures = [row for row in qa_rows if row["status"] != "PASS"]
    if failures:
        raise SystemExit(f"QA failed: {failures}")
    print(f"wrote {OUT_CSV.relative_to(ROOT)} rows={len(rows)}")


if __name__ == "__main__":
    main()
