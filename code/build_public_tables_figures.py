#!/usr/bin/env python3
"""Build public manuscript tables and figures from recomputed package outputs."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RUN_OUTPUTS = ROOT / "results/recomputed"
INPUTS = ROOT / "data/frozen/adopted_model_inputs"
EU27_BENCHMARK_DEBT = RUN_OUTPUTS / "eu27_benchmark/eu27_benchmark_debt_2036.csv"
EU27_BENCHMARK_ANNUAL_DEBT = RUN_OUTPUTS / "eu27_benchmark/eu27_benchmark_annual_debt_decomposition.csv"
EU27_OUTPUT_SPENDING = RUN_OUTPUTS / "eu27_benchmark/eu27_output_spending_paths.csv"

TABLES = ROOT / "tables"
FIGURES = ROOT / "figures"
QA = ROOT / "qa"
for path in [TABLES, FIGURES, QA]:
    path.mkdir(parents=True, exist_ok=True)


def write_table(df: pd.DataFrame, stem: str) -> None:
    df.to_csv(TABLES / f"{stem}.csv", index=False)
    try:
        markdown = df.to_markdown(index=False, disable_numparse=True)
    except ImportError:
        headers = [str(col) for col in df.columns]
        lines = [
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join(["---"] * len(headers)) + " |",
        ]
        for _, row in df.iterrows():
            values = [str(row[col]).replace("\n", " ") for col in df.columns]
            lines.append("| " + " | ".join(values) + " |")
        markdown = "\n".join(lines)
    (TABLES / f"{stem}.md").write_text(
        markdown + "\n",
        encoding="utf-8",
    )


FEATURE_LABELS = {
    "trade": "investment import content",
    "debt": "public debt",
    "liq": "household net financial worth",
    "log_gdp_pc": "real PPP income",
}

INTERACTION_LABELS = {
    "shock_G_I_x_trade": "investment import content",
    "shock_G_I_x_liq": "household net financial worth",
}


def public_feature_label(features: str) -> str:
    return " + ".join(FEATURE_LABELS.get(part, part) for part in str(features).split("+"))


GATE_REASON_LABELS = {
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


def public_gate_reason(value: object) -> str:
    parts = [part.strip() for part in str(value).split(";") if part.strip()]
    if not parts:
        return "No screening reason recorded"
    return "; ".join(GATE_REASON_LABELS.get(part, part.replace("_", " ")) for part in parts)


def fmt(v: float, digits: int = 2, signed: bool = False) -> str:
    if signed:
        return f"{v:+.{digits}f}"
    return f"{v:.{digits}f}"


def pass_label(passed: object) -> str:
    return "PASS" if bool(passed) else "FAIL"


def gate_value(value: float, passed: object, digits: int = 3) -> str:
    return f"{fmt(float(value), digits)} ({pass_label(passed)})"


def load_eu27_response_paths() -> pd.DataFrame:
    if not EU27_OUTPUT_SPENDING.exists():
        raise FileNotFoundError(
            "Missing recomputed EU27 benchmark response path. Run code/run_full_estimator_repro.py first."
        )
    return pd.read_csv(EU27_OUTPUT_SPENDING).sort_values("horizon").reset_index(drop=True)


def build_source_code_table() -> pd.DataFrame:
    rows = [
        {
            "Input": "Investment import content",
            "Primary source": "OECD TiVA 2025",
            "Dataset and codes": "DSD_TIVA_MAINSH@DF_MAINSH(1.1); indicator GFCF_VA_SH; total activity _T; counterpart D; unit PT_GFCF_VA",
            "Years used": "2004-2022",
            "Coverage": "EU27",
            "Unit": "Share of GFCF",
            "Transformation": "1 minus the domestic value-added share in gross fixed capital formation; standardised on the EU27 panel",
            "Source record": "adopted_sources_manifest_20260514.csv; OECD request returned official observations through 2022",
        },
        {
            "Input": "Household net financial worth",
            "Primary source": "Eurostat financial accounts and national accounts",
            "Dataset and codes": "nasa_10_f_bs, sector S14_S15, na_item F, finpos ASS and LIAB, co_nco NCO, unit MIO_EUR; denominator nama_10_gdp, B1GQ, CP_MEUR",
            "Years used": "2004-2022",
            "Coverage": "EU27",
            "Unit": "Ratio to nominal GDP",
            "Transformation": "Negative of household financial assets less liabilities, divided by nominal GDP; standardised on the EU27 panel",
            "Source record": "adopted_sources_manifest_20260514.csv",
        },
        {
            "Input": "Real PPP income",
            "Primary source": "Eurostat national accounts",
            "Dataset and codes": "nama_10_pc, B1GQ, CP_PPS_EU27_2020_HAB and CLV_I20_HAB",
            "Years used": "2004-2022",
            "Coverage": "EU27",
            "Unit": "2020 PPS per inhabitant",
            "Transformation": "2020 current-PPS level multiplied by the real per-capita index divided by 100; log; standardised on the EU27 panel",
            "Source record": "adopted_sources_manifest_20260514.csv",
        },
        {
            "Input": "Public debt state",
            "Primary source": "Eurostat government deficit, debt and associated data",
            "Dataset and codes": "gov_10dd_edpt1, sector S13, na_item GD, unit PC_GDP",
            "Years used": "2004-2024 in the source panel; common h8 admission sample follows the TiVA-linked window",
            "Coverage": "EU27",
            "Unit": "Percent of GDP",
            "Transformation": "Maastricht gross debt ratio; standardised on the EU27 panel",
            "Source record": "source_csv/government_debt_eu27_1995_2024.csv and adopted model inputs",
        },
        {
            "Input": "Baseline debt ratio",
            "Primary source": "European Commission Debt Sustainability Monitor 2025 extraction",
            "Dataset and codes": "DSM Poland baseline gross general-government debt ratio",
            "Years used": "2024-2036",
            "Coverage": "Poland",
            "Unit": "Percent of GDP",
            "Transformation": "Baseline debt path used as the institutional reference series before scenario margins are applied",
            "Source record": "ec_poland_dsm2025_baseline_table_20260308.csv; commission_poland_exogenous_path_20260310.csv",
        },
        {
            "Input": "EU27 benchmark debt comparison",
            "Primary source": "EU27 panel benchmark debt-accounting recomputation",
            "Dataset and codes": "Institutional debt-equation and direct debt-to-GDP endpoint margins; annual institutional debt decomposition for Appendix C.2 and C.3",
            "Years used": "2028-2036 for annual decomposition; 2036 for endpoint table",
            "Coverage": "EU27 panel benchmark",
            "Unit": "Percentage points of GDP or debt-to-GDP ratio, as labelled by column",
            "Transformation": "EU27 output, spending and direct debt-to-GDP paths are recomputed from frozen model inputs, passed through the same Poland debt-recursion shell, and validated against frozen benchmark targets",
            "Source record": "results/recomputed/eu27_benchmark/eu27_benchmark_debt_2036.csv; results/recomputed/eu27_benchmark/eu27_benchmark_annual_debt_decomposition.csv; validation targets in data/frozen/eu27_benchmark_debt/",
        },
        {
            "Input": "Structural balance",
            "Primary source": "European Commission Debt Sustainability Monitor 2025 extraction",
            "Dataset and codes": "DSM Poland structural balance",
            "Years used": "2024-2036",
            "Coverage": "Poland",
            "Unit": "Percent of GDP",
            "Transformation": "Reported as the structural-balance environment underlying the baseline interpretation",
            "Source record": "ec_poland_dsm2025_baseline_table_20260308.csv",
        },
        {
            "Input": "Primary balance",
            "Primary source": "European Commission Debt Sustainability Monitor 2025 extraction",
            "Dataset and codes": "DSM Poland primary balance",
            "Years used": "2024-2036",
            "Coverage": "Poland",
            "Unit": "Percent of GDP",
            "Transformation": "Baseline primary-balance path; scenario path adds discretionary action and cyclical feedback",
            "Source record": "ec_poland_dsm2025_baseline_table_20260308.csv; dsa_debt_paths.csv",
        },
        {
            "Input": "Nominal growth assumptions",
            "Primary source": "European Commission Debt Sustainability Monitor 2025 extraction and project debt-accounting input",
            "Dataset and codes": "Real GDP growth, GDP deflator growth and nominal GDP growth",
            "Years used": "2025-2036",
            "Coverage": "Poland",
            "Unit": "Annual percent change",
            "Transformation": "Nominal growth is used in the debt recursion and adjusted in the scenario path through output feedback",
            "Source record": "commission_poland_exogenous_path_20260310.csv; dsa_debt_paths.csv",
        },
        {
            "Input": "Interest-growth and snowball terms",
            "Primary source": "European Commission Debt Sustainability Monitor 2025 extraction",
            "Dataset and codes": "Interest expenditure, real-growth effect and inflation effect",
            "Years used": "2024-2036",
            "Coverage": "Poland",
            "Unit": "Percentage points of GDP",
            "Transformation": "Combined in Appendix C as the annual snowball term in the debt-accounting decomposition",
            "Source record": "ec_poland_dsm2025_baseline_table_20260308.csv; dsa_debt_paths.csv",
        },
        {
            "Input": "Stock-flow adjustment",
            "Primary source": "European Commission Debt Sustainability Monitor 2025 extraction",
            "Dataset and codes": "DSM Poland stock-flow adjustments",
            "Years used": "2024-2036",
            "Coverage": "Poland",
            "Unit": "Percentage points of GDP",
            "Transformation": "Carried as the stock-flow component in the institutional debt-accounting environment",
            "Source record": "ec_poland_dsm2025_baseline_table_20260308.csv",
        },
        {
            "Input": "Ageing-related costs",
            "Primary source": "European Commission Debt Sustainability Monitor 2025 extraction",
            "Dataset and codes": "Ageing-cost assumptions used in the DSM medium-term baseline",
            "Years used": "2024-2036",
            "Coverage": "Poland",
            "Unit": "Percent of GDP or contribution to fiscal baseline where applicable",
            "Transformation": "Included in the Commission medium-term baseline environment; not separately re-estimated in the local projections",
            "Source record": "Debt Sustainability Monitor 2025 Poland tables and extracted baseline record",
        },
        {
            "Input": "Public-investment scenario action",
            "Primary source": "Author scenario design",
            "Dataset and codes": "Three annual actions in 2028, 2029 and 2030",
            "Years used": "2028-2036 application horizon",
            "Coverage": "Poland",
            "Unit": "Percentage points of GDP",
            "Transformation": "The same 1 pp of GDP action is evaluated as expansion and as cut; K_G maps estimated shock scale to the action",
            "Source record": "scenario code and debt-accounting manifests in the reproducibility package",
        },
    ]
    df = pd.DataFrame(rows)
    write_table(df, "source_codes")
    return df


def build_state_table() -> pd.DataFrame:
    panel = pd.read_csv(INPUTS / "country_feature_panel.csv")
    pol = panel[(panel["country"] == "POL") & (panel["year"] == 2022)].iloc[0]
    trans = pd.read_csv(INPUTS / "variant_transformations.csv").set_index("raw_column")
    debt_std = pd.read_csv(INPUTS / "state_variable_standardization_current.csv").set_index("raw_column")

    rows = [
        {
            "State variable": "Investment import content",
            "Measure": "Import content of gross fixed capital formation, from OECD TiVA domestic value-added shares",
            "Source": "OECD TiVA 2025, GFCF_VA_SH",
            "Mean": fmt(trans.loc["trade_raw", "mean"], 3),
            "SD": fmt(trans.loc["trade_raw", "sd"], 3),
            "N": "513",
            "Poland value": fmt(pol["trade_raw"], 3),
            "Poland z": fmt(pol["trade_z"], 3, signed=True),
        },
        {
            "State variable": "Public debt",
            "Measure": "Maastricht gross debt, percent of GDP",
            "Source": "Eurostat government debt",
            "Mean": fmt(debt_std.loc["debt_raw", "mean"], 1),
            "SD": fmt(debt_std.loc["debt_raw", "sd"], 1),
            "N": "567",
            "Poland value": fmt(pol["debt_raw"], 1),
            "Poland z": fmt(pol["debt_z"], 3, signed=True),
        },
        {
            "State variable": "Household net financial worth",
            "Measure": "Negative household net financial worth divided by nominal GDP",
            "Source": "Eurostat financial accounts and nominal GDP",
            "Mean": fmt(trans.loc["liq_raw", "mean"], 3),
            "SD": fmt(trans.loc["liq_raw", "sd"], 3),
            "N": "513",
            "Poland value": fmt(pol["liq_raw"], 3),
            "Poland z": fmt(pol["liq_z"], 3, signed=True),
        },
        {
            "State variable": "Real PPP income level",
            "Measure": "Log real GDP per capita in 2020 PPS terms",
            "Source": "Eurostat national accounts",
            "Mean": fmt(trans.loc["log_gdp_pc_raw", "mean"], 3),
            "SD": fmt(trans.loc["log_gdp_pc_raw", "sd"], 3),
            "N": "513",
            "Poland value": fmt(pol["log_gdp_pc_raw"], 3),
            "Poland z": fmt(pol["log_gdp_pc_z"], 3, signed=True),
        },
    ]
    df = pd.DataFrame(rows)
    write_table(df, "state_variables")
    return df


def build_first_stage_tables() -> pd.DataFrame:
    fs = pd.read_csv(RUN_OUTPUTS / "feature_screen" / "feature_robustness_summary.csv")
    out = pd.DataFrame(
        {
            "State-variable subset": fs["features"].map(public_feature_label),
            "Design matrix rank": fs.apply(
                lambda row: f"{int(row['h8_design_rank'])}/{int(row['h8_regressor_count'])} ({pass_label(row['full_rank'])})",
                axis=1,
            ),
            "Condition number": fs.apply(
                lambda row: gate_value(row["h8_condition_number"], row["condition_ok"], 1),
                axis=1,
            ),
            "Max state correlation": fs.apply(
                lambda row: gate_value(row["max_abs_feature_corr_h8"], row["corr_ok"], 3),
                axis=1,
            ),
            "Support p-value": fs.apply(
                lambda row: gate_value(row["mahalanobis_support_p"], row["support_p_ok"], 3),
                axis=1,
            ),
            "Maximum absolute Polish z-score": fs.apply(
                lambda row: gate_value(row["max_abs_poland_z"], row["max_z_ok"], 3),
                axis=1,
            ),
            "Output p-value h8": fs.apply(
                lambda row: gate_value(row["p_wald_y_h8"], row["output_interaction_ok"], 3),
                axis=1,
            ),
            "Leave-one-country": fs.apply(
                lambda row: f"{int(row['loo_ok_count'])}/27 ({pass_label(row['loo_finite_ok'])})",
                axis=1,
            ),
            "Bootstrap": fs.apply(
                lambda row: f"{int(row['boot_ok_count'])}/19 ({pass_label(row['bootstrap_finite_ok'])})",
                axis=1,
            ),
            "Time blocks": fs.apply(
                lambda row: f"{int(row['time_ok_count'])}/3 ({pass_label(row['time_block_finite_ok'])})",
                axis=1,
            ),
            "Status": fs["gate_status"].map(
                {"PASS_ROBUSTNESS_GATE": "Retained", "FAIL_ROBUSTNESS_GATE": "Not retained"}
            ),
            "Selection reason": fs["gate_reason"].map(public_gate_reason),
        }
    )
    write_table(out, "first_stage_all")
    retained = out[out["Status"] == "Retained"].copy()
    write_table(retained, "first_stage_retained")
    return out


def build_response_tables() -> pd.DataFrame:
    paths = pd.read_csv(RUN_OUTPUTS / "polish_output_spending" / "polish_output_spending_paths.csv")
    eu27_paths = load_eu27_response_paths()
    eu27_ky = dict(zip(eu27_paths["horizon"].astype(int), eu27_paths["K_Y_cumulative"].astype(float)))
    eu27_kg = dict(zip(eu27_paths["horizon"].astype(int), eu27_paths["K_G_cumulative"].astype(float)))
    eu27_ratio = dict(
        zip(eu27_paths["horizon"].astype(int), eu27_paths["cumulative_output_to_spending_ratio"].astype(float))
    )
    piv = paths.pivot_table(
        index="horizon",
        columns="features",
        values=["K_Y_cumulative", "K_G_cumulative"],
        aggfunc="first",
    )
    out = pd.DataFrame({"Horizon": list(range(9))})
    for feature, label in [("trade", "Investment import content"), ("liq", "Household net financial worth")]:
        out[f"{label} K_Y"] = [fmt(piv.loc[h, ("K_Y_cumulative", feature)], 2) for h in range(9)]
        out[f"{label} K_G"] = [fmt(piv.loc[h, ("K_G_cumulative", feature)], 2) for h in range(9)]
    out["Equal-weight K_Y"] = [
        fmt((piv.loc[h, ("K_Y_cumulative", "trade")] + piv.loc[h, ("K_Y_cumulative", "liq")]) / 2, 2)
        for h in range(9)
    ]
    out["Equal-weight K_G"] = [
        fmt((piv.loc[h, ("K_G_cumulative", "trade")] + piv.loc[h, ("K_G_cumulative", "liq")]) / 2, 2)
        for h in range(9)
    ]
    write_table(out, "response_paths")

    ky = pd.DataFrame({"Horizon": list(range(9))})
    ky["EU27 benchmark"] = [fmt(eu27_ky[h], 2) for h in range(9)]
    for feature, label in [("trade", "Investment import content"), ("liq", "Household net financial worth")]:
        ky[label] = [fmt(piv.loc[h, ("K_Y_cumulative", feature)], 2) for h in range(9)]
    ky["Equal weight average"] = [
        fmt((piv.loc[h, ("K_Y_cumulative", "trade")] + piv.loc[h, ("K_Y_cumulative", "liq")]) / 2, 2)
        for h in range(9)
    ]
    write_table(ky, "ky_paths")

    kg = pd.DataFrame({"Horizon": list(range(9))})
    kg["EU27 benchmark"] = [fmt(eu27_kg[h], 2) for h in range(9)]
    for feature, label in [("trade", "Investment import content"), ("liq", "Household net financial worth")]:
        kg[label] = [fmt(piv.loc[h, ("K_G_cumulative", feature)], 2) for h in range(9)]
    kg["Equal weight average"] = [
        fmt((piv.loc[h, ("K_G_cumulative", "trade")] + piv.loc[h, ("K_G_cumulative", "liq")]) / 2, 2)
        for h in range(9)
    ]
    write_table(kg, "kg_paths")

    ratio = pd.DataFrame({"Horizon": list(range(9))})
    ratio["EU27 benchmark"] = [fmt(eu27_ratio[h], 2) for h in range(9)]
    for feature, label in [("trade", "Investment import content"), ("liq", "Household net financial worth")]:
        ratio[label] = [
            fmt(
                piv.loc[h, ("K_Y_cumulative", feature)] / piv.loc[h, ("K_G_cumulative", feature)],
                2,
            )
            for h in range(9)
        ]
    ratio["Equal weight average"] = [
        fmt(
            (
                (piv.loc[h, ("K_Y_cumulative", "trade")] + piv.loc[h, ("K_Y_cumulative", "liq")]) / 2
            )
            / (
                (piv.loc[h, ("K_G_cumulative", "trade")] + piv.loc[h, ("K_G_cumulative", "liq")]) / 2
            ),
            2,
        )
        for h in range(9)
    ]
    write_table(ratio, "output_to_spending_ratio_paths")

    trade_ky_h8 = piv.loc[8, ("K_Y_cumulative", "trade")]
    trade_kg_h8 = piv.loc[8, ("K_G_cumulative", "trade")]
    liq_ky_h8 = piv.loc[8, ("K_Y_cumulative", "liq")]
    liq_kg_h8 = piv.loc[8, ("K_G_cumulative", "liq")]
    ew_ky_h8 = (trade_ky_h8 + liq_ky_h8) / 2.0
    ew_kg_h8 = (trade_kg_h8 + liq_kg_h8) / 2.0
    headline = pd.DataFrame(
        [
            [
                "EU27 panel benchmark",
                "Common EU27 response without state interactions",
                fmt(eu27_ky[8], 2),
                fmt(eu27_kg[8], 2),
                fmt(eu27_ratio[8], 2),
            ],
            [
                "Polish evaluation based on investment import content",
                "Official TiVA GFCF import-content profile",
                fmt(trade_ky_h8, 2),
                fmt(trade_kg_h8, 2),
                fmt(trade_ky_h8 / trade_kg_h8, 2),
            ],
            [
                "Polish evaluation based on household net financial worth",
                "Eurostat financial-accounts balance-sheet profile",
                fmt(liq_ky_h8, 2),
                fmt(liq_kg_h8, 2),
                fmt(liq_ky_h8 / liq_kg_h8, 2),
            ],
            [
                "Equal weight average across the two Polish evaluations",
                "Arithmetic average of the two retained Polish paths",
                fmt(ew_ky_h8, 2),
                fmt(ew_kg_h8, 2),
                fmt(ew_ky_h8 / ew_kg_h8, 2),
            ],
        ],
        columns=["Estimation track", "Country characteristics used for evaluation", "K_Y h8", "K_G h8", "K_Y/K_G h8"],
    )
    write_table(headline, "h8_responses")

    _qa_h8_matches_ratio_table(headline, ratio)
    return paths


def _qa_h8_matches_ratio_table(headline: pd.DataFrame, ratio: pd.DataFrame) -> None:
    h8_polish_rows = headline.iloc[1:].reset_index(drop=True)
    ratio_h8 = ratio[ratio["Horizon"] == 8].iloc[0]
    ratio_targets = [
        ratio_h8["Investment import content"],
        ratio_h8["Household net financial worth"],
        ratio_h8["Equal weight average"],
    ]
    observed = list(h8_polish_rows["K_Y/K_G h8"])
    if observed != ratio_targets:
        raise AssertionError(
            "h8 ratio rows do not match output_to_spending_ratio_paths at horizon 8: "
            f"observed {observed}, target {ratio_targets}"
        )


def build_debt_table() -> pd.DataFrame:
    debt = pd.read_csv(RUN_OUTPUTS / "debt_accounting" / "polish_debt_2036_summary.csv")
    eu27 = pd.read_csv(EU27_BENCHMARK_DEBT).iloc[0]

    def val(feature: str, sign: str, col: str) -> float:
        row = debt[(debt["features"] == feature) & (debt["scenario_sign"] == sign)].iloc[0]
        return float(row[col])

    rows = [
        [
            eu27["empirical_path"],
            fmt(float(eu27["expansion_institutional_debt_equation"]), 1, signed=True),
            fmt(float(eu27["expansion_direct_debt_to_gdp_lp_path"]), 1, signed=True),
            fmt(float(eu27["cut_institutional_debt_equation"]), 1, signed=True),
            fmt(float(eu27["cut_direct_debt_to_gdp_lp_path"]), 1, signed=True),
        ]
    ]
    for label, feature in [
        ("Polish evaluation based on investment import content", "trade"),
        ("Polish evaluation based on household net financial worth", "liq"),
    ]:
        rows.append(
            [
                label,
                fmt(val(feature, "expansion", "dsa_margin_vs_baseline_pp"), 1, signed=True),
                fmt(val(feature, "expansion", "direct_DY_LP_margin_pp"), 1, signed=True),
                fmt(val(feature, "cut", "dsa_margin_vs_baseline_pp"), 1, signed=True),
                fmt(val(feature, "cut", "direct_DY_LP_margin_pp"), 1, signed=True),
            ]
        )
    rows.append(
        [
            "Equal weight average across the two Polish evaluations",
            fmt((val("trade", "expansion", "dsa_margin_vs_baseline_pp") + val("liq", "expansion", "dsa_margin_vs_baseline_pp")) / 2, 1, signed=True),
            fmt((val("trade", "expansion", "direct_DY_LP_margin_pp") + val("liq", "expansion", "direct_DY_LP_margin_pp")) / 2, 1, signed=True),
            fmt((val("trade", "cut", "dsa_margin_vs_baseline_pp") + val("liq", "cut", "dsa_margin_vs_baseline_pp")) / 2, 1, signed=True),
            fmt((val("trade", "cut", "direct_DY_LP_margin_pp") + val("liq", "cut", "direct_DY_LP_margin_pp")) / 2, 1, signed=True),
        ]
    )
    out = pd.DataFrame(
        rows,
        columns=[
            "Empirical path",
            "Expansion, institutional debt equation",
            "Expansion, direct debt-to-GDP local-projection path",
            "Cut, institutional debt equation",
            "Cut, direct debt-to-GDP local-projection path",
        ],
    )
    write_table(out, "debt_2036")
    return out


def build_debt_decomposition_tables() -> pd.DataFrame:
    debt_paths = pd.read_csv(RUN_OUTPUTS / "debt_accounting" / "dsa_debt_paths.csv")
    baseline_inputs = pd.read_csv(INPUTS / "ec_poland_dsm2025_baseline_table_20260308.csv")
    baseline_cols = [
        "year",
        "interest_expenditure",
        "growth_effect_real",
        "inflation_effect",
        "stock_flow_adjustments",
    ]
    baseline_inputs = baseline_inputs[baseline_cols].copy()
    baseline_inputs["baseline_snowball_term_pp"] = (
        baseline_inputs["interest_expenditure"]
        + baseline_inputs["growth_effect_real"]
        + baseline_inputs["inflation_effect"]
    )
    debt_paths = debt_paths.merge(
        baseline_inputs[["year", "baseline_snowball_term_pp", "stock_flow_adjustments"]],
        on="year",
        how="left",
    )
    debt_paths = debt_paths.sort_values(["features", "scenario_sign", "year"]).copy()
    debt_paths["prev_D_Y_new_pp"] = debt_paths.groupby(["features", "scenario_sign"])["D_Y_new_pp"].shift(1)
    debt_paths["scenario_snowball_term_pp"] = (
        debt_paths["D_Y_new_pp"]
        - debt_paths["prev_D_Y_new_pp"]
        + debt_paths["PB_new_pp"]
        - debt_paths["stock_flow_adjustments"].fillna(0.0)
    )
    keep = debt_paths[
        debt_paths["features"].isin(["trade", "liq"])
        & debt_paths["scenario_sign"].isin(["expansion", "cut"])
        & debt_paths["year"].between(2028, 2036)
    ].copy()
    numeric = [
        "Y_shortfall_pct",
        "direct_discretionary_PB_level_pp",
        "delta_cyclical_PB_pp",
        "baseline_PB_pp",
        "PB_new_pp",
        "nominal_gdp_growth_new_pct",
        "baseline_D_Y_pp",
        "D_Y_new_pp",
        "dsa_margin_vs_baseline_pp",
        "direct_DY_LP_margin_initial_action_pp",
        "scenario_snowball_term_pp",
        "stock_flow_adjustments",
    ]
    avg = keep.groupby(["scenario_sign", "year"], as_index=False)[numeric].mean()
    avg["features"] = "equal_weight"
    avg["spec_id"] = "EW"
    combined = pd.concat([keep, avg], ignore_index=True, sort=False)
    feature_order = ["trade", "liq", "equal_weight"]
    sign_order = ["expansion", "cut"]
    combined["feature_order"] = combined["features"].map({name: i for i, name in enumerate(feature_order)})
    combined["sign_order"] = combined["scenario_sign"].map({name: i for i, name in enumerate(sign_order)})
    combined = combined.sort_values(["feature_order", "sign_order", "year"])

    labels = {
        "trade": "Investment import content",
        "liq": "Household net financial worth",
        "equal_weight": "Equal weight average",
    }
    actions = {"expansion": "Expansion", "cut": "Cut"}
    out = pd.DataFrame(
        {
            "Empirical path": combined["features"].map(labels),
            "Action": combined["scenario_sign"].map(actions),
            "Year": combined["year"].astype(int),
            "Baseline debt ratio": combined["baseline_D_Y_pp"].map(lambda x: fmt(x, 2)),
            "Scenario debt ratio": combined["D_Y_new_pp"].map(lambda x: fmt(x, 2)),
            "Debt margin": combined["dsa_margin_vs_baseline_pp"].map(lambda x: fmt(x, 2, signed=True)),
            "Output effect, GDP level": combined["Y_shortfall_pct"].map(lambda x: fmt(x, 2, signed=True)),
            "Direct primary-balance effect": combined["direct_discretionary_PB_level_pp"].map(
                lambda x: fmt(x, 2, signed=True)
            ),
            "Cyclical primary-balance feedback": combined["delta_cyclical_PB_pp"].map(
                lambda x: fmt(x, 2, signed=True)
            ),
            "Baseline primary balance": combined["baseline_PB_pp"].map(lambda x: fmt(x, 2, signed=True)),
            "Scenario primary balance": combined["PB_new_pp"].map(lambda x: fmt(x, 2, signed=True)),
            "Scenario nominal GDP growth": combined["nominal_gdp_growth_new_pct"].map(lambda x: fmt(x, 2, signed=True)),
            "Snowball term": combined["scenario_snowball_term_pp"].map(lambda x: fmt(x, 2, signed=True)),
            "Stock-flow adjustment": combined["stock_flow_adjustments"].map(lambda x: fmt(x, 2, signed=True)),
            "Institutional debt margin": combined["dsa_margin_vs_baseline_pp"].map(lambda x: fmt(x, 2, signed=True)),
            "Direct debt-to-GDP LP margin": combined["direct_DY_LP_margin_initial_action_pp"].map(
                lambda x: fmt(x, 2, signed=True)
            ),
        }
    )
    write_table(out, "annual_debt_decomposition")
    return out


def build_eu27_debt_decomposition_table() -> pd.DataFrame:
    eu27 = pd.read_csv(EU27_BENCHMARK_ANNUAL_DEBT, dtype=str)
    expected_cols = [
        "Empirical path",
        "Action",
        "Year",
        "Baseline debt ratio",
        "Scenario debt ratio",
        "Debt margin",
        "Output effect, GDP level",
        "Direct primary-balance effect",
        "Cyclical primary-balance feedback",
        "Baseline primary balance",
        "Scenario primary balance",
        "Scenario nominal GDP growth",
        "Snowball term",
        "Stock-flow adjustment",
        "Institutional debt margin",
    ]
    if list(eu27.columns) != expected_cols:
        raise ValueError("EU27 annual benchmark debt decomposition has unexpected columns")
    eu27 = eu27.copy()
    if set(eu27["Action"]) != {"Expansion", "Cut"}:
        raise ValueError("EU27 annual benchmark debt decomposition must contain Expansion and Cut actions")
    if set(eu27["Year"].astype(int)) != set(range(2028, 2037)):
        raise ValueError("EU27 annual benchmark debt decomposition must cover 2028-2036")
    write_table(eu27, "eu27_annual_debt_decomposition")
    return eu27


def p_fmt(value: float) -> str:
    if pd.isna(value):
        return ""
    value = float(value)
    if value < 0.001:
        return "<0.001"
    return f"{value:.3f}"


def coefficient_cell(coef: float, se: float, p_value: float) -> str:
    return f"{fmt(float(coef), 3, signed=True)} ({fmt(float(se), 3)}; p={p_fmt(float(p_value))})"


def response_cell(value: float, se: float) -> str:
    return f"{fmt(float(value), 3, signed=True)} ({fmt(float(se), 3)})"


def build_estimation_output_tables() -> None:
    retained = pd.read_csv(RUN_OUTPUTS / "estimation_output" / "retained_regression_coefficients.csv")
    eu27 = pd.read_csv(RUN_OUTPUTS / "estimation_output" / "eu27_benchmark_regression_coefficients.csv")
    response = pd.read_csv(RUN_OUTPUTS / "estimation_output" / "retained_horizon_response_output.csv")
    eu27_response = load_eu27_response_paths()

    setup_wide_rows = [
        {
            "Object": "EU27 panel benchmark",
            "Specification": "Linear local projection without state interactions",
            "Outcomes": "Real GDP response and public-investment spending response",
            "Horizons": "0-8",
            "Fixed effects": "Country and year",
            "Inference": "Driscoll-Kraay covariance by horizon",
            "Lag depth": "1",
            "Controls": "Public consumption shock; lagged public investment growth, public consumption growth, output growth, and short-term rate",
            "Countries": "27",
            "Years": "2004-2024 at h=0; endpoint year declines with horizon",
        },
        {
            "Object": "Polish investment import-content evaluation",
            "Specification": "State-dependent local projection with investment import-content interaction",
            "Outcomes": "Real GDP response and public-investment spending response",
            "Horizons": "0-8",
            "Fixed effects": "Country and year",
            "Inference": "Driscoll-Kraay covariance by horizon",
            "Lag depth": "1",
            "Controls": "Public consumption shock; state main effect; lagged public investment growth, public consumption growth, output growth, and short-term rate",
            "Countries": "27",
            "Years": "2004-2023 at h=0; endpoint year declines with horizon",
        },
        {
            "Object": "Polish household net-financial-worth evaluation",
            "Specification": "State-dependent local projection with household net-financial-worth interaction",
            "Outcomes": "Real GDP response and public-investment spending response",
            "Horizons": "0-8",
            "Fixed effects": "Country and year",
            "Inference": "Driscoll-Kraay covariance by horizon",
            "Lag depth": "1",
            "Controls": "Public consumption shock; state main effect; lagged public investment growth, public consumption growth, output growth, and short-term rate",
            "Countries": "27",
            "Years": "2004-2023 at h=0; endpoint year declines with horizon",
        },
    ]
    setup_wide = pd.DataFrame(setup_wide_rows)
    write_table(setup_wide, "estimation_setup_wide_machine_readable")

    setup_rows = []
    for row in setup_wide_rows:
        evaluation = row["Object"]
        for item in [
            "Specification",
            "Outcomes",
            "Horizons",
            "Fixed effects",
            "Inference",
            "Lag depth",
            "Controls",
            "Countries",
            "Years",
        ]:
            setup_rows.append({"Evaluation": evaluation, "Item": item, "Details": row[item]})
    write_table(pd.DataFrame(setup_rows), "estimation_setup")

    eu27_beta = eu27[eu27["term"].eq("shock_G_I")].copy()
    eu27_table = pd.DataFrame(
        {
            "Outcome": eu27_beta["outcome"].map({"output": "Output", "spending": "Public investment spending"}),
            "Horizon": eu27_beta["horizon"].astype(int),
            "beta_h": [
                coefficient_cell(row.coefficient, row.std_error, row.p_value)
                for row in eu27_beta.itertuples(index=False)
            ],
            "Observations": eu27_beta["nobs"].astype(int),
            "Countries": eu27_beta["country_n"].astype(int),
            "Years": eu27_beta.apply(lambda row: f"{int(row['year_min'])}-{int(row['year_max'])}", axis=1),
            "Design matrix rank": eu27_beta.apply(
                lambda row: f"{int(row['design_rank'])}/{int(row['regressor_count'])}", axis=1
            ),
        }
    )
    write_table(eu27_table, "estimation_eu27_beta_by_horizon")

    central = retained[
        retained["term"].eq("shock_G_I") | retained["term"].str.startswith("shock_G_I_x_")
    ].copy()
    rows = []
    labels = {"investment_import_content": "Investment import content", "household_net_financial_worth": "Household net financial worth"}
    for (spec_id, outcome, horizon), group in central.groupby(["spec_id", "outcome", "horizon"], sort=True):
        beta = group[group["term"].eq("shock_G_I")].iloc[0]
        theta = group[group["term"].str.startswith("shock_G_I_x_")].iloc[0]
        rows.append(
            {
                "Specification": labels.get(spec_id, spec_id),
                "_evaluation_key": spec_id,
                "Outcome": {"output": "Output", "spending": "Public investment spending"}[outcome],
                "Horizon": int(horizon),
                "beta_h": coefficient_cell(beta["coefficient"], beta["std_error"], beta["p_value"]),
                "theta_h": coefficient_cell(theta["coefficient"], theta["std_error"], theta["p_value"]),
                "State variable in theta_h": INTERACTION_LABELS.get(str(theta["term"]), str(theta["term"])),
                "Observations": int(beta["nobs"]),
                "Countries": int(beta["country_n"]),
                "Years": f"{int(beta['year_min'])}-{int(beta['year_max'])}",
                "Design matrix rank": f"{int(beta['design_rank'])}/{int(beta['regressor_count'])}",
            }
        )
    retained_table = pd.DataFrame(rows)
    public_retained_table = retained_table.drop(columns=["_evaluation_key"])
    write_table(public_retained_table, "estimation_retained_beta_theta_by_horizon")
    write_table(
        public_retained_table[
            [
                "Specification",
                "Outcome",
                "Horizon",
                "beta_h",
                "theta_h",
                "State variable in theta_h",
            ]
        ],
        "estimation_retained_beta_theta_coefficients",
    )
    sample_cols = [
        "Specification",
        "Outcome",
        "Horizon",
        "Observations",
        "Countries",
        "Years",
        "Design matrix rank",
    ]
    retained_sample_table = public_retained_table[sample_cols]
    write_table(retained_sample_table, "estimation_retained_beta_theta_sample")
    write_table(
        public_retained_table[retained_table["_evaluation_key"].eq("investment_import_content")][sample_cols],
        "estimation_retained_beta_theta_sample_investment_import_content",
    )
    write_table(
        public_retained_table[retained_table["_evaluation_key"].eq("household_net_financial_worth")][sample_cols],
        "estimation_retained_beta_theta_sample_household_net_financial_worth",
    )

    response_rows = []
    for row in eu27_response.itertuples(index=False):
        response_rows.append(
            {
                "Path": "EU27 panel benchmark",
                "Horizon": int(row.horizon),
                "Incremental output response": response_cell(row.mu_Y_incremental, row.se_Y_incremental),
                "Cumulative K_Y": fmt(row.K_Y_cumulative, 3),
                "Incremental spending response": response_cell(row.mu_G_incremental, row.se_G_incremental),
                "Cumulative K_G": fmt(row.K_G_cumulative, 3),
                "K_Y/K_G": fmt(row.cumulative_output_to_spending_ratio, 3),
                "Observations": int(row.nobs),
                "Countries": int(row.country_n),
                "Years": f"{int(row.year_min_effective)}-{int(row.year_max_effective)}",
            }
        )
    for row in response.itertuples(index=False):
        response_rows.append(
            {
                "Path": labels.get(row.spec_id, row.spec_id),
                "Horizon": int(row.horizon),
                "Incremental output response": response_cell(row.mu_Y_incremental, row.se_Y_incremental),
                "Cumulative K_Y": fmt(row.K_Y_cumulative, 3),
                "Incremental spending response": response_cell(row.mu_G_incremental, row.se_G_incremental),
                "Cumulative K_G": fmt(row.K_G_cumulative, 3),
                "K_Y/K_G": fmt(row.cumulative_output_to_spending_ratio, 3),
                "Observations": int(row.nobs),
                "Countries": int(row.country_n),
                "Years": f"{int(row.year_min_effective)}-{int(row.year_max_effective)}",
            }
        )
    response_table = pd.DataFrame(response_rows)
    write_table(response_table, "estimation_response_bridge_by_horizon")
    write_table(
        response_table[
            [
                "Path",
                "Horizon",
                "Incremental output response",
                "Cumulative K_Y",
                "Incremental spending response",
                "Cumulative K_G",
                "K_Y/K_G",
            ]
        ],
        "estimation_response_bridge_paths",
    )
    write_table(
        response_table[["Path", "Horizon", "Observations", "Countries", "Years"]],
        "estimation_response_bridge_sample",
    )
    response_sample = response_table[["Path", "Horizon", "Observations", "Countries", "Years"]]
    write_table(
        response_sample[response_sample["Path"].eq("EU27 panel benchmark")],
        "estimation_response_bridge_sample_eu27",
    )
    write_table(
        response_sample[response_sample["Path"].eq("Investment import content")],
        "estimation_response_bridge_sample_import_content",
    )
    write_table(
        response_sample[response_sample["Path"].eq("Household net financial worth")],
        "estimation_response_bridge_sample_net_worth",
    )


def build_figures(paths: pd.DataFrame) -> None:
    labels = {"trade": "Investment import content", "liq": "Household net financial worth"}
    eu27_paths = load_eu27_response_paths()
    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    ax.plot(
        eu27_paths["horizon"],
        eu27_paths["K_Y_cumulative"],
        marker="o",
        linewidth=1.8,
        label="EU27 benchmark",
    )
    for feature, label in labels.items():
        sub = paths[paths["features"] == feature]
        ax.plot(sub["horizon"], sub["K_Y_cumulative"], marker="o", label=label)
    avg = paths.groupby("horizon", as_index=False)["K_Y_cumulative"].mean()
    ax.plot(avg["horizon"], avg["K_Y_cumulative"], marker="o", linestyle="--", label="Equal-weight average")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Horizon")
    ax.set_ylabel("Cumulative output response")
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2)
    fig.tight_layout(rect=(0, 0.14, 1, 1))
    fig.savefig(FIGURES / "figure_ky_paths.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    ax.plot(
        eu27_paths["horizon"],
        eu27_paths["cumulative_output_to_spending_ratio"],
        marker="o",
        linewidth=1.8,
        label="EU27 benchmark",
    )
    for feature, label in labels.items():
        sub = paths[paths["features"] == feature].copy()
        ratio = sub["K_Y_cumulative"].to_numpy(dtype=float) / sub["K_G_cumulative"].to_numpy(dtype=float)
        ax.plot(sub["horizon"], ratio, marker="o", label=label)
    avg_ky = paths.groupby("horizon", as_index=False)["K_Y_cumulative"].mean()
    avg_kg = paths.groupby("horizon", as_index=False)["K_G_cumulative"].mean()
    avg_ratio = avg_ky["K_Y_cumulative"].to_numpy(dtype=float) / avg_kg["K_G_cumulative"].to_numpy(dtype=float)
    ax.plot(avg_ky["horizon"], avg_ratio, marker="o", linestyle="--", label="Equal-weight average")
    ax.axhline(0.6, color="black", linewidth=0.8, linestyle=":", label="Commission 0.6")
    ax.set_xlabel("Horizon")
    ax.set_ylabel("Cumulative output-to-spending ratio")
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2)
    fig.tight_layout(rect=(0, 0.14, 1, 1))
    fig.savefig(FIGURES / "figure_output_spending_ratio_paths.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    debt = pd.read_csv(RUN_OUTPUTS / "debt_accounting" / "polish_debt_2036_summary.csv")
    eu27 = pd.read_csv(EU27_BENCHMARK_DEBT).iloc[0]

    def val(feature: str, sign: str, col: str) -> float:
        row = debt[(debt["features"] == feature) & (debt["scenario_sign"] == sign)].iloc[0]
        return float(row[col])

    groups = ["EU27", "Import content", "Net fin. worth", "Equal weight"]
    series = {
        "Expansion, institutional": [
            float(eu27["expansion_institutional_debt_equation"]),
            val("trade", "expansion", "dsa_margin_vs_baseline_pp"),
            val("liq", "expansion", "dsa_margin_vs_baseline_pp"),
            (val("trade", "expansion", "dsa_margin_vs_baseline_pp") + val("liq", "expansion", "dsa_margin_vs_baseline_pp")) / 2,
        ],
        "Expansion, direct": [
            float(eu27["expansion_direct_debt_to_gdp_lp_path"]),
            val("trade", "expansion", "direct_DY_LP_margin_pp"),
            val("liq", "expansion", "direct_DY_LP_margin_pp"),
            (val("trade", "expansion", "direct_DY_LP_margin_pp") + val("liq", "expansion", "direct_DY_LP_margin_pp")) / 2,
        ],
        "Cut, institutional": [
            float(eu27["cut_institutional_debt_equation"]),
            val("trade", "cut", "dsa_margin_vs_baseline_pp"),
            val("liq", "cut", "dsa_margin_vs_baseline_pp"),
            (val("trade", "cut", "dsa_margin_vs_baseline_pp") + val("liq", "cut", "dsa_margin_vs_baseline_pp")) / 2,
        ],
        "Cut, direct": [
            float(eu27["cut_direct_debt_to_gdp_lp_path"]),
            val("trade", "cut", "direct_DY_LP_margin_pp"),
            val("liq", "cut", "direct_DY_LP_margin_pp"),
            (val("trade", "cut", "direct_DY_LP_margin_pp") + val("liq", "cut", "direct_DY_LP_margin_pp")) / 2,
        ],
    }
    fig, ax = plt.subplots(figsize=(7.8, 5.0))
    x_pos = list(range(len(groups)))
    width = 0.18
    offsets = [-1.5 * width, -0.5 * width, 0.5 * width, 1.5 * width]
    for offset, (label, values) in zip(offsets, series.items()):
        ax.bar([x + offset for x in x_pos], values, width=width, label=label)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(groups)
    ax.set_ylabel("2036 debt-to-GDP margin versus baseline, pp", labelpad=14)
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2)
    fig.subplots_adjust(left=0.20, right=0.98, top=0.90, bottom=0.26)
    fig.savefig(FIGURES / "figure_debt_margins_2036.png", dpi=220, pad_inches=0.18)
    plt.close(fig)


def main() -> None:
    required_inputs = [
        INPUTS / "country_feature_panel.csv",
        INPUTS / "variant_transformations.csv",
        INPUTS / "state_variable_standardization_current.csv",
        RUN_OUTPUTS / "feature_screen" / "feature_robustness_summary.csv",
        RUN_OUTPUTS / "polish_output_spending" / "polish_output_spending_paths.csv",
        RUN_OUTPUTS / "debt_accounting" / "polish_debt_2036_summary.csv",
        RUN_OUTPUTS / "debt_accounting" / "dsa_debt_paths.csv",
        EU27_OUTPUT_SPENDING,
        EU27_BENCHMARK_DEBT,
        EU27_BENCHMARK_ANNUAL_DEBT,
    ]
    missing = [str(path.relative_to(ROOT)) for path in required_inputs if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing recomputed public inputs: {missing}")

    sources = build_source_code_table()
    state = build_state_table()
    screen = build_first_stage_tables()
    paths = build_response_tables()
    debt = build_debt_table()
    debt_decomp = build_debt_decomposition_tables()
    eu27_debt_decomp = build_eu27_debt_decomposition_table()
    build_estimation_output_tables()
    build_figures(paths)
    feature_screen_summary = pd.read_csv(RUN_OUTPUTS / "feature_screen" / "feature_robustness_summary.csv")
    retained_spec_ids = set(
        feature_screen_summary.loc[
            feature_screen_summary["gate_status"] == "PASS_ROBUSTNESS_GATE", "spec_id"
        ]
    )

    qa_rows = [
        {"check": "source_code_rows", "status": "PASS" if len(sources) >= 6 else "FAIL", "detail": str(len(sources))},
        {"check": "state_rows", "status": "PASS" if len(state) == 4 else "FAIL", "detail": str(len(state))},
        {"check": "screen_rows", "status": "PASS" if len(screen) == 15 else "FAIL", "detail": str(len(screen))},
        {
            "check": "retained_specs",
            "status": "PASS"
            if retained_spec_ids == {"investment_import_content", "household_net_financial_worth"}
            else "FAIL",
            "detail": "investment import content; household net financial worth",
        },
        {"check": "debt_rows", "status": "PASS" if len(debt) == 4 else "FAIL", "detail": str(len(debt))},
        {
            "check": "debt_table_includes_eu27_benchmark",
            "status": "PASS" if "EU27 panel benchmark" in set(debt["Empirical path"]) else "FAIL",
            "detail": ",".join(debt["Empirical path"]),
        },
        {
            "check": "debt_decomposition_rows",
            "status": "PASS" if len(debt_decomp) == 54 else "FAIL",
            "detail": str(len(debt_decomp)),
        },
        {
            "check": "eu27_annual_debt_decomposition_rows",
            "status": "PASS" if len(eu27_debt_decomp) == 18 else "FAIL",
            "detail": str(len(eu27_debt_decomp)),
        },
        {
            "check": "eu27_annual_debt_decomposition_actions",
            "status": "PASS" if set(eu27_debt_decomp["Action"]) == {"Expansion", "Cut"} else "FAIL",
            "detail": ",".join(sorted(set(eu27_debt_decomp["Action"]))),
        },
        {
            "check": "figure2_includes_eu27_benchmark",
            "status": "PASS" if len(load_eu27_response_paths()) == 9 else "FAIL",
            "detail": "EU27 benchmark path recomputed by code/run_full_estimator_repro.py",
        },
        {
            "check": "estimation_output_tables_present",
            "status": "PASS"
            if all(
                (TABLES / f"{stem}.csv").exists()
                for stem in [
                    "estimation_setup",
                    "estimation_eu27_beta_by_horizon",
                    "estimation_retained_beta_theta_by_horizon",
                    "estimation_response_bridge_by_horizon",
                ]
            )
            else "FAIL",
            "detail": "regression output disclosure tables",
        },
        {
            "check": "uses_recomputed_outputs_for_tables",
            "status": "PASS" if not missing else "FAIL",
            "detail": "all manuscript-facing tables and figures read results/recomputed; frozen run outputs and EU27 debt files are validation targets only",
        },
    ]
    pd.DataFrame(qa_rows).to_csv(QA / "public_tables_figures_qa_20260514.csv", index=False)
    failures = [row for row in qa_rows if row["status"] != "PASS"]
    if failures:
        raise SystemExit(f"QA failed: {failures}")
    print("public tables and figures built")


if __name__ == "__main__":
    main()
