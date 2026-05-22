#!/usr/bin/env python3
"""Run the local reproducibility chain for the public package."""

from __future__ import annotations

import csv
import hashlib
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / "code"
QA = ROOT / "qa"
RESULTS = ROOT / "results"
PROV = ROOT / "data/provenance"

SCRIPT_STEPS = [
    ("rebuild_adopted_state_variables_from_sources.py", []),
    ("run_full_estimator_repro.py", []),
    ("build_adopted_specification_summary.py", []),
    ("build_public_tables_figures.py", []),
    ("build_public_notebook.py", []),
    ("audit_hardcoded_validation_targets.py", []),
    ("execute_public_notebook.py", [str(ROOT / "notebooks/self_defeating_public_investment_cuts_full_repro_20260521.ipynb")]),
]

QA_FILES = [
    QA / "source_rebuild_adopted_state_variables_qa_20260514.csv",
    QA / "full_estimator_model_input_rebuild_qa.csv",
    QA / "full_estimator_repro_validation.csv",
    QA / "adopted_specification_summary_qa_20260514.csv",
    QA / "public_tables_figures_qa_20260514.csv",
    QA / "hardcoded_validation_target_scan_20260521.csv",
    RESULTS / "notebook_check_summary.csv",
    QA / "jupyterlite_files_sync_qa_20260521.csv",
    QA / "download_archives_qa_20260521.csv",
]

FROZEN_DIRS = [
    ROOT / "data/frozen/adopted_sources",
    ROOT / "data/frozen/adopted_model_inputs",
    ROOT / "data/frozen/adopted_run_outputs",
    ROOT / "data/frozen/eu27_benchmark_debt",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_manifest() -> None:
    rows = []
    for folder in FROZEN_DIRS:
        for path in sorted(p for p in folder.rglob("*") if p.is_file()):
            rows.append(
                {
                    "path": str(path.relative_to(ROOT)),
                    "bytes": path.stat().st_size,
                    "sha256": sha256(path),
                }
            )
    PROV.mkdir(parents=True, exist_ok=True)
    out = PROV / "frozen_inputs_manifest_20260514.csv"
    with out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["path", "bytes", "sha256"], lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def refresh_feature_screen_manifest() -> None:
    manifest = ROOT / "data/frozen/adopted_run_outputs/feature_screen/source_manifest.csv"
    rows = []
    for folder in [ROOT / "data/frozen/adopted_model_inputs", ROOT / "data/frozen/adopted_run_outputs/feature_screen"]:
        for path in sorted(p for p in folder.rglob("*") if p.is_file()):
            if path == manifest:
                continue
            rows.append(
                {
                    "path": str(path.relative_to(ROOT)),
                    "sha256": sha256(path),
                    "bytes": path.stat().st_size,
                }
            )
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["path", "sha256", "bytes"], lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def refresh_notebook_manifest() -> None:
    rows = [
        ("code/build_public_notebook.py", "notebook builder"),
        ("code/execute_public_notebook.py", "notebook execution checker"),
        ("notebooks/self_defeating_public_investment_cuts_full_repro_20260521.ipynb", "public reproducibility notebook"),
        ("docs/notebook_executed_preview.md", "executed notebook preview"),
        ("results/notebook_check_summary.csv", "notebook QA ledger"),
        ("results/notebook_selected_horizons.csv", "notebook selected-horizon output table"),
        ("results/notebook_debt_margins_2036.csv", "notebook debt endpoint output table"),
        ("qa/notebook_execution_log.txt", "notebook execution log"),
    ]
    out = RESULTS / "notebook_manifest_20260521.csv"
    manifest_rows = []
    for rel_path, role in rows:
        path = ROOT / rel_path
        if not path.exists():
            raise FileNotFoundError(f"missing notebook manifest input: {rel_path}")
        manifest_rows.append({"path": rel_path, "role": role, "sha256": sha256(path)})
    with out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["path", "role", "sha256"], lineterminator="\n")
        writer.writeheader()
        writer.writerows(manifest_rows)


def validate_manifest(manifest: Path, base: Path, path_col: str) -> dict[str, str]:
    if not manifest.exists():
        return {
            "check": f"manifest_valid:{manifest.relative_to(ROOT)}",
            "status": "FAIL",
            "detail": "missing_manifest",
        }
    with manifest.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    bad: list[str] = []
    for row in rows:
        rel_path = row.get(path_col, "")
        target = base / rel_path
        if not target.exists():
            bad.append(f"missing:{rel_path}")
            continue
        expected_hash = row.get("sha256", "")
        if expected_hash and expected_hash != sha256(target):
            bad.append(f"sha256:{rel_path}")
        expected_bytes = row.get("bytes", "")
        if expected_bytes and str(target.stat().st_size) != str(expected_bytes):
            bad.append(f"bytes:{rel_path}")
    return {
        "check": f"manifest_valid:{manifest.relative_to(ROOT)}",
        "status": "PASS" if not bad else "FAIL",
        "detail": f"rows={len(rows)} bad={len(bad)}" + ("" if not bad else f" first={bad[0]}"),
    }


def manifest_validation_rows() -> list[dict[str, str]]:
    return [
        validate_manifest(PROV / "adopted_sources_manifest_20260514.csv", ROOT, "path"),
        validate_manifest(PROV / "frozen_inputs_manifest_20260514.csv", ROOT, "path"),
        validate_manifest(RESULTS / "adopted_specification_summary_manifest_20260514.csv", ROOT, "path"),
        validate_manifest(RESULTS / "notebook_manifest_20260521.csv", ROOT, "path"),
        validate_manifest(RESULTS / "recomputed/full_estimator_manifest.csv", ROOT, "path"),
        validate_manifest(ROOT / "data/frozen/adopted_run_outputs/feature_screen/source_manifest.csv", ROOT, "path"),
    ]


def run_script(name: str, args: list[str]) -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    proc = subprocess.run(
        [sys.executable, str(CODE / name), *args],
        cwd=ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    output_tail = proc.stdout[-800:].rstrip().replace(str(ROOT), ".").replace("\n", " | ")
    if proc.returncode == 0 and name == "run_full_estimator_repro.py":
        output_tail = (
            "Full estimator repro completed. Verbose internal estimator stdout is "
            "suppressed in this public run ledger; see results/recomputed/, tables/ "
            "and qa/full_estimator_repro_validation.csv for reproduced outputs."
        )
    if proc.returncode == 0 and name == "build_public_downloads.py":
        output_tail = (
            "Download archive rebuilt. The external docs/downloads/*.sha256 file "
            "is the authoritative archive checksum; archive-internal QA ledgers "
            "do not certify the containing ZIP hash."
        )
    return {
        "script": name,
        "status": "PASS" if proc.returncode == 0 else "FAIL",
        "returncode": str(proc.returncode),
        "output_tail": output_tail,
    }


def qa_passes(path: Path) -> tuple[bool, str]:
    if not path.exists():
        return False, "missing"
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return False, "empty"
    fields = rows[0].keys()
    if "status" in fields:
        bad = [row for row in rows if row.get("status") != "PASS"]
        return not bad, f"rows={len(rows)} bad={len(bad)}"
    if "passed" in fields:
        bad = [row for row in rows if str(row.get("passed")).lower() != "true"]
        return not bad, f"rows={len(rows)} bad={len(bad)}"
    return False, "no status/passed column"


def write_csv(path: Path, rows: list[dict[str, str]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def build_qa_rows() -> list[dict[str, str]]:
    qa_rows = []
    for path in QA_FILES:
        ok, detail = qa_passes(path)
        qa_rows.append(
            {
                "check": f"qa_passes:{path.relative_to(ROOT)}",
                "status": "PASS" if ok else "FAIL",
                "detail": detail,
            }
        )

    external_refs = []
    for script in [CODE / "build_public_tables_figures.py"]:
        text = script.read_text(encoding="utf-8")
        for banned in ["ROOT.parent"]:
            if banned in text:
                external_refs.append(f"{script.relative_to(ROOT)}:{banned}")
    qa_rows.append(
        {
            "check": "no_external_parent_dependency_in_table_builder",
            "status": "PASS" if not external_refs else "FAIL",
            "detail": ";".join(external_refs),
        }
    )
    qa_rows.extend(manifest_validation_rows())
    return qa_rows


def mirror_final_public_qa_to_lite() -> None:
    source = QA / "public_repro_qa_20260521.csv"
    target = ROOT / "docs/lite/files/qa/public_repro_qa_20260521.csv"
    if target.parent.exists() and source.exists():
        target.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")


def refresh_reader_facing_frozen_reports() -> None:
    """Mirror cleaned reader-facing reports while leaving machine CSV targets intact."""

    source = ROOT / "results/recomputed/feature_screen/REPORT.md"
    target = ROOT / "data/frozen/adopted_run_outputs/feature_screen/REPORT.md"
    if source.exists() and target.exists():
        target.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")


def main() -> None:
    refresh_feature_screen_manifest()
    write_manifest()
    run_rows = [run_script(name, args) for name, args in SCRIPT_STEPS]
    refresh_reader_facing_frozen_reports()
    refresh_feature_screen_manifest()
    write_manifest()
    refresh_notebook_manifest()
    write_csv(QA / "public_repro_run_log_20260521.csv", run_rows, ["script", "status", "returncode", "output_tail"])

    download_row = run_script("build_public_downloads.py", [])
    run_rows.append(download_row)
    sync_row = run_script("sync_jupyterlite_files.py", [])
    run_rows.append(sync_row)
    write_csv(QA / "public_repro_run_log_20260521.csv", run_rows, ["script", "status", "returncode", "output_tail"])

    qa_rows = build_qa_rows()
    write_csv(QA / "public_repro_qa_20260521.csv", qa_rows, ["check", "status", "detail"])

    # Rebuild public payloads after the final run log and QA ledger exist.
    final_download_row = run_script("build_public_downloads.py", [])
    final_sync_row = run_script("sync_jupyterlite_files.py", [])
    mirror_final_public_qa_to_lite()

    qa_rows = build_qa_rows()
    write_csv(QA / "public_repro_qa_20260521.csv", qa_rows, ["check", "status", "detail"])
    mirror_final_public_qa_to_lite()

    ok, detail = qa_passes(QA / "download_archives_qa_20260521.csv")
    final_ok, final_detail = qa_passes(QA / "jupyterlite_files_sync_qa_20260521.csv")
    failures = (
        [row for row in run_rows if row["status"] != "PASS"]
        + [row for row in qa_rows if row["status"] != "PASS"]
        + [row for row in [final_download_row, final_sync_row] if row["status"] != "PASS"]
    )
    if failures:
        raise SystemExit(f"public reproducibility failed: {failures}")
    if not ok or not final_ok:
        raise SystemExit(f"final public payload QA failed: download={detail}; jupyterlite={final_detail}")
    print("public reproducibility PASS")


if __name__ == "__main__":
    main()
