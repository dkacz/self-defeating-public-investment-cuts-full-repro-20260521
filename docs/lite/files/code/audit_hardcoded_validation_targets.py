#!/usr/bin/env python3
"""Scan the public package for validation-target anti-patterns.

The public notebook may validate recomputed outputs against frozen benchmark
files. It must not validate against inline magic constants or old public
delivery paths.
"""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
QA_OUT = ROOT / "qa/hardcoded_validation_target_scan_20260521.csv"

SCAN_PATHS = [
    ROOT / "code",
    ROOT / "notebooks",
    ROOT / "README.md",
    ROOT / "REPORT.md",
    ROOT / "REPRO_STATUS.md",
    ROOT / "PUBLIC_RELEASE_QA_20260521.md",
    ROOT / "docs/index.html",
]

FORBIDDEN_PATTERNS = [
    (re.compile(r"\b" + "EXPECTED" + r"_[A-Z0-9_]+\b"), "inline_expected_constant_family"),
    (re.compile(r"\b" + "hard" + r"_assert\b"), "hard" + "_assert_validation_label"),
    (re.compile("self_defeating_public_investment_cuts_" + r"repro\.ipynb"), "old_notebook_filename"),
    (re.compile("self-defeating-public-investment-cuts-" + "repro-20260514"), "old_public_repository_path"),
    (re.compile("investment import content " + "Poland z"), "old_inline_state_target_text"),
    (re.compile("net financial worth " + "Poland z"), "old_inline_state_target_text"),
    (re.compile("real PPP income " + "Poland z"), "old_inline_state_target_text"),
    (re.compile("expected" + "_states"), "old_inline_state_target_variable"),
]


def iter_files() -> list[Path]:
    files: list[Path] = []
    for path in SCAN_PATHS:
        if not path.exists():
            continue
        if path.is_file():
            if path.resolve() != Path(__file__).resolve():
                files.append(path)
            continue
        for item in path.rglob("*"):
            if (
                item.is_file()
                and item.resolve() != Path(__file__).resolve()
                and item.suffix not in {".pyc", ".pyo"}
                and "__pycache__" not in item.parts
            ):
                files.append(item)
    return sorted(set(files), key=lambda p: p.relative_to(ROOT).as_posix())


def text_for_scan(path: Path) -> str:
    raw = path.read_text(encoding="utf-8", errors="ignore")
    if path.suffix == ".ipynb":
        try:
            notebook = json.loads(raw)
        except json.JSONDecodeError:
            return raw
        chunks: list[str] = []
        for cell in notebook.get("cells", []):
            source = cell.get("source", "")
            chunks.append("".join(source) if isinstance(source, list) else str(source))
        return "\n".join(chunks)
    return raw


def main() -> None:
    rows: list[dict[str, str]] = []
    for path in iter_files():
        text = text_for_scan(path)
        rel = path.relative_to(ROOT).as_posix()
        for pattern, label in FORBIDDEN_PATTERNS:
            for match in pattern.finditer(text):
                line_no = text[: match.start()].count("\n") + 1
                rows.append(
                    {
                        "check": label,
                        "status": "FAIL",
                        "path": rel,
                        "line": str(line_no),
                        "detail": match.group(0),
                    }
                )
    if not rows:
        rows.append(
            {
                "check": "no_inline_validation_target_antipatterns",
                "status": "PASS",
                "path": ".",
                "line": "",
                "detail": "no forbidden validation-target or stale-public-path patterns found",
            }
        )
    QA_OUT.parent.mkdir(parents=True, exist_ok=True)
    with QA_OUT.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["check", "status", "path", "line", "detail"], lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    failures = [row for row in rows if row["status"] != "PASS"]
    if failures:
        first = failures[0]
        raise SystemExit(
            "hardcoded validation target scan failed: "
            f"{first['check']} in {first['path']}:{first['line']} ({first['detail']})"
        )


if __name__ == "__main__":
    main()
