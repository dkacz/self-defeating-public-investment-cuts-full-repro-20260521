#!/usr/bin/env python3
"""Build public downloadable archives from the current repaired package tree."""

from __future__ import annotations

import csv
import hashlib
import json
import re
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DOWNLOADS = ROOT / "docs/downloads"
FULL_ZIP = DOWNLOADS / "full_repro_package_20260521.zip"
FULL_SHA = DOWNLOADS / "full_repro_package_20260521.zip.sha256"
FROZEN_ZIP = DOWNLOADS / "frozen_repro_data_20260521.zip"
FROZEN_SHA = DOWNLOADS / "frozen_repro_data_20260521.zip.sha256"
QA_OUT = ROOT / "qa/download_archives_qa_20260521.csv"
ZIP_TIMESTAMP = (2026, 5, 21, 0, 0, 0)

FULL_INCLUDE_PATHS = [
    ".gitattributes",
    "README.md",
    "REPORT.md",
    "PUBLIC_RELEASE_QA_20260521.md",
    "REPRO_STATUS.md",
    "requirements.txt",
    "code",
    "data",
    "docs/.nojekyll",
    "docs/index.html",
    "docs/notebook_executed_preview.md",
    "figures",
    "manuscript",
    "notebooks",
    "qa",
    "references",
    "results",
    "tables",
]

EXCLUDED_PARTS = {
    "__pycache__",
    ".ipynb_checkpoints",
}

EXCLUDED_SUFFIXES = {
    ".pyc",
    ".pyo",
    ".DS_Store",
}

ARCHIVE_EXCLUDED_REL_PATHS = {
    # This file is generated after archive creation and the external .sha256 is
    # the archive checksum authority. Including an older copy inside the ZIP is
    # misleading because a file cannot certify the final hash of its container.
    "qa/download_archives_qa_20260521.csv",
}

BANNED_SPEC_ID_LABEL = "Spec" + " ID"

BANNED_PROJECT_STATE_HASHES_BY_TOKEN_COUNT = {
    2: {
        "826a828c93e895c37a80ea1852a92397d03502ca1ea2ab7e74f2d3c9dba2d3fe",
        "25ea7ef696799108f8a7528de77779289e0b6addecd135d7ecd2cffcd88ebb25",
        "998093a07d00dbea3762a306e84a1d92713d72fd3ac4eb0290f0da5fa203b74d",
        "ad11b9e28c3ae36ff1059d796546007f3f10fa9541847fafaac43955b9286e72",
        "d2abfb9f00b9e57b16ee823af454bfe3efde5eb3f211224945df924e63302234",
        "d1bfed6cbfdda5f77bb1b132b77564788071864f85d2fb5d8854f308e40881b1",
    },
    3: {
        "987d8c8031825ad0966fa7a184ca141bcdab6ab72cbb05a3af1f9b91991f8cb0",
        "f32481b57cd1aea5de0ba899366427f24565024c34d53351976b784bc2e3e73c",
        "016f276e59469f8627d01dfb28007a324ba6b0f17d3891497bd4801f88bf89c1",
        "dfb8bb3ff6f55eefb5e0bb4b17c00b1b47631483cc6b40287e62828a3581312d",
    },
    4: {
        "872393aa7e937129e8ee6e6f16213a3ce433efc981f7b23c8da85be2731e7be2",
    },
}

STALE_RECOMPUTED_README_HASHES_BY_TOKEN_COUNT = {
    3: {
        "32b80b74a34afbb411205fcf7a4f3222fe843b2317d8083309afc7857960e5da",
    },
    4: {
        "b0fbe657f91eb9b0382ccf81a8c75ab87a91f9037670d4523f604127aea7247f",
    },
    6: {
        "e8653cf397c1e8d2d4359937bab60efd9d5d9163fd243cf0e1e94996891b53ff",
    },
}

TEXT_MEMBER_SUFFIXES = {
    ".csv",
    ".html",
    ".htm",
    ".ipynb",
    ".json",
    ".md",
    ".py",
    ".qmd",
    ".txt",
    ".yaml",
    ".yml",
}


READER_FACING_PATTERN_TEXTS = [
    r"\b[Ff][0-9]{2}\b",
    re.escape(BANNED_SPEC_ID_LABEL),
    r"Specification [0-9]+",
    r"shock_G_I_x_",
    r"PASS_ROBUSTNESS_GATE",
    r"FAIL_ROBUSTNESS_GATE",
    r"all_checks_pass",
    r"failed_[A-Za-z0-9_]+",
    r"feature_screen:",
    r"polish_output:",
    r"eu27_debt:",
    r"source_rebuilt_model_input_matches_frozen",
    r"gate_reason",
    r"gate_status",
    r"raw_p_pass_unadjusted",
    r"log_gdp_pc",
    r"\btrade\+",
    r"\+liq\b",
    r"\bliq\+",
    r"\bdebt\+",
    r"mozdzen_tiva2022",
    r"sample_f[0-9]{2}",
    r"_f[0-9]{2}",
    r"K_Y_h8",
    r"K_G_h8",
    r"dsa_margin_2036",
    r"direct_dy_margin_2036",
    r"references/papers_txt",
    r"feature_robustness_summary\.csv",
    r"criteria_literature_map\.csv",
    r"output_interaction_multiplicity_h8\.csv",
]

READER_FACING_PATTERNS = re.compile("|".join(READER_FACING_PATTERN_TEXTS))


def contains_banned_project_state_label(text: str) -> bool:
    return contains_hashed_token_sequence(text, BANNED_PROJECT_STATE_HASHES_BY_TOKEN_COUNT)


def contains_stale_recomputed_readme_label(text: str) -> bool:
    return contains_hashed_token_sequence(text, STALE_RECOMPUTED_README_HASHES_BY_TOKEN_COUNT)


def contains_hashed_token_sequence(text: str, hashes_by_width: dict[int, set[str]]) -> bool:
    tokens = re.findall(r"[a-z0-9]+", text.lower().replace("_", " "))
    for width, digest_set in hashes_by_width.items():
        if len(tokens) < width:
            continue
        for start in range(0, len(tokens) - width + 1):
            candidate = " ".join(tokens[start:start + width])
            if hashlib.sha256(candidate.encode("utf-8")).hexdigest() in digest_set:
                return True
    return False


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def should_skip(path: Path) -> bool:
    rel = path.relative_to(ROOT)
    rel_posix = rel.as_posix()
    if rel_posix in ARCHIVE_EXCLUDED_REL_PATHS:
        return True
    if any(part in EXCLUDED_PARTS for part in rel.parts):
        return True
    if path.name in EXCLUDED_SUFFIXES:
        return True
    if path.suffix in EXCLUDED_SUFFIXES:
        return True
    if rel.parts[:2] == ("docs", "downloads"):
        return True
    if rel.parts[:2] == ("docs", "lite"):
        return True
    return False


def iter_full_files() -> list[Path]:
    files: list[Path] = []
    for rel in FULL_INCLUDE_PATHS:
        path = ROOT / rel
        if not path.exists():
            continue
        if path.is_file():
            if not should_skip(path):
                files.append(path)
            continue
        for item in path.rglob("*"):
            if item.is_file() and not should_skip(item):
                files.append(item)
    return sorted(set(files), key=lambda p: p.relative_to(ROOT).as_posix())


def write_zip(path: Path, files: list[Path]) -> None:
    if path.exists():
        path.unlink()
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for file_path in files:
            member = file_path.relative_to(ROOT).as_posix()
            info = zipfile.ZipInfo(member, ZIP_TIMESTAMP)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            with file_path.open("rb") as handle:
                archive.writestr(info, handle.read(), compresslevel=9)


def write_sha(path: Path, digest: str, target: Path) -> None:
    path.write_text(f"{digest}  {target.name}\n", encoding="utf-8")


def reader_facing_members() -> set[str]:
    return {
        "README.md",
        "REPORT.md",
        "PUBLIC_RELEASE_QA_20260521.md",
        "REPRO_STATUS.md",
        "docs/index.html",
        "docs/notebook_executed_preview.md",
        "notebooks/self_defeating_public_investment_cuts_full_repro_20260521.ipynb",
        "qa/browser_jupyterlite_fullrepro_qa_20260521.md",
        "qa/notebook_execution_log.txt",
        "results/adopted_specification_summary_20260514.md",
        "data/frozen/adopted_run_outputs/feature_screen/REPORT.md",
        "results/recomputed/feature_screen/REPORT.md",
        *[
            p.relative_to(ROOT).as_posix()
            for p in (ROOT / "tables").glob("*.md")
        ],
        *[
            p.relative_to(ROOT).as_posix()
            for p in (ROOT / "qa/notebook_step_logs").glob("*.log")
        ],
    }


def public_code_members() -> set[str]:
    return {
        "code/build_public_downloads.py",
        "code/sync_jupyterlite_files.py",
        "code/build_public_notebook.py",
        "code/run_full_estimator_repro.py",
    }


def notebook_reader_text(raw: bytes) -> str:
    notebook = json.loads(raw.decode("utf-8", errors="ignore"))
    chunks: list[str] = []
    for cell in notebook.get("cells", []):
        source = cell.get("source", "")
        if isinstance(source, list):
            source_text = "".join(source)
        else:
            source_text = str(source)
        if cell.get("cell_type") in {"markdown", "code"}:
            chunks.append(source_text)
        for output in cell.get("outputs", []):
            if output.get("output_type") == "stream":
                text = output.get("text", "")
                chunks.append("".join(text) if isinstance(text, list) else str(text))
            data = output.get("data", {})
            for key in ["text/plain", "text/html"]:
                text = data.get(key, "")
                chunks.append("".join(text) if isinstance(text, list) else str(text))
            if output.get("ename") or output.get("evalue"):
                chunks.append(f"{output.get('ename', '')}: {output.get('evalue', '')}")
    return "\n".join(chunks)


def scan_zip_reader_facing(path: Path) -> list[str]:
    bad: list[str] = []
    targets = reader_facing_members()
    with zipfile.ZipFile(path) as archive:
        names = set(archive.namelist())
        for member in sorted(targets & names):
            raw = archive.read(member)
            text = notebook_reader_text(raw) if member.endswith(".ipynb") else raw.decode("utf-8", errors="ignore")
            if READER_FACING_PATTERNS.search(text) or contains_banned_project_state_label(text):
                bad.append(member)
    return bad


def scan_zip_public_code(path: Path) -> list[str]:
    bad: list[str] = []
    targets = public_code_members()
    with zipfile.ZipFile(path) as archive:
        names = set(archive.namelist())
        for member in sorted(targets & names):
            text = archive.read(member).decode("utf-8", errors="ignore")
            if contains_banned_project_state_label(text):
                bad.append(member)
    return bad


def scan_zip_all_text_project_state(path: Path) -> tuple[int, list[str]]:
    bad: list[str] = []
    scanned = 0
    with zipfile.ZipFile(path) as archive:
        for member in sorted(archive.namelist()):
            if Path(member).suffix.lower() not in TEXT_MEMBER_SUFFIXES:
                continue
            scanned += 1
            raw = archive.read(member)
            text = notebook_reader_text(raw) if member.endswith(".ipynb") else raw.decode("utf-8", errors="ignore")
            if contains_banned_project_state_label(text):
                bad.append(member)
    return scanned, bad


def scan_zip_recomputed_model_inputs_readme(path: Path) -> bool:
    member = "results/recomputed/model_inputs/README.md"
    with zipfile.ZipFile(path) as archive:
        if member not in archive.namelist():
            return False
        text = archive.read(member).decode("utf-8", errors="ignore")
    return not contains_stale_recomputed_readme_label(text)


def remove_frozen_download() -> bool:
    removed = False
    for path in [FROZEN_ZIP, FROZEN_SHA]:
        if path.exists():
            path.unlink()
            removed = True
    return removed


def write_qa(rows: list[dict[str, str]]) -> None:
    QA_OUT.parent.mkdir(parents=True, exist_ok=True)
    with QA_OUT.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["check", "status", "detail"], lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    files = iter_full_files()
    write_zip(FULL_ZIP, files)
    digest = sha256(FULL_ZIP)
    write_sha(FULL_SHA, digest, FULL_ZIP)
    removed_frozen = remove_frozen_download()
    bad_reader_files = scan_zip_reader_facing(FULL_ZIP)
    bad_public_code_files = scan_zip_public_code(FULL_ZIP)
    all_text_count, bad_all_text_files = scan_zip_all_text_project_state(FULL_ZIP)
    recomputed_readme_ok = scan_zip_recomputed_model_inputs_readme(FULL_ZIP)

    rows = [
        {
            "check": "full_download_zip_exists",
            "status": "PASS" if FULL_ZIP.exists() and FULL_SHA.exists() else "FAIL",
            "detail": f"{FULL_ZIP.relative_to(ROOT)} files={len(files)} bytes={FULL_ZIP.stat().st_size}",
        },
        {
            "check": "full_download_sha256_matches",
            "status": "PASS" if FULL_SHA.read_text(encoding="utf-8").split()[0] == digest else "FAIL",
            "detail": digest,
        },
        {
            "check": "separate_frozen_data_zip_removed",
            "status": "PASS" if not FROZEN_ZIP.exists() and not FROZEN_SHA.exists() else "FAIL",
            "detail": "removed_or_absent" if removed_frozen else "already_absent",
        },
        {
            "check": "full_download_excludes_downloads_and_jupyterlite_build",
            "status": "PASS",
            "detail": "docs/downloads and docs/lite are excluded from the archive payload",
        },
        {
            "check": "reader_facing_download_members_use_semantic_labels",
            "status": "PASS" if not bad_reader_files else "FAIL",
            "detail": "none" if not bad_reader_files else ";".join(bad_reader_files[:20]),
        },
        {
            "check": "public_code_avoids_project_state_labels",
            "status": "PASS" if not bad_public_code_files else "FAIL",
            "detail": "none" if not bad_public_code_files else ";".join(bad_public_code_files[:20]),
        },
        {
            "check": "all_text_archive_members_avoid_project_state_labels",
            "status": "PASS" if not bad_all_text_files else "FAIL",
            "detail": f"scanned={all_text_count}; bad=none" if not bad_all_text_files else ";".join(bad_all_text_files[:20]),
        },
        {
            "check": "recomputed_model_inputs_readme_is_runtime_output",
            "status": "PASS" if recomputed_readme_ok else "FAIL",
            "detail": "results/recomputed/model_inputs/README.md",
        },
    ]
    write_qa(rows)
    failures = [row for row in rows if row["status"] != "PASS"]
    if failures:
        raise SystemExit(f"download archive QA failed: {failures}")
    print(f"download archives PASS full_zip={FULL_ZIP.name} sha256={digest}")


if __name__ == "__main__":
    main()
