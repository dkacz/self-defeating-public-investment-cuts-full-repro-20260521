#!/usr/bin/env python3
"""Refresh the JupyterLite file payload from the current public package."""

from __future__ import annotations

import csv
import hashlib
import json
import mimetypes
import re
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LITE_FILES = ROOT / "docs/lite/files"
LITE_API_CONTENTS = ROOT / "docs/lite/api/contents"
QA_OUT = ROOT / "qa/jupyterlite_files_sync_qa_20260521.csv"
PUBLIC_RELEASE_TIMESTAMP = "2026-05-21T00:00:00Z"

INCLUDE_PATHS = [
    ".gitattributes",
    "README.md",
    "REPORT.md",
    "PUBLIC_RELEASE_QA_20260521.md",
    "REPRO_STATUS.md",
    "requirements.txt",
    "code",
    "data",
    "figures",
    "notebooks",
    "qa",
    "references",
    "results",
    "tables",
]

EXCLUDED_PARTS = {
    "__pycache__",
    ".ipynb_checkpoints",
    "downloads",
}

EXCLUDED_SUFFIXES = {
    ".pyc",
    ".pyo",
    ".DS_Store",
}

EXCLUDED_REL_PREFIXES = {
    "docs/lite",
    "docs/downloads",
    "results/recomputed",
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
    tokens = re.findall(r"[a-z0-9]+", text.lower().replace("_", " "))
    for width, digest_set in BANNED_PROJECT_STATE_HASHES_BY_TOKEN_COUNT.items():
        if len(tokens) < width:
            continue
        for start in range(0, len(tokens) - width + 1):
            candidate = " ".join(tokens[start:start + width])
            if hashlib.sha256(candidate.encode("utf-8")).hexdigest() in digest_set:
                return True
    return False


def rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def should_skip(path: Path) -> bool:
    rel_path = rel(path)
    if any(rel_path == prefix or rel_path.startswith(prefix + "/") for prefix in EXCLUDED_REL_PREFIXES):
        return True
    if any(part in EXCLUDED_PARTS for part in path.relative_to(ROOT).parts):
        return True
    if path.name in EXCLUDED_SUFFIXES:
        return True
    if path.suffix in EXCLUDED_SUFFIXES:
        return True
    return False


def iter_payload_files() -> list[Path]:
    files: list[Path] = []
    for rel_path in INCLUDE_PATHS:
        path = ROOT / rel_path
        if not path.exists():
            continue
        if path.is_file():
            if not should_skip(path):
                files.append(path)
            continue
        for item in path.rglob("*"):
            if item.is_file() and not should_skip(item):
                files.append(item)
    return sorted(set(files), key=rel)


def reader_facing_files() -> list[Path]:
    candidates: list[Path] = []
    for rel_path in [
        "README.md",
        "REPORT.md",
        "PUBLIC_RELEASE_QA_20260521.md",
        "REPRO_STATUS.md",
        "notebooks/self_defeating_public_investment_cuts_full_repro_20260521.ipynb",
        "qa/browser_jupyterlite_fullrepro_qa_20260521.md",
        "qa/notebook_execution_log.txt",
        "results/adopted_specification_summary_20260514.md",
        "data/frozen/adopted_run_outputs/feature_screen/REPORT.md",
        "results/recomputed/feature_screen/REPORT.md",
    ]:
        path = LITE_FILES / rel_path
        if path.exists():
            candidates.append(path)
    for pattern in ["tables/*.md", "qa/notebook_step_logs/*.log"]:
        candidates.extend(sorted(LITE_FILES.glob(pattern)))
    return sorted(set(candidates), key=lambda p: p.relative_to(LITE_FILES).as_posix())


def public_code_files() -> list[Path]:
    candidates: list[Path] = []
    for rel_path in [
        "code/build_public_downloads.py",
        "code/sync_jupyterlite_files.py",
        "code/build_public_notebook.py",
        "code/run_full_estimator_repro.py",
    ]:
        path = LITE_FILES / rel_path
        if path.exists():
            candidates.append(path)
    return sorted(candidates, key=lambda p: p.relative_to(LITE_FILES).as_posix())


def notebook_reader_text(path: Path) -> str:
    notebook = json.loads(path.read_text(encoding="utf-8", errors="ignore"))
    chunks: list[str] = []
    for cell in notebook.get("cells", []):
        source = cell.get("source", "")
        source_text = "".join(source) if isinstance(source, list) else str(source)
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


def scan_reader_facing_files() -> list[str]:
    bad: list[str] = []
    for path in reader_facing_files():
        text = notebook_reader_text(path) if path.suffix == ".ipynb" else path.read_text(encoding="utf-8", errors="ignore")
        if READER_FACING_PATTERNS.search(text) or contains_banned_project_state_label(text):
            bad.append(path.relative_to(LITE_FILES).as_posix())
    return bad


def scan_public_code_files() -> list[str]:
    bad: list[str] = []
    for path in public_code_files():
        text = path.read_text(encoding="utf-8", errors="ignore")
        if contains_banned_project_state_label(text):
            bad.append(path.relative_to(LITE_FILES).as_posix())
    return bad


def timestamp_for(path: Path) -> str:
    return PUBLIC_RELEASE_TIMESTAMP


def mime_for(path: Path) -> str | None:
    suffix_map = {
        ".md": "text/markdown",
        ".py": "text/x-python",
        ".csv": "text/csv",
        ".txt": "text/plain",
        ".log": "text/plain",
        ".json": "application/json",
        ".ipynb": "application/x-ipynb+json",
        ".png": "image/png",
    }
    if path.suffix in suffix_map:
        return suffix_map[path.suffix]
    return mimetypes.guess_type(path.name)[0]


def contents_entry(path: Path) -> dict[str, object]:
    rel_path = path.relative_to(LITE_FILES).as_posix()
    stamp = timestamp_for(path)
    is_dir = path.is_dir()
    return {
        "content": None,
        "created": stamp,
        "format": None,
        "hash": None,
        "hash_algorithm": None,
        "last_modified": stamp,
        "mimetype": None if is_dir else mime_for(path),
        "name": path.name,
        "path": rel_path,
        "size": None if is_dir else path.stat().st_size,
        "type": "directory" if is_dir else "file",
        "writable": True,
    }


def directory_payload(path: Path) -> dict[str, object]:
    rel_path = "" if path == LITE_FILES else path.relative_to(LITE_FILES).as_posix()
    stamp = timestamp_for(path)
    children = sorted(path.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))
    return {
        "content": [contents_entry(child) for child in children],
        "created": stamp,
        "format": "json",
        "hash": None,
        "hash_algorithm": None,
        "last_modified": stamp,
        "mimetype": None,
        "name": "" if path == LITE_FILES else path.name,
        "path": rel_path,
        "size": None,
        "type": "directory",
        "writable": True,
    }


def regenerate_api_contents_metadata() -> None:
    if LITE_API_CONTENTS.exists():
        shutil.rmtree(LITE_API_CONTENTS)
    for directory in [LITE_FILES] + sorted([p for p in LITE_FILES.rglob("*") if p.is_dir()]):
        rel_dir = "" if directory == LITE_FILES else directory.relative_to(LITE_FILES).as_posix()
        target_dir = LITE_API_CONTENTS / rel_dir
        target_dir.mkdir(parents=True, exist_ok=True)
        (target_dir / "all.json").write_text(
            json.dumps(directory_payload(directory), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )


def actual_jupyterlite_payload_paths() -> set[str]:
    paths: set[str] = set()
    if not LITE_FILES.exists():
        return paths
    for item in LITE_FILES.rglob("*"):
        paths.add(item.relative_to(LITE_FILES).as_posix())
    return paths


def advertised_api_contents_paths() -> set[str]:
    paths: set[str] = set()
    if not LITE_API_CONTENTS.exists():
        return paths
    for index_path in LITE_API_CONTENTS.rglob("all.json"):
        try:
            payload = json.loads(index_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            paths.add(index_path.relative_to(ROOT).as_posix() + ":invalid-json")
            continue
        for item in payload.get("content", []):
            item_path = str(item.get("path", ""))
            if item_path:
                paths.add(item_path)
    return paths


def scan_api_contents_mirror() -> list[str]:
    actual = actual_jupyterlite_payload_paths()
    advertised = advertised_api_contents_paths()
    missing = sorted(actual - advertised)
    stale = sorted(advertised - actual)
    return [f"missing:{path}" for path in missing] + [f"stale:{path}" for path in stale]


def clean_excluded_api_contents_metadata() -> None:
    if not LITE_API_CONTENTS.exists():
        return
    for rel_prefix in EXCLUDED_REL_PREFIXES:
        stale_dir = LITE_API_CONTENTS / rel_prefix
        if stale_dir.exists():
            shutil.rmtree(stale_dir)
    for index_path in LITE_API_CONTENTS.rglob("all.json"):
        try:
            payload = json.loads(index_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        content = payload.get("content")
        if not isinstance(content, list):
            continue
        filtered = []
        changed = False
        for item in content:
            item_path = str(item.get("path", ""))
            excluded = any(
                item_path == rel_prefix or item_path.startswith(rel_prefix + "/")
                for rel_prefix in EXCLUDED_REL_PREFIXES
            )
            if excluded:
                changed = True
                continue
            filtered.append(item)
        if changed:
            payload["content"] = filtered
            index_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def scan_excluded_api_contents_metadata() -> list[str]:
    bad: list[str] = []
    if not LITE_API_CONTENTS.exists():
        return bad
    for rel_prefix in EXCLUDED_REL_PREFIXES:
        stale_path = LITE_API_CONTENTS / rel_prefix
        if stale_path.exists():
            bad.append(stale_path.relative_to(ROOT).as_posix())
    for index_path in LITE_API_CONTENTS.rglob("all.json"):
        try:
            payload = json.loads(index_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            bad.append(index_path.relative_to(ROOT).as_posix() + ":invalid-json")
            continue
        for item in payload.get("content", []):
            item_path = str(item.get("path", ""))
            if any(item_path == rel_prefix or item_path.startswith(rel_prefix + "/") for rel_prefix in EXCLUDED_REL_PREFIXES):
                bad.append(index_path.relative_to(ROOT).as_posix() + ":" + item_path)
    return sorted(set(bad))


def write_qa(rows: list[dict[str, str]]) -> None:
    QA_OUT.parent.mkdir(parents=True, exist_ok=True)
    with QA_OUT.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["check", "status", "detail"], lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    payload_files = iter_payload_files()
    if LITE_FILES.exists():
        shutil.rmtree(LITE_FILES)
    LITE_FILES.mkdir(parents=True, exist_ok=True)
    for source in payload_files:
        target = LITE_FILES / source.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    regenerate_api_contents_metadata()
    clean_excluded_api_contents_metadata()

    bad_reader_files = scan_reader_facing_files()
    bad_public_code_files = scan_public_code_files()
    bad_api_metadata = scan_excluded_api_contents_metadata()
    bad_api_mirror = scan_api_contents_mirror()
    rows = [
        {
            "check": "jupyterlite_files_payload_refreshed",
            "status": "PASS" if payload_files else "FAIL",
            "detail": f"files={len(payload_files)}",
        },
        {
            "check": "jupyterlite_reader_facing_files_use_semantic_labels",
            "status": "PASS" if not bad_reader_files else "FAIL",
            "detail": "none" if not bad_reader_files else ";".join(bad_reader_files[:20]),
        },
        {
            "check": "jupyterlite_public_code_avoids_project_state_labels",
            "status": "PASS" if not bad_public_code_files else "FAIL",
            "detail": "none" if not bad_public_code_files else ";".join(bad_public_code_files[:20]),
        },
        {
            "check": "jupyterlite_excludes_precomputed_recomputed_results",
            "status": "PASS" if not (LITE_FILES / "results/recomputed").exists() and not bad_api_metadata else "FAIL",
            "detail": "results/recomputed omitted from files and api/contents metadata" if not bad_api_metadata else ";".join(bad_api_metadata[:20]),
        },
        {
            "check": "jupyterlite_api_contents_excludes_omitted_payloads",
            "status": "PASS" if not bad_api_metadata else "FAIL",
            "detail": "none" if not bad_api_metadata else ";".join(bad_api_metadata[:20]),
        },
        {
            "check": "jupyterlite_api_contents_mirrors_file_payload",
            "status": "PASS" if not bad_api_mirror else "FAIL",
            "detail": "none" if not bad_api_mirror else ";".join(bad_api_mirror[:40]),
        },
    ]
    write_qa(rows)
    qa_payload = LITE_FILES / QA_OUT.relative_to(ROOT)
    qa_payload.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(QA_OUT, qa_payload)
    regenerate_api_contents_metadata()
    clean_excluded_api_contents_metadata()
    failures = [row for row in rows if row["status"] != "PASS"]
    if failures:
        raise SystemExit(f"JupyterLite file sync QA failed: {failures}")
    print(f"jupyterlite files sync PASS files={len(payload_files)}")


if __name__ == "__main__":
    main()
