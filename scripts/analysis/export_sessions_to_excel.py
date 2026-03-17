#!/usr/bin/env python3
"""Download DS study sessions and export to Excel.

Outputs:
- A local mirror of downloaded JSON files (default: outputs/session_exports/sessions)
- An Excel workbook with:
  - query_level sheet: one row per query interaction
  - session_level sheet: one row per session file

Identifier policy:
- Prolific ID is the primary identifier (`prolific_id` column).

Usage examples:
  python scripts/analysis/export_sessions_to_excel.py --source auto
  python scripts/analysis/export_sessions_to_excel.py --source github --repo ksauka/hicxai-data-private
  python scripts/analysis/export_sessions_to_excel.py --source dropbox --dropbox-folder /ds_project_sessions/sessions
"""

from __future__ import annotations

import argparse
import base64
import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd
import requests


def _load_local_secrets(project_root: Path) -> Dict[str, str]:
    """Load `.streamlit/secrets.toml` when present (local convenience only)."""
    secrets_path = project_root / ".streamlit" / "secrets.toml"
    if not secrets_path.exists():
        return {}

    try:
        import tomllib  # Python 3.11+
    except Exception:
        return {}

    try:
        with secrets_path.open("rb") as f:
            parsed = tomllib.load(f)
        return {k: str(v) for k, v in parsed.items() if isinstance(v, (str, int, float, bool))}
    except Exception:
        return {}


def _normalize_repo(repo: str) -> str:
    repo = repo.strip()
    if repo.endswith(".git"):
        repo = repo[:-4]
    if "github.com/" in repo:
        repo = repo.split("github.com/")[-1]
    return repo


def _get_credential(
    key_names: Iterable[str],
    local_secrets: Dict[str, str],
) -> Optional[str]:
    for k in key_names:
        v = os.getenv(k) or local_secrets.get(k)
        if v:
            return str(v)
    return None


def _github_list_session_paths(repo: str, token: str, branch: str = "main") -> List[str]:
    url = f"https://api.github.com/repos/{repo}/git/trees/{branch}?recursive=1"
    headers = {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github+json",
    }
    r = requests.get(url, headers=headers, timeout=30)
    if r.status_code != 200:
        raise RuntimeError(f"GitHub tree listing failed ({r.status_code}): {r.text}")

    tree = r.json().get("tree", [])
    paths = [
        obj["path"]
        for obj in tree
        if obj.get("type") == "blob"
        and isinstance(obj.get("path"), str)
        and obj["path"].startswith("sessions/")
        and obj["path"].endswith(".json")
    ]
    return sorted(paths)


def _github_download_file(repo: str, token: str, path: str, branch: str = "main") -> bytes:
    url = f"https://api.github.com/repos/{repo}/contents/{path}?ref={branch}"
    headers = {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github+json",
    }
    r = requests.get(url, headers=headers, timeout=30)
    if r.status_code != 200:
        raise RuntimeError(f"GitHub content read failed for {path} ({r.status_code}): {r.text}")

    payload = r.json()
    content_b64 = payload.get("content", "")
    if not content_b64:
        raise RuntimeError(f"No content found for {path}")

    return base64.b64decode(content_b64)


def _download_sessions_from_github(
    repo: str,
    token: str,
    out_sessions_dir: Path,
    branch: str = "main",
) -> List[Path]:
    paths = _github_list_session_paths(repo, token, branch=branch)
    print(f"Found {len(paths)} session files in GitHub repo {repo}")

    local_paths: List[Path] = []
    for p in paths:
        data = _github_download_file(repo, token, p, branch=branch)
        local_file = out_sessions_dir / p
        local_file.parent.mkdir(parents=True, exist_ok=True)
        local_file.write_bytes(data)
        local_paths.append(local_file)

    return local_paths


def _download_sessions_from_dropbox(dropbox_folder: str, out_sessions_dir: Path) -> List[Path]:
    try:
        import importlib
        dropbox = importlib.import_module("dropbox")
        dropbox_files = importlib.import_module("dropbox.files")
        FileMetadata = getattr(dropbox_files, "FileMetadata")
        FolderMetadata = getattr(dropbox_files, "FolderMetadata")
    except Exception as e:
        raise RuntimeError(f"Dropbox SDK not available: {e}")

    app_key = os.getenv("DROPBOX_APP_KEY")
    app_secret = os.getenv("DROPBOX_APP_SECRET")
    refresh_token = os.getenv("DROPBOX_REFRESH_TOKEN")

    if not all([app_key, app_secret, refresh_token]):
        raise RuntimeError("Missing DROPBOX_APP_KEY / DROPBOX_APP_SECRET / DROPBOX_REFRESH_TOKEN")

    dbx = dropbox.Dropbox(
        app_key=app_key,
        app_secret=app_secret,
        oauth2_refresh_token=refresh_token,
    )

    local_paths: List[Path] = []
    stack = [dropbox_folder]

    while stack:
        current = stack.pop()
        result = dbx.files_list_folder(current)
        while True:
            for entry in result.entries:
                if isinstance(entry, FolderMetadata):
                    stack.append(entry.path_lower or entry.path_display)
                elif isinstance(entry, FileMetadata):
                    if entry.name.endswith(".json") and "/sessions/" in (entry.path_lower or ""):
                        rel = (entry.path_display or entry.path_lower).lstrip("/")
                        local_file = out_sessions_dir / rel
                        local_file.parent.mkdir(parents=True, exist_ok=True)
                        _, resp = dbx.files_download(entry.path_display or entry.path_lower)
                        local_file.write_bytes(resp.content)
                        local_paths.append(local_file)

            if not result.has_more:
                break
            result = dbx.files_list_folder_continue(result.cursor)

    print(f"Found {len(local_paths)} session files in Dropbox folder {dropbox_folder}")
    return sorted(local_paths)


def _safe_json_load(path: Path) -> Optional[Dict[str, Any]]:
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"Skipping unreadable JSON {path}: {e}")
        return None


def _join_ranking(v: Any) -> str:
    if isinstance(v, list):
        return " | ".join(str(x) for x in v)
    return "" if v is None else str(v)


def _extract_app_number_from_path(path: Path) -> str:
    """Extract app folder token (e.g., app_4) from a session file path."""
    for part in path.parts:
        if part.startswith("app_"):
            return part
    return ""


def _is_test_like(value: Any) -> bool:
    if value is None:
        return False
    s = str(value).upper()
    markers = ("TEST", "DIAG", "DEBUG", "UNKNOWN", "DUMMY")
    return any(m in s for m in markers)


def _get_final_clarification_resolution_mode(query_record: Dict[str, Any]) -> str:
    match_log = query_record.get("clarification_match_log", []) or []
    if match_log:
        final_mode = match_log[-1].get("resolution_mode")
        if final_mode:
            return str(final_mode)

    num_turns = query_record.get("num_clarification_turns")
    if isinstance(num_turns, (int, float)) and num_turns > 0:
        return "clarification_without_logged_mode"
    return "no_clarification"


def _get_clarification_intent_cluster(query_record: Dict[str, Any]) -> str:
    option_sets = query_record.get("clarification_option_sets", []) or []
    if option_sets:
        final_option_set = option_sets[-1] or []
        if final_option_set:
            return " | ".join(str(option) for option in final_option_set)
    return "no_clarification_cluster"


def _build_clarification_summary(query_df: pd.DataFrame) -> pd.DataFrame:
    if query_df.empty:
        return pd.DataFrame(columns=[
            "clarification_intent_cluster",
            "final_clarification_resolution_mode",
            "query_count",
        ])

    summary = (
        query_df.groupby(
            ["clarification_intent_cluster", "final_clarification_resolution_mode"],
            dropna=False,
        )
        .size()
        .reset_index(name="query_count")
        .sort_values(
            by=["query_count", "clarification_intent_cluster", "final_clarification_resolution_mode"],
            ascending=[False, True, True],
        )
    )
    return summary


def _flatten_records(
    json_files: List[Path],
    include_empty_sessions: bool = False,
    include_test_sessions: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    query_rows: List[Dict[str, Any]] = []
    session_rows: List[Dict[str, Any]] = []

    for jf in json_files:
        payload = _safe_json_load(jf)
        if not payload:
            continue

        metadata = payload.get("metadata", {}) or {}
        summary = payload.get("summary_statistics", {}) or {}
        final_feedback = payload.get("final_feedback", {}) or {}

        query_results = payload.get("query_results") or final_feedback.get("query_results") or []

        # Recompute summary metrics from query-level data when available.
        if query_results:
            n_queries = len(query_results)
            n_correct = sum(1 for r in query_results if r.get("is_correct", False))
            n_incorrect = n_queries - n_correct
            total_clar_turns = sum(r.get("num_clarification_turns", 0) for r in query_results)
            total_clar = sum(1 for r in query_results if (r.get("num_clarification_turns", 0) or 0) > 0)
            total_time = sum(r.get("interaction_time_seconds", 0) for r in query_results)
            accuracy = n_correct / n_queries if n_queries else 0
            avg_clar = total_clar / n_queries if n_queries else 0
            avg_clar_turns = total_clar_turns / n_queries if n_queries else 0
            avg_time = total_time / n_queries if n_queries else 0
            clarity = [r.get("feedback_clarity") for r in query_results if r.get("feedback_clarity") is not None]
            confidence = [r.get("feedback_confidence") for r in query_results if r.get("feedback_confidence") is not None]
            avg_clarity = (sum(clarity) / len(clarity)) if clarity else None
            avg_confidence = (sum(confidence) / len(confidence)) if confidence else None
        else:
            n_queries = summary.get("total_queries", None)
            n_correct = summary.get("queries_correct", None)
            n_incorrect = summary.get("queries_incorrect", None)
            accuracy = summary.get("accuracy", None)
            total_clar_turns = summary.get("total_clarifications", None)
            total_clar = summary.get("total_clarifications", None)
            avg_clar = summary.get("avg_clarifications_per_query", None)
            avg_clar_turns = summary.get("avg_clarifications_per_query", None)
            total_time = summary.get("total_interaction_time_seconds", None)
            avg_time = summary.get("avg_time_per_query_seconds", None)
            avg_clarity = summary.get("avg_feedback_clarity", None)
            avg_confidence = summary.get("avg_feedback_confidence", None)

        prolific_id = (
            metadata.get("participant_id")
            or final_feedback.get("prolific_pid")
            or final_feedback.get("participant_id")
            or ""
        )
        app_number = metadata.get("app_number") or _extract_app_number_from_path(jf)
        session_id = metadata.get("session_id") or final_feedback.get("session_id") or ""

        if not include_empty_sessions and not query_results:
            continue
        if not include_test_sessions and (
            _is_test_like(prolific_id)
            or _is_test_like(session_id)
            or _is_test_like(jf.name)
        ):
            continue

        session_base = {
            "source_file": str(jf),
            "prolific_id": prolific_id,
            "session_id": session_id,
            "condition": metadata.get("condition") or final_feedback.get("condition") or "",
            "dataset": metadata.get("dataset", ""),
            "system": metadata.get("system", ""),
            "app_number": app_number,
            "session_start": metadata.get("session_start", ""),
            "session_end": metadata.get("session_end", ""),
            "duration_seconds": metadata.get("duration_seconds", None),
            "total_queries": n_queries,
            "queries_correct": n_correct,
            "queries_incorrect": n_incorrect,
            "accuracy": accuracy,
            "total_clarifications": total_clar,
            "total_clarification_turns": total_clar_turns,
            "avg_clarifications_per_query": avg_clar,
            "avg_clarification_turns_per_query": avg_clar_turns,
            "total_why_questions": summary.get("total_why_questions", None),
            "total_interaction_time_seconds": total_time,
            "avg_time_per_query_seconds": avg_time,
            "total_interaction_time_minutes": (total_time / 60.0) if total_time is not None else None,
            "avg_time_per_query_minutes": (avg_time / 60.0) if avg_time is not None else None,
            "avg_feedback_clarity": avg_clarity,
            "avg_feedback_confidence": avg_confidence,
        }
        session_rows.append(session_base)

        for q in query_results:
            final_clarification_resolution_mode = _get_final_clarification_resolution_mode(q)
            clarification_intent_cluster = _get_clarification_intent_cluster(q)
            row = dict(session_base)
            row.update({
                "query_index": q.get("query_index"),
                "query_text": q.get("query_text"),
                "true_intent": q.get("true_intent"),
                "predicted_intent": q.get("predicted_intent"),
                "confidence": q.get("confidence"),
                "num_clarification_turns": q.get("num_clarification_turns"),
                "is_correct": q.get("is_correct"),
                "interaction_time_seconds": q.get("interaction_time_seconds"),
                "interaction_time_minutes": (
                    q.get("interaction_time_seconds", 0) / 60.0
                    if q.get("interaction_time_seconds") is not None else None
                ),
                "conversation": q.get("conversation_transcript", ""),
                "timestamp": q.get("timestamp"),
                # Live DS output fields (prefer explicit ds_agent_* keys, fallback to core fields).
                "ds_agent_predicted_intent": q.get("ds_agent_predicted_intent", q.get("predicted_intent")),
                "ds_agent_num_interactions": q.get("ds_agent_num_interactions", q.get("num_clarification_turns")),
                "ds_agent_confidence": q.get("ds_agent_confidence", q.get("confidence")),
                "ds_agent_was_correct": q.get("ds_agent_was_correct", q.get("is_correct")),
                # Static study-set/reference values (supports legacy key names).
                "studyset_predicted_intent": q.get("studyset_predicted_intent", q.get("benchmark_predicted_intent", q.get("llm_predicted_intent"))),
                "studyset_num_interactions": q.get("studyset_num_interactions", q.get("benchmark_num_interactions", q.get("llm_num_interactions"))),
                "studyset_confidence": q.get("studyset_confidence", q.get("benchmark_confidence", q.get("llm_confidence"))),
                "studyset_was_correct": q.get("studyset_was_correct", q.get("benchmark_was_correct", q.get("llm_was_correct"))),
                "ds_agent_agrees_with_oracle": q.get("ds_agent_agrees_with_oracle", q.get("is_correct")),
                "ds_agent_agrees_with_studyset": q.get(
                    "ds_agent_agrees_with_studyset",
                    (q.get("predicted_intent") == q.get("studyset_predicted_intent"))
                    if q.get("studyset_predicted_intent") is not None else None,
                ),
                "clarification_option_sets": json.dumps(q.get("clarification_option_sets", []), ensure_ascii=False),
                "clarification_match_log": json.dumps(q.get("clarification_match_log", []), ensure_ascii=False),
                "clarification_matched_options": " | ".join(
                    str(item.get("matched_option"))
                    for item in (q.get("clarification_match_log", []) or [])
                    if item.get("matched_option")
                ),
                "clarification_normalization_used": any(
                    item.get("matched_option") is not None
                    for item in (q.get("clarification_match_log", []) or [])
                ),
                "final_clarification_resolution_mode": final_clarification_resolution_mode,
                "clarification_intent_cluster": clarification_intent_cluster,
                "user_validated_intent": q.get("user_validated_intent"),
                "user_ranking": _join_ranking(q.get("user_ranking")),
                "user_agrees_with_system": q.get("user_agrees_with_system"),
                "user_agrees_with_oracle": q.get("user_agrees_with_oracle"),
                "feedback_clarity": q.get("feedback_clarity"),
                "feedback_confidence": q.get("feedback_confidence"),
                "feedback_comment": q.get("feedback_comment", ""),
                "feedback_submitted": q.get("feedback_submitted"),
            })
            query_rows.append(row)

    query_df = pd.DataFrame(query_rows)
    session_df = pd.DataFrame(session_rows)

    if not query_df.empty:
        query_df = query_df.sort_values(by=["prolific_id", "session_id", "query_index"], na_position="last")
    if not session_df.empty:
        session_df = session_df.sort_values(by=["prolific_id", "session_start"], na_position="last")

    return query_df, session_df


def main() -> int:
    parser = argparse.ArgumentParser(description="Download session JSON files and export to Excel.")
    parser.add_argument("--source", choices=["auto", "github", "dropbox"], default="auto")
    parser.add_argument("--repo", default=None, help="GitHub repo in owner/repo or URL format")
    parser.add_argument("--branch", default="main", help="GitHub branch to read from")
    parser.add_argument("--dropbox-folder", default="/ds_project_sessions", help="Dropbox root folder to scan")
    parser.add_argument("--out-dir", default="outputs/session_exports", help="Output directory")
    parser.add_argument("--excel-name", default="sessions_export.xlsx", help="Excel output filename")
    parser.add_argument(
        "--include-empty-sessions",
        action="store_true",
        help="Include sessions with no query_results (disabled by default).",
    )
    parser.add_argument(
        "--include-test-sessions",
        action="store_true",
        help="Include test/diagnostic sessions (disabled by default).",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[2]
    out_dir = (project_root / args.out_dir).resolve()
    out_sessions_dir = out_dir / "sessions"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_sessions_dir.mkdir(parents=True, exist_ok=True)

    local_secrets = _load_local_secrets(project_root)

    github_repo = args.repo or _get_credential(["GITHUB_DATA_REPO", "GITHUB_REPO"], local_secrets)
    github_token = _get_credential(["GITHUB_DATA_TOKEN", "GITHUB_TOKEN"], local_secrets)

    if github_repo:
        github_repo = _normalize_repo(github_repo)

    json_files: List[Path] = []

    if args.source in ("auto", "github"):
        if github_repo and github_token:
            try:
                json_files = _download_sessions_from_github(
                    repo=github_repo,
                    token=github_token,
                    out_sessions_dir=out_sessions_dir,
                    branch=args.branch,
                )
            except Exception as e:
                print(f"GitHub download failed: {e}")
                if args.source == "github":
                    return 1
        elif args.source == "github":
            print("Missing GitHub credentials. Set GITHUB_REPO/GITHUB_TOKEN in env or .streamlit/secrets.toml")
            return 1

    if not json_files and args.source in ("auto", "dropbox"):
        try:
            json_files = _download_sessions_from_dropbox(args.dropbox_folder, out_sessions_dir)
        except Exception as e:
            print(f"Dropbox download failed: {e}")
            if args.source == "dropbox":
                return 1

    if not json_files:
        print("No session JSON files found from selected source(s).")
        return 1

    query_df, session_df = _flatten_records(
        json_files,
        include_empty_sessions=args.include_empty_sessions,
        include_test_sessions=args.include_test_sessions,
    )

    excel_path = out_dir / args.excel_name
    clarification_summary_df = _build_clarification_summary(query_df)
    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        query_df.to_excel(writer, sheet_name="query_level", index=False)
        session_df.to_excel(writer, sheet_name="session_level", index=False)
        clarification_summary_df.to_excel(writer, sheet_name="clarification_summary", index=False)

    print(f"Downloaded {len(json_files)} JSON files")
    print(f"Query rows: {len(query_df)}")
    print(f"Session rows: {len(session_df)}")
    print(f"Excel written to: {excel_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
