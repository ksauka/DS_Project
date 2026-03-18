"""Analyze Zadeh-like conflict persistence from DS interaction logs.

This script replays user turns from an evaluation CSV, computes per-turn conflict
coefficient K, and reports:
1) Correlation of high/mean K with clarification depth
2) Correlation of high/mean K with unknown prediction rate
3) Cluster-specific summaries (order and PTO families)
4) Ablation across combination rules

Usage example:
python scripts/analysis/analyze_conflict_zadeh.py \
  --results-file results/clinc150/workflow_demo/ds_evaluation/clinc150_predictions.csv \
  --dataset clinc150 \
  --output-dir outputs/zadeh_conflict
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config.hierarchy_loader import load_hierarchical_intents_from_json, load_hierarchy_from_json
from config.threshold_loader import load_thresholds_from_json
from src.models.classifier import IntentClassifier
from src.models.ds_mass_function import DSMassFunction
from src.models.embeddings import IntentEmbeddings, SentenceEmbedder


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)
logging.getLogger("src.models.ds_mass_function").setLevel(logging.WARNING)


CLUSTERS = {
    "order_cluster": {"order", "order_status", "shopping_list_update", "shopping_list"},
    "pto_cluster": {"pto_used", "pto_balance", "pto_request_status", "pto_request"},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze DS conflict persistence (Zadeh-like)")
    parser.add_argument(
        "--results-file",
        type=str,
        default="",
        help="Path to predictions CSV (simulated evaluation).",
    )
    parser.add_argument(
        "--sessions-dir",
        type=str,
        default="",
        help="Path to exported session JSON directory (real interactions).",
    )
    parser.add_argument("--dataset", type=str, default="clinc150", choices=["banking77", "clinc150"])
    parser.add_argument("--output-dir", type=str, default="outputs/zadeh_conflict")
    parser.add_argument(
        "--rules",
        type=str,
        default="dempster,no_norm,yager",
        help="Comma-separated combination rules for ablation",
    )
    parser.add_argument(
        "--k-high-quantile",
        type=float,
        default=0.9,
        help="Quantile used to mark high-K queries",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=0,
        help="Optional cap for debugging (0 = all rows)",
    )
    return parser.parse_args()


def load_rows_from_sessions_dir(sessions_dir: str) -> pd.DataFrame:
    """Load query-level rows from exported private-session JSON files.

    Expected schema includes either top-level `query_results` or
    `final_feedback.query_results`.
    """
    root = Path(sessions_dir)
    json_files = sorted(root.rglob("*.json"))
    rows = []

    for fpath in json_files:
        try:
            with fpath.open("r", encoding="utf-8") as f:
                payload = json.load(f)
        except Exception:
            continue

        qrs = payload.get("query_results")
        if not isinstance(qrs, list) or not qrs:
            qrs = payload.get("final_feedback", {}).get("query_results", [])

        meta = payload.get("metadata", {})
        participant = meta.get("participant_id", "")
        dataset = meta.get("dataset", "")

        for r in qrs:
            if not isinstance(r, dict):
                continue
            rows.append(
                {
                    "query": r.get("query_text", ""),
                    "true_intent": r.get("true_intent", ""),
                    "predicted_intent": r.get("predicted_intent", ""),
                    "confidence": r.get("confidence", 0.0),
                    "interaction": r.get("conversation_transcript", ""),
                    "query_index": r.get("query_index", -1),
                    "timestamp": r.get("timestamp", ""),
                    "participant_id": participant,
                    "dataset": dataset,
                    "source_file": str(fpath),
                }
            )

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    # Deduplicate when both top-level and final_feedback copies exist.
    dedup_cols = ["source_file", "query_index", "query", "timestamp"]
    keep_cols = [c for c in dedup_cols if c in df.columns]
    if keep_cols:
        df = df.drop_duplicates(subset=keep_cols).reset_index(drop=True)
    return df


def model_config(dataset: str) -> Dict[str, str]:
    root = Path(__file__).parent.parent.parent
    if dataset == "clinc150":
        return {
            "hierarchy": str(root / "config/hierarchies/clinc150_hierarchy.json"),
            "intents": str(root / "config/hierarchies/clinc150_intents.json"),
            "thresholds": str(root / "results/clinc150/workflow_demo/clinc150_optimal_thresholds.json"),
            "classifier": str(root / "experiments/clinc150/clinc150_logistic_model.pkl"),
        }
    return {
        "hierarchy": str(root / "config/hierarchies/banking77_hierarchy.json"),
        "intents": str(root / "config/hierarchies/banking77_intents.json"),
        "thresholds": str(root / "results/banking77/workflow_demo/banking77_optimal_thresholds.json"),
        "classifier": str(root / "experiments/banking77/banking77_logistic_model.pkl"),
    }


def load_ds_system(dataset: str) -> DSMassFunction:
    cfg = model_config(dataset)
    hierarchy = load_hierarchy_from_json(cfg["hierarchy"])
    intents = load_hierarchical_intents_from_json(cfg["intents"])
    thresholds = load_thresholds_from_json(cfg["thresholds"])
    classifier = IntentClassifier.from_pretrained(cfg["classifier"])
    embedder = SentenceEmbedder(model_name="intfloat/e5-base")
    intent_embeddings = IntentEmbeddings(intents, embedder=embedder)
    return DSMassFunction(
        intent_embeddings=intent_embeddings.get_all_embeddings(),
        hierarchy=hierarchy,
        classifier=classifier,
        custom_thresholds=thresholds,
        enable_belief_tracking=False,
        embedder=embedder,
    )


def extract_user_turns(interaction: str, query: str) -> List[str]:
    if not isinstance(interaction, str) or not interaction.strip():
        return [query] if isinstance(query, str) and query.strip() else []

    turns = []
    for line in interaction.splitlines():
        line = line.strip()
        if line.startswith("User:"):
            text = line.replace("User:", "", 1).strip()
            if text:
                turns.append(text)

    if not turns and isinstance(query, str) and query.strip():
        turns = [query]
    return turns


def conflict_k(ds: DSMassFunction, mass1: Dict[str, float], mass2: Dict[str, float]) -> float:
    conflict = 0.0
    for a, ma in mass1.items():
        for b, mb in mass2.items():
            hcd = ds.find_highest_common_descendant(a, b)
            if hcd is None:
                conflict += ma * mb
    return conflict


def combine_rule(
    ds: DSMassFunction,
    mass1: Dict[str, float],
    mass2: Dict[str, float],
    rule: str,
) -> Tuple[Dict[str, float], float]:
    combined = {}
    k = 0.0

    for a, ma in mass1.items():
        for b, mb in mass2.items():
            contrib = ma * mb
            hcd = ds.find_highest_common_descendant(a, b)
            if hcd is None:
                k += contrib
                if rule == "yager":
                    combined["Uncertainty"] = combined.get("Uncertainty", 0.0) + contrib
            else:
                combined[hcd] = combined.get(hcd, 0.0) + contrib

    if rule == "dempster" and k < 1.0:
        norm = 1.0 - k
        for key in list(combined.keys()):
            combined[key] = combined[key] / norm

    return combined, k


def cluster_name(true_intent: str) -> str:
    for name, intents in CLUSTERS.items():
        if true_intent in intents:
            return name
    return "other"


def safe_corr(x: pd.Series, y: pd.Series) -> float:
    """Compute correlation safely; return NaN for degenerate inputs."""
    if x is None or y is None:
        return float("nan")
    if len(x) < 2 or len(y) < 2:
        return float("nan")
    if x.nunique(dropna=True) < 2 or y.nunique(dropna=True) < 2:
        return float("nan")
    return float(x.corr(y))


def replay_and_measure(df: pd.DataFrame, ds: DSMassFunction, rule: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    mass_cache: Dict[str, Dict[str, float]] = {}
    query_rows = []
    turn_rows = []

    for idx, row in df.iterrows():
        turns = extract_user_turns(row.get("interaction", ""), row.get("query", ""))
        if not turns:
            continue

        ds.conversation_history = []
        combined = None
        k_values: List[float] = []

        for turn_idx, text in enumerate(turns, start=1):
            if text not in mass_cache:
                mass_cache[text] = ds.compute_mass_function(text)
            current_mass = mass_cache[text]

            if combined is None:
                combined = current_mass
                k_val = 0.0
            else:
                combined, k_val = combine_rule(ds, combined, current_mass, rule)
            k_values.append(k_val)

            q, opts, pred, conf = ds.get_clarification_step(combined)
            turn_rows.append(
                {
                    "row_id": idx,
                    "turn": turn_idx,
                    "text": text,
                    "K": k_val,
                    "ds_pred": pred if pred is not None else "",
                    "ds_conf": float(conf),
                    "ds_asked_clarification": int(pred is None),
                    "num_options": len(opts) if isinstance(opts, list) else 0,
                    "rule": rule,
                }
            )

        interaction = row.get("interaction", "")
        clar_cnt = interaction.count("Could you clarify") if isinstance(interaction, str) else 0
        query_rows.append(
            {
                "row_id": idx,
                "query": row.get("query", ""),
                "true_intent": row.get("true_intent", ""),
                "predicted_intent": row.get("predicted_intent", ""),
                "unknown": int(row.get("predicted_intent", "") == "unknown"),
                "clarification_depth": int(clar_cnt),
                "n_user_turns": len(turns),
                "mean_K": float(pd.Series(k_values).mean()),
                "max_K": float(pd.Series(k_values).max()),
                "last_K": float(k_values[-1]),
                "sum_K": float(pd.Series(k_values).sum()),
                "cluster": cluster_name(row.get("true_intent", "")),
                "participant_id": row.get("participant_id", ""),
                "source_file": row.get("source_file", ""),
                "session_dataset": row.get("dataset", ""),
                "rule": rule,
            }
        )

    query_df = pd.DataFrame(query_rows)
    turn_df = pd.DataFrame(turn_rows)
    return query_df, turn_df


def summarize(
    query_df: pd.DataFrame,
    k_high_quantile: float,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if query_df.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    q = query_df.copy()
    positive_k = q.loc[q["max_K"] > 0, "max_K"]
    if not positive_k.empty:
        k_cut = float(positive_k.quantile(k_high_quantile))
        q["high_K"] = ((q["max_K"] >= k_cut) & (q["max_K"] > 0)).astype(int)
    else:
        k_cut = 0.0
        q["high_K"] = 0

    overall = pd.DataFrame(
        [
            {
                "rule": q["rule"].iloc[0],
                "n_queries": len(q),
                "corr_meanK_vs_clar_depth": safe_corr(q["mean_K"], q["clarification_depth"]),
                "corr_maxK_vs_clar_depth": safe_corr(q["max_K"], q["clarification_depth"]),
                "corr_meanK_vs_unknown": safe_corr(q["mean_K"], q["unknown"]),
                "corr_maxK_vs_unknown": safe_corr(q["max_K"], q["unknown"]),
                "highK_unknown_rate": q.loc[q["high_K"] == 1, "unknown"].mean(),
                "lowK_unknown_rate": q.loc[q["high_K"] == 0, "unknown"].mean(),
                "highK_avg_clar_depth": q.loc[q["high_K"] == 1, "clarification_depth"].mean(),
                "lowK_avg_clar_depth": q.loc[q["high_K"] == 0, "clarification_depth"].mean(),
                "k_high_quantile": k_high_quantile,
                "k_high_cutoff": k_cut,
            }
        ]
    )

    cluster = (
        q.groupby(["rule", "cluster"], as_index=False)
        .agg(
            n_queries=("row_id", "count"),
            unknown_rate=("unknown", "mean"),
            avg_clar_depth=("clarification_depth", "mean"),
            avg_mean_K=("mean_K", "mean"),
            avg_max_K=("max_K", "mean"),
        )
        .sort_values(["rule", "cluster"])
    )

    # Per-intent/leaf view for rigorous reporting.
    intent_rows = []
    for intent, sub in q.groupby("true_intent"):
        intent_rows.append(
            {
                "rule": sub["rule"].iloc[0],
                "true_intent": intent,
                "n_queries": len(sub),
                "unknown_rate": float(sub["unknown"].mean()),
                "avg_clar_depth": float(sub["clarification_depth"].mean()),
                "avg_mean_K": float(sub["mean_K"].mean()),
                "avg_max_K": float(sub["max_K"].mean()),
                "corr_meanK_vs_clar_depth": safe_corr(sub["mean_K"], sub["clarification_depth"]),
                "corr_maxK_vs_clar_depth": safe_corr(sub["max_K"], sub["clarification_depth"]),
                "corr_meanK_vs_unknown": safe_corr(sub["mean_K"], sub["unknown"]),
                "corr_maxK_vs_unknown": safe_corr(sub["max_K"], sub["unknown"]),
                "highK_unknown_rate": float(sub.loc[sub["high_K"] == 1, "unknown"].mean()) if (sub["high_K"] == 1).any() else float("nan"),
            }
        )

    intent_df = (
        pd.DataFrame(intent_rows)
        .sort_values(["rule", "n_queries", "true_intent"], ascending=[True, False, True])
        .reset_index(drop=True)
    )

    return overall, cluster, intent_df


def summarize_session_level(query_df: pd.DataFrame) -> pd.DataFrame:
    """Compute explicit session-file level correlation metrics.

    These are derived analytics and are intentionally not expected in raw session logs.
    """
    if query_df.empty or "source_file" not in query_df.columns:
        return pd.DataFrame()

    rows = []
    group_cols = ["rule", "source_file"]
    for (rule, source_file), sub in query_df.groupby(group_cols):
        rows.append(
            {
                "rule": rule,
                "source_file": source_file,
                "participant_id": str(sub["participant_id"].iloc[0]) if "participant_id" in sub.columns else "",
                "session_dataset": str(sub["session_dataset"].iloc[0]) if "session_dataset" in sub.columns else "",
                "n_queries": int(len(sub)),
                "unknown_rate": float(sub["unknown"].mean()),
                "avg_clar_depth": float(sub["clarification_depth"].mean()),
                "avg_mean_K": float(sub["mean_K"].mean()),
                "avg_max_K": float(sub["max_K"].mean()),
                "corr_meanK_vs_clar_depth": safe_corr(sub["mean_K"], sub["clarification_depth"]),
                "corr_maxK_vs_clar_depth": safe_corr(sub["max_K"], sub["clarification_depth"]),
                "corr_meanK_vs_unknown": safe_corr(sub["mean_K"], sub["unknown"]),
                "corr_maxK_vs_unknown": safe_corr(sub["max_K"], sub["unknown"]),
            }
        )

    return pd.DataFrame(rows).sort_values(["rule", "source_file"]).reset_index(drop=True)


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.sessions_dir:
        df = load_rows_from_sessions_dir(args.sessions_dir)
        if df.empty:
            raise ValueError(f"No query rows found in sessions dir: {args.sessions_dir}")
        source_note = f"sessions_dir={args.sessions_dir}"
    elif args.results_file:
        df = pd.read_csv(args.results_file)
        source_note = f"results_file={args.results_file}"
    else:
        raise ValueError("Provide either --results-file or --sessions-dir")

    if args.max_rows > 0:
        df = df.head(args.max_rows).copy()
        logger.info("Using first %d rows for analysis", len(df))

    rules = [r.strip() for r in args.rules.split(",") if r.strip()]
    logger.info("Rules for ablation: %s", rules)

    ds = load_ds_system(args.dataset)

    ablation_overall = []
    ablation_cluster = []
    ablation_intent = []
    ablation_session = []

    for rule in rules:
        logger.info("Replaying interactions with rule=%s", rule)
        query_df, turn_df = replay_and_measure(df, ds, rule)

        query_path = out_dir / f"query_level_{rule}.csv"
        turn_path = out_dir / f"turn_level_{rule}.csv"
        query_df.to_csv(query_path, index=False)
        turn_df.to_csv(turn_path, index=False)

        overall_df, cluster_df, intent_df = summarize(query_df, args.k_high_quantile)
        session_df = summarize_session_level(query_df)
        ablation_overall.append(overall_df)
        ablation_cluster.append(cluster_df)
        ablation_intent.append(intent_df)
        ablation_session.append(session_df)

        logger.info("Saved %s and %s", query_path, turn_path)

    overall = pd.concat(ablation_overall, ignore_index=True) if ablation_overall else pd.DataFrame()
    cluster = pd.concat(ablation_cluster, ignore_index=True) if ablation_cluster else pd.DataFrame()
    intent = pd.concat(ablation_intent, ignore_index=True) if ablation_intent else pd.DataFrame()
    session = pd.concat(ablation_session, ignore_index=True) if ablation_session else pd.DataFrame()

    overall_path = out_dir / "ablation_overall.csv"
    cluster_path = out_dir / "ablation_cluster.csv"
    intent_path = out_dir / "ablation_intent.csv"
    session_path = out_dir / "ablation_session.csv"
    overall.to_csv(overall_path, index=False)
    cluster.to_csv(cluster_path, index=False)
    intent.to_csv(intent_path, index=False)
    session.to_csv(session_path, index=False)

    report_lines = [
        "Zadeh-like Conflict Persistence Analysis",
        "=" * 44,
        f"Input source: {source_note}",
        f"Dataset: {args.dataset}",
        f"Rules: {', '.join(rules)}",
        "",
    ]
    if not overall.empty:
        report_lines.append("Overall Metrics")
        report_lines.append(overall.to_string(index=False))
        report_lines.append("")
    if not cluster.empty:
        report_lines.append("Cluster Metrics")
        report_lines.append(cluster.to_string(index=False))
        report_lines.append("")
    if not intent.empty:
        report_lines.append("Intent Metrics (Top 30 by n_queries)")
        report_lines.append(intent.head(30).to_string(index=False))
        report_lines.append("")
    if not session.empty:
        report_lines.append("Session Metrics")
        report_lines.append(session.to_string(index=False))

    report_path = out_dir / "zadeh_conflict_report.txt"
    report_path.write_text("\n".join(report_lines), encoding="utf-8")

    logger.info("Saved overall ablation to %s", overall_path)
    logger.info("Saved cluster ablation to %s", cluster_path)
    logger.info("Saved intent ablation to %s", intent_path)
    logger.info("Saved session ablation to %s", session_path)
    logger.info("Saved report to %s", report_path)


if __name__ == "__main__":
    main()
