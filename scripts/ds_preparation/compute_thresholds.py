"""Script to compute optimal thresholds from belief values."""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Optional, Dict, Set
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.utils.file_io import load_csv, save_json, ensure_dir

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Compute optimal thresholds from belief values'
    )
    parser.add_argument(
        '--belief-file',
        type=str,
        required=True,
        help='Path to CSV file with belief values'
    )
    parser.add_argument(
        '--output-file',
        type=str,
        required=True,
        help='Path to save optimal thresholds JSON'
    )
    parser.add_argument(
        '--min-threshold',
        type=float,
        default=0.0,
        help='Minimum threshold value'
    )
    parser.add_argument(
        '--max-threshold',
        type=float,
        default=1.0,
        help='Maximum threshold value'
    )
    parser.add_argument(
        '--num-steps',
        type=int,
        default=101,
        help='Number of threshold values to test'
    )
    return parser.parse_args()


def get_ancestors(intent: str, hierarchy: Dict[str, list]) -> Set[str]:
    """Get all ancestors of an intent in the hierarchy.
    
    Args:
        intent: Intent to find ancestors for
        hierarchy: Hierarchy dictionary
        
    Returns:
        Set of ancestor intents including the intent itself
    """
    ancestors = {intent}
    for parent, children in hierarchy.items():
        if intent in children:
            ancestors.add(parent)
            ancestors.update(get_ancestors(parent, hierarchy))
    return ancestors


def add_ancestor_labels(df: pd.DataFrame, hierarchy: Dict[str, list]) -> pd.DataFrame:
    """Add is_correct_{intent} columns based on ancestor relationships.
    
    Args:
        df: DataFrame with belief values
        hierarchy: Hierarchy dictionary
        
    Returns:
        DataFrame with added is_correct columns
    """
    # Get all unique intents from hierarchy
    all_intents = set(hierarchy.keys())
    for children in hierarchy.values():
        all_intents.update(children)
    
    # For each intent, create is_correct column
    for intent in all_intents:
        if intent not in df.columns:
            continue
        
        # Check if true_intent is intent or any of its descendants
        df[f'is_correct_{intent}'] = df['true_intent'].apply(
            lambda true_intent: int(intent in get_ancestors(true_intent, hierarchy))
        )
    
    return df


def compute_optimal_thresholds(
    df: pd.DataFrame,
    intent_columns: list,
    min_threshold: float = 0.0,
    max_threshold: float = 1.0,
    num_steps: int = 101
) -> dict:
    """Compute optimal thresholds for all intents.

    Args:
        df: DataFrame with belief values and ground truth
        intent_columns: List of intent column names
        min_threshold: Minimum threshold to test
        max_threshold: Maximum threshold to test
        num_steps: Number of threshold values to test

    Returns:
        Dictionary with optimal thresholds and F1 scores
    """
    thresholds = np.linspace(min_threshold, max_threshold, num_steps)
    optimal_results = {}

    logger.info(f"Computing optimal thresholds for {len(intent_columns)} intents...")

    for intent in tqdm(intent_columns, desc="Processing intents"):
        if intent not in df.columns:
            logger.warning(f"Intent '{intent}' not found in DataFrame")
            continue

        belief_col = intent
        ground_truth_col = f"is_correct_{intent}"

        if ground_truth_col not in df.columns:
            # If hierarchy wasn't loaded, skip this intent
            logger.debug(f"Ground truth column '{ground_truth_col}' not found, skipping")
            continue

        belief_values = df[belief_col].values
        ground_truth = df[ground_truth_col].values

        best_threshold = 0.0
        best_f1 = 0.0

        for threshold in thresholds:
            predictions = (belief_values >= threshold).astype(int)
            try:
                f1 = f1_score(ground_truth, predictions, average='weighted')
                if f1 > best_f1:
                    best_f1 = f1
                    best_threshold = threshold
            except:
                continue

        optimal_results[intent] = {
            'threshold': float(best_threshold),
            'f1_score': float(best_f1)
        }

    return optimal_results


def main():
    """Main function."""
    args = parse_args()

    logger.info(f"Loading belief values from {args.belief_file}")
    df = load_csv(args.belief_file)
    
    # Load hierarchy if available
    hierarchy = None
    belief_file_path = Path(args.belief_file)
    hierarchy_file = belief_file_path.parent.parent.parent / 'config' / 'hierarchies' / f"{belief_file_path.stem.split('_')[0]}_hierarchy.json"
    
    if hierarchy_file.exists():
        logger.info(f"Loading hierarchy from {hierarchy_file}")
        import json
        with open(hierarchy_file) as f:
            hierarchy = json.load(f)
        
        # Add ancestor-aware ground truth labels
        logger.info("Adding ancestor-aware ground truth labels...")
        df = add_ancestor_labels(df, hierarchy)
    else:
        logger.warning(f"Hierarchy file not found: {hierarchy_file}")
        logger.warning("Computing thresholds without ancestor relationships")

    # Identify intent columns (those without 'is_correct_' prefix)
    all_columns = df.columns.tolist()
    intent_columns = [
        col for col in all_columns
        if not col.startswith('is_correct_') and
        col not in ['query', 'true_intent', 'belief_values']
    ]

    logger.info(f"Found {len(intent_columns)} intent columns")

    # Compute optimal thresholds
    optimal_results = compute_optimal_thresholds(
        df=df,
        intent_columns=intent_columns,
        min_threshold=args.min_threshold,
        max_threshold=args.max_threshold,
        num_steps=args.num_steps
    )

    # Save results
    output_path = Path(args.output_file)
    ensure_dir(output_path.parent)
    save_json(optimal_results, output_path)

    logger.info(f"\nSaved optimal thresholds to {output_path}")

    # Print summary statistics
    thresholds = [r['threshold'] for r in optimal_results.values()]
    f1_scores = [r['f1_score'] for r in optimal_results.values()]

    logger.info("\n" + "="*50)
    logger.info("SUMMARY STATISTICS")
    logger.info("="*50)
    logger.info(f"Number of intents: {len(optimal_results)}")
    logger.info(f"Threshold range: [{np.min(thresholds):.2f}, {np.max(thresholds):.2f}]")
    logger.info(f"Mean threshold: {np.mean(thresholds):.2f}")
    logger.info(f"Mean F1 score: {np.mean(f1_scores):.4f}")


if __name__ == '__main__':
    main()
