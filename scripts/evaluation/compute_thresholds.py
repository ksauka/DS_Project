"""Script to compute optimal thresholds from belief values."""

import argparse
import logging
import sys
from pathlib import Path
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
            logger.warning(f"Ground truth column '{ground_truth_col}' not found")
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

    # Identify intent columns (those without 'is_correct_' prefix)
    all_columns = df.columns.tolist()
    intent_columns = [
        col for col in all_columns
        if not col.startswith('is_correct_') and
        col not in ['user_question', 'correct_intent', 'belief_values']
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
