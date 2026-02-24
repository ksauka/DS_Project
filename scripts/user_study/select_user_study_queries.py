"""Script to select queries for user study based on LLM simulation results."""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.utils.query_selector import QuerySelector
from src.utils.file_io import load_csv, save_csv, save_json, ensure_dir

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Select problematic queries for user study'
    )
    parser.add_argument(
        '--results-file',
        type=str,
        required=True,
        help='Path to LLM simulation results CSV'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='outputs/user_study',
        help='Output directory for selected queries'
    )
    parser.add_argument(
        '--max-samples',
        type=int,
        default=100,
        help='Maximum number of queries to select'
    )
    parser.add_argument(
        '--strategy',
        type=str,
        default='balanced',
        choices=[
            'balanced',
            'worst',
            'high_interaction',
            'high_interaction_quadrant_balanced'
        ],
        help='Selection strategy'
    )
    parser.add_argument(
        '--min-interactions',
        type=int,
        default=2,
        help='Minimum clarification turns for high-interaction category'
    )
    parser.add_argument(
        '--max-confidence',
        type=float,
        default=0.7,
        help='Maximum confidence for low-confidence category'
    )
    parser.add_argument(
        '--problematic-ratio',
        type=float,
        default=0.5,
        help='For high_interaction strategy: fraction reserved for high_interaction_problematic'
    )
    parser.add_argument(
        '--entropy-threshold',
        type=float,
        help='Optional fixed threshold for high entropy H (else median-based)'
    )
    parser.add_argument(
        '--conflict-threshold',
        type=float,
        help='Optional fixed threshold for high conflict K (else median-based)'
    )
    parser.add_argument(
        '--include-llm-history',
        action='store_true',
        help='Include LLM conversation history in export'
    )
    return parser.parse_args()


def main():
    """Main function."""
    args = parse_args()

    logger.info(f"Loading LLM simulation results from {args.results_file}")
    results_df = load_csv(args.results_file)

    # Initialize selector
    selector = QuerySelector(
        min_interactions=args.min_interactions,
        max_confidence=args.max_confidence,
        include_incorrect=True,
        high_interaction_problematic_ratio=args.problematic_ratio,
        entropy_threshold=args.entropy_threshold,
        conflict_threshold=args.conflict_threshold
    )

    # Analyze and categorize
    logger.info("Analyzing results...")
    categorized = selector.analyze_results(results_df)

    # Save categorized results
    output_dir = ensure_dir(args.output_dir)
    categories_dir = ensure_dir(output_dir / 'categories')

    for category, df in categorized.items():
        category_path = categories_dir / f'{category}.csv'
        save_csv(df, category_path, index=False)
        logger.info(f"Saved {len(df)} queries to {category_path}")

    # Select queries for user study
    logger.info(f"Selecting queries using '{args.strategy}' strategy...")
    selected_df = selector.select_for_user_study(
        results_df,
        max_samples=args.max_samples,
        strategy=args.strategy
    )

    # Generate summary
    summary = selector.generate_study_summary(selected_df)
    
    logger.info("\n" + "="*60)
    logger.info("SELECTION SUMMARY")
    logger.info("="*60)
    logger.info(f"Total selected: {summary['total_selected']}")
    logger.info(f"Avg interactions: {summary['avg_interactions']:.2f}")
    logger.info(f"Avg confidence: {summary['avg_confidence']:.4f}")
    logger.info(f"LLM Accuracy: {summary['accuracy']:.4f}")
    logger.info("\nCategory distribution:")
    for cat, count in summary['categories'].items():
        logger.info(f"  {cat}: {count}")
    logger.info("\nTop intents:")
    for intent, count in summary['intent_distribution'].items():
        logger.info(f"  {intent}: {count}")

    # Save summary
    summary_path = output_dir / 'selection_summary.json'
    save_json(summary, summary_path)

    # Export for user study
    study_file = output_dir / 'user_study_queries.csv'
    selector.export_for_user_study(
        selected_df,
        study_file,
        include_llm_interaction=args.include_llm_history
    )

    logger.info(f"\n✓ User study queries exported to {study_file}")
    logger.info(f"✓ Ready for human evaluation!")


if __name__ == '__main__':
    main()
