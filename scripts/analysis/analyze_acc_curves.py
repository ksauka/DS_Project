"""
Script to analyze accuracy-coverage-burden curves.
Evaluates tradeoffs for selective prediction.
"""

import argparse
import logging
import sys
from pathlib import Path
import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.utils.evaluation_curves import AccuracyCoverageBurdenAnalyzer
from src.utils.file_io import load_csv, save_json, ensure_dir

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Analyze accuracy-coverage-burden curves'
    )
    parser.add_argument(
        '--results-file',
        type=str,
        required=True,
        help='Path to evaluation results CSV'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='outputs/acc_curves',
        help='Output directory for ACC analysis'
    )
    parser.add_argument(
        '--target-coverage',
        type=float,
        default=0.9,
        help='Target coverage for optimal threshold (default: 0.9)'
    )
    parser.add_argument(
        '--target-accuracy',
        type=float,
        default=0.85,
        help='Target accuracy for optimal threshold (default: 0.85)'
    )
    parser.add_argument(
        '--num-thresholds',
        type=int,
        default=21,
        help='Number of thresholds to evaluate (default: 21)'
    )
    return parser.parse_args()


def extract_interaction_count(interaction_str):
    """Extract interaction count from conversation history string."""
    if pd.isna(interaction_str) or not interaction_str:
        return 0
    return interaction_str.count("Chatbot:")


def main():
    """Main function."""
    args = parse_args()
    
    logger.info("Loading evaluation results...")
    results_df = load_csv(args.results_file)
    
    # Create output directory
    output_dir = ensure_dir(args.output_dir)
    
    # Extract data
    predictions = results_df['predicted_intent'].tolist()
    true_labels = results_df['true_intent'].tolist()
    confidences = results_df['confidence'].tolist()
    
    # Extract interaction counts if available
    interactions = None
    if 'interaction' in results_df.columns:
        interactions = results_df['interaction'].apply(extract_interaction_count).tolist()
        logger.info("Interaction data available - will include burden analysis")
    elif 'num_interactions' in results_df.columns:
        interactions = results_df['num_interactions'].tolist()
        logger.info("Interaction data available - will include burden analysis")
    else:
        logger.warning("No interaction data found - burden analysis will be skipped")
    
    # Initialize analyzer
    analyzer = AccuracyCoverageBurdenAnalyzer()
    
    # Generate curves
    logger.info(f"Generating ACC curves with {args.num_thresholds} thresholds...")
    thresholds = np.linspace(0, 1, args.num_thresholds)
    
    curves_df = analyzer.generate_acc_curves(
        predictions=predictions,
        true_labels=true_labels,
        confidences=confidences,
        interactions=interactions,
        thresholds=thresholds
    )
    
    # Save curves data
    curves_path = output_dir / 'acc_curves_data.csv'
    curves_df.to_csv(curves_path, index=False)
    logger.info(f"Saved curves data to {curves_path}")
    
    # Find optimal threshold
    logger.info("Finding optimal threshold...")
    optimal = analyzer.find_optimal_threshold(
        curves_df,
        target_coverage=args.target_coverage,
        target_accuracy=args.target_accuracy
    )
    
    # Display optimal threshold
    print("\n" + "="*70)
    print("OPTIMAL THRESHOLD ANALYSIS")
    print("="*70)
    print(f"\nOptimal Threshold: {optimal['optimal_threshold']:.3f}")
    print(f"Accuracy: {optimal['accuracy']:.1%}")
    print(f"Coverage: {optimal['coverage']:.1%}")
    print(f"Accepted Queries: {optimal['num_accepted']}")
    print(f"Average Confidence: {optimal['avg_confidence']:.3f}")
    if optimal.get('avg_burden') is not None:
        print(f"Average Burden: {optimal['avg_burden']:.2f} interactions")
    print("="*70)
    
    # Save optimal threshold
    optimal_path = output_dir / 'optimal_threshold.json'
    save_json(optimal, optimal_path)
    logger.info(f"Saved optimal threshold to {optimal_path}")
    
    # Generate plots
    logger.info("Generating visualizations...")
    analyzer.plot_acc_curves(curves_df, output_dir)
    
    # Generate report
    logger.info("Generating report...")
    report_path = output_dir / 'acc_report.txt'
    analyzer.generate_acc_report(curves_df, optimal, report_path)
    
    # Additional analysis
    print("\n" + "="*70)
    print("CURVE STATISTICS")
    print("="*70)
    print(f"\nAccuracy Range: {curves_df['accuracy'].min():.1%} - {curves_df['accuracy'].max():.1%}")
    print(f"Coverage Range: {curves_df['coverage'].min():.1%} - {curves_df['coverage'].max():.1%}")
    
    if 'avg_burden' in curves_df.columns:
        print(f"Burden Range: {curves_df['avg_burden'].min():.2f} - {curves_df['avg_burden'].max():.2f}")
        
        # Correlation analysis
        acc_burden_corr = curves_df['accuracy'].corr(curves_df['avg_burden'])
        print(f"\nAccuracy-Burden Correlation: {acc_burden_corr:.3f}")
        
        if acc_burden_corr > 0:
            print("  → Higher accuracy requires more interactions")
        else:
            print("  → Higher accuracy achieved with fewer interactions")
    
    print("="*70)
    
    logger.info("\n✓ ACC curve analysis complete!")
    print(f"\nOutputs saved to: {output_dir}")
    print("  - acc_curves_data.csv: Curves data")
    print("  - optimal_threshold.json: Optimal threshold info")
    print("  - *.png: Visualization plots")
    print("  - acc_report.txt: Detailed report")


if __name__ == '__main__':
    main()
