"""
Script to test faithfulness of explanations.
Validates that belief progressions align with predictions.
"""

import argparse
import logging
import sys
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.utils.faithfulness import FaithfulnessValidator
from src.utils.file_io import load_csv, save_json, ensure_dir

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Test faithfulness of explanations'
    )
    parser.add_argument(
        '--results-file',
        type=str,
        required=True,
        help='Path to evaluation results CSV'
    )
    parser.add_argument(
        '--belief-logs-dir',
        type=str,
        help='Directory containing belief log JSON files'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='outputs/faithfulness',
        help='Output directory for faithfulness reports'
    )
    return parser.parse_args()


def main():
    """Main function."""
    args = parse_args()
    
    logger.info("Loading evaluation results...")
    results_df = load_csv(args.results_file)
    
    # Create output directory
    output_dir = ensure_dir(args.output_dir)
    
    # Initialize validator
    validator = FaithfulnessValidator()
    
    # Run validation
    logger.info("Running faithfulness tests...")
    belief_logs_dir = Path(args.belief_logs_dir) if args.belief_logs_dir else None
    
    validation_summary = validator.validate_results(
        results_df,
        belief_logs_dir=belief_logs_dir
    )
    
    # Display summary
    print("\n" + "="*70)
    print("FAITHFULNESS VALIDATION SUMMARY")
    print("="*70)
    print(f"\nTotal Tests: {validation_summary['total_tests']}")
    print(f"Passed: {validation_summary['passed_tests']}")
    print(f"Failed: {validation_summary['failed_tests']}")
    print(f"Pass Rate: {validation_summary['pass_rate']:.1%}")
    
    # Per-test breakdown
    print("\nTest Breakdown:")
    test_types = [k for k in validation_summary.keys() if k.endswith('_pass_rate')]
    for test_key in test_types:
        test_name = test_key.replace('_pass_rate', '').replace('_', ' ').title()
        pass_rate = validation_summary[test_key]
        count = validation_summary.get(f"{test_key.replace('_pass_rate', '_count')}", 0)
        print(f"  {test_name}: {pass_rate:.1%} ({count} tests)")
    
    print("="*70)
    
    # Save results
    results_path = output_dir / 'faithfulness_summary.json'
    save_json(validation_summary, results_path)
    logger.info(f"Saved faithfulness summary to {results_path}")
    
    # Generate detailed report
    report_path = output_dir / 'faithfulness_report.txt'
    validator.generate_faithfulness_report(validation_summary, report_path)
    
    # Save detailed results as CSV
    if validation_summary['detailed_results']:
        detailed_df = pd.DataFrame(validation_summary['detailed_results'])
        detailed_path = output_dir / 'faithfulness_detailed.csv'
        detailed_df.to_csv(detailed_path, index=False)
        logger.info(f"Saved detailed results to {detailed_path}")
    
    logger.info("\n✓ Faithfulness validation complete!")


if __name__ == '__main__':
    main()
