"""Analyze and visualize explainability aspects of DS system."""

import argparse
import logging
import sys
from pathlib import Path
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.utils.explainability import BeliefTracker, BeliefVisualizer
from src.utils.faithfulness import FaithfulnessValidator
from src.utils.file_io import load_json, save_json, ensure_dir

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Analyze explainability of DS system'
    )
    parser.add_argument(
        '--belief-logs-dir',
        type=str,
        required=True,
        help='Directory containing belief progression logs'
    )
    parser.add_argument(
        '--predictions-file',
        type=str,
        required=True,
        help='Path to predictions CSV file'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='outputs/explainability',
        help='Output directory for explainability analysis'
    )
    parser.add_argument(
        '--max-samples',
        type=int,
        default=50,
        help='Maximum number of samples to visualize'
    )
    return parser.parse_args()


def analyze_belief_progression(belief_logs_dir: Path, max_samples: int = 50) -> Dict:
    """Analyze belief progression patterns across queries."""
    belief_files = sorted(belief_logs_dir.glob('query_*_belief_log.json'))
    
    if not belief_files:
        logger.warning(f"No belief logs found in {belief_logs_dir}")
        return {}
    
    logger.info(f"Found {len(belief_files)} belief logs")
    
    # Statistics
    num_turns_list = []
    belief_convergence = []
    top_intent_changes = []
    
    for i, log_file in enumerate(belief_files[:max_samples]):
        try:
            tracker = BeliefTracker()
            tracker.load_from_json(str(log_file))
            history = tracker.get_history()
            
            if not history:
                continue
            
            num_turns = len(history)
            num_turns_list.append(num_turns)
            
            # Track top intent changes
            top_intents = []
            for belief, label in history:
                if belief:
                    top_intent = max(belief.items(), key=lambda x: x[1])
                    top_intents.append(top_intent[0])
            
            # Count how many times top intent changed
            changes = sum(1 for i in range(1, len(top_intents)) if top_intents[i] != top_intents[i-1])
            top_intent_changes.append(changes)
            
            # Measure convergence (belief increase in final intent)
            if len(history) > 1:
                initial_belief, _ = history[0]
                final_belief, _ = history[-1]
                if initial_belief and final_belief:
                    final_intent = max(final_belief.items(), key=lambda x: x[1])[0]
                    initial_val = initial_belief.get(final_intent, 0)
                    final_val = final_belief.get(final_intent, 0)
                    convergence = final_val - initial_val
                    belief_convergence.append(convergence)
        
        except Exception as e:
            logger.error(f"Error processing {log_file}: {e}")
            continue
    
    return {
        'total_queries': len(belief_files),
        'analyzed_queries': len(num_turns_list),
        'avg_turns': float(np.mean(num_turns_list)) if num_turns_list else 0,
        'max_turns': int(np.max(num_turns_list)) if num_turns_list else 0,
        'avg_intent_changes': float(np.mean(top_intent_changes)) if top_intent_changes else 0,
        'avg_belief_convergence': float(np.mean(belief_convergence)) if belief_convergence else 0,
        'queries_with_changes': sum(1 for c in top_intent_changes if c > 0)
    }


def validate_faithfulness(predictions_df: pd.DataFrame, belief_logs_dir: Path) -> Dict:
    """Validate faithfulness of explanations."""
    validator = FaithfulnessValidator()
    
    faithfulness_results = []
    
    for idx, row in predictions_df.iterrows():
        query_id = idx + 1
        log_file = belief_logs_dir / f"query_{query_id}_belief_log.json"
        
        if not log_file.exists():
            continue
        
        try:
            # Load belief progression
            tracker = BeliefTracker()
            tracker.load_from_json(str(log_file))
            
            final_belief = tracker.get_final_belief()
            if not final_belief:
                continue
            
            prediction = row.get('predicted_intent', row.get('final_predicted_intent'))
            
            # Test alignment
            test_result = validator.test_prediction_belief_alignment(
                prediction=prediction,
                final_belief=final_belief,
                query_id=str(query_id)
            )
            
            faithfulness_results.append(test_result)
            
        except Exception as e:
            logger.error(f"Error validating query {query_id}: {e}")
            continue
    
    if not faithfulness_results:
        return {}
    
    # Calculate statistics
    passed = sum(1 for r in faithfulness_results if r['passed'])
    total = len(faithfulness_results)
    
    return {
        'total_validated': total,
        'faithful_predictions': passed,
        'faithfulness_rate': passed / total if total > 0 else 0,
        'unfaithful_predictions': total - passed
    }


def visualize_explainability(
    belief_progression_stats: Dict,
    faithfulness_stats: Dict,
    output_dir: Path
):
    """Create explainability visualizations."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    sns.set_style("whitegrid")
    
    # Plot 1: Belief progression statistics
    if belief_progression_stats:
        stats = [
            ('Avg Turns', belief_progression_stats.get('avg_turns', 0)),
            ('Max Turns', belief_progression_stats.get('max_turns', 0)),
            ('Avg Intent\nChanges', belief_progression_stats.get('avg_intent_changes', 0))
        ]
        labels, values = zip(*stats)
        
        axes[0, 0].bar(labels, values, color=['steelblue', 'coral', 'mediumseagreen'])
        axes[0, 0].set_ylabel('Count', fontsize=11)
        axes[0, 0].set_title('Belief Progression Statistics', fontsize=13, fontweight='bold')
        
        for i, v in enumerate(values):
            axes[0, 0].text(i, v + 0.1, f'{v:.2f}', ha='center', fontsize=10)
    
    # Plot 2: Belief convergence
    if belief_progression_stats and belief_progression_stats.get('avg_belief_convergence'):
        conv = belief_progression_stats['avg_belief_convergence']
        axes[0, 1].barh(['Avg Belief\nConvergence'], [conv], color='purple')
        axes[0, 1].set_xlabel('Belief Increase', fontsize=11)
        axes[0, 1].set_title('Average Belief Convergence', fontsize=13, fontweight='bold')
        axes[0, 1].text(conv + 0.01, 0, f'{conv:.3f}', va='center', fontsize=10)
    
    # Plot 3: Faithfulness validation
    if faithfulness_stats:
        faithful = faithfulness_stats.get('faithful_predictions', 0)
        unfaithful = faithfulness_stats.get('unfaithful_predictions', 0)
        
        sizes = [faithful, unfaithful]
        labels_pie = ['Faithful', 'Unfaithful']
        colors = ['#66b266', '#ff6666']
        
        axes[1, 0].pie(sizes, labels=labels_pie, autopct='%1.1f%%', 
                       colors=colors, startangle=90)
        axes[1, 0].set_title('Prediction Faithfulness', fontsize=13, fontweight='bold')
    
    # Plot 4: Summary statistics
    if belief_progression_stats and faithfulness_stats:
        summary_text = f"""
EXPLAINABILITY SUMMARY

Belief Progression:
• Queries analyzed: {belief_progression_stats.get('analyzed_queries', 0)}
• Avg turns per query: {belief_progression_stats.get('avg_turns', 0):.2f}
• Queries with intent changes: {belief_progression_stats.get('queries_with_changes', 0)}

Faithfulness:
• Faithful predictions: {faithfulness_stats.get('faithful_predictions', 0)}
• Faithfulness rate: {faithfulness_stats.get('faithfulness_rate', 0):.2%}
• Validated queries: {faithfulness_stats.get('total_validated', 0)}
        """
        axes[1, 1].text(0.1, 0.5, summary_text, fontsize=10, 
                        verticalalignment='center', family='monospace')
        axes[1, 1].axis('off')
    
    plt.tight_layout()
    plot_path = output_dir / 'explainability_analysis.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    logger.info(f"Saved explainability plot to {plot_path}")
    plt.close()


def main():
    """Main explainability analysis function."""
    args = parse_args()
    
    logger.info("Starting explainability analysis...")
    
    # Create output directory
    output_dir = ensure_dir(args.output_dir)
    
    # Load predictions
    logger.info(f"Loading predictions from {args.predictions_file}")
    predictions_df = pd.read_csv(args.predictions_file)
    
    # Analyze belief progression
    belief_logs_dir = Path(args.belief_logs_dir)
    logger.info(f"Analyzing belief progression from {belief_logs_dir}")
    belief_stats = analyze_belief_progression(belief_logs_dir, args.max_samples)
    
    # Validate faithfulness
    logger.info("Validating prediction faithfulness...")
    faithfulness_stats = validate_faithfulness(predictions_df, belief_logs_dir)
    
    # Create visualizations
    logger.info("Creating explainability visualizations...")
    visualize_explainability(belief_stats, faithfulness_stats, output_dir)
    
    # Save analysis results
    analysis_results = {
        'belief_progression': belief_stats,
        'faithfulness': faithfulness_stats
    }
    
    results_path = output_dir / 'explainability_metrics.json'
    save_json(analysis_results, results_path)
    
    # Print summary
    logger.info("\n" + "="*60)
    logger.info("EXPLAINABILITY ANALYSIS COMPLETE")
    logger.info("="*60)
    
    if belief_stats:
        logger.info("\nBelief Progression:")
        logger.info(f"  Queries analyzed: {belief_stats.get('analyzed_queries', 0)}")
        logger.info(f"  Avg turns: {belief_stats.get('avg_turns', 0):.2f}")
        logger.info(f"  Avg intent changes: {belief_stats.get('avg_intent_changes', 0):.2f}")
        logger.info(f"  Avg belief convergence: {belief_stats.get('avg_belief_convergence', 0):.3f}")
    
    if faithfulness_stats:
        logger.info("\nFaithfulness Validation:")
        logger.info(f"  Total validated: {faithfulness_stats.get('total_validated', 0)}")
        logger.info(f"  Faithful predictions: {faithfulness_stats.get('faithful_predictions', 0)}")
        logger.info(f"  Faithfulness rate: {faithfulness_stats.get('faithfulness_rate', 0):.2%}")
    
    logger.info("\n" + "="*60)
    logger.info(f"Results saved to: {results_path}")


if __name__ == '__main__':
    main()
