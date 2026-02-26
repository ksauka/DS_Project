"""Script to compare LLM simulation vs real human interaction results."""

import argparse
import logging
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.utils.file_io import load_csv, save_json, ensure_dir
from src.utils.explainability import BeliefVisualizer, BeliefTracker

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Compare LLM vs Human interaction results'
    )
    parser.add_argument(
        '--user-study-results',
        type=str,
        required=True,
        help='Path to user study results CSV'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='outputs/user_study/comparison',
        help='Output directory for comparison results'
    )
    parser.add_argument(
        '--belief-logs-dir',
        type=str,
        help='Directory containing belief progression logs (optional)'
    )
    parser.add_argument(
        '--compare-belief-progression',
        action='store_true',
        help='Compare belief progressions between LLM and Human'
    )
    return parser.parse_args()


def compute_comparison_metrics(df: pd.DataFrame) -> dict:
    """Compute comparison metrics between LLM and human.

    Args:
        df: DataFrame with both LLM and human results

    Returns:
        Dictionary with comparison metrics
    """
    metrics = {}

    # Accuracy comparison
    metrics['human_accuracy'] = df['is_correct'].mean()
    
    # Interaction count comparison
    metrics['human_avg_interactions'] = df['human_interaction_count'].mean()
    metrics['llm_avg_interactions'] = df['llm_interaction_count'].mean()
    metrics['interaction_reduction'] = (
        (metrics['llm_avg_interactions'] - metrics['human_avg_interactions']) /
        metrics['llm_avg_interactions'] * 100
    )

    # Confidence comparison
    metrics['human_avg_confidence'] = df['confidence'].mean()

    # Agreement rate (when predictions match between human session and LLM session)
    metrics['prediction_agreement_rate'] = (
        df['predicted_intent'] == df['llm_prediction']
    ).mean()

    # Cases where human got right but LLM was wrong
    df['human_correct_llm_wrong'] = (
        df['is_correct'] & 
        (df['predicted_intent'] != df['llm_prediction'])
    )
    metrics['human_improvement_rate'] = df['human_correct_llm_wrong'].mean()

    # Statistical tests
    from scipy import stats
    
    # Test if interaction counts differ significantly
    t_stat, p_value = stats.ttest_rel(
        df['human_interaction_count'],
        df['llm_interaction_count']
    )
    metrics['interaction_count_ttest'] = {
        't_statistic': float(t_stat),
        'p_value': float(p_value),
        'significant': p_value < 0.05
    }

    return metrics


def create_comparison_plots(df: pd.DataFrame, output_dir: Path):
    """Create visualization plots comparing LLM vs Human.

    Args:
        df: DataFrame with results
        output_dir: Output directory
    """
    sns.set_style("whitegrid")
    
    # Plot 1: Interaction count comparison
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Histogram of interactions
    axes[0].hist(
        df['llm_interaction_count'],
        bins=10,
        alpha=0.5,
        label='LLM',
        color='blue'
    )
    axes[0].hist(
        df['human_interaction_count'],
        bins=10,
        alpha=0.5,
        label='Human',
        color='green'
    )
    axes[0].set_xlabel('Number of Interactions')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('Interaction Count Distribution')
    axes[0].legend()

    # Scatter plot: LLM vs Human interactions
    axes[1].scatter(
        df['llm_interaction_count'],
        df['human_interaction_count'],
        alpha=0.6
    )
    axes[1].plot([0, df['llm_interaction_count'].max()],
                 [0, df['llm_interaction_count'].max()],
                 'r--', label='Equal')
    axes[1].set_xlabel('LLM Interactions')
    axes[1].set_ylabel('Human Interactions')
    axes[1].set_title('LLM vs Human Interaction Count')
    axes[1].legend()

    # Box plot comparison
    interaction_data = pd.DataFrame({
        'LLM': df['llm_interaction_count'],
        'Human': df['human_interaction_count']
    })
    axes[2].boxplot([interaction_data['LLM'], interaction_data['Human']],
                    labels=['LLM', 'Human'])
    axes[2].set_ylabel('Number of Interactions')
    axes[2].set_title('Interaction Count Comparison')

    plt.tight_layout()
    plt.savefig(output_dir / 'interaction_comparison.png', dpi=300)
    logger.info(f"Saved interaction comparison plot")
    plt.close()

    # Plot 2: Accuracy by interaction count
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Group by interaction bins
    df['human_interaction_bin'] = pd.cut(
        df['human_interaction_count'],
        bins=[0, 1, 2, 3, 10],
        labels=['0-1', '2', '3', '4+']
    )
    
    accuracy_by_interactions = df.groupby('human_interaction_bin')['is_correct'].agg(['mean', 'count'])
    
    ax.bar(range(len(accuracy_by_interactions)), accuracy_by_interactions['mean'])
    ax.set_xticks(range(len(accuracy_by_interactions)))
    ax.set_xticklabels(accuracy_by_interactions.index)
    ax.set_xlabel('Number of Human Interactions')
    ax.set_ylabel('Accuracy')
    ax.set_title('Accuracy vs Interaction Count (Human)')
    ax.set_ylim([0, 1])
    
    # Add count labels on bars
    for i, (idx, row) in enumerate(accuracy_by_interactions.iterrows()):
        ax.text(i, row['mean'] + 0.02, f"n={row['count']}", ha='center')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'accuracy_by_interactions.png', dpi=300)
    logger.info(f"Saved accuracy by interactions plot")
    plt.close()

    # Plot 3: Confidence distribution
    fig, ax = plt.subplots(figsize=(10, 6))
    
    correct = df[df['is_correct']]['confidence']
    incorrect = df[~df['is_correct']]['confidence']
    
    ax.hist(correct, bins=20, alpha=0.5, label='Correct', color='green')
    ax.hist(incorrect, bins=20, alpha=0.5, label='Incorrect', color='red')
    ax.set_xlabel('Confidence')
    ax.set_ylabel('Frequency')
    ax.set_title('Confidence Distribution (Human Study)')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(output_dir / 'confidence_distribution.png', dpi=300)
    logger.info(f"Saved confidence distribution plot")
    plt.close()


def compare_belief_progressions(
    belief_logs_dir: Path,
    user_study_df: pd.DataFrame,
    output_dir: Path
):
    """Compare belief progressions between LLM and Human interactions.

    Args:
        belief_logs_dir: Directory with belief log JSON files
        user_study_df: DataFrame with user study results
        output_dir: Output directory for comparison plots
    """
    logger.info("Comparing belief progressions...")
    
    # Create subdirectory for belief comparisons
    belief_comparison_dir = ensure_dir(output_dir / 'belief_comparisons')
    
    # Look for matching LLM and Human belief logs
    llm_logs_dir = Path(belief_logs_dir) / 'llm' if (Path(belief_logs_dir) / 'llm').exists() else Path(belief_logs_dir)
    human_logs_dir = Path(belief_logs_dir) / 'human' if (Path(belief_logs_dir) / 'human').exists() else Path(belief_logs_dir)
    
    comparisons_made = 0
    
    for idx, row in user_study_df.iterrows():
        query_index = row['query_index']
        user_id = row['user_id']
        
        # Try to find matching belief logs
        # Human log pattern: {user_id}_query_{idx+1}_belief.json
        human_log_pattern = f"{user_id}_query_{query_index+1}_belief.json"
        human_log_path = human_logs_dir / human_log_pattern
        
        # LLM log pattern: query_{idx+1}_belief_log.json  
        llm_log_pattern = f"query_{query_index+1}_belief_log.json"
        llm_log_path = llm_logs_dir / llm_log_pattern
        
        if human_log_path.exists() and llm_log_path.exists():
            # Load belief histories
            llm_tracker = BeliefTracker()
            llm_tracker.load_from_json(str(llm_log_path))
            llm_history = llm_tracker.get_history()
            
            human_tracker = BeliefTracker()
            human_tracker.load_from_json(str(human_log_path))
            human_history = human_tracker.get_history()
            
            # Create comparison plot
            belief_histories = [
                (llm_history, "LLM Simulation"),
                (human_history, f"Human ({user_id})")
            ]
            
            plot_path = belief_comparison_dir / f"query_{query_index+1}_comparison.png"
            BeliefVisualizer.compare_belief_progressions(
                belief_histories,
                title=f"Query {query_index+1}: LLM vs Human Belief Progression",
                save_path=str(plot_path)
            )
            
            comparisons_made += 1
            logger.info(f"Created comparison for query {query_index+1}")
    
    logger.info(f"Created {comparisons_made} belief progression comparisons")


def main():
    """Main function."""
    args = parse_args()

    logger.info(f"Loading user study results from {args.user_study_results}")
    df = load_csv(args.user_study_results)

    # Create output directory
    output_dir = ensure_dir(args.output_dir)

    # Compute metrics
    logger.info("Computing comparison metrics...")
    metrics = compute_comparison_metrics(df)

    # Display metrics
    print("\n" + "="*60)
    print("LLM vs HUMAN COMPARISON RESULTS")
    print("="*60)
    print(f"\nTotal queries evaluated: {len(df)}")
    print(f"\nHuman Accuracy: {metrics['human_accuracy']:.4f}")
    print(f"Human Avg Interactions: {metrics['human_avg_interactions']:.2f}")
    print(f"LLM Avg Interactions: {metrics['llm_avg_interactions']:.2f}")
    print(f"Interaction Reduction: {metrics['interaction_reduction']:.1f}%")
    print(f"\nHuman Avg Confidence: {metrics['human_avg_confidence']:.4f}")
    print(f"Prediction Agreement Rate: {metrics['prediction_agreement_rate']:.4f}")
    print(f"Human Improvement Rate: {metrics['human_improvement_rate']:.4f}")
    print(f"\nInteraction Count T-Test:")
    print(f"  t-statistic: {metrics['interaction_count_ttest']['t_statistic']:.4f}")
    print(f"  p-value: {metrics['interaction_count_ttest']['p_value']:.4f}")
    print(f"  Significant: {metrics['interaction_count_ttest']['significant']}")
    print("="*60)

    # Save metrics
    metrics_path = output_dir / 'comparison_metrics.json'
    save_json(metrics, metrics_path)
    logger.info(f"Saved metrics to {metrics_path}")

    # Create plots
    logger.info("Creating comparison plots...")
    create_comparison_plots(df, output_dir)
    
    # Compare belief progressions if requested
    if args.compare_belief_progression and args.belief_logs_dir:
        compare_belief_progressions(
            belief_logs_dir=Path(args.belief_logs_dir),
            user_study_df=df,
            output_dir=output_dir
        )
    elif args.compare_belief_progression:
        logger.warning("--compare-belief-progression requires --belief-logs-dir")

    # Generate detailed report
    report_path = output_dir / 'comparison_report.txt'
    with open(report_path, 'w') as f:
        f.write("="*60 + "\n")
        f.write("LLM vs HUMAN INTERACTION COMPARISON REPORT\n")
        f.write("="*60 + "\n\n")
        
        f.write(f"Total queries: {len(df)}\n\n")
        
        f.write("KEY FINDINGS:\n")
        f.write("-"*60 + "\n")
        f.write(f"1. Human accuracy: {metrics['human_accuracy']:.1%}\n")
        f.write(f"2. Average interactions (Human): {metrics['human_avg_interactions']:.2f}\n")
        f.write(f"3. Average interactions (LLM): {metrics['llm_avg_interactions']:.2f}\n")
        f.write(f"4. Interaction reduction: {metrics['interaction_reduction']:.1f}%\n")
        if metrics['interaction_reduction'] > 0:
            f.write("   → Humans needed FEWER clarifications than LLM\n")
        else:
            f.write("   → Humans needed MORE clarifications than LLM\n")
        f.write(f"\n5. Prediction agreement: {metrics['prediction_agreement_rate']:.1%}\n")
        f.write(f"6. Human improvement over LLM: {metrics['human_improvement_rate']:.1%}\n\n")
        
        f.write("STATISTICAL SIGNIFICANCE:\n")
        f.write("-"*60 + "\n")
        f.write(f"Interaction count difference is ")
        if metrics['interaction_count_ttest']['significant']:
            f.write("STATISTICALLY SIGNIFICANT\n")
        else:
            f.write("NOT statistically significant\n")
        f.write(f"(p-value = {metrics['interaction_count_ttest']['p_value']:.4f})\n\n")
        
        f.write("IMPLICATIONS:\n")
        f.write("-"*60 + "\n")
        f.write("This comparison reveals how real human interaction differs from\n")
        f.write("LLM-simulated customer behavior, providing insights into:\n")
        f.write("- Whether LLM simulation accurately represents real users\n")
        f.write("- Model performance differences with real vs simulated users\n")
        f.write("- Areas where the system needs improvement for real deployment\n")

    logger.info(f"Saved detailed report to {report_path}")
    logger.info("\n✓ Comparison complete!")


if __name__ == '__main__':
    main()
