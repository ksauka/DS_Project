"""Compare results across all evaluation methods."""

import argparse
import logging
import sys
from pathlib import Path
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.utils.file_io import load_json, save_json, ensure_dir

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Compare results across all evaluation methods'
    )
    parser.add_argument(
        '--vanilla-metrics',
        type=str,
        required=True,
        help='Path to vanilla baseline metrics JSON'
    )
    parser.add_argument(
        '--ds-no-thresh-metrics',
        type=str,
        help='Path to DS without thresholds metrics JSON (optional)'
    )
    parser.add_argument(
        '--ds-with-thresh-metrics',
        type=str,
        help='Path to DS with thresholds metrics JSON (optional)'
    )
    parser.add_argument(
        '--llm-simulation-results',
        type=str,
        help='Path to LLM simulation results CSV (optional)'
    )
    parser.add_argument(
        '--user-study-results',
        type=str,
        help='Path to user study results CSV (optional)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='outputs/comparison',
        help='Output directory for comparison results'
    )
    return parser.parse_args()


def load_metrics_safely(path: str) -> dict:
    """Load metrics JSON safely, return empty dict if not found."""
    if not path or not Path(path).exists():
        return {}
    try:
        return load_json(path)
    except Exception as e:
        logger.warning(f"Failed to load {path}: {e}")
        return {}


def analyze_llm_simulation(csv_path: str) -> dict:
    """Analyze LLM simulation results."""
    if not csv_path or not Path(csv_path).exists():
        return {}
    
    try:
        df = pd.read_csv(csv_path)
        
        # Calculate metrics
        accuracy = (df['final_predicted_intent'] == df['true_intent']).mean()
        avg_clarifications = df['num_clarifications'].mean()
        max_clarifications = df['num_clarifications'].max()
        
        # Queries that needed clarification
        needed_clarification = (df['num_clarifications'] > 0).sum()
        
        # Improvement from clarifications
        if 'initial_predicted_intent' in df.columns:
            initial_correct = (df['initial_predicted_intent'] == df['true_intent']).sum()
            final_correct = (df['final_predicted_intent'] == df['true_intent']).sum()
            improvement = final_correct - initial_correct
        else:
            improvement = 0
        
        return {
            'accuracy': float(accuracy),
            'avg_clarifications': float(avg_clarifications),
            'max_clarifications': int(max_clarifications),
            'queries_needing_clarification': int(needed_clarification),
            'total_queries': len(df),
            'improvement_from_clarifications': int(improvement)
        }
    except Exception as e:
        logger.error(f"Error analyzing LLM simulation: {e}")
        return {}


def analyze_user_study(csv_path: str) -> dict:
    """Analyze user study results."""
    if not csv_path or not Path(csv_path).exists():
        return {}
    
    try:
        df = pd.read_csv(csv_path)
        
        # Similar analysis as LLM simulation
        accuracy = (df['final_predicted_intent'] == df['true_intent']).mean()
        avg_clarifications = df['num_clarifications'].mean()
        
        # User satisfaction if available
        satisfaction = df['user_satisfaction'].mean() if 'user_satisfaction' in df.columns else None
        
        return {
            'accuracy': float(accuracy),
            'avg_clarifications': float(avg_clarifications),
            'total_queries': len(df),
            'user_satisfaction': float(satisfaction) if satisfaction is not None else None
        }
    except Exception as e:
        logger.error(f"Error analyzing user study: {e}")
        return {}


def create_comparison_table(results: dict) -> pd.DataFrame:
    """Create comparison table from all results."""
    rows = []
    
    # Helper to safely get metric
    def get_metric(data, key, default=0):
        return data.get(key, default) if data else default
    
    # Vanilla baseline
    if 'vanilla' in results and results['vanilla']:
        v = results['vanilla']
        rows.append({
            'Method': 'Vanilla Baseline',
            'Accuracy': get_metric(v, 'accuracy'),
            'F1 (macro)': get_metric(v, 'f1_macro'),
            'F1 (weighted)': get_metric(v, 'f1_weighted'),
            'Faithfulness': get_metric(v, 'faithfulness_rate'),
            'Avg Clarifications': 0,
            'Description': 'LogisticRegression + BERT (no hierarchy)'
        })
    
    # DS without thresholds
    if 'ds_no_thresh' in results and results['ds_no_thresh']:
        rows.append({
            'Method': 'DS (No Thresholds)',
            'Accuracy': results['ds_no_thresh'].get('accuracy', 0),
            'F1 (macro)': results['ds_no_thresh'].get('f1_macro', 0),
            'F1 (weighted)': results['ds_no_thresh'].get('f1_weighted', 0),
            'Avg Clarifications': 0,
            'Description': 'Direct prediction (threshold=0.0)'
        })
    
    # DS with thresholds
    if 'ds_with_thresh' in results and results['ds_with_thresh']:
        rows.append({
            'Method': 'DS (With Thresholds)',
            'Accuracy': results['ds_with_thresh'].get('accuracy', 0),
            'F1 (macro)': results['ds_with_thresh'].get('f1_macro', 0),
            'F1 (weighted)': results['ds_with_thresh'].get('f1_weighted', 0),
            'Avg Clarifications': 0,
            'Description': 'With optimal thresholds'
        })
    
    # LLM simulation
    if 'llm_simulation' in results and results['llm_simulation']:
        rows.append({
            'Method': 'LLM Simulation',
            'Accuracy': results['llm_simulation'].get('accuracy', 0),
            'F1 (macro)': None,
            'F1 (weighted)': None,
            'Avg Clarifications': results['llm_simulation'].get('avg_clarifications', 0),
            'Description': 'Automated user simulation'
        })
    
    # User study
    if 'user_study' in results and results['user_study']:
        rows.append({
            'Method': 'User Study',
            'Accuracy': results['user_study'].get('accuracy', 0),
            'F1 (macro)': None,
            'F1 (weighted)': None,
            'Avg Clarifications': results['user_study'].get('avg_clarifications', 0),
            'Description': 'Real human interaction'
        })
    
    return pd.DataFrame(rows)


def plot_comparison(df: pd.DataFrame, output_dir: Path):
    """Create comparison visualizations."""
    sns.set_style("whitegrid")
    
    # Plot 1: Accuracy comparison
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Accuracy bar chart
    methods = df['Method'].tolist()
    accuracies = df['Accuracy'].tolist()
    
    axes[0].barh(methods, accuracies, color='steelblue')
    axes[0].set_xlabel('Accuracy', fontsize=12)
    axes[0].set_title('Accuracy Comparison Across Methods', fontsize=14, fontweight='bold')
    axes[0].set_xlim(0, 1)
    
    # Add value labels
    for i, (method, acc) in enumerate(zip(methods, accuracies)):
        axes[0].text(acc + 0.01, i, f'{acc:.4f}', va='center', fontsize=10)
    
    # Clarifications bar chart (if available)
    clarif_df = df[df['Avg Clarifications'] > 0]
    if not clarif_df.empty:
        methods_with_clarif = clarif_df['Method'].tolist()
        avg_clarif = clarif_df['Avg Clarifications'].tolist()
        
        axes[1].barh(methods_with_clarif, avg_clarif, color='coral')
        axes[1].set_xlabel('Avg Number of Clarifications', fontsize=12)
        axes[1].set_title('Average Clarifications per Query', fontsize=14, fontweight='bold')
        
        # Add value labels
        for i, (method, clarif) in enumerate(zip(methods_with_clarif, avg_clarif)):
            axes[1].text(clarif + 0.05, i, f'{clarif:.2f}', va='center', fontsize=10)
    else:
        axes[1].text(0.5, 0.5, 'No clarification data available', 
                     ha='center', va='center', transform=axes[1].transAxes)
        axes[1].set_title('Average Clarifications per Query', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plot_path = output_dir / 'comparison_plots.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    logger.info(f"Saved comparison plot to {plot_path}")
    plt.close()


def generate_summary_report(results: dict, comparison_df: pd.DataFrame, output_dir: Path):
    """Generate text summary report."""
    report_lines = []
    report_lines.append("="*70)
    report_lines.append("EVALUATION RESULTS COMPARISON")
    report_lines.append("="*70)
    report_lines.append("")
    
    # Vanilla baseline
    if 'vanilla' in results and results['vanilla']:
        report_lines.append("1. VANILLA BASELINE (LogisticRegression + BERT)")
        report_lines.append("-" * 70)
        v = results['vanilla']
        report_lines.append(f"   Accuracy:          {v.get('accuracy', 0):.4f}")
        report_lines.append(f"   F1 (macro):        {v.get('f1_macro', 0):.4f}")
        report_lines.append(f"   F1 (weighted):     {v.get('f1_weighted', 0):.4f}")
        report_lines.append(f"   Avg Confidence:    {v.get('avg_confidence', 0):.4f}")
        report_lines.append("")
    
    # DS evaluations
    if 'ds_no_thresh' in results and results['ds_no_thresh']:
        report_lines.append("2. DS WITHOUT THRESHOLDS (Direct Prediction)")
        report_lines.append("-" * 70)
        d = results['ds_no_thresh']
        report_lines.append(f"   Accuracy:          {d.get('accuracy', 0):.4f}")
        report_lines.append(f"   F1 (macro):        {d.get('f1_macro', 0):.4f}")
        report_lines.append("")
    
    if 'ds_with_thresh' in results and results['ds_with_thresh']:
        report_lines.append("3. DS WITH OPTIMAL THRESHOLDS")
        report_lines.append("-" * 70)
        d = results['ds_with_thresh']
        report_lines.append(f"   Accuracy:          {d.get('accuracy', 0):.4f}")
        report_lines.append(f"   F1 (macro):        {d.get('f1_macro', 0):.4f}")
        report_lines.append("")
    
    # LLM simulation
    if 'llm_simulation' in results and results['llm_simulation']:
        report_lines.append("4. LLM SIMULATION (Automated User Responses)")
        report_lines.append("-" * 70)
        l = results['llm_simulation']
        report_lines.append(f"   Accuracy:                    {l.get('accuracy', 0):.4f}")
        report_lines.append(f"   Avg Clarifications:          {l.get('avg_clarifications', 0):.2f}")
        report_lines.append(f"   Queries Needing Clarif:      {l.get('queries_needing_clarification', 0)}")
        report_lines.append(f"   Improvement from Clarif:     {l.get('improvement_from_clarifications', 0)}")
        report_lines.append("")
    
    # User study
    if 'user_study' in results and results['user_study']:
        report_lines.append("5. USER STUDY (Real Human Interaction)")
        report_lines.append("-" * 70)
        u = results['user_study']
        report_lines.append(f"   Accuracy:                    {u.get('accuracy', 0):.4f}")
        report_lines.append(f"   Avg Clarifications:          {u.get('avg_clarifications', 0):.2f}")
        if u.get('user_satisfaction') is not None:
            report_lines.append(f"   User Satisfaction:           {u.get('user_satisfaction', 0):.2f}/5")
        report_lines.append("")
    
    # Key findings
    report_lines.append("="*70)
    report_lines.append("KEY FINDINGS")
    report_lines.append("="*70)
    
    # Calculate improvements if possible
    if 'vanilla' in results and 'ds_with_thresh' in results:
        vanilla_acc = results['vanilla'].get('accuracy', 0)
        ds_acc = results['ds_with_thresh'].get('accuracy', 0)
        improvement = (ds_acc - vanilla_acc) / vanilla_acc * 100 if vanilla_acc > 0 else 0
        report_lines.append(f"DS Improvement over Vanilla: {improvement:+.2f}%")
    
    if 'llm_simulation' in results:
        l = results['llm_simulation']
        if l.get('improvement_from_clarifications', 0) > 0:
            report_lines.append(f"Queries improved by clarifications: {l['improvement_from_clarifications']}")
    
    report_lines.append("="*70)
    
    # Save report
    report_path = output_dir / 'comparison_summary.txt'
    with open(report_path, 'w') as f:
        f.write('\n'.join(report_lines))
    
    logger.info(f"Saved summary report to {report_path}")
    
    # Print to console
    print('\n'.join(report_lines))


def main():
    """Main comparison function."""
    args = parse_args()
    
    logger.info("Starting results comparison...")
    
    # Create output directory
    output_dir = ensure_dir(args.output_dir)
    
    # Load all metrics
    results = {
        'vanilla': load_metrics_safely(args.vanilla_metrics),
        'ds_no_thresh': load_metrics_safely(args.ds_no_thresh_metrics),
        'ds_with_thresh': load_metrics_safely(args.ds_with_thresh_metrics),
        'llm_simulation': analyze_llm_simulation(args.llm_simulation_results),
        'user_study': analyze_user_study(args.user_study_results)
    }
    
    # Create comparison table
    comparison_df = create_comparison_table(results)
    
    # Save comparison table
    table_path = output_dir / 'comparison_table.csv'
    comparison_df.to_csv(table_path, index=False)
    logger.info(f"Saved comparison table to {table_path}")
    
    # Create visualizations
    if not comparison_df.empty:
        plot_comparison(comparison_df, output_dir)
    
    # Generate summary report
    generate_summary_report(results, comparison_df, output_dir)
    
    # Save all results as JSON
    all_results_path = output_dir / 'all_results.json'
    save_json(results, all_results_path)
    logger.info(f"Saved all results to {all_results_path}")
    
    logger.info("\n✅ Comparison complete!")


if __name__ == '__main__':
    main()
