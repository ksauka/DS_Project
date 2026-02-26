"""
Accuracy-Coverage-Burden (ACC) curve analysis for selective prediction.
Evaluates tradeoff between accuracy, coverage, and interaction burden at different confidence thresholds.
"""

import logging
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

logger = logging.getLogger(__name__)


class AccuracyCoverageBurdenAnalyzer:
    """
    Generates accuracy-coverage-burden curves for threshold analysis.
    Supports selective prediction evaluation and cost-benefit analysis.
    """
    
    def __init__(self):
        """Initialize the analyzer."""
        self.curves_data = None
    
    def compute_coverage(
        self,
        confidences: List[float],
        threshold: float
    ) -> float:
        """
        Compute coverage: percentage of predictions with confidence >= threshold.
        
        Args:
            confidences: List of confidence scores
            threshold: Confidence threshold
            
        Returns:
            Coverage percentage (0.0 to 1.0)
        """
        if not confidences:
            return 0.0
        
        accepted = sum(1 for conf in confidences if conf >= threshold)
        return accepted / len(confidences)
    
    def accuracy_at_threshold(
        self,
        predictions: List[str],
        true_labels: List[str],
        confidences: List[float],
        threshold: float
    ) -> Dict:
        """
        Compute accuracy for predictions above threshold.
        
        Args:
            predictions: Predicted labels
            true_labels: Ground truth labels
            confidences: Confidence scores
            threshold: Confidence threshold
            
        Returns:
            Dictionary with accuracy and coverage metrics
        """
        # Filter predictions above threshold
        accepted_preds = []
        accepted_true = []
        accepted_confs = []
        
        for pred, true, conf in zip(predictions, true_labels, confidences):
            if conf >= threshold:
                accepted_preds.append(pred)
                accepted_true.append(true)
                accepted_confs.append(conf)
        
        # Compute metrics
        if accepted_preds:
            accuracy = sum(1 for p, t in zip(accepted_preds, accepted_true) if p == t) / len(accepted_preds)
            avg_confidence = np.mean(accepted_confs)
        else:
            accuracy = 0.0
            avg_confidence = 0.0
        
        coverage = len(accepted_preds) / len(predictions) if predictions else 0.0
        
        return {
            'threshold': threshold,
            'accuracy': accuracy,
            'coverage': coverage,
            'num_accepted': len(accepted_preds),
            'num_total': len(predictions),
            'avg_confidence': avg_confidence
        }
    
    def burden_at_threshold(
        self,
        interactions: List[int],
        confidences: List[float],
        threshold: float
    ) -> Dict:
        """
        Compute interaction burden for predictions above threshold.
        
        Args:
            interactions: Number of interactions per query
            confidences: Confidence scores
            threshold: Confidence threshold
            
        Returns:
            Dictionary with burden metrics
        """
        # Filter interactions above threshold
        accepted_interactions = []
        
        for interact, conf in zip(interactions, confidences):
            if conf >= threshold:
                accepted_interactions.append(interact)
        
        if accepted_interactions:
            avg_burden = np.mean(accepted_interactions)
            total_burden = sum(accepted_interactions)
            max_burden = max(accepted_interactions)
        else:
            avg_burden = 0.0
            total_burden = 0
            max_burden = 0
        
        return {
            'threshold': threshold,
            'avg_burden': avg_burden,
            'total_burden': total_burden,
            'max_burden': max_burden,
            'num_accepted': len(accepted_interactions)
        }
    
    def generate_acc_curves(
        self,
        predictions: List[str],
        true_labels: List[str],
        confidences: List[float],
        interactions: Optional[List[int]] = None,
        thresholds: Optional[np.ndarray] = None
    ) -> pd.DataFrame:
        """
        Generate accuracy-coverage-burden curves across thresholds.
        
        Args:
            predictions: Predicted labels
            true_labels: Ground truth labels
            confidences: Confidence scores
            interactions: Optional interaction counts per query
            thresholds: Optional array of thresholds to test (default: 0 to 1 by 0.05)
            
        Returns:
            DataFrame with curves data
        """
        if thresholds is None:
            thresholds = np.arange(0.0, 1.01, 0.05)
        
        curves_data = []
        
        for threshold in thresholds:
            # Accuracy and coverage
            acc_metrics = self.accuracy_at_threshold(
                predictions, true_labels, confidences, threshold
            )
            
            row = {
                'threshold': threshold,
                'accuracy': acc_metrics['accuracy'],
                'coverage': acc_metrics['coverage'],
                'num_accepted': acc_metrics['num_accepted'],
                'avg_confidence': acc_metrics['avg_confidence']
            }
            
            # Add burden metrics if available
            if interactions is not None:
                burden_metrics = self.burden_at_threshold(
                    interactions, confidences, threshold
                )
                row['avg_burden'] = burden_metrics['avg_burden']
                row['total_burden'] = burden_metrics['total_burden']
                row['max_burden'] = burden_metrics['max_burden']
            
            curves_data.append(row)
        
        self.curves_data = pd.DataFrame(curves_data)
        return self.curves_data
    
    def find_optimal_threshold(
        self,
        curves_df: pd.DataFrame,
        target_coverage: float = 0.9,
        target_accuracy: float = 0.85
    ) -> Dict:
        """
        Find optimal threshold balancing accuracy and coverage targets.
        
        Args:
            curves_df: Curves data from generate_acc_curves()
            target_coverage: Minimum coverage target
            target_accuracy: Minimum accuracy target
            
        Returns:
            Dictionary with optimal threshold info
        """
        # Filter by coverage target
        valid = curves_df[curves_df['coverage'] >= target_coverage]
        
        if valid.empty:
            logger.warning(f"No threshold achieves {target_coverage:.0%} coverage")
            # Find highest coverage
            best = curves_df.loc[curves_df['coverage'].idxmax()]
        else:
            # Among valid, find highest accuracy
            accuracy_filtered = valid[valid['accuracy'] >= target_accuracy]
            
            if not accuracy_filtered.empty:
                # Find lowest threshold achieving both targets
                best = accuracy_filtered.loc[accuracy_filtered['threshold'].idxmin()]
            else:
                # Find highest accuracy among coverage-valid
                best = valid.loc[valid['accuracy'].idxmax()]
        
        return {
            'optimal_threshold': best['threshold'],
            'accuracy': best['accuracy'],
            'coverage': best['coverage'],
            'num_accepted': best['num_accepted'],
            'avg_confidence': best['avg_confidence'],
            'avg_burden': best.get('avg_burden', None)
        }
    
    def plot_acc_curves(
        self,
        curves_df: pd.DataFrame,
        output_dir: Path,
        title_prefix: str = ""
    ):
        """
        Generate visualizations of ACC curves.
        
        Args:
            curves_df: Curves data from generate_acc_curves()
            output_dir: Directory to save plots
            title_prefix: Optional prefix for plot titles
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Plot 1: Accuracy vs Coverage
        plt.figure(figsize=(10, 6))
        plt.plot(curves_df['coverage'], curves_df['accuracy'], 
                marker='o', linewidth=2, markersize=4, label='Accuracy')
        plt.xlabel('Coverage', fontsize=12)
        plt.ylabel('Accuracy', fontsize=12)
        plt.title(f'{title_prefix}Accuracy vs Coverage', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        
        # Add diagonal reference line
        plt.plot([0, 1], [0, 1], 'k--', alpha=0.3, label='Random')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(output_dir / 'accuracy_vs_coverage.png', dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved accuracy vs coverage plot")
        
        # Plot 2: Accuracy and Coverage vs Threshold
        fig, ax1 = plt.subplots(figsize=(10, 6))
        
        color_acc = 'tab:blue'
        ax1.set_xlabel('Confidence Threshold', fontsize=12)
        ax1.set_ylabel('Accuracy', fontsize=12, color=color_acc)
        ax1.plot(curves_df['threshold'], curves_df['accuracy'], 
                color=color_acc, marker='o', linewidth=2, markersize=4, label='Accuracy')
        ax1.tick_params(axis='y', labelcolor=color_acc)
        ax1.grid(True, alpha=0.3)
        
        ax2 = ax1.twinx()
        color_cov = 'tab:orange'
        ax2.set_ylabel('Coverage', fontsize=12, color=color_cov)
        ax2.plot(curves_df['threshold'], curves_df['coverage'], 
                color=color_cov, marker='s', linewidth=2, markersize=4, label='Coverage')
        ax2.tick_params(axis='y', labelcolor=color_cov)
        
        plt.title(f'{title_prefix}Accuracy and Coverage vs Threshold', 
                 fontsize=14, fontweight='bold')
        fig.tight_layout()
        plt.savefig(output_dir / 'acc_cov_vs_threshold.png', dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved accuracy and coverage vs threshold plot")
        
        # Plot 3: Burden analysis (if available)
        if 'avg_burden' in curves_df.columns:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
            
            # Accuracy vs Burden
            ax1.scatter(curves_df['avg_burden'], curves_df['accuracy'], 
                       c=curves_df['coverage'], cmap='viridis', s=100, alpha=0.7)
            ax1.set_xlabel('Average Interaction Burden', fontsize=12)
            ax1.set_ylabel('Accuracy', fontsize=12)
            ax1.set_title('Accuracy vs Burden (colored by coverage)', fontsize=12, fontweight='bold')
            ax1.grid(True, alpha=0.3)
            cbar1 = plt.colorbar(ax1.collections[0], ax=ax1)
            cbar1.set_label('Coverage', fontsize=10)
            
            # Burden vs Threshold
            ax2.plot(curves_df['threshold'], curves_df['avg_burden'], 
                    marker='o', linewidth=2, color='tab:red')
            ax2.set_xlabel('Confidence Threshold', fontsize=12)
            ax2.set_ylabel('Average Interaction Burden', fontsize=12)
            ax2.set_title('Interaction Burden vs Threshold', fontsize=12, fontweight='bold')
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(output_dir / 'burden_analysis.png', dpi=300, bbox_inches='tight')
            plt.close()
            logger.info(f"Saved burden analysis plot")
        
        # Plot 4: Combined ACC curve with annotations
        fig, ax = plt.subplots(figsize=(12, 7))
        
        # Create color gradient based on burden (if available)
        if 'avg_burden' in curves_df.columns:
            scatter = ax.scatter(curves_df['coverage'], curves_df['accuracy'],
                               c=curves_df['avg_burden'], cmap='RdYlGn_r', 
                               s=150, alpha=0.7, edgecolors='black', linewidth=0.5)
            cbar = plt.colorbar(scatter, ax=ax)
            cbar.set_label('Avg Interaction Burden', fontsize=11)
        else:
            ax.plot(curves_df['coverage'], curves_df['accuracy'], 
                   marker='o', linewidth=2, markersize=8, color='tab:blue')
        
        ax.set_xlabel('Coverage', fontsize=13, fontweight='bold')
        ax.set_ylabel('Accuracy', fontsize=13, fontweight='bold')
        ax.set_title(f'{title_prefix}Accuracy-Coverage-Burden Trade-off', 
                    fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 1.05)
        ax.set_ylim(0, 1.05)
        
        # Add reference lines
        ax.axhline(y=0.8, color='gray', linestyle='--', alpha=0.5, label='80% Accuracy')
        ax.axhline(y=0.9, color='gray', linestyle='--', alpha=0.5, label='90% Accuracy')
        ax.axvline(x=0.8, color='gray', linestyle='--', alpha=0.5, label='80% Coverage')
        ax.axvline(x=0.9, color='gray', linestyle='--', alpha=0.5, label='90% Coverage')
        ax.legend(loc='lower left', fontsize=9)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'acc_combined.png', dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved combined ACC plot")
    
    def generate_acc_report(
        self,
        curves_df: pd.DataFrame,
        optimal_threshold: Dict,
        output_path: Path
    ):
        """
        Generate a human-readable ACC analysis report.
        
        Args:
            curves_df: Curves data from generate_acc_curves()
            optimal_threshold: Optimal threshold from find_optimal_threshold()
            output_path: Path to save the report
        """
        with open(output_path, 'w') as f:
            f.write("="*70 + "\n")
            f.write("ACCURACY-COVERAGE-BURDEN ANALYSIS REPORT\n")
            f.write("="*70 + "\n\n")
            
            f.write("OPTIMAL THRESHOLD\n")
            f.write("-"*70 + "\n")
            f.write(f"Threshold: {optimal_threshold['optimal_threshold']:.3f}\n")
            f.write(f"Accuracy: {optimal_threshold['accuracy']:.1%}\n")
            f.write(f"Coverage: {optimal_threshold['coverage']:.1%}\n")
            f.write(f"Accepted Queries: {optimal_threshold['num_accepted']}\n")
            f.write(f"Average Confidence: {optimal_threshold['avg_confidence']:.3f}\n")
            if optimal_threshold.get('avg_burden') is not None:
                f.write(f"Average Burden: {optimal_threshold['avg_burden']:.2f} interactions\n")
            f.write("\n")
            
            f.write("THRESHOLD ANALYSIS\n")
            f.write("-"*70 + "\n\n")
            
            # High coverage scenario
            high_cov = curves_df[curves_df['coverage'] >= 0.9]
            if not high_cov.empty:
                best_high = high_cov.loc[high_cov['accuracy'].idxmax()]
                f.write("High Coverage (≥90%):\n")
                f.write(f"  Best Threshold: {best_high['threshold']:.3f}\n")
                f.write(f"  Accuracy: {best_high['accuracy']:.1%}\n")
                f.write(f"  Coverage: {best_high['coverage']:.1%}\n\n")
            
            # High accuracy scenario
            high_acc = curves_df[curves_df['accuracy'] >= 0.9]
            if not high_acc.empty:
                best_high_acc = high_acc.loc[high_acc['coverage'].idxmax()]
                f.write("High Accuracy (≥90%):\n")
                f.write(f"  Best Threshold: {best_high_acc['threshold']:.3f}\n")
                f.write(f"  Accuracy: {best_high_acc['accuracy']:.1%}\n")
                f.write(f"  Coverage: {best_high_acc['coverage']:.1%}\n\n")
            
            # Burden analysis
            if 'avg_burden' in curves_df.columns:
                f.write("BURDEN ANALYSIS\n")
                f.write("-"*70 + "\n")
                f.write(f"Overall Average Burden: {curves_df['avg_burden'].mean():.2f}\n")
                f.write(f"Min Burden (threshold={curves_df.loc[curves_df['avg_burden'].idxmin(), 'threshold']:.2f}): "
                       f"{curves_df['avg_burden'].min():.2f}\n")
                f.write(f"Max Burden (threshold={curves_df.loc[curves_df['avg_burden'].idxmax(), 'threshold']:.2f}): "
                       f"{curves_df['avg_burden'].max():.2f}\n\n")
            
            f.write("RECOMMENDATIONS\n")
            f.write("-"*70 + "\n")
            
            opt_acc = optimal_threshold['accuracy']
            opt_cov = optimal_threshold['coverage']
            
            if opt_acc >= 0.9 and opt_cov >= 0.9:
                f.write("✓ EXCELLENT: System achieves high accuracy and coverage.\n")
                f.write("  Recommended for production deployment.\n")
            elif opt_acc >= 0.85 and opt_cov >= 0.85:
                f.write("✓ GOOD: System shows strong performance.\n")
                f.write("  Consider slight threshold adjustment based on use case.\n")
            elif opt_acc >= 0.75 or opt_cov >= 0.75:
                f.write("⚠ MODERATE: Tradeoff between accuracy and coverage.\n")
                f.write("  Consider domain-specific threshold tuning.\n")
            else:
                f.write("✗ NEEDS IMPROVEMENT: Low accuracy or coverage.\n")
                f.write("  Model retraining or feature engineering recommended.\n")
            
            f.write("\n" + "="*70 + "\n")
        
        logger.info(f"ACC report saved to {output_path}")
