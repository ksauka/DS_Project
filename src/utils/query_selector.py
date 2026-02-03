"""Query selector for identifying problematic samples for user study."""

import logging
from typing import List, Dict, Tuple
import pandas as pd
import numpy as np
from pathlib import Path

logger = logging.getLogger(__name__)


class QuerySelector:
    """Select problematic queries for user study based on model performance."""

    def __init__(
        self,
        min_interactions: int = 2,
        max_confidence: float = 0.7,
        include_incorrect: bool = True
    ):
        """Initialize query selector.

        Args:
            min_interactions: Minimum clarification turns to consider
            max_confidence: Maximum confidence for low-confidence queries
            include_incorrect: Whether to include incorrectly classified queries
        """
        self.min_interactions = min_interactions
        self.max_confidence = max_confidence
        self.include_incorrect = include_incorrect

    def analyze_results(
        self,
        results_df: pd.DataFrame
    ) -> Dict[str, pd.DataFrame]:
        """Analyze results and categorize problematic queries.

        Args:
            results_df: DataFrame with evaluation results containing:
                - query, true_intent, predicted_intent, confidence, interaction

        Returns:
            Dictionary with categorized query DataFrames
        """
        # Count interactions
        results_df['num_interactions'] = results_df['interaction'].apply(
            lambda x: x.count('Chatbot:') if pd.notna(x) else 0
        )

        # Check correctness
        results_df['is_correct'] = (
            results_df['true_intent'] == results_df['predicted_intent']
        )

        categorized = {}

        # Category 1: High interaction count (many clarifications)
        high_interaction = results_df[
            results_df['num_interactions'] >= self.min_interactions
        ].copy()
        categorized['high_interaction'] = high_interaction
        logger.info(
            f"Found {len(high_interaction)} queries with "
            f">= {self.min_interactions} interactions"
        )

        # Category 2: Low confidence predictions
        low_confidence = results_df[
            results_df['confidence'] <= self.max_confidence
        ].copy()
        categorized['low_confidence'] = low_confidence
        logger.info(
            f"Found {len(low_confidence)} queries with "
            f"confidence <= {self.max_confidence}"
        )

        # Category 3: Incorrect predictions
        if self.include_incorrect:
            incorrect = results_df[~results_df['is_correct']].copy()
            categorized['incorrect'] = incorrect
            logger.info(f"Found {len(incorrect)} incorrect predictions")

        # Category 4: High interaction AND incorrect
        problematic = results_df[
            (results_df['num_interactions'] >= self.min_interactions) &
            (~results_df['is_correct'])
        ].copy()
        categorized['problematic'] = problematic
        logger.info(
            f"Found {len(problematic)} queries that are both "
            f"high-interaction and incorrect"
        )

        # Category 5: Low confidence AND incorrect
        uncertain = results_df[
            (results_df['confidence'] <= self.max_confidence) &
            (~results_df['is_correct'])
        ].copy()
        categorized['uncertain'] = uncertain
        logger.info(
            f"Found {len(uncertain)} queries that are both "
            f"low-confidence and incorrect"
        )

        return categorized

    def select_for_user_study(
        self,
        results_df: pd.DataFrame,
        max_samples: int = 100,
        strategy: str = 'balanced'
    ) -> pd.DataFrame:
        """Select specific queries for user study.

        Args:
            results_df: DataFrame with evaluation results
            max_samples: Maximum number of queries to select
            strategy: Selection strategy:
                - 'balanced': Mix of all categories
                - 'worst': Most problematic queries
                - 'high_interaction': Focus on multi-turn dialogues

        Returns:
            DataFrame with selected queries
        """
        categorized = self.analyze_results(results_df)

        if strategy == 'balanced':
            selected = self._balanced_selection(categorized, max_samples)
        elif strategy == 'worst':
            selected = self._worst_queries(categorized, max_samples)
        elif strategy == 'high_interaction':
            selected = self._high_interaction_focus(categorized, max_samples)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        logger.info(f"Selected {len(selected)} queries for user study")
        return selected

    def _balanced_selection(
        self,
        categorized: Dict[str, pd.DataFrame],
        max_samples: int
    ) -> pd.DataFrame:
        """Select balanced mix from all categories."""
        samples_per_category = max_samples // len(categorized)
        selected_dfs = []

        for category, df in categorized.items():
            if len(df) > 0:
                n_samples = min(samples_per_category, len(df))
                sample = df.sample(n=n_samples, random_state=42)
                sample['selection_category'] = category
                selected_dfs.append(sample)

        selected = pd.concat(selected_dfs, ignore_index=True)
        
        # Remove duplicates, keeping first occurrence
        selected = selected.drop_duplicates(subset=['query'], keep='first')
        
        return selected.head(max_samples)

    def _worst_queries(
        self,
        categorized: Dict[str, pd.DataFrame],
        max_samples: int
    ) -> pd.DataFrame:
        """Select most problematic queries."""
        # Prioritize: incorrect + high interaction + low confidence
        if 'problematic' in categorized and len(categorized['problematic']) > 0:
            df = categorized['problematic'].copy()
        elif 'uncertain' in categorized and len(categorized['uncertain']) > 0:
            df = categorized['uncertain'].copy()
        else:
            df = categorized['incorrect'].copy()

        # Sort by: interaction count (desc), confidence (asc)
        df = df.sort_values(
            by=['num_interactions', 'confidence'],
            ascending=[False, True]
        )
        
        selected = df.head(max_samples).copy()
        selected['selection_category'] = 'worst'
        return selected

    def _high_interaction_focus(
        self,
        categorized: Dict[str, pd.DataFrame],
        max_samples: int
    ) -> pd.DataFrame:
        """Focus on queries with most interactions."""
        df = categorized['high_interaction'].copy()
        
        # Sort by interaction count
        df = df.sort_values(by='num_interactions', ascending=False)
        
        selected = df.head(max_samples).copy()
        selected['selection_category'] = 'high_interaction'
        return selected

    def generate_study_summary(
        self,
        selected_df: pd.DataFrame
    ) -> Dict:
        """Generate summary statistics for selected queries.

        Args:
            selected_df: Selected queries DataFrame

        Returns:
            Dictionary with summary statistics
        """
        summary = {
            'total_selected': len(selected_df),
            'avg_interactions': selected_df['num_interactions'].mean(),
            'avg_confidence': selected_df['confidence'].mean(),
            'accuracy': selected_df['is_correct'].mean(),
            'categories': selected_df['selection_category'].value_counts().to_dict()
        }

        # Intent distribution
        summary['intent_distribution'] = (
            selected_df['true_intent'].value_counts().head(10).to_dict()
        )

        return summary

    def export_for_user_study(
        self,
        selected_df: pd.DataFrame,
        output_path: Path,
        include_llm_interaction: bool = False
    ):
        """Export selected queries in format suitable for user study.

        Args:
            selected_df: Selected queries DataFrame
            output_path: Path to save export file
            include_llm_interaction: Whether to include LLM conversation history
        """
        # Prepare export data
        export_columns = [
            'query',
            'true_intent',
            'predicted_intent',
            'confidence',
            'num_interactions',
            'is_correct',
            'selection_category'
        ]

        if include_llm_interaction:
            export_columns.append('interaction')

        export_df = selected_df[export_columns].copy()
        
        # Add study fields
        export_df['user_id'] = ''
        export_df['user_response_1'] = ''
        export_df['user_response_2'] = ''
        export_df['user_response_3'] = ''
        export_df['final_prediction'] = ''
        export_df['human_interaction_count'] = ''
        export_df['notes'] = ''

        # Save
        output_path.parent.mkdir(parents=True, exist_ok=True)
        export_df.to_csv(output_path, index=False)
        
        logger.info(f"Exported {len(export_df)} queries to {output_path}")
