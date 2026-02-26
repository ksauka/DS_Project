"""Query selector for identifying problematic samples for user study."""

import logging
from typing import Dict
import pandas as pd

logger = logging.getLogger(__name__)


class QuerySelector:
    """Select problematic queries for user study based on model performance."""

    def __init__(
        self,
        min_interactions: int = 2,
        max_confidence: float = 0.7,
        problematic_ratio: float = 0.5
    ):
        """Initialize query selector.

        Args:
            min_interactions: Minimum clarification turns to consider
            max_confidence: Maximum confidence for low-confidence queries
            problematic_ratio: Fraction of samples allocated to problematic queries
        """
        self.min_interactions = min_interactions
        self.max_confidence = max_confidence
        self.problematic_ratio = min(max(problematic_ratio, 0.0), 1.0)
    
    def _preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add computed columns: num_interactions, is_correct.
        
        IMPORTANT: is_correct validates predictions against GROUND TRUTH labels,
        not confidence scores. Model can be confidently wrong!
        """
        df = df.copy()
        
        # Count interactions
        if 'interaction' in df.columns:
            df['num_interactions'] = df['interaction'].apply(
                lambda x: x.count('Chatbot:') if pd.notna(x) else 0
            )
        elif 'num_interactions' not in df.columns:
            df['num_interactions'] = 0
        
        # Check correctness: Compare prediction to ground truth label
        # NOTE: High confidence ≠ correct! We validate against true_intent.
        df['is_correct'] = df['true_intent'] == df['predicted_intent']
        
        return df



    def select_by_interaction_levels(
        self,
        results_df: pd.DataFrame,
        max_samples: int
    ) -> pd.DataFrame:
        """Select queries by success and interaction level.
        
        Categories are based on GROUND TRUTH CORRECTNESS (not confidence):
        - successful_level1: CORRECT predictions (true_intent == predicted_intent) with exactly 2 turns
        - successful_level2: CORRECT predictions with >2 turns
        - problematic_level1: INCORRECT predictions with exactly 2 turns  
        - problematic_level2: INCORRECT predictions with >2 turns
        
        NOTE: Model can have high confidence but be wrong! We validate against
        dataset labels to identify actual successes vs failures.
        """
        df = self._preprocess(results_df)
        
        # Filter high-interaction only
        df = df[df['num_interactions'] >= self.min_interactions].copy()
        logger.info(f"Filtered to {len(df)} high-interaction queries")
        
        # Split by level and correctness
        categories = {
            'successful_level1': df[(df['is_correct']) & (df['num_interactions'] == 2)],
            'successful_level2': df[(df['is_correct']) & (df['num_interactions'] > 2)],
            'problematic_level1': df[(~df['is_correct']) & (df['num_interactions'] == 2)],
            'problematic_level2': df[(~df['is_correct']) & (df['num_interactions'] > 2)]
        }
        
        # Remove empty categories
        categories = {k: v for k, v in categories.items() if len(v) > 0}
        
        if not categories:
            logger.warning("No queries found!")
            return pd.DataFrame()
        
        # Log availability
        for cat, cat_df in categories.items():
            logger.info(f"{cat}: {len(cat_df)} available")
        
        # Balanced sampling
        per_cat = max_samples // len(categories)
        remainder = max_samples % len(categories)
        
        selected = []
        for idx, (cat, cat_df) in enumerate(categories.items()):
            n = per_cat + (1 if idx < remainder else 0)
            n = min(n, len(cat_df))
            sample = cat_df.sample(n=n, random_state=42).copy()
            sample['selection_category'] = cat
            selected.append(sample)
            logger.info(f"{cat}: selected {n}")
        
        result = pd.concat(selected, ignore_index=True) if selected else pd.DataFrame()
        logger.info(f"Total: {len(result)} queries")
        return result

    def select_worst_queries(
        self,
        results_df: pd.DataFrame,
        max_samples: int
    ) -> pd.DataFrame:
        """Select most problematic queries."""
        df = self._preprocess(results_df)
        
        # Prioritize: high interaction + incorrect + low confidence
        problematic = df[
            (df['num_interactions'] >= self.min_interactions) &
            (~df['is_correct'])
        ]
        
        if len(problematic) == 0:
            # Fallback to incorrect only
            problematic = df[~df['is_correct']]
        
        if len(problematic) == 0:
            # Fallback to low confidence
            problematic = df[df['confidence'] <= self.max_confidence]
        
        # Sort by interactions (desc) then confidence (asc)
        problematic = problematic.sort_values(
            by=['num_interactions', 'confidence'],
            ascending=[False, True]
        )
        
        selected = problematic.head(max_samples).copy()
        selected['selection_category'] = 'worst'
        logger.info(f"Selected {len(selected)} worst queries")
        return selected

    def select_high_interaction(
        self,
        results_df: pd.DataFrame,
        max_samples: int
    ) -> pd.DataFrame:
        """Mix of problematic and successful high-interaction queries."""
        df = self._preprocess(results_df)
        
        # Filter high-interaction
        df = df[df['num_interactions'] >= self.min_interactions]
        
        problematic = df[~df['is_correct']].sort_values(
            by=['num_interactions', 'confidence'],
            ascending=[False, True]
        )
        successful = df[df['is_correct']].sort_values(
            by=['num_interactions', 'confidence'],
            ascending=[False, False]
        )
        
        # Allocate samples
        n_prob = int(round(max_samples * self.problematic_ratio))
        n_succ = max_samples - n_prob
        
        prob_sample = problematic.head(n_prob).copy()
        succ_sample = successful.head(n_succ).copy()
        
        # Handle shortfalls
        if len(prob_sample) < n_prob and len(successful) > n_succ:
            extra = successful.iloc[n_succ:n_succ + (n_prob - len(prob_sample))]
            succ_sample = pd.concat([succ_sample, extra], ignore_index=True)
        
        if len(succ_sample) < n_succ and len(problematic) > n_prob:
            extra = problematic.iloc[n_prob:n_prob + (n_succ - len(succ_sample))]
            prob_sample = pd.concat([prob_sample, extra], ignore_index=True)
        
        # Label and combine
        if len(prob_sample) > 0:
            prob_sample['selection_category'] = 'problematic'
        if len(succ_sample) > 0:
            succ_sample['selection_category'] = 'successful'
        
        selected = pd.concat(
            [df for df in [prob_sample, succ_sample] if len(df) > 0],
            ignore_index=True
        )
        
        logger.info(f"Selected {len(selected)} high-interaction queries")
        return selected

    def select_for_user_study(
        self,
        results_df: pd.DataFrame,
        max_samples: int = 100,
        strategy: str = 'interaction_levels'
    ) -> pd.DataFrame:
        """Select specific queries for user study.

        Args:
            results_df: DataFrame with evaluation results
            max_samples: Maximum number of queries to select
            strategy: Selection strategy:
                - 'interaction_levels': Split by success and level (default)
                - 'worst': Most problematic queries
                - 'high_interaction': Mix of problematic and successful

        Returns:
            DataFrame with selected queries
        """
        if strategy == 'interaction_levels':
            selected = self.select_by_interaction_levels(results_df, max_samples)
        elif strategy == 'worst':
            selected = self.select_worst_queries(results_df, max_samples)
        elif strategy == 'high_interaction':
            selected = self.select_high_interaction(results_df, max_samples)
        else:
            raise ValueError(
                f"Unknown strategy '{strategy}'. "
                f"Valid options: 'interaction_levels', 'worst', 'high_interaction'"
            )

        logger.info(f"Selected {len(selected)} queries using '{strategy}' strategy")
        return selected

    def generate_study_summary(self, selected_df: pd.DataFrame) -> Dict:
        """Generate summary statistics."""
        if selected_df.empty:
            return {
                'total_selected': 0,
                'avg_interactions': 0.0,
                'avg_confidence': 0.0,
                'accuracy': 0.0,
                'categories': {}
            }

        return {
            'total_selected': len(selected_df),
            'avg_interactions': float(selected_df['num_interactions'].mean()),
            'avg_confidence': float(selected_df['confidence'].mean()),
            'accuracy': float(selected_df['is_correct'].mean()),
            'categories': selected_df['selection_category'].value_counts().to_dict(),
            'intent_distribution': (
                selected_df['true_intent'].value_counts().head(10).to_dict()
                if 'true_intent' in selected_df.columns else {}
            )
        }

