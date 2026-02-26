"""
Faithfulness validation for explainable intent classification.
Tests that connect prediction changes to quantified belief shifts.
"""

import logging
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
from pathlib import Path

logger = logging.getLogger(__name__)


class FaithfulnessValidator:
    """
    Validates faithfulness of explanations in DS reasoning system.
    Tests whether belief progressions align with predictions and support counterfactual analysis.
    """
    
    def __init__(self):
        """Initialize the faithfulness validator."""
        self.test_results = []
    
    def test_prediction_belief_alignment(
        self,
        prediction: str,
        final_belief: Dict[str, float],
        query_id: Optional[str] = None
    ) -> Dict:
        """
        Test: Does the predicted intent have the highest belief?
        
        Args:
            prediction: Predicted intent label
            final_belief: Final belief distribution
            query_id: Optional query identifier
            
        Returns:
            Dictionary with test results
        """
        if not final_belief:
            return {
                'query_id': query_id,
                'test': 'prediction_belief_alignment',
                'passed': False,
                'reason': 'Empty belief distribution',
                'predicted_intent': prediction,
                'predicted_belief': 0.0,
                'highest_belief_intent': None,
                'highest_belief_value': 0.0
            }
        
        # Find intent with highest belief
        max_intent = max(final_belief.items(), key=lambda x: x[1])
        highest_intent, highest_value = max_intent
        
        # Get belief in predicted intent
        predicted_belief = final_belief.get(prediction, 0.0)
        
        # Test passes if prediction matches highest belief
        passed = (prediction == highest_intent)
        
        result = {
            'query_id': query_id,
            'test': 'prediction_belief_alignment',
            'passed': passed,
            'predicted_intent': prediction,
            'predicted_belief': predicted_belief,
            'highest_belief_intent': highest_intent,
            'highest_belief_value': highest_value,
            'belief_rank': sorted(final_belief.items(), key=lambda x: x[1], reverse=True).index((prediction, predicted_belief)) + 1 if prediction in final_belief else None
        }
        
        if not passed:
            result['reason'] = f"Predicted '{prediction}' (belief={predicted_belief:.3f}) but '{highest_intent}' has highest belief ({highest_value:.3f})"
        
        return result
    
    def test_belief_monotonicity(
        self,
        belief_history: List[Tuple[Dict[str, float], str]],
        true_intent: str,
        query_id: Optional[str] = None
    ) -> Dict:
        """
        Test: Does belief in the correct intent generally increase over turns?
        
        Args:
            belief_history: List of (belief_dict, turn_label) tuples
            true_intent: Ground truth intent
            query_id: Optional query identifier
            
        Returns:
            Dictionary with test results
        """
        if not belief_history:
            return {
                'query_id': query_id,
                'test': 'belief_monotonicity',
                'passed': False,
                'reason': 'Empty belief history'
            }
        
        # Track belief values for true intent across turns
        true_intent_beliefs = []
        for belief_dict, turn_label in belief_history:
            belief_value = belief_dict.get(true_intent, 0.0)
            true_intent_beliefs.append(belief_value)
        
        # Check if trend is generally increasing
        if len(true_intent_beliefs) < 2:
            return {
                'query_id': query_id,
                'test': 'belief_monotonicity',
                'passed': True,  # Pass by default if only one turn
                'true_intent': true_intent,
                'belief_progression': true_intent_beliefs,
                'num_turns': len(true_intent_beliefs)
            }
        
        # Compute increases vs decreases
        increases = sum(1 for i in range(1, len(true_intent_beliefs)) 
                       if true_intent_beliefs[i] > true_intent_beliefs[i-1])
        decreases = sum(1 for i in range(1, len(true_intent_beliefs)) 
                       if true_intent_beliefs[i] < true_intent_beliefs[i-1])
        
        # Compute net change
        net_change = true_intent_beliefs[-1] - true_intent_beliefs[0]
        
        # Test passes if more increases than decreases, or net positive change
        passed = (increases >= decreases) or (net_change > 0)
        
        result = {
            'query_id': query_id,
            'test': 'belief_monotonicity',
            'passed': passed,
            'true_intent': true_intent,
            'belief_progression': true_intent_beliefs,
            'initial_belief': true_intent_beliefs[0],
            'final_belief': true_intent_beliefs[-1],
            'net_change': net_change,
            'num_increases': increases,
            'num_decreases': decreases,
            'num_turns': len(true_intent_beliefs)
        }
        
        if not passed:
            result['reason'] = f"Belief decreased more often ({decreases}) than increased ({increases}), net change={net_change:.3f}"
        
        return result
    
    def compute_belief_delta(
        self,
        before: Dict[str, float],
        after: Dict[str, float],
        top_k: int = 5
    ) -> Dict[str, float]:
        """
        Quantify belief changes between two turns.
        
        Args:
            before: Belief distribution before
            after: Belief distribution after
            top_k: Number of top changes to return
            
        Returns:
            Dictionary mapping intent -> belief delta
        """
        # Get all intents
        all_intents = set(before.keys()) | set(after.keys())
        
        # Compute deltas
        deltas = {}
        for intent in all_intents:
            before_val = before.get(intent, 0.0)
            after_val = after.get(intent, 0.0)
            deltas[intent] = after_val - before_val
        
        # Sort by absolute change
        sorted_deltas = sorted(deltas.items(), key=lambda x: abs(x[1]), reverse=True)
        
        return dict(sorted_deltas[:top_k])
    
    def test_uncertainty_reduction(
        self,
        belief_history: List[Tuple[Dict[str, float], str]],
        query_id: Optional[str] = None
    ) -> Dict:
        """
        Test: Does uncertainty decrease over turns?
        
        Args:
            belief_history: List of (belief_dict, turn_label) tuples
            query_id: Optional query identifier
            
        Returns:
            Dictionary with test results
        """
        if not belief_history:
            return {
                'query_id': query_id,
                'test': 'uncertainty_reduction',
                'passed': False,
                'reason': 'Empty belief history'
            }
        
        # Track uncertainty across turns
        uncertainty_values = []
        for belief_dict, turn_label in belief_history:
            uncertainty = belief_dict.get('Uncertainty', 0.0)
            uncertainty_values.append(uncertainty)
        
        if len(uncertainty_values) < 2:
            return {
                'query_id': query_id,
                'test': 'uncertainty_reduction',
                'passed': True,
                'uncertainty_progression': uncertainty_values,
                'num_turns': len(uncertainty_values)
            }
        
        # Check if uncertainty decreased
        initial_uncertainty = uncertainty_values[0]
        final_uncertainty = uncertainty_values[-1]
        net_reduction = initial_uncertainty - final_uncertainty
        
        # Test passes if uncertainty decreased or stayed low
        passed = (net_reduction >= 0) or (final_uncertainty < 0.2)
        
        result = {
            'query_id': query_id,
            'test': 'uncertainty_reduction',
            'passed': passed,
            'uncertainty_progression': uncertainty_values,
            'initial_uncertainty': initial_uncertainty,
            'final_uncertainty': final_uncertainty,
            'net_reduction': net_reduction,
            'num_turns': len(uncertainty_values)
        }
        
        if not passed:
            result['reason'] = f"Uncertainty increased from {initial_uncertainty:.3f} to {final_uncertainty:.3f}"
        
        return result
    
    def counterfactual_clarification(
        self,
        ds_calculator,
        query: str,
        initial_mass: Dict[str, float],
        alternative_responses: List[str],
        true_intent: str
    ) -> Dict:
        """
        Simulate alternative user responses and compare outcomes.
        
        Args:
            ds_calculator: DSMassFunction instance
            query: Original query
            initial_mass: Initial mass function
            alternative_responses: List of alternative responses to simulate
            true_intent: Ground truth intent
            
        Returns:
            Dictionary with counterfactual analysis
        """
        results = []
        
        for alt_response in alternative_responses:
            # Save original state
            original_history = ds_calculator.conversation_history.copy()
            if ds_calculator.belief_tracker:
                ds_calculator.belief_tracker.clear_history()
            
            # Simulate with alternative response
            try:
                # Create a modified callback that returns the alternative response
                def alt_callback(history, question):
                    return alt_response
                
                original_callback = ds_calculator.customer_agent_callback
                ds_calculator.customer_agent_callback = alt_callback
                
                # Evaluate with alternative
                prediction = ds_calculator.evaluate_with_clarifications(initial_mass)
                
                # Get results
                if prediction:
                    pred_intent, confidence = prediction
                else:
                    pred_intent, confidence = "unknown", 0.0
                
                # Get belief history
                belief_tracker = ds_calculator.get_belief_tracker()
                belief_history = belief_tracker.get_history() if belief_tracker else []
                
                results.append({
                    'alternative_response': alt_response,
                    'predicted_intent': pred_intent,
                    'confidence': confidence,
                    'is_correct': pred_intent == true_intent,
                    'num_turns': len(belief_history),
                    'final_belief': belief_history[-1][0] if belief_history else {}
                })
                
            finally:
                # Restore original state
                ds_calculator.customer_agent_callback = original_callback
                ds_calculator.conversation_history = original_history
        
        # Analyze counterfactuals
        correct_outcomes = sum(1 for r in results if r['is_correct'])
        
        return {
            'query': query,
            'true_intent': true_intent,
            'num_alternatives': len(alternative_responses),
            'counterfactual_results': results,
            'correct_outcomes': correct_outcomes,
            'outcome_stability': correct_outcomes / len(results) if results else 0.0
        }
    
    def test_option_set_consistency(
        self,
        ds_calculator,
        query: str,
        true_intent: str,
        num_trials: int = 5
    ) -> Dict:
        """
        Test: Do different clarification options lead to the correct intent?
        
        Args:
            ds_calculator: DSMassFunction instance
            query: Query to test
            true_intent: Ground truth intent
            num_trials: Number of random trials
            
        Returns:
            Dictionary with consistency test results
        """
        outcomes = []
        
        for trial in range(num_trials):
            # Reset state
            ds_calculator.conversation_history = []
            if ds_calculator.belief_tracker:
                ds_calculator.belief_tracker.clear_history()
            
            # Evaluate
            initial_mass = ds_calculator.compute_mass_function(query)
            prediction = ds_calculator.evaluate_with_clarifications(initial_mass)
            
            if prediction:
                pred_intent, confidence = prediction
            else:
                pred_intent, confidence = "unknown", 0.0
            
            outcomes.append({
                'trial': trial + 1,
                'predicted_intent': pred_intent,
                'confidence': confidence,
                'is_correct': pred_intent == true_intent
            })
        
        # Analyze consistency
        unique_predictions = set(o['predicted_intent'] for o in outcomes)
        correct_count = sum(1 for o in outcomes if o['is_correct'])
        
        return {
            'query': query,
            'true_intent': true_intent,
            'num_trials': num_trials,
            'outcomes': outcomes,
            'unique_predictions': len(unique_predictions),
            'correct_count': correct_count,
            'accuracy': correct_count / num_trials,
            'consistent': len(unique_predictions) == 1
        }
    
    def validate_results(
        self,
        results_df: pd.DataFrame,
        belief_logs_dir: Optional[Path] = None
    ) -> Dict:
        """
        Run all faithfulness tests on evaluation results.
        
        Args:
            results_df: DataFrame with evaluation results
            belief_logs_dir: Optional directory with belief log JSON files
            
        Returns:
            Dictionary with aggregated test results
        """
        all_tests = []
        
        for idx, row in results_df.iterrows():
            query_id = f"query_{idx}"
            pred_intent = row['predicted_intent']
            true_intent = row['true_intent']
            confidence = row['confidence']
            
            # Test 1: Prediction-belief alignment (if we have belief logs)
            if belief_logs_dir:
                from .explainability import BeliefTracker
                belief_log_path = belief_logs_dir / f"query_{idx+1}_belief_log.json"
                
                if belief_log_path.exists():
                    tracker = BeliefTracker()
                    tracker.load_from_json(str(belief_log_path))
                    belief_history = tracker.get_history()
                    
                    if belief_history:
                        final_belief = belief_history[-1][0]
                        
                        # Test alignment
                        alignment_result = self.test_prediction_belief_alignment(
                            pred_intent, final_belief, query_id
                        )
                        all_tests.append(alignment_result)
                        
                        # Test monotonicity
                        monotonicity_result = self.test_belief_monotonicity(
                            belief_history, true_intent, query_id
                        )
                        all_tests.append(monotonicity_result)
                        
                        # Test uncertainty reduction
                        uncertainty_result = self.test_uncertainty_reduction(
                            belief_history, query_id
                        )
                        all_tests.append(uncertainty_result)
        
        # Aggregate results by test type
        tests_df = pd.DataFrame(all_tests)
        
        summary = {
            'total_tests': len(all_tests),
            'passed_tests': sum(1 for t in all_tests if t.get('passed', False)),
            'failed_tests': sum(1 for t in all_tests if not t.get('passed', True)),
            'pass_rate': sum(1 for t in all_tests if t.get('passed', False)) / len(all_tests) if all_tests else 0.0
        }
        
        # Per-test-type summary
        if not tests_df.empty:
            for test_type in tests_df['test'].unique():
                test_subset = tests_df[tests_df['test'] == test_type]
                summary[f'{test_type}_pass_rate'] = test_subset['passed'].mean()
                summary[f'{test_type}_count'] = len(test_subset)
        
        summary['detailed_results'] = all_tests
        
        return summary
    
    def generate_faithfulness_report(
        self,
        validation_summary: Dict,
        output_path: Path
    ):
        """
        Generate a human-readable faithfulness report.
        
        Args:
            validation_summary: Summary from validate_results()
            output_path: Path to save the report
        """
        with open(output_path, 'w') as f:
            f.write("="*70 + "\n")
            f.write("FAITHFULNESS VALIDATION REPORT\n")
            f.write("="*70 + "\n\n")
            
            f.write(f"Total Tests: {validation_summary['total_tests']}\n")
            f.write(f"Passed: {validation_summary['passed_tests']}\n")
            f.write(f"Failed: {validation_summary['failed_tests']}\n")
            f.write(f"Overall Pass Rate: {validation_summary['pass_rate']:.1%}\n\n")
            
            f.write("-"*70 + "\n")
            f.write("TEST BREAKDOWN\n")
            f.write("-"*70 + "\n\n")
            
            # Per-test summaries
            test_types = [k for k in validation_summary.keys() if k.endswith('_pass_rate')]
            for test_key in test_types:
                test_name = test_key.replace('_pass_rate', '')
                pass_rate = validation_summary[test_key]
                count = validation_summary.get(f'{test_name}_count', 0)
                
                f.write(f"{test_name.replace('_', ' ').title()}:\n")
                f.write(f"  Tests: {count}\n")
                f.write(f"  Pass Rate: {pass_rate:.1%}\n\n")
            
            f.write("-"*70 + "\n")
            f.write("INTERPRETATION\n")
            f.write("-"*70 + "\n\n")
            
            overall_pass = validation_summary['pass_rate']
            
            if overall_pass >= 0.9:
                f.write("✓ EXCELLENT: Explanations are highly faithful to model behavior.\n")
            elif overall_pass >= 0.75:
                f.write("✓ GOOD: Explanations generally align with model behavior.\n")
            elif overall_pass >= 0.6:
                f.write("⚠ MODERATE: Some inconsistencies between explanations and behavior.\n")
            else:
                f.write("✗ POOR: Significant faithfulness issues detected.\n")
            
            f.write("\n" + "="*70 + "\n")
        
        logger.info(f"Faithfulness report saved to {output_path}")
