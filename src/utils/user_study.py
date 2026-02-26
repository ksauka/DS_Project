"""User study interface for real human interaction."""

import logging
from typing import Optional, Dict, List, Tuple
from pathlib import Path
import pandas as pd
from datetime import datetime

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.models.ds_mass_function import DSMassFunction
from src.utils.explainability import BeliefVisualizer

logger = logging.getLogger(__name__)


class UserStudyInterface:
    """Interface for conducting user studies with real humans."""

    def __init__(
        self,
        ds_calculator: DSMassFunction,
        study_data_path: Path,
        output_dir: Path,
        save_belief_plots: bool = True,
        save_belief_logs: bool = True
    ):
        """Initialize user study interface.

        Args:
            ds_calculator: DS Mass Function calculator
            study_data_path: Path to CSV with queries for study
            output_dir: Directory to save results
            save_belief_plots: Whether to save belief progression plots
            save_belief_logs: Whether to save belief logs as JSON
        """
        self.ds_calculator = ds_calculator
        self.study_data = pd.read_csv(study_data_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Belief tracking options
        self.save_belief_plots = save_belief_plots
        self.save_belief_logs = save_belief_logs
        
        # Create subdirectories for belief tracking
        if save_belief_plots or save_belief_logs:
            self.belief_dir = self.output_dir / 'belief_progressions'
            self.belief_dir.mkdir(exist_ok=True)
            if save_belief_plots:
                self.plots_dir = self.belief_dir / 'plots'
                self.plots_dir.mkdir(exist_ok=True)
            if save_belief_logs:
                self.logs_dir = self.belief_dir / 'logs'
                self.logs_dir.mkdir(exist_ok=True)
        
        # Results tracking
        self.results = []
        self.current_index = 0

    def run_interactive_session(
        self,
        user_id: str,
        start_index: int = 0,
        max_queries: Optional[int] = None
    ):
        """Run interactive session with a human user.

        Args:
            user_id: Identifier for the user
            start_index: Starting query index
            max_queries: Maximum number of queries to process
        """
        logger.info(f"Starting user study session for user: {user_id}")
        
        self.current_index = start_index
        end_index = min(
            len(self.study_data),
            start_index + (max_queries or len(self.study_data))
        )

        print("\n" + "="*60)
        print(f"USER STUDY SESSION - User ID: {user_id}")
        print("="*60)
        print(f"\nYou will be shown {end_index - start_index} queries.")
        print("For each query, you'll interact with the system naturally.")
        print("Type your responses as a real user would.\n")
        input("Press Enter to start...")

        for idx in range(start_index, end_index):
            self._process_query(idx, user_id)
            
            # Save after each query
            self._save_progress()

            # Ask if user wants to continue
            if idx < end_index - 1:
                cont = input("\nContinue to next query? (y/n): ").strip().lower()
                if cont != 'y':
                    logger.info("User ended session early")
                    break

        print("\n" + "="*60)
        print("SESSION COMPLETE")
        print("="*60)
        logger.info(f"Completed {len(self.results)} queries")

    def _process_query(self, idx: int, user_id: str):
        """Process a single query with human interaction.

        Args:
            idx: Query index in study data
            user_id: User identifier
        """
        query_data = self.study_data.iloc[idx]
        query = query_data['query']
        true_intent = query_data['true_intent']
        llm_prediction = query_data['predicted_intent']
        llm_interactions = query_data['num_interactions']

        print("\n" + "-"*60)
        print(f"Query {idx + 1}/{len(self.study_data)}")
        print("-"*60)
        print(f"\nInitial Query: {query}")
        print(f"\n[Hidden from user: True Intent = {true_intent}]")
        print(f"[LLM needed {llm_interactions} interactions]")
        print("-"*60)

        # Reset conversation and belief tracking
        self.ds_calculator.conversation_history = []
        self.ds_calculator.clear_belief_history()
        
        # Use human input instead of LLM
        def human_input_callback(conversation_history: str, chatbot_question: str) -> str:
            """Callback for human input."""
            print(f"\nChatbot: {chatbot_question}")
            response = input("Your response: ").strip()
            return response

        # Replace callback with human input
        original_callback = self.ds_calculator.customer_agent_callback
        self.ds_calculator.customer_agent_callback = human_input_callback

        try:
            # Run DS calculation
            initial_mass = self.ds_calculator.compute_mass_function(query)
            prediction = self.ds_calculator.evaluate_with_clarifications(
                initial_mass,
                maximum_depth=5
            )

            # Display result
            print("\n" + "="*60)
            if prediction:
                pred_intent, confidence = prediction
                print(f"System Prediction: {pred_intent}")
                print(f"Confidence: {confidence:.4f}")
            else:
                pred_intent, confidence = "unknown", 0.0
                print("System could not determine intent")
            
            is_correct = (pred_intent == true_intent)
            print(f"\n{'✓ CORRECT' if is_correct else '✗ INCORRECT'}")
            print(f"True Intent: {true_intent}")
            print("="*60)

            # Record result
            result = {
                'timestamp': datetime.now().isoformat(),
                'user_id': user_id,
                'query_index': idx,
                'query': query,
                'true_intent': true_intent,
                'predicted_intent': pred_intent,
                'confidence': confidence,
                'is_correct': is_correct,
                'human_interaction_count': len([
                    h for h in self.ds_calculator.conversation_history 
                    if 'Chatbot:' in h
                ]),
                'conversation_history': '\n'.join(
                    self.ds_calculator.conversation_history
                ),
                'llm_prediction': llm_prediction,
                'llm_interaction_count': llm_interactions
            }

            self.results.append(result)
            
            # Save belief progression visualizations
            belief_tracker = self.ds_calculator.get_belief_tracker()
            if belief_tracker and belief_tracker.get_history():
                belief_history = belief_tracker.get_history()
                
                if self.save_belief_plots:
                    plot_filename = f"{user_id}_query_{idx+1}_belief.png"
                    plot_path = self.plots_dir / plot_filename
                    BeliefVisualizer.plot_belief_progression(
                        belief_history,
                        title=f"User {user_id} - Query {idx+1}",
                        save_path=str(plot_path)
                    )
                    logger.info(f"Saved belief plot to {plot_path}")
                
                if self.save_belief_logs:
                    log_filename = f"{user_id}_query_{idx+1}_belief.json"
                    log_path = self.logs_dir / log_filename
                    belief_tracker.save_to_json(str(log_path))
                    logger.info(f"Saved belief log to {log_path}")

        finally:
            # Restore original callback
            self.ds_calculator.customer_agent_callback = original_callback

    def _save_progress(self):
        """Save current progress to file."""
        if not self.results:
            return

        results_df = pd.DataFrame(self.results)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = self.output_dir / f"user_study_results_{timestamp}.csv"
        
        results_df.to_csv(output_path, index=False)
        logger.info(f"Saved {len(results_df)} results to {output_path}")

    def get_results_summary(self) -> Dict:
        """Get summary of user study results.

        Returns:
            Dictionary with summary statistics
        """
        if not self.results:
            return {}

        results_df = pd.DataFrame(self.results)

        summary = {
            'total_queries': len(results_df),
            'accuracy': results_df['is_correct'].mean(),
            'avg_confidence': results_df['confidence'].mean(),
            'avg_human_interactions': results_df['human_interaction_count'].mean(),
            'avg_llm_interactions': results_df['llm_interaction_count'].mean(),
            'human_vs_llm_interaction_ratio': (
                results_df['human_interaction_count'].mean() /
                results_df['llm_interaction_count'].mean()
                if results_df['llm_interaction_count'].mean() > 0 else 0
            )
        }

        return summary


def run_batch_user_study(
    study_data_path: Path,
    model_path: Path,
    hierarchy_path: Path,
    intents_path: Path,
    output_dir: Path,
    num_users: int = 5,
    queries_per_user: int = 20
):
    """Run batch user study with multiple users.

    Args:
        study_data_path: Path to selected queries CSV
        model_path: Path to trained model
        hierarchy_path: Path to hierarchy JSON
        intents_path: Path to intents JSON
        output_dir: Output directory
        num_users: Number of users
        queries_per_user: Queries per user
    """
    from src.models.embeddings import SentenceEmbedder, IntentEmbeddings
    from src.models.classifier import IntentClassifier
    from config.hierarchy_loader import (
        load_hierarchy_from_json,
        load_hierarchical_intents_from_json
    )

    # Load components
    hierarchy = load_hierarchy_from_json(str(hierarchy_path))
    hierarchical_intents = load_hierarchical_intents_from_json(str(intents_path))
    
    embedder = SentenceEmbedder()
    intent_embeddings = IntentEmbeddings(hierarchical_intents, embedder=embedder)
    classifier = IntentClassifier.from_pretrained(model_path)

    ds_calculator = DSMassFunction(
        intent_embeddings=intent_embeddings.get_all_embeddings(),
        hierarchy=hierarchy,
        classifier=classifier,
        customer_agent_callback=None  # Will be set per session
    )

    # Create interface
    interface = UserStudyInterface(ds_calculator, study_data_path, output_dir)

    # Run sessions for each user
    for user_num in range(num_users):
        user_id = f"user_{user_num + 1:03d}"
        start_idx = user_num * queries_per_user
        
        print(f"\n\nStarting session for {user_id}...")
        interface.run_interactive_session(
            user_id=user_id,
            start_index=start_idx,
            max_queries=queries_per_user
        )

    # Final summary
    summary = interface.get_results_summary()
    print("\n" + "="*60)
    print("STUDY SUMMARY")
    print("="*60)
    for key, value in summary.items():
        print(f"{key}: {value}")
