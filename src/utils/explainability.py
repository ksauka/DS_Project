"""
Explainability Module for DS Intent Classification
Tracks and visualizes belief progression during intent disambiguation
"""

import json
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np


class BeliefTracker:
    """
    Tracks belief values across multiple turns during intent disambiguation.
    Provides methods to record, save, and retrieve belief progression.
    """
    
    def __init__(self):
        """Initialize an empty belief history."""
        self.belief_history: List[Tuple[Dict[str, float], str]] = []
    
    def record_belief(self, belief_dict: Dict[str, float], turn_label: str):
        """
        Record belief values for a specific turn.
        
        Args:
            belief_dict: Dictionary mapping intent names to belief values
            turn_label: Label for this turn (e.g., "Turn 1", "After Clarification")
        """
        self.belief_history.append((belief_dict.copy(), turn_label))
    
    def clear_history(self):
        """Clear all recorded belief history."""
        self.belief_history = []
    
    def get_history(self) -> List[Tuple[Dict[str, float], str]]:
        """
        Get the complete belief history.
        
        Returns:
            List of tuples containing (belief_dict, turn_label)
        """
        return self.belief_history.copy()
    
    def get_latest_belief(self) -> Optional[Dict[str, float]]:
        """Get the most recent belief state.
        
        Returns:
            Latest belief dictionary, or None if no history
        """
        if not self.belief_history:
            return None
        return self.belief_history[-1][0]
    
    def save_to_json(self, filepath: str):
        """
        Save belief history to a JSON file.
        
        Args:
            filepath: Path to save the JSON file
        """
        history_export = [
            {"turn": label, "belief": belief_dict}
            for belief_dict, label in self.belief_history
        ]
        Path(filepath).write_text(json.dumps(history_export, indent=2))
    
    def load_from_json(self, filepath: str):
        """
        Load belief history from a JSON file.
        
        Args:
            filepath: Path to the JSON file
        """
        data = json.loads(Path(filepath).read_text())
        self.belief_history = [
            (entry["belief"], entry["turn"]) 
            for entry in data
        ]
    
    def get_final_belief(self) -> Optional[Dict[str, float]]:
        """
        Get the belief values from the last recorded turn.
        
        Returns:
            Dictionary of belief values from the final turn, or None if empty
        """
        if not self.belief_history:
            return None
        return self.belief_history[-1][0].copy()
    
    def get_belief_at_turn(self, turn_index: int) -> Optional[Dict[str, float]]:
        """
        Get belief values at a specific turn index.
        
        Args:
            turn_index: Index of the turn (0-based)
            
        Returns:
            Dictionary of belief values, or None if index is invalid
        """
        if 0 <= turn_index < len(self.belief_history):
            return self.belief_history[turn_index][0].copy()
        return None


class BeliefVisualizer:
    """
    Visualizes belief progression during intent disambiguation.
    Creates plots showing how belief values change across turns.
    """
    
    @staticmethod
    def plot_belief_progression(
        belief_history: List[Tuple[Dict[str, float], str]], 
        title: str = "Belief Progression Over Turns",
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (12, 6),
        show_uncertainty: bool = True
    ):
        """
        Create a bar plot showing belief progression across turns.
        
        Args:
            belief_history: List of tuples [(belief_dict, "Turn 1"), (belief_dict, "Turn 2"), ...]
            title: Title for the plot
            save_path: Optional path to save the figure
            figsize: Figure size as (width, height)
            show_uncertainty: Whether to highlight uncertainty in red
        """
        if not belief_history:
            print("No belief history to plot.")
            return
        
        # Collect all intents across all turns
        all_intents = set()
        for belief_dict, _ in belief_history:
            all_intents.update(belief_dict.keys())
        
        if show_uncertainty:
            all_intents.add("Uncertainty")  # Ensure uncertainty is always included
        
        # Create records for DataFrame
        records = []
        for belief_dict, label in belief_history:
            for intent in all_intents:
                belief_value = belief_dict.get(intent, 0.0)
                records.append({
                    "Intent": intent, 
                    "Belief": belief_value, 
                    "Turn": label
                })
        
        df = pd.DataFrame(records)
        
        # Create the plot
        plt.figure(figsize=figsize)
        sns.barplot(data=df, x="Intent", y="Belief", hue="Turn", palette="tab10")
        plt.title(title, fontsize=14, fontweight='bold')
        plt.ylabel("Belief Value", fontsize=12)
        plt.xlabel("Intent", fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.legend(title="Turn", bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.ylim(0, 1.0)
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    @staticmethod
    def plot_top_intents_progression(
        belief_history: List[Tuple[Dict[str, float], str]], 
        top_k: int = 5,
        title: str = "Top Intent Belief Progression",
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (10, 6)
    ):
        """
        Plot progression of top-k intents across turns using grouped bars.
        
        Args:
            belief_history: List of tuples [(belief_dict, "Turn 1"), (belief_dict, "Turn 2"), ...]
            top_k: Number of top intents to display
            title: Title for the plot
            save_path: Optional path to save the figure
            figsize: Figure size as (width, height)
        """
        if not belief_history:
            print("No belief history to plot.")
            return
        
        # Get all intents from final turn
        final_beliefs = belief_history[-1][0]
        top_intents = sorted(final_beliefs.items(), key=lambda x: x[1], reverse=True)[:top_k]
        top_intent_names = [intent for intent, _ in top_intents]

        turn_labels = [label for _, label in belief_history]
        turn_count = len(turn_labels)
        intent_count = len(top_intent_names)

        x = np.arange(turn_count)
        bar_width = 0.8 / max(intent_count, 1)

        plt.figure(figsize=figsize)
        for idx, intent in enumerate(top_intent_names):
            intent_beliefs = [belief_dict.get(intent, 0.0) for belief_dict, _ in belief_history]
            offsets = x - 0.4 + (idx + 0.5) * bar_width
            plt.bar(offsets, intent_beliefs, width=bar_width, label=intent)

        plt.title(title, fontsize=14, fontweight='bold')
        plt.ylabel("Belief Value", fontsize=12)
        plt.xlabel("Turn", fontsize=12)
        plt.xticks(x, turn_labels, rotation=45, ha='right')
        plt.legend(title="Intent", bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.ylim(0, 1.0)
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
        else:
            plt.show()

        plt.close()
    
    @staticmethod
    def compare_belief_progressions(
        belief_histories: List[Tuple[List[Tuple[Dict[str, float], str]], str]],
        title: str = "Belief Progression Comparison",
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (14, 6)
    ):
        """
        Compare belief progressions from multiple sessions (e.g., LLM vs Human).
        
        Args:
            belief_histories: List of (belief_history, label) tuples
                e.g., [(llm_history, "LLM"), (human_history, "Human")]
            title: Title for the plot
            save_path: Optional path to save the figure
            figsize: Figure size as (width, height)
        """
        if not belief_histories:
            print("No belief histories to compare.")
            return
        
        # Collect all intents across all sessions
        all_intents = set()
        for history, _ in belief_histories:
            for belief_dict, _ in history:
                all_intents.update(belief_dict.keys())
        
        # Create subplots for each session
        fig, axes = plt.subplots(1, len(belief_histories), figsize=figsize, sharey=True)
        if len(belief_histories) == 1:
            axes = [axes]
        
        for idx, (history, session_label) in enumerate(belief_histories):
            ax = axes[idx]
            
            # Create records for this session
            records = []
            for belief_dict, turn_label in history:
                for intent in all_intents:
                    belief_value = belief_dict.get(intent, 0.0)
                    records.append({
                        "Intent": intent,
                        "Belief": belief_value,
                        "Turn": turn_label
                    })
            
            df = pd.DataFrame(records)
            
            # Plot
            palette = {intent: "#ff6961" if intent == "Uncertainty" else None for intent in all_intents}
            sns.barplot(data=df, x="Intent", y="Belief", hue="Turn", ax=ax, palette=palette)
            ax.set_title(session_label, fontsize=12, fontweight='bold')
            ax.set_ylabel("Belief Value" if idx == 0 else "", fontsize=10)
            ax.set_xlabel("Intent", fontsize=10)
            ax.tick_params(axis='x', rotation=45)
            ax.set_ylim(0, 1.0)
            ax.grid(axis='y', alpha=0.3)
            
            if idx < len(belief_histories) - 1:
                ax.get_legend().remove()
            else:
                ax.legend(title="Turn", bbox_to_anchor=(1.05, 1), loc='upper left')
        
        fig.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Comparison plot saved to {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    @staticmethod
    def generate_belief_summary(
        belief_history: List[Tuple[Dict[str, float], str]]
    ) -> Dict:
        """
        Generate a statistical summary of belief progression.
        
        Args:
            belief_history: List of tuples [(belief_dict, "Turn 1"), (belief_dict, "Turn 2"), ...]
            
        Returns:
            Dictionary containing summary statistics
        """
        if not belief_history:
            return {"error": "No belief history available"}
        
        # Extract data
        initial_beliefs = belief_history[0][0]
        final_beliefs = belief_history[-1][0]
        num_turns = len(belief_history)
        
        # Find most confident intent at each turn
        top_intents_per_turn = []
        for belief_dict, turn_label in belief_history:
            if belief_dict:
                top_intent = max(belief_dict.items(), key=lambda x: x[1])
                top_intents_per_turn.append({
                    "turn": turn_label,
                    "intent": top_intent[0],
                    "belief": top_intent[1]
                })
        
        # Calculate uncertainty progression
        uncertainty_progression = [
            {
                "turn": turn_label,
                "uncertainty": belief_dict.get("Uncertainty", 0.0)
            }
            for belief_dict, turn_label in belief_history
        ]
        
        summary = {
            "num_turns": num_turns,
            "initial_top_intent": max(initial_beliefs.items(), key=lambda x: x[1])[0] if initial_beliefs else None,
            "initial_top_belief": max(initial_beliefs.items(), key=lambda x: x[1])[1] if initial_beliefs else 0,
            "final_top_intent": max(final_beliefs.items(), key=lambda x: x[1])[0] if final_beliefs else None,
            "final_top_belief": max(final_beliefs.items(), key=lambda x: x[1])[1] if final_beliefs else 0,
            "top_intents_per_turn": top_intents_per_turn,
            "uncertainty_progression": uncertainty_progression,
            "initial_uncertainty": initial_beliefs.get("Uncertainty", 0.0),
            "final_uncertainty": final_beliefs.get("Uncertainty", 0.0)
        }
        
        return summary
