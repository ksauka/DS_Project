"""Dempster-Shafer mass function for hierarchical intent disambiguation."""

import logging
import random
from typing import Dict, List, Optional, Tuple, Callable
import numpy as np
from .embeddings import SentenceEmbedder
from .classifier import IntentClassifier

logger = logging.getLogger(__name__)

# Import BeliefTracker at runtime to avoid circular import issues
def _get_belief_tracker():
    """Import BeliefTracker at runtime."""
    try:
        from src.utils.explainability import BeliefTracker
        return BeliefTracker
    except ImportError:
        # Fallback - return a mock class if import fails
        logger.warning("BeliefTracker import failed, using mock")
        class MockBeliefTracker:
            def record_belief(self, *args, **kwargs): pass
            def get_history(self): return []
            def get_final_belief(self): return {}
        return MockBeliefTracker


class DSMassFunction:
    """Dempster-Shafer Theory implementation for hierarchical reasoning."""

    def __init__(
        self,
        intent_embeddings: Dict[str, np.ndarray],
        hierarchy: Dict[str, List[str]],
        classifier: IntentClassifier,
        custom_thresholds: Optional[Dict[str, float]] = None,
        customer_agent_callback: Optional[Callable] = None,
        enable_belief_tracking: bool = True,
        embedder: Optional[object] = None,
    ):
        """Initialize DS Mass Function.

        Args:
            intent_embeddings: Dictionary of intent embeddings
            hierarchy: Hierarchical structure of intents
            classifier: Trained intent classifier
            custom_thresholds: Optional custom confidence thresholds per intent
            customer_agent_callback: Optional function to simulate user responses
            enable_belief_tracking: Whether to enable belief progression tracking
            embedder: Optional pre-built SentenceEmbedder to reuse (avoids duplicate model load)
        """
        self.intent_embeddings = intent_embeddings
        # Reuse caller-supplied embedder if provided; otherwise create a fresh one.
        self.embedder = embedder if embedder is not None else SentenceEmbedder()
        self.hierarchy = hierarchy
        self.custom_thresholds = custom_thresholds or {}
        self.conversation_history = []
        self.user_response = None
        self.classifier = classifier
        self.customer_agent_callback = customer_agent_callback
        
        # Belief tracking for explainability
        self.enable_belief_tracking = enable_belief_tracking
        if enable_belief_tracking:
            BeliefTrackerClass = _get_belief_tracker()
            self.belief_tracker = BeliefTrackerClass()
        else:
            self.belief_tracker = None

    def is_leaf(self, intent: str) -> bool:
        """Check if a node is a leaf in the hierarchy.

        Args:
            intent: Intent name

        Returns:
            True if leaf node, False otherwise
        """
        return len(self.hierarchy.get(intent, [])) == 0

    def get_threshold(self, intent: str) -> float:
        """Get threshold based on the level of the intent.

        Args:
            intent: Intent name

        Returns:
            Threshold value
        """
        if self.is_leaf(intent):
            return 0.1
        elif intent in self.hierarchy:
            return 0.2
        else:
            return 0.3

    def get_confidence_threshold(self, intent: str) -> float:
        """Get confidence threshold for an intent.

        Args:
            intent: Intent name

        Returns:
            Confidence threshold
        """
        # If custom thresholds are provided, use them
        if self.custom_thresholds and intent in self.custom_thresholds:
            return self.custom_thresholds[intent]
        
        # If NO custom thresholds at all (None or empty dict), 
        # return 0.0 to skip clarifications (baseline evaluation)
        if not self.custom_thresholds:
            return 0.0
            
        # Otherwise use defaults
        if self.is_leaf(intent):
            return 0.3
        elif intent in self.hierarchy:
            return 0.5
        else:
            return 0.7

    def get_node_depth(self, node: str) -> int:
        """Compute the depth of a node in the hierarchy.

        Args:
            node: Node name

        Returns:
            Depth of the node
        """
        depth = 0
        current = node
        while current in self.hierarchy and self.hierarchy[current]:
            depth += 1
            current = self.hierarchy[current][0]
        return depth

    def find_lowest_common_ancestor(self, nodes: List[str]) -> Optional[str]:
        """Find the lowest common ancestor (LCA) of given nodes.

        Args:
            nodes: List of node names

        Returns:
            LCA node name or None
        """
        if not nodes:
            return None

        ancestors = []
        for node in nodes:
            node_ancestors = set()
            current = node
            while True:
                node_ancestors.add(current)
                parent = next(
                    (parent for parent, children in self.hierarchy.items() if current in children), None
                )
                if parent is None:
                    break
                current = parent
            ancestors.append(node_ancestors)

        common_ancestors = set.intersection(*ancestors)
        lca, min_depth = None, float("inf")
        for ancestor in common_ancestors:
            depth = self.get_node_depth(ancestor)
            if depth < min_depth:
                min_depth = depth
                lca = ancestor
        return lca

    def compute_mass_function(self, user_query: str) -> Dict[str, float]:
        """Compute normalized Dempster-Shafer mass function.

        Args:
            user_query: User query text

        Returns:
            Dictionary mapping intents to mass values
        """
        self.conversation_history.append(f"User: {user_query}")
        # NOTE: No prefix used here for consistency with training.
        # Old notebook had bug: trained WITHOUT prefix, inferred WITH prefix.
        # This implementation is consistent: no prefix in both training and inference.
        query_embedding = self.embedder.get_embedding(user_query)
        probs = self.classifier.predict_proba(
            query_embedding.reshape(1, -1)
        )[0]

        mass_function = {intent: 0.0 for intent in self.intent_embeddings.keys()}

        for intent_name, prob in zip(self.classifier.get_classes(), probs):
            if intent_name in mass_function:
                mass_function[intent_name] = prob

        total_mass = sum(mass_function.values())
        if total_mass > 0:
            mass_function = {intent: mass / total_mass for intent, mass in mass_function.items()}
        else:
            mass_function = {intent: 0.0 for intent in self.intent_embeddings.keys()}
            mass_function["Uncertainty"] = 1.0
        return mass_function

    def compute_belief(self, mass_function: Dict[str, float]) -> Dict[str, float]:
        """Compute belief values for all intents bottom-up.

        Args:
            mass_function: Mass function dictionary

        Returns:
            Dictionary of belief values
        """
        belief = {}

        def compute_node_belief(intent: str) -> float:
            if intent in belief:
                return belief[intent]

            node_belief = mass_function.get(intent, 0)
            children = self.hierarchy.get(intent, [])

            for child in children:
                node_belief += compute_node_belief(child)

            belief[intent] = node_belief
            return node_belief

        for intent in self.hierarchy:
            compute_node_belief(intent)

        return belief

    def combine_mass_functions(
        self,
        mass1: Dict[str, float],
        mass2: Dict[str, float]
    ) -> Dict[str, float]:
        """Combine two mass functions using Dempster's rule.

        Args:
            mass1: First mass function
            mass2: Second mass function

        Returns:
            Combined mass function
        """
        combined_mass = {}
        conflict = 0

        for a in mass1:
            for b in mass2:
                hcd = self.find_highest_common_descendant(a, b)
                intersection = hcd if hcd else "Uncertainty"
                contribution = mass1[a] * mass2[b]

                if intersection == "Uncertainty":
                    conflict += contribution
                else:
                    combined_mass[intersection] = (combined_mass.get(intersection, 0) + contribution)
        if conflict < 1:
            for key in combined_mass:
                combined_mass[key] /= (1 - conflict)

        return combined_mass

    def find_highest_common_descendant(
        self,
        node1: str,
        node2: str
    ) -> Optional[str]:
        """Find the highest common descendant (HCD) of two nodes.

        Args:
            node1: First node name
            node2: Second node name

        Returns:
            HCD node name or None
        """
        descendants1 = self.get_all_descendants(node1)
        descendants2 = self.get_all_descendants(node2)
        common_descendants = descendants1.intersection(descendants2)

        if not common_descendants:
            return None

        hcd, max_depth = None, -1
        for descendant in common_descendants:
            depth = self.get_node_depth(descendant)
            if depth > max_depth:
                max_depth = depth
                hcd = descendant

        return hcd

    def get_all_descendants(self, node: str) -> set:
        """Get all descendants of a node in the hierarchy.

        Args:
            node: Node name

        Returns:
            Set of descendant node names
        """
        descendants = set()
        stack = [node]

        while stack:
            current = stack.pop()
            descendants.add(current)
            children = self.hierarchy.get(current, [])
            for child in children:
                stack.append(child)

        return descendants

    def evaluate_hierarchy(self,nodes: List[str],mass_function: Dict[str, float]) -> Tuple[List[Tuple[str, float]], Dict[str, float]]:
        """Return confident nodes and belief values.

        Args:
            nodes: List of nodes to evaluate
            mass_function: Current mass function

        Returns:
            Tuple of (confident_nodes, belief_values)
        """
        belief = self.compute_belief(mass_function)
        confident_nodes = []

        for intent in nodes:
            intent_belief = belief.get(intent, 0)
            threshold = self.get_confidence_threshold(intent)
            if intent_belief >= threshold:
                confident_nodes.append((intent, intent_belief))

        return confident_nodes, belief

    def ask_clarification(
        self,
        parent_nodes: List[Tuple[str, float]],
        belief: Dict[str, float]
    ) -> List[Tuple[str, List[str]]]:
        """Generate clarification queries for ambiguous parent nodes.

        Args:
            parent_nodes: List of (parent, belief) tuples
            belief: Current belief values

        Returns:
            List of (parent, children) tuples for clarification
        """
        clarification_queries = []

        for parent, _ in parent_nodes:
            children = self.hierarchy.get(parent, [])
            if len(children) < 4:
                shuffled = list(children)
                random.shuffle(shuffled)
                clarification_queries.append((parent, shuffled))
            else:
                children_with_belief = [
                    (child, belief.get(child, 0)) for child in children
                ]
                children_with_belief.sort(key=lambda x: x[1], reverse=True)
                top_children = [child for child, _ in children_with_belief[:3]]
                random.shuffle(top_children)
                clarification_queries.append((parent, top_children))

        return clarification_queries

    def evaluate_with_clarifications(self, initial_mass: Dict[str, float], depth: int = 0, maximum_depth: int = 5) -> Optional[Tuple[str, float]]:
        """Evaluate recursively with clarification queries until a confident leaf is found.
        
        IMPORTANT: Stops immediately when ONE confident leaf node is found.
        Does NOT continue asking clarifications after finding a single confident answer.
        """
        if depth >= maximum_depth:
            return None

        def evaluate_from_leaves(current_mass: Dict[str, float], depth: int = 0, maximum_depth: int = 5) -> Optional[Tuple[str, float]]:
            if depth >= maximum_depth:
                return None

            # Find all leaf nodes
            leaf_nodes = [intent for intent in self.hierarchy if self.is_leaf(intent)]
            confident_nodes, belief = self.evaluate_hierarchy(leaf_nodes, current_mass)

            # Record belief state for this turn (enables visualisation / explainability)
            if self.belief_tracker is not None:
                turn_label = "Initial" if depth == 0 else f"Turn {depth}"
                self.belief_tracker.record_belief(belief, turn_label)

            if confident_nodes:
                # --- CASE 1: Confident leaf nodes ---
                confident_leaf_nodes = [n for n in confident_nodes if self.is_leaf(n[0])]
                if len(confident_leaf_nodes) == 1:
                    logger.info(f"Single confident leaf node found: {confident_leaf_nodes[0][0]}")
                    # CRITICAL: Return immediately - do NOT ask clarifications for a single confident answer
                    return confident_leaf_nodes[0]

                if len(confident_leaf_nodes) > 1:
                    lca = self.find_lowest_common_ancestor([i for i, _ in confident_leaf_nodes])
                    if lca:
                        logger.info(f"Multiple confident leaf nodes found. LCA: {lca}")
                        clarification_queries = self.ask_clarification([(lca, belief.get(lca, 0))], belief)
                        for parent, children in clarification_queries:
                            chatbot_question = (
                                f"It seems like you're looking for something related to {parent}. "
                                f"Could you clarify which specific thing you're interested in? "
                                f"Here are a few suggestions: ({children})"
                            )
                            self.conversation_history.append(f"Chatbot: {chatbot_question}")
                            if self.customer_agent_callback:
                                self.user_response = self.customer_agent_callback(
                                    "\n".join(self.conversation_history), chatbot_question
                                )
                            else:
                                self.user_response = input("User: ")
                            user_mass = self.compute_mass_function(self.user_response)
                            current_mass = self.combine_mass_functions(current_mass, user_mass)
                        return evaluate_from_leaves(current_mass, depth + 1, maximum_depth)
                    else:
                        top_nodes = sorted(confident_leaf_nodes, key=lambda x: x[1], reverse=True)[:3]
                        options = [i for i, _ in top_nodes]
                        chatbot_question = (
                            f"There are a few things that might match: ({options}). "
                            f"Could you clarify a bit more?"
                        )
                        self.conversation_history.append(f"Chatbot: {chatbot_question}")
                        if self.customer_agent_callback:
                            self.user_response = self.customer_agent_callback(
                                "\n".join(self.conversation_history), chatbot_question
                            )
                        else:
                            self.user_response = input("User: ")
                        user_mass = self.compute_mass_function(self.user_response)
                        current_mass = self.combine_mass_functions(current_mass, user_mass)

                # --- CASE 2: No confident leaf nodes, handle non-leaf nodes ---
                else:
                    confident_non_leaf_nodes = [(i, v) for i, v in confident_nodes if not self.is_leaf(i)]
                    if confident_non_leaf_nodes:
                        logger.info("confident non-leaf")
                        def height_from_leaves(node: str) -> int:
                            if self.is_leaf(node) or node not in self.hierarchy:
                                return 0
                            return 1 + max(height_from_leaves(c) for c in self.hierarchy[node])

                        heights = {i: height_from_leaves(i) for i, _ in confident_non_leaf_nodes}
                        min_height = min(heights.values())
                        lowest_nodes = [(i, v) for i, v in confident_non_leaf_nodes if heights[i] == min_height]

                        lca_non_leaf = self.find_lowest_common_ancestor([i for i, _ in lowest_nodes])
                        if lca_non_leaf:
                            logger.info(f"LCA among lowest non-leaf nodes: {lca_non_leaf}")
                            clarification_queries = self.ask_clarification([(lca_non_leaf, belief.get(lca_non_leaf, 0))], belief)
                            for parent, children in clarification_queries:
                                chatbot_question = (
                                    f"It seems like you're looking for something related to {parent}. "
                                    f"Could you clarify which specific thing you're interested in? "
                                    f"Here are a few suggestions: ({children})"
                                )
                                self.conversation_history.append(f"Chatbot: {chatbot_question}")
                                if self.customer_agent_callback:
                                    self.user_response = self.customer_agent_callback(
                                        "\n".join(self.conversation_history), chatbot_question
                                    )
                                else:
                                    self.user_response = input("User: ")
                                user_mass = self.compute_mass_function(self.user_response)
                                current_mass = self.combine_mass_functions(current_mass, user_mass)
                            return evaluate_from_leaves(current_mass, depth + 1, maximum_depth)
                        else:
                            options = [i for i, _ in lowest_nodes]
                            chatbot_question = (
                                f"There are several possibilities at this level: ({options}). "
                                f"Could you clarify which one fits best?"
                            )
                            self.conversation_history.append(f"Chatbot: {chatbot_question}")
                            if self.customer_agent_callback:
                                self.user_response = self.customer_agent_callback(
                                    "\n".join(self.conversation_history), chatbot_question
                                )
                            else:
                                self.user_response = input("User: ")
                            user_mass = self.compute_mass_function(self.user_response)
                            current_mass = self.combine_mass_functions(current_mass, user_mass)

            else:
                # --- CASE 3: No confident nodes ---
                if depth >= maximum_depth - 1:
                    # Max depth reached - return None to signal no confident answer found
                    logger.info(f"Max depth ({maximum_depth}) reached without finding confident answer")
                    return None
                    
                chatbot_question = (
                    "I'm not entirely sure what you're asking. "
                    "Could you rephrase your question a bit?"
                )
                self.conversation_history.append(f"Chatbot: {chatbot_question}")
                if self.customer_agent_callback:
                    self.user_response = self.customer_agent_callback(
                        "\n".join(self.conversation_history), chatbot_question
                    )
                else:
                    self.user_response = input("User: ")
                user_mass = self.compute_mass_function(self.user_response)
                current_mass = self.combine_mass_functions(current_mass, user_mass)

            # Recurse with incremented depth
            return evaluate_from_leaves(current_mass, depth + 1, maximum_depth)

        # Start evaluation
        return evaluate_from_leaves(initial_mass, depth=depth)
    
    def clear_belief_history(self):
        """Clear the belief tracking history."""
        if self.belief_tracker is not None:
            self.belief_tracker.clear_history()
    
    def get_belief_tracker(self) -> Optional[object]:
        """
        Get the belief tracker instance.
        
        Returns:
            BeliefTracker instance if tracking is enabled, None otherwise
        """
        return self.belief_tracker
    
    def get_current_belief(self) -> Optional[Dict[str, float]]:
        """Get the current/final belief state.
        
        Returns:
            Current belief dictionary, or None if not available
        """
        if self.belief_tracker is not None:
            return self.belief_tracker.get_latest_belief()
        return None
    
    def save_belief_log(self, filepath: str):
        """
        Save belief progression history to JSON file.
        
        Args:
            filepath: Path to save the belief log
        """
        if self.belief_tracker is not None:
            self.belief_tracker.save_to_json(filepath)
        else:
            logger.warning("Belief tracking is not enabled")
    
    def get_clarification_step(self, mass_function: Dict[str, float]) -> Tuple[Optional[str], List[str], Optional[str], float]:
        """Single non-blocking step that exactly mirrors the notebook's evaluate_from_leaves() logic.

        evaluate_from_leaves() in logistic_DS_B77.ipynb evaluates ONLY leaf nodes, so:
          Case 1 — confident leaf nodes exist:
            - Single → return prediction immediately
            - Multiple → find LCA → return clarification question
          Case 3 — no confident leaf nodes → return rephrase request

        Returns:
            (None, [], intent, confidence)       — single confident leaf: predict now
            (question, options_list, None, 0.0)  — ask this clarification question
        """
        belief = self.compute_belief(mass_function)
        leaf_nodes = [intent for intent in self.hierarchy if self.is_leaf(intent)]
        confident_nodes, _ = self.evaluate_hierarchy(leaf_nodes, mass_function)

        if confident_nodes:
            # --- CASE 1: Confident leaf nodes ---
            confident_leaf_nodes = [n for n in confident_nodes if self.is_leaf(n[0])]

            if len(confident_leaf_nodes) == 1:
                intent, confidence = confident_leaf_nodes[0]
                logger.info(f"get_clarification_step: single confident leaf {intent} ({confidence:.3f})")
                return None, [], intent, confidence

            if len(confident_leaf_nodes) > 1:
                lca = self.find_lowest_common_ancestor([i for i, _ in confident_leaf_nodes])
                if lca:
                    logger.info(f"get_clarification_step: {len(confident_leaf_nodes)} confident leaves, LCA={lca}")
                    clarification_queries = self.ask_clarification([(lca, belief.get(lca, 0))], belief)
                    for parent, children in clarification_queries:
                        return (
                            f"It seems like you're looking for something related to {parent}. "
                            f"Could you clarify which specific thing you're interested in?"
                        ), children, None, 0.0
                else:
                    top_nodes = sorted(confident_leaf_nodes, key=lambda x: x[1], reverse=True)[:3]
                    options = [i for i, _ in top_nodes]
                    return (
                        f"There are a few things that might match. "
                        f"Could you clarify a bit more?"
                    ), options, None, 0.0

        else:
            # --- CASE 3: No confident leaf nodes → ask to rephrase ---
            logger.info("get_clarification_step: no confident nodes, asking to rephrase")
            return (
                "I'm not entirely sure what you're asking. "
                "Could you rephrase your question a bit?"
            ), [], None, 0.0

        # Fallback (should not be reached in normal operation)
        return (
            "I'm not entirely sure what you're asking. "
            "Could you rephrase your question a bit?"
        ), [], None, 0.0

    def should_ask_clarification(self, mass_function: Dict[str, float]) -> bool:
        """
        Determine if clarification is needed based on current mass function.
        
        Args:
            mass_function: Current mass function
            
        Returns:
            True if clarification is needed, False otherwise
        """
        # Compute belief values
        belief = self.compute_belief(mass_function)
        
        # Check if any intent meets confidence threshold
        leaf_nodes = [intent for intent in self.hierarchy if self.is_leaf(intent)]
        for intent in leaf_nodes:
            intent_belief = belief.get(intent, 0)
            threshold = self.get_confidence_threshold(intent)
            if intent_belief >= threshold:
                return False  # Confident enough, no clarification needed
        
        return True  # Need clarification
    
    def generate_clarification_question(self, mass_function: Dict[str, float]) -> Tuple[str, List[str]]:
        """Generate a clarification question, exactly mirroring the notebook's evaluate_from_leaves() logic.

        evaluate_from_leaves() evaluates ONLY leaf nodes, so:
          Case 1b — multiple confident leaves → LCA-based question with child options
          Case 3  — no confident leaves       → generic rephrase request

        Args:
            mass_function: Current mass function

        Returns:
            (question_str, options_list) where options_list is empty for rephrase requests
        """
        belief = self.compute_belief(mass_function)
        leaf_nodes = [intent for intent in self.hierarchy if self.is_leaf(intent)]
        confident_nodes, _ = self.evaluate_hierarchy(leaf_nodes, mass_function)

        if confident_nodes:
            # --- CASE 1b: Multiple confident leaf nodes ---
            confident_leaf_nodes = [n for n in confident_nodes if self.is_leaf(n[0])]
            if len(confident_leaf_nodes) > 1:
                lca = self.find_lowest_common_ancestor([i for i, _ in confident_leaf_nodes])
                if lca:
                    clarification_queries = self.ask_clarification([(lca, belief.get(lca, 0))], belief)
                    for parent, children in clarification_queries:
                        return (
                            f"It seems like you're looking for something related to {parent}. "
                            f"Could you clarify which specific thing you're interested in?"
                        ), children
                top_nodes = sorted(confident_leaf_nodes, key=lambda x: x[1], reverse=True)[:3]
                options = [i for i, _ in top_nodes]
                return (
                    f"There are a few things that might match. "
                    f"Could you clarify a bit more?"
                ), options

        # --- CASE 3: No confident leaf nodes → ask to rephrase ---
        return (
            "I'm not entirely sure what you're asking. "
            "Could you rephrase your question a bit?"
        ), []
    
    def update_mass_with_clarification(self, current_mass: Dict[str, float], user_response: str) -> Dict[str, float]:
        """
        Update mass function with user clarification response.
        
        Args:
            current_mass: Current mass function
            user_response: User's clarification response
            
        Returns:
            Updated mass function
        """
        # Record conversation
        self.conversation_history.append(f"User: {user_response}")
        
        # Compute new mass from user response
        user_mass = self.compute_mass_function(user_response)
        
        # Combine using Dempster's rule
        combined_mass = self.combine_mass_functions(current_mass, user_mass)
        
        # Record updated belief for tracking
        if self.belief_tracker:
            belief = self.compute_belief(combined_mass)
            turn_label = f"Turn {len(self.conversation_history)//2 + 1}"
            self.belief_tracker.record_belief(belief, turn_label)
        
        return combined_mass
    
    def get_prediction_from_mass(self, mass_function: Dict[str, float]) -> Tuple[str, float]:
        """
        Get final prediction and confidence from mass function.
        
        Args:
            mass_function: Current mass function
            
        Returns:
            Tuple of (predicted_intent, confidence_score)
        """
        belief = self.compute_belief(mass_function)
        
        # Find highest belief leaf node
        leaf_beliefs = [(leaf, belief.get(leaf, 0)) for leaf in self.hierarchy if self.is_leaf(leaf)]
        leaf_beliefs.sort(key=lambda x: x[1], reverse=True)
        
        if leaf_beliefs:
            predicted_intent, confidence = leaf_beliefs[0]
            return predicted_intent, confidence
        else:
            return "unknown", 0.0
