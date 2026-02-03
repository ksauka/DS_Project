"""Dempster-Shafer mass function for hierarchical intent disambiguation."""

import logging
from typing import Dict, List, Optional, Tuple, Callable
import numpy as np
from .embeddings import SentenceEmbedder
from .classifier import IntentClassifier
from ..utils.explainability import BeliefTracker

logger = logging.getLogger(__name__)


class DSMassFunction:
    """Dempster-Shafer Theory implementation for hierarchical reasoning."""

    def __init__(
        self,
        intent_embeddings: Dict[str, np.ndarray],
        hierarchy: Dict[str, List[str]],
        classifier: IntentClassifier,
        custom_thresholds: Optional[Dict[str, float]] = None,
        customer_agent_callback: Optional[Callable] = None,
        enable_belief_tracking: bool = True
    ):
        """Initialize DS Mass Function.

        Args:
            intent_embeddings: Dictionary of intent embeddings
            hierarchy: Hierarchical structure of intents
            classifier: Trained intent classifier
            custom_thresholds: Optional custom confidence thresholds per intent
            customer_agent_callback: Optional function to simulate user responses
            enable_belief_tracking: Whether to enable belief progression tracking
        """
        self.intent_embeddings = intent_embeddings
        self.embedder = SentenceEmbedder()
        self.hierarchy = hierarchy
        self.custom_thresholds = custom_thresholds or {}
        self.conversation_history = []
        self.user_response = None
        self.classifier = classifier
        self.customer_agent_callback = customer_agent_callback
        
        # Belief tracking for explainability
        self.enable_belief_tracking = enable_belief_tracking
        self.belief_tracker = BeliefTracker() if enable_belief_tracking else None

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
        if intent in self.custom_thresholds:
            return self.custom_thresholds[intent]
        elif self.is_leaf(intent):
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
                    (parent for parent, children in self.hierarchy.items()
                     if current in children),
                    None
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
            mass_function = {
                intent: mass / total_mass
                for intent, mass in mass_function.items()
            }
        else:
            mass_function = {
                intent: 0.0 for intent in self.intent_embeddings.keys()
            }
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
                    combined_mass[intersection] = (
                        combined_mass.get(intersection, 0) + contribution
                    )

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

    def evaluate_hierarchy(
        self,
        nodes: List[str],
        mass_function: Dict[str, float]
    ) -> Tuple[List[Tuple[str, float]], Dict[str, float]]:
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
                clarification_queries.append((parent, children))
            else:
                children_with_belief = [
                    (child, belief.get(child, 0)) for child in children
                ]
                children_with_belief.sort(key=lambda x: x[1], reverse=True)
                top_children = [child for child, _ in children_with_belief[:3]]
                clarification_queries.append((parent, top_children))

        return clarification_queries

    def evaluate_with_clarifications(
        self,
        initial_mass: Dict[str, float],
        depth: int = 0,
        maximum_depth: int = 5
    ) -> Optional[Tuple[str, float]]:
        """Evaluate recursively with clarifications until confident leaf found.

        Args:
            initial_mass: Initial mass function
            depth: Current recursion depth
            maximum_depth: Maximum recursion depth

        Returns:
            Tuple of (intent, confidence) or None
        """
        if depth >= maximum_depth:
            logger.warning("Maximum clarification depth reached")
            return None

        return self._evaluate_from_leaves(initial_mass, depth, maximum_depth)

    def _evaluate_from_leaves(
        self,
        current_mass: Dict[str, float],
        depth: int,
        maximum_depth: int
    ) -> Optional[Tuple[str, float]]:
        """Internal recursive evaluation starting from leaf nodes.

        Args:
            current_mass: Current mass function
            depth: Current recursion depth
            maximum_depth: Maximum recursion depth

        Returns:
            Tuple of (intent, confidence) or None
        """
        if depth >= maximum_depth:
            return None

        # Find all leaf nodes
        leaf_nodes = [
            intent for intent in self.hierarchy if self.is_leaf(intent)
        ]
        confident_nodes, belief = self.evaluate_hierarchy(leaf_nodes, current_mass)
        
        # Track belief at this turn for explainability
        if self.enable_belief_tracking and self.belief_tracker is not None:
            turn_label = f"Turn {depth + 1}"
            if depth == 0:
                turn_label = "Initial Query"
            self.belief_tracker.record_belief(belief, turn_label)

        if confident_nodes:
            # CASE 1: Confident leaf nodes
            confident_leaf_nodes = [
                n for n in confident_nodes if self.is_leaf(n[0])
            ]

            if len(confident_leaf_nodes) == 1:
                logger.info(f"Found confident leaf: {confident_leaf_nodes[0][0]}")
                return confident_leaf_nodes[0]

            if len(confident_leaf_nodes) > 1:
                lca = self.find_lowest_common_ancestor(
                    [i for i, _ in confident_leaf_nodes]
                )
                if lca:
                    logger.info(f"Multiple confident leaves. LCA: {lca}")
                    current_mass = self._handle_clarification(
                        lca, belief, current_mass
                    )
                    return self._evaluate_from_leaves(
                        current_mass, depth + 1, maximum_depth
                    )

            # CASE 2: Handle non-leaf nodes
            confident_non_leaf = [
                (i, v) for i, v in confident_nodes if not self.is_leaf(i)
            ]
            if confident_non_leaf:
                current_mass = self._handle_non_leaf_nodes(
                    confident_non_leaf, belief, current_mass
                )
                return self._evaluate_from_leaves(
                    current_mass, depth + 1, maximum_depth
                )
        else:
            # CASE 3: No confident nodes - ask for clarification
            current_mass = self._handle_no_confidence(current_mass)

        return self._evaluate_from_leaves(current_mass, depth + 1, maximum_depth)

    def _handle_clarification(
        self,
        parent: str,
        belief: Dict[str, float],
        current_mass: Dict[str, float]
    ) -> Dict[str, float]:
        """Handle clarification for a parent node.

        Args:
            parent: Parent node name
            belief: Current belief values
            current_mass: Current mass function

        Returns:
            Updated mass function
        """
        clarification_queries = self.ask_clarification(
            [(parent, belief.get(parent, 0))], belief
        )

        for parent_node, children in clarification_queries:
            chatbot_question = (
                f"It seems like you're looking for something related to "
                f"{parent_node}. Could you clarify which specific thing "
                f"you're interested in? Here are a few suggestions: {children}"
            )
            self.conversation_history.append(f"Chatbot: {chatbot_question}")

            if self.customer_agent_callback:
                self.user_response = self.customer_agent_callback(
                    "\n".join(self.conversation_history),
                    chatbot_question
                )
            else:
                self.user_response = input("User: ")

            user_mass = self.compute_mass_function(self.user_response)
            current_mass = self.combine_mass_functions(current_mass, user_mass)

        return current_mass

    def _handle_non_leaf_nodes(
        self,
        confident_non_leaf: List[Tuple[str, float]],
        belief: Dict[str, float],
        current_mass: Dict[str, float]
    ) -> Dict[str, float]:
        """Handle confident non-leaf nodes.

        Args:
            confident_non_leaf: List of confident non-leaf nodes
            belief: Current belief values
            current_mass: Current mass function

        Returns:
            Updated mass function
        """
        def height_from_leaves(node: str) -> int:
            if self.is_leaf(node) or node not in self.hierarchy:
                return 0
            return 1 + max(
                height_from_leaves(c) for c in self.hierarchy[node]
            )

        heights = {i: height_from_leaves(i) for i, _ in confident_non_leaf}
        min_height = min(heights.values())
        lowest_nodes = [
            (i, v) for i, v in confident_non_leaf if heights[i] == min_height
        ]

        lca_non_leaf = self.find_lowest_common_ancestor(
            [i for i, _ in lowest_nodes]
        )

        if lca_non_leaf:
            return self._handle_clarification(lca_non_leaf, belief, current_mass)

        return current_mass

    def _handle_no_confidence(
        self,
        current_mass: Dict[str, float]
    ) -> Dict[str, float]:
        """Handle case with no confident nodes.

        Args:
            current_mass: Current mass function

        Returns:
            Updated mass function
        """
        chatbot_question = (
            "I'm not entirely sure what you're asking. "
            "Could you rephrase your question a bit?"
        )
        self.conversation_history.append(f"Chatbot: {chatbot_question}")

        if self.customer_agent_callback:
            self.user_response = self.customer_agent_callback(
                "\n".join(self.conversation_history),
                chatbot_question
            )
        else:
            self.user_response = input("User: ")

        user_mass = self.compute_mass_function(self.user_response)
        return self.combine_mass_functions(current_mass, user_mass)
    
    def clear_belief_history(self):
        """Clear the belief tracking history."""
        if self.belief_tracker is not None:
            self.belief_tracker.clear_history()
    
    def get_belief_tracker(self) -> Optional[BeliefTracker]:
        """
        Get the belief tracker instance.
        
        Returns:
            BeliefTracker instance if tracking is enabled, None otherwise
        """
        return self.belief_tracker
    
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
