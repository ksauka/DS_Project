"""Interactive user study script with real human participants."""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.models.embeddings import SentenceEmbedder, IntentEmbeddings
from src.models.classifier import IntentClassifier
from src.models.ds_mass_function import DSMassFunction
from config.hierarchy_loader import (
    load_hierarchy_from_json,
    load_hierarchical_intents_from_json
)
from config.threshold_loader import load_thresholds_from_json

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Interactive user study with real human participants'
    )
    parser.add_argument(
        '--model-path',
        type=str,
        required=True,
        help='Path to trained classifier model'
    )
    parser.add_argument(
        '--hierarchy-file',
        type=str,
        required=True,
        help='Path to hierarchy JSON file'
    )
    parser.add_argument(
        '--intents-file',
        type=str,
        required=True,
        help='Path to hierarchical intents JSON file'
    )
    parser.add_argument(
        '--thresholds-file',
        type=str,
        help='Path to optimal thresholds JSON file'
    )
    parser.add_argument(
        '--embedding-model',
        type=str,
        default='intfloat/e5-base',
        help='Sentence transformer model name'
    )
    parser.add_argument(
        '--max-depth',
        type=int,
        default=5,
        help='Maximum clarification depth'
    )
    return parser.parse_args()


def main():
    """Main prediction function."""
    args = parse_args()

    logger.info("Loading models and configuration...")

    # Load hierarchy and intents
    hierarchy = load_hierarchy_from_json(args.hierarchy_file)
    hierarchical_intents = load_hierarchical_intents_from_json(args.intents_file)

    # Load thresholds if provided
    custom_thresholds = None
    if args.thresholds_file:
        custom_thresholds = load_thresholds_from_json(args.thresholds_file)

    # Initialize embedder and intent embeddings
    embedder = SentenceEmbedder(model_name=args.embedding_model)
    intent_embeddings = IntentEmbeddings(hierarchical_intents, embedder=embedder)

    # Load trained classifier
    classifier = IntentClassifier.from_pretrained(args.model_path)

    # Initialize DS Mass Function
    ds_calculator = DSMassFunction(
        intent_embeddings=intent_embeddings.get_all_embeddings(),
        hierarchy=hierarchy,
        classifier=classifier,
        custom_thresholds=custom_thresholds,
        customer_agent_callback=None  # Use manual input
    )

    logger.info("\n" + "="*50)
    logger.info("INTERACTIVE INTENT PREDICTION")
    logger.info("="*50)
    logger.info("Type your query and press Enter.")
    logger.info("Type 'quit' or 'exit' to stop.\n")

    while True:
        try:
            # Get user input
            user_query = input("\nYour query: ").strip()

            if user_query.lower() in ['quit', 'exit', 'q']:
                logger.info("Goodbye!")
                break

            if not user_query:
                continue

            # Reset conversation history
            ds_calculator.conversation_history = []

            # Compute initial mass and evaluate
            initial_mass = ds_calculator.compute_mass_function(user_query)
            prediction = ds_calculator.evaluate_with_clarifications(
                initial_mass,
                maximum_depth=args.max_depth
            )

            # Display result
            print("\n" + "-"*50)
            if prediction:
                pred_intent, confidence = prediction
                print(f"Predicted Intent: {pred_intent}")
                print(f"Confidence: {confidence:.4f}")
            else:
                print("Could not determine intent after maximum clarifications.")

            # Display conversation history
            if ds_calculator.conversation_history:
                print("\nConversation History:")
                for turn in ds_calculator.conversation_history:
                    print(f"  {turn}")
            print("-"*50)

        except KeyboardInterrupt:
            logger.info("\n\nInterrupted. Goodbye!")
            break
        except Exception as e:
            logger.error(f"Error: {e}")


if __name__ == '__main__':
    main()
