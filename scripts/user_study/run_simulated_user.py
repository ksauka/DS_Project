"""Script to run simulated user testing with LLM agent."""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.agents.customer_agent import CustomerAgent
from src.utils.user_study import UserStudyInterface
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
        description='Run simulated user testing with LLM agent (GPT-4o-mini)'
    )
    parser.add_argument(
        '--study-queries',
        type=str,
        required=True,
        help='Path to selected queries CSV for user study'
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
        '--output-dir',
        type=str,
        default='outputs/user_study/sessions',
        help='Output directory for session results'
    )
    parser.add_argument(
        '--user-id',
        type=str,
        required=True,
        help='User identifier for this session'
    )
    parser.add_argument(
        '--start-index',
        type=int,
        default=0,
        help='Starting query index'
    )
    parser.add_argument(
        '--max-queries',
        type=int,
        help='Maximum number of queries to process'
    )
    parser.add_argument(
        '--embedding-model',
        type=str,
        default='intfloat/e5-base',
        help='Sentence transformer model name'
    )
    return parser.parse_args()


def main():
    """Main function."""
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

    # Initialize LLM customer agent for simulation
    logger.info("Initializing LLM customer agent...")
    customer_agent = CustomerAgent()
    
    # Initialize DS Mass Function with LLM callback
    ds_calculator = DSMassFunction(
        intent_embeddings=intent_embeddings.get_all_embeddings(),
        hierarchy=hierarchy,
        classifier=classifier,
        custom_thresholds=custom_thresholds,
        customer_agent_callback=customer_agent.generate_response  # Use LLM agent
    )

    # Create user study interface
    interface = UserStudyInterface(
        ds_calculator=ds_calculator,
        study_data_path=Path(args.study_queries),
        output_dir=Path(args.output_dir)
    )

    # Run interactive session
    interface.run_interactive_session(
        user_id=args.user_id,
        start_index=args.start_index,
        max_queries=args.max_queries
    )

    # Display summary
    summary = interface.get_results_summary()
    
    print("\n" + "="*60)
    print("SESSION SUMMARY")
    print("="*60)
    for key, value in summary.items():
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")
    print("="*60)


if __name__ == '__main__':
    main()
