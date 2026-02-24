"""Extract belief values from test data for threshold computation."""

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Dict
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.data.data_loader import DataLoader
from src.models.embeddings import SentenceEmbedder, IntentEmbeddings
from src.models.classifier import IntentClassifier
from src.models.ds_mass_function import DSMassFunction
from src.utils.file_io import save_csv, ensure_dir
from config.hierarchy_loader import (
    load_hierarchy_from_json,
    load_hierarchical_intents_from_json
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Extract belief values from test data'
    )
    parser.add_argument(
        '--dataset',
        type=str,
        required=True,
        choices=['banking77', 'clinc150', 'snips', 'atis'],
        help='Dataset name'
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
        '--embedding-model',
        type=str,
        default='intfloat/e5-base',
        help='Sentence transformer model name'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='outputs/evaluation',
        help='Output directory for results'
    )
    parser.add_argument(
        '--num-samples',
        type=int,
        help='Number of samples to process (default: all)'
    )
    parser.add_argument(
        '--n-chunks',
        type=int,
        default=4,
        help='Number of chunks to split data into (default: 4)'
    )
    return parser.parse_args()


def extract_beliefs_from_data(
    data_loader: DataLoader,
    ds_calculator: DSMassFunction,
    dataset_name: str,
    split: str = 'test',
    num_samples: int = None,
    n_chunks: int = 4,
    output_dir: Path = None
) -> List[Dict]:
    """Extract belief values from test data.
    
    Args:
        data_loader: DataLoader instance
        ds_calculator: DSMassFunction instance
        dataset_name: Dataset name
        split: Dataset split ('test' or 'validation')
        num_samples: Number of samples to process (None for all)
        n_chunks: Number of chunks to split data into
        output_dir: Output directory for results
        
    Returns:
        List of belief extraction results
    """
    logger.info(f"Loading {split} data...")
    # Get the raw dataset from the DataLoader
    # It has the proper structure with 'text' and label fields
    raw_dataset = data_loader.dataset[split]
    
    # Limit samples first if specified (before conversion)
    if num_samples:
        raw_dataset = raw_dataset.select(range(min(num_samples, len(raw_dataset))))
        logger.info(f"Processing {num_samples} samples")
    else:
        logger.info(f"Processing all {len(raw_dataset)} samples")
    
    # Convert to list of dicts with standardized keys
    test_data = []
    text_field = data_loader.config.text_field
    label_field = data_loader.config.label_field
    label_feature = raw_dataset.features.get(label_field)
    label_names = list(label_feature.names) if hasattr(label_feature, 'names') else None
    
    for i in range(len(raw_dataset)):
        example = raw_dataset[i]
        label = example[label_field]
        # Convert label to intent name if it's an integer
        if isinstance(label, int):
            if label_names:
                intent_name = label_names[label]
            else:
                intent_name = data_loader.index_to_name.get(label, str(label))
        else:
            intent_name = label

        if data_loader.config.has_oos and data_loader.config.oos_label:
            if intent_name == data_loader.config.oos_label:
                continue
        
        test_data.append({
            'text': example[text_field],
            'intent': intent_name
        })
    
    # Split into chunks
    chunk_size = len(test_data) // n_chunks
    chunks = []
    for i in range(n_chunks - 1):
        chunks.append(test_data[i * chunk_size:(i + 1) * chunk_size])
    chunks.append(test_data[(n_chunks - 1) * chunk_size:])  # Last chunk includes remainder
    
    all_results = []
    
    # Process each chunk
    for chunk_idx, chunk in enumerate(chunks):
        logger.info(f"\nProcessing chunk {chunk_idx + 1}/{n_chunks} ({len(chunk)} samples)")
        
        chunk_results = []
        
        with tqdm(total=len(chunk), desc=f"Chunk {chunk_idx + 1}") as pbar:
            for ex in chunk:
                text = ex['text']
                true_intent = ex['intent']
                
                # Compute mass function and beliefs
                try:
                    mass_function = ds_calculator.compute_mass_function(text)
                    
                    if mass_function is None:
                        logger.warning(f"No mass function for query: {text[:50]}...")
                        pbar.update(1)
                        continue
                    
                    belief = ds_calculator.compute_belief(mass_function)
                    
                    if not belief:
                        logger.warning(f"No belief values for query: {text[:50]}...")
                        pbar.update(1)
                        continue
                    
                    # Store result
                    result = {
                        'query': text,
                        'true_intent': true_intent
                    }
                    # Add belief values for all intents
                    result.update(belief)
                    
                    chunk_results.append(result)
                    
                except Exception as e:
                    logger.error(f"Error processing query '{text[:50]}...': {e}")
                    continue
                
                pbar.update(1)
        
        # Save chunk to CSV
        if output_dir:
            chunk_df = pd.DataFrame(chunk_results)
            chunk_file = output_dir / f'{dataset_name}_beliefs_chunk_{chunk_idx + 1}.csv'
            save_csv(chunk_df, chunk_file, index=False)
            logger.info(f"Saved chunk {chunk_idx + 1} to {chunk_file}")
        
        all_results.extend(chunk_results)
    
    return all_results


def main():
    """Main extraction function."""
    args = parse_args()
    
    logger.info(f"Starting belief extraction for {args.dataset}")
    
    # Create output directory
    output_dir = ensure_dir(args.output_dir)
    
    # Load dataset
    logger.info("Loading dataset...")
    data_loader = DataLoader(args.dataset)
    data_loader.load()
    
    # Load hierarchy and intents
    logger.info("Loading configuration...")
    hierarchy = load_hierarchy_from_json(args.hierarchy_file)
    hierarchical_intents = load_hierarchical_intents_from_json(args.intents_file)
    
    # Initialize embedder and intent embeddings
    logger.info("Initializing embedder...")
    embedder = SentenceEmbedder(model_name=args.embedding_model)
    intent_embeddings = IntentEmbeddings(hierarchical_intents, embedder=embedder)
    
    # Load trained classifier
    logger.info(f"Loading classifier from {args.model_path}")
    classifier = IntentClassifier()
    classifier.load(args.model_path)
    logger.info(f"Classifier loaded with {len(classifier.get_classes())} classes")
    
    # Initialize DS Mass Function WITHOUT thresholds (for belief extraction only)
    logger.info("Initializing DS Mass Function...")
    ds_calculator = DSMassFunction(
        intent_embeddings=intent_embeddings.get_all_embeddings(),
        hierarchy=hierarchy,
        classifier=classifier,
        custom_thresholds=None,  # No thresholds for belief extraction
        enable_belief_tracking=False  # Disable tracking for efficiency
    )
    
    # Extract beliefs
    results = extract_beliefs_from_data(
        data_loader=data_loader,
        ds_calculator=ds_calculator,
        dataset_name=args.dataset,
        split='test',
        num_samples=args.num_samples,
        n_chunks=args.n_chunks,
        output_dir=output_dir
    )
    
    # Save merged results
    logger.info("\nMerging all chunks...")
    results_df = pd.DataFrame(results)
    merged_path = output_dir / f'{args.dataset}_beliefs.csv'
    save_csv(results_df, merged_path, index=False)
    
    logger.info("\n" + "="*60)
    logger.info("BELIEF EXTRACTION COMPLETE")
    logger.info("="*60)
    logger.info(f"Total samples processed: {len(results)}")
    logger.info(f"Number of intents: {len(results_df.columns) - 2}")  # Excluding 'query' and 'true_intent'
    logger.info(f"Beliefs saved to: {merged_path}")
    logger.info("="*60)
    logger.info("\nNext step: Run compute_thresholds.py to find optimal thresholds")


if __name__ == '__main__':
    main()
