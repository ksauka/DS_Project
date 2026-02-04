"""Evaluation script with DS Mass Function and clarifications."""

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Dict
import pandas as pd
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.data.data_loader import DataLoader
from src.models.embeddings import SentenceEmbedder, IntentEmbeddings
from src.models.classifier import IntentClassifier
from src.models.ds_mass_function import DSMassFunction
from src.agents.customer_agent import CustomerAgent
from src.utils.metrics import compute_all_metrics, analyze_predictions, count_interactions
from src.utils.file_io import save_csv, save_json, ensure_dir
from src.utils.explainability import BeliefVisualizer
from src.utils.faithfulness import FaithfulnessValidator
from src.utils.evaluation_curves import AccuracyCoverageBurdenAnalyzer
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
        description='Evaluate intent classification with DS reasoning'
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
        '--output-dir',
        type=str,
        default='outputs/evaluation',
        help='Output directory for results'
    )
    parser.add_argument(
        '--max-depth',
        type=int,
        default=5,
        help='Maximum clarification depth'
    )
    parser.add_argument(
        '--num-samples',
        type=int,
        help='Number of samples to evaluate (default: all)'
    )
    parser.add_argument(
        '--n-chunks',
        type=int,
        default=4,
        help='Number of chunks to split data into for incremental saving (default: 4)'
    )
    parser.add_argument(
        '--use-customer-agent',
        action='store_true',
        help='Use customer agent for clarifications'
    )
    parser.add_argument(
        '--save-belief-plots',
        action='store_true',
        help='Save belief progression plots for each query'
    )
    parser.add_argument(
        '--save-belief-logs',
        action='store_true',
        help='Save belief progression logs as JSON'
    )
    parser.add_argument(
        '--test-faithfulness',
        action='store_true',
        help='Run faithfulness validation tests'
    )
    parser.add_argument(
        '--generate-acc-curves',
        action='store_true',
        help='Generate accuracy-coverage-burden curves'
    )
    return parser.parse_args()


def evaluate_with_ds(
    data_loader: DataLoader,
    ds_calculator: DSMassFunction,
    dataset_name: str,
    split: str = 'test',
    num_samples: int = None,
    n_chunks: int = 4,
    save_belief_plots: bool = False,
    save_belief_logs: bool = False,
    output_dir: Path = None
) -> List[Dict]:
    """Evaluate using DS Mass Function with clarifications.

    Args:
        data_loader: Loaded dataset
        ds_calculator: DS Mass Function calculator
        dataset_name: Name of the dataset (for saving chunk files)
        split: Dataset split to evaluate
        num_samples: Number of samples to evaluate
        n_chunks: Number of chunks to split data into
        save_belief_plots: Whether to save belief progression plots
        save_belief_logs: Whether to save belief progression logs
        output_dir: Output directory for belief visualizations

    Returns:
        List of result dictionaries
    """
    # Get test data
    texts, true_intents, _ = data_loader.get_split_data(split)

    if num_samples:
        texts = texts[:num_samples]
        true_intents = true_intents[:num_samples]

    results = []
    logger.info(f"Evaluating {len(texts)} samples in {n_chunks} chunks...")
    
    # Create subdirectories for belief tracking
    if save_belief_plots or save_belief_logs:
        belief_dir = ensure_dir(output_dir / 'belief_progressions')
        if save_belief_plots:
            plots_dir = ensure_dir(belief_dir / 'plots')
        if save_belief_logs:
            logs_dir = ensure_dir(belief_dir / 'logs')
    
    # Split data into chunks
    def chunk_data(data1, data2, n):
        """Split two lists into n equal chunks."""
        chunk_size = len(data1) // n
        chunks = []
        for i in range(n):
            start = i * chunk_size
            end = start + chunk_size if i < n - 1 else len(data1)
            chunks.append((data1[start:end], data2[start:end]))
        return chunks
    
    chunks = chunk_data(texts, true_intents, n_chunks)
    
    # Process each chunk
    for chunk_idx, (chunk_texts, chunk_labels) in enumerate(chunks):
        logger.info(f"\nProcessing chunk {chunk_idx + 1}/{n_chunks} ({len(chunk_texts)} samples)...")
        
        chunk_results = []
        pbar = tqdm(zip(chunk_texts, chunk_labels), 
                    desc=f"Chunk {chunk_idx + 1}/{n_chunks}", 
                    total=len(chunk_texts),
                    unit="query")
        
        for idx, (text, true_intent) in enumerate(pbar):
            global_idx = sum(len(c[0]) for c in chunks[:chunk_idx]) + idx
            
            # Update progress bar with current query info
            pbar.set_postfix({'intent': true_intent[:20], 'processing': '...'}, refresh=True)
            
            # Reset conversation history and belief tracking
            ds_calculator.conversation_history = []
            ds_calculator.clear_belief_history()

            # Compute initial mass function (classifier probabilities)
            initial_mass = ds_calculator.compute_mass_function(text)
            
            # IMPORTANT: Extract INITIAL beliefs (before clarifications) for threshold optimization
            # This matches the old notebook behavior where beliefs are extracted from classifier output only
            initial_belief = ds_calculator.compute_belief(initial_mass)
            
            # Now evaluate with clarifications using thresholds (if provided)
            prediction = ds_calculator.evaluate_with_clarifications(initial_mass)

            # Store result
            if prediction:
                pred_intent, confidence = prediction
            else:
                pred_intent, confidence = "unknown", 0.0

            # DEBUG: Check conversation history
            conv_history = ds_calculator.conversation_history
            conv_text = '\n'.join(conv_history) if conv_history else ""
            
            result = {
                'query': text,
                'true_intent': true_intent,
                'predicted_intent': pred_intent,
                'confidence': confidence,
                'interaction': conv_text
            }
            
            # Store INITIAL belief values for threshold computation (not final belief after clarifications)
            if initial_belief:
                result['belief_values'] = initial_belief
            
            # Print conversation history with more visibility
            if conv_history:
                import sys
                print(f"\n{'='*70}")
                print(f"Query: {text[:80]}...")
                print(f"True intent: {true_intent}")
                print(f"Predicted: {pred_intent}")
                print(f"Conversation ({len(conv_history)} turns):")
                for i, turn in enumerate(conv_history):
                    print(f"  [{i}] {turn}")
                print(f"{'='*70}\n")
                sys.stdout.flush()
                logger.info(f"Conversation logged for query with {len(conv_history)} turns")
            
            chunk_results.append(result)
            
            # Update progress bar with result
            pbar.set_postfix({
                'pred': pred_intent[:15] if pred_intent else 'N/A',
                'conf': f'{confidence:.2f}'
            }, refresh=False)
            
            # Save belief progression visualizations
            belief_tracker = ds_calculator.get_belief_tracker()
            if belief_tracker and belief_tracker.get_history():
                belief_history = belief_tracker.get_history()
                
                if save_belief_plots:
                    plot_path = plots_dir / f"query_{global_idx+1}_belief_progression.png"
                    BeliefVisualizer.plot_belief_progression(
                        belief_history,
                        title=f"Query {global_idx+1}: {text[:50]}...",
                        save_path=str(plot_path)
                    )
                
                if save_belief_logs:
                    log_path = logs_dir / f"query_{global_idx+1}_belief_log.json"
                    belief_tracker.save_to_json(str(log_path))
        
        # Save chunk results to disk
        results.extend(chunk_results)
        chunk_df = pd.DataFrame(chunk_results)
        chunk_file = output_dir / f"{dataset_name}_chunk_{chunk_idx+1}_predictions.csv"
        chunk_df.to_csv(chunk_file, index=False)
        logger.info(f"Saved chunk {chunk_idx + 1} results to {chunk_file}")

    # Save combined predictions file for all chunks
    if output_dir:
        all_results_df = pd.DataFrame(results)
        combined_file = output_dir / f"{dataset_name}_predictions.csv"
        all_results_df.to_csv(combined_file, index=False)
        logger.info(f"Saved combined results to {combined_file}")

    return results


def main():
    """Main evaluation function."""
    args = parse_args()

    logger.info(f"Starting evaluation for {args.dataset}")

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

    # Load thresholds if provided
    custom_thresholds = None
    if args.thresholds_file:
        custom_thresholds = load_thresholds_from_json(args.thresholds_file)
        logger.info(f"Loaded {len(custom_thresholds)} custom thresholds")

    # Initialize embedder and intent embeddings
    logger.info("Initializing embedder...")
    embedder = SentenceEmbedder(model_name=args.embedding_model)
    intent_embeddings = IntentEmbeddings(hierarchical_intents, embedder=embedder)

    # Load trained classifier
    logger.info(f"Loading model from {args.model_path}")
    classifier = IntentClassifier.from_pretrained(args.model_path)

    # Initialize customer agent if requested
    customer_agent = None
    if args.use_customer_agent:
        logger.info("Initializing customer agent...")
        customer_agent = CustomerAgent()

    # Initialize DS Mass Function
    logger.info("Initializing DS Mass Function...")
    ds_calculator = DSMassFunction(
        intent_embeddings=intent_embeddings.get_all_embeddings(),
        hierarchy=hierarchy,
        classifier=classifier,
        custom_thresholds=custom_thresholds,
        customer_agent_callback=customer_agent if customer_agent else None
    )

    # Evaluate
    results = evaluate_with_ds(
        data_loader=data_loader,
        ds_calculator=ds_calculator,
        dataset_name=args.dataset,
        split='test',
        num_samples=args.num_samples,
        n_chunks=args.n_chunks,
        save_belief_plots=args.save_belief_plots,
        save_belief_logs=args.save_belief_logs,
        output_dir=output_dir
    )

    # Convert to DataFrame
    results_df = pd.DataFrame(results)

    # Save predictions
    predictions_path = output_dir / f'{args.dataset}_predictions.csv'
    save_csv(results_df, predictions_path, index=False)
    logger.info(f"Saved predictions to {predictions_path}")
    
    # Save per-intent belief values for threshold computation
    if 'belief_values' in results_df.columns:
        logger.info("Extracting per-intent belief values...")
        # Expand belief_values dict into separate columns
        belief_records = []
        for idx, row in results_df.iterrows():
            record = {
                'query': row['query'],
                'true_intent': row['true_intent']
            }
            # Add belief values
            if isinstance(row['belief_values'], dict):
                record.update(row['belief_values'])
            belief_records.append(record)
        
        beliefs_df = pd.DataFrame(belief_records)
        beliefs_path = output_dir / f'{args.dataset}_beliefs.csv'
        save_csv(beliefs_df, beliefs_path, index=False)
        logger.info(f"Saved belief values to {beliefs_path}")

    # Compute metrics
    logger.info("\nComputing metrics...")
    predictions = [
        (r['predicted_intent'], r['confidence'])
        for r in results
    ]
    true_labels = [r['true_intent'] for r in results]

    metrics = analyze_predictions(predictions, true_labels)

    logger.info("\n" + "="*50)
    logger.info("EVALUATION RESULTS")
    logger.info("="*50)
    logger.info(f"Accuracy: {metrics['accuracy']:.4f}")
    logger.info(f"F1 Score (macro): {metrics['f1_macro']:.4f}")
    logger.info(f"Average Confidence: {metrics['avg_confidence']:.4f}")
    logger.info(f"Correct Predictions: {metrics['num_correct']}/{metrics['total']}")
    logger.info(f"Avg Confidence (Correct): {metrics['avg_correct_confidence']:.4f}")
    logger.info(f"Avg Confidence (Incorrect): {metrics['avg_incorrect_confidence']:.4f}")

    # Analyze interactions
    interaction_stats = count_interactions(
        [r['interaction'] for r in results]
    )
    logger.info(f"\nAverage Clarifications: {interaction_stats['avg_turns_per_conversation']:.2f}")

    # Save metrics
    metrics['interaction_stats'] = interaction_stats
    metrics_path = output_dir / f'{args.dataset}_metrics.json'
    save_json(metrics, metrics_path)
    logger.info(f"\nSaved metrics to {metrics_path}")
    
    # Run faithfulness tests if requested
    if args.test_faithfulness and (args.save_belief_logs or args.save_belief_plots):
        logger.info("\nRunning faithfulness validation...")
        faithfulness_dir = ensure_dir(output_dir / 'faithfulness')
        
        validator = FaithfulnessValidator()
        belief_logs_dir = output_dir / 'belief_progressions' / 'logs' if args.save_belief_logs else None
        
        validation_summary = validator.validate_results(
            results_df,
            belief_logs_dir=belief_logs_dir
        )
        
        # Save faithfulness results
        faith_path = faithfulness_dir / 'faithfulness_summary.json'
        save_json(validation_summary, faith_path)
        
        validator.generate_faithfulness_report(
            validation_summary,
            faithfulness_dir / 'faithfulness_report.txt'
        )
        
        logger.info(f"Faithfulness Pass Rate: {validation_summary['pass_rate']:.1%}")
    elif args.test_faithfulness:
        logger.warning("--test-faithfulness requires --save-belief-logs or --save-belief-plots")
    
    # Generate ACC curves if requested
    if args.generate_acc_curves:
        logger.info("\nGenerating ACC curves...")
        acc_dir = ensure_dir(output_dir / 'acc_curves')
        
        analyzer = AccuracyCoverageBurdenAnalyzer()
        
        # Extract data from results
        pred_labels = [r['predicted_intent'] for r in results]
        true_labels = [r['true_intent'] for r in results]
        confidences = [r['confidence'] for r in results]
        interactions = [r['interaction'].count("Chatbot:") for r in results]
        
        # Generate curves
        curves_df = analyzer.generate_acc_curves(
            predictions=pred_labels,
            true_labels=true_labels,
            confidences=confidences,
            interactions=interactions
        )
        
        # Save curves data
        curves_df.to_csv(acc_dir / 'acc_curves_data.csv', index=False)
        
        # Find optimal threshold
        optimal = analyzer.find_optimal_threshold(curves_df)
        save_json(optimal, acc_dir / 'optimal_threshold.json')
        
        # Generate plots and report
        analyzer.plot_acc_curves(curves_df, acc_dir, title_prefix=f"{args.dataset.upper()} ")
        analyzer.generate_acc_report(curves_df, optimal, acc_dir / 'acc_report.txt')
        
        logger.info(f"Optimal Threshold: {optimal['optimal_threshold']:.3f} (Acc: {optimal['accuracy']:.1%}, Cov: {optimal['coverage']:.1%})")

    logger.info("\nEvaluation completed!")


if __name__ == '__main__':
    main()
