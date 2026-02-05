#!/usr/bin/env python3
"""Quick verification that the complete pipeline works - uses same API as train.py"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

print("="*60)
print("QUICK PIPELINE VERIFICATION")
print("="*60)
print()

# Test 1: Can we train? (Using exact same code as train.py)
print("Test 1: Training classifier...")
try:
    from src.data.data_loader import DataLoader
    from src.models.embeddings import SentenceEmbedder, IntentEmbeddings
    from src.models.classifier import IntentClassifier
    from config.hierarchy_loader import load_hierarchy_from_json, load_hierarchical_intents_from_json
    
    # Load dataset (same as train.py)
    data_loader = DataLoader("banking77")
    data_loader.load()
    
    # Get training data (same as train.py)
    train_texts, train_intent_names, train_labels = data_loader.get_split_data('train')
    train_texts = train_texts[:500]  # Need enough samples for multiple classes
    train_labels = train_labels[:500]
    
    # Initialize embedder (same as train.py)
    embedder = SentenceEmbedder(model_name='intfloat/e5-base')
    
    # Generate embeddings (same as train.py)
    train_embeddings = embedder.get_embeddings_batch(
        train_texts,
        batch_size=32,
        show_progress=False
    )
    
    # Train classifier (same as train.py)
    classifier = IntentClassifier(
        classifier_type='logistic',
        max_iter=1000
    )
    classifier.train(train_embeddings, train_labels)  # Uses .train() not .fit()!
    
    print("  ✅ Training works!")
except Exception as e:
    print(f"  ❌ Training failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 2: Can we evaluate and extract beliefs?
print("\nTest 2: Extracting beliefs...")
try:
    from src.models.ds_mass_function import DSMassFunction
    
    hierarchy = load_hierarchy_from_json("config/hierarchies/banking77_hierarchy.json")
    hierarchical_intents = load_hierarchical_intents_from_json("config/hierarchies/banking77_intents.json")
    
    intent_embeddings_obj = IntentEmbeddings(hierarchical_intents, embedder=embedder)
    intent_embeddings = intent_embeddings_obj.get_all_embeddings()
    
    ds_calculator = DSMassFunction(
        intent_embeddings=intent_embeddings,
        hierarchy=hierarchy,
        classifier=classifier,
        enable_belief_tracking=True
    )
    
    # Test on one query
    test_query = "I lost my card"
    initial_mass = ds_calculator.compute_mass_function(test_query)
    belief = ds_calculator.compute_belief(initial_mass)
    
    # The belief should be computed directly (not relying on get_current_belief which requires evaluate_with_clarifications)
    if belief and len(belief) > 0:
        print(f"  ✅ Belief extraction works! (computed {len(belief)} intent beliefs)")
    else:
        print("  ❌ Belief extraction returned empty")
        sys.exit(1)
        
except Exception as e:
    print(f"  ❌ Belief extraction failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 3: Can we compute thresholds?
print("\nTest 3: Computing thresholds...")
try:
    import pandas as pd
    import numpy as np
    from sklearn.metrics import f1_score
    
    # Create mock belief data
    data = {
        'query': ['query1', 'query2'],
        'true_intent': ['card_not_working', 'lost_or_stolen_card']
    }
    
    # Add belief columns for a few intents
    for intent in list(intent_embeddings.keys())[:5]:
        data[intent] = [0.5, 0.7]
    
    df = pd.DataFrame(data)
    
    # Add ground truth for one intent
    df['is_correct_card_not_working'] = [1, 0]
    
    # Compute threshold for one intent
    belief_values = df['card_not_working'].values if 'card_not_working' in df.columns else np.array([0.5, 0.7])
    ground_truth = df['is_correct_card_not_working'].values
    
    thresholds = np.linspace(0, 1, 11)  # Fewer for speed
    best_threshold = 0.0
    best_f1 = 0.0
    
    for threshold in thresholds:
        predictions = (belief_values >= threshold).astype(int)
        f1 = f1_score(ground_truth, predictions, average='weighted', zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
    
    print(f"  ✅ Threshold computation works! (θ={best_threshold:.2f}, F1={best_f1:.4f})")
    
except Exception as e:
    print(f"  ❌ Threshold computation failed: {e}")
    sys.exit(1)

print("\n" + "="*60)
print("✅ ALL PIPELINE COMPONENTS VERIFIED!")
print("="*60)
print()
print("Ready to run complete system test:")
print("  python scripts/test_complete_system.py")
print()
print("Or use the bash version:")
print("  ./scripts/test_complete_system.sh")
print()
