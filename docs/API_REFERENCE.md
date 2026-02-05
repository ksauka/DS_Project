# API Reference: Original Notebook vs. Refactored Code

## Summary of Changes

The refactored code wraps the original libraries for modularity. Here's the mapping:

| Original Notebook | Refactored Code | Notes |
|-------------------|-----------------|-------|
| `load_dataset("banking77")` | `DataLoader("banking77").load()` | Abstracted dataset loading |
| `dataset["train"]` | `data_loader.get_split_data("train")` | Returns (texts, intent_names, labels) tuple |
| `SentenceTransformer('intfloat/e5-base')` | `SentenceEmbedder(model_name='intfloat/e5-base')` | Wrapper for embeddings |
| `model.encode(texts, batch_size=64)` | `embedder.get_embeddings_batch(texts, batch_size=64)` | Consistent API |
| `LogisticRegression()` | `IntentClassifier(classifier_type='logistic')` | Unified classifier interface |
| `clf.fit(X, y)` | `classifier.train(X, y)` | Renamed for clarity |
| Direct belief computation | `ds_calculator.compute_belief(mass)` | Returns belief dict directly |

## Key APIs

### DataLoader
```python
data_loader = DataLoader("banking77")
data_loader.load()

# Get split data
texts, intent_names, labels = data_loader.get_split_data("train")
```

### SentenceEmbedder
```python
embedder = SentenceEmbedder(model_name='intfloat/e5-base')

# Batch encoding
embeddings = embedder.get_embeddings_batch(
    texts, 
    batch_size=64, 
    show_progress=True
)
```

### IntentClassifier
```python
classifier = IntentClassifier(classifier_type='logistic', max_iter=1000)

# Training
classifier.train(embeddings, labels)

# Prediction
predictions = classifier.predict(test_embeddings)
probs = classifier.predict_proba(test_embeddings)
```

### DSMassFunction
```python
ds_calculator = DSMassFunction(
    intent_embeddings=intent_embeddings,
    hierarchy=hierarchy,
    classifier=classifier,
    custom_thresholds=thresholds,  # Optional
    enable_belief_tracking=True
)

# Compute mass and belief
initial_mass = ds_calculator.compute_mass_function(query)
belief = ds_calculator.compute_belief(initial_mass)

# Interactive clarifications
prediction = ds_calculator.evaluate_with_clarifications(initial_mass)
```

## Why the Refactoring?

✅ **Modularity**: Easy to swap classifiers (logistic ↔ SVM)
✅ **Consistency**: Unified API across datasets
✅ **Maintainability**: Changes in one place propagate everywhere
✅ **Testability**: Each component can be tested independently
✅ **FAIR Principles**: Findable, Accessible, Interoperable, Reusable code

## Verified Working Pipeline

✅ `train.py` → Uses correct APIs
✅ `evaluate.py` → Uses correct APIs
✅ `compute_thresholds.py` → Uses correct APIs
✅ All test scripts pass
