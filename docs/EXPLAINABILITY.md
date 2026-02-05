# Explainability Feature Documentation

## Overview

The explainability module provides **belief progression tracking** during intent disambiguation, making the Dempster-Shafer reasoning process transparent and interpretable. This feature tracks how belief values for different intents evolve across multi-turn clarification dialogues.

## Architecture

### Core Components

1. **BeliefTracker** (`src/utils/explainability.py`)
   - Records belief values at each turn during disambiguation
   - Saves/loads belief history to/from JSON
   - Provides access to historical belief states

2. **BeliefVisualizer** (`src/utils/explainability.py`)
   - Creates visualizations of belief progression
   - Supports multiple visualization types (bar plots, line plots, comparisons)
   - Generates statistical summaries

3. **DSMassFunction Integration** (`src/models/ds_mass_function.py`)
   - Automatic belief tracking at each evaluation turn
   - Optional enable/disable tracking via `enable_belief_tracking` parameter
   - Methods: `clear_belief_history()`, `get_belief_tracker()`, `save_belief_log()`

## Usage

### Basic Usage in Evaluation

```python
# evaluate.py with belief tracking
python evaluate.py \
    --dataset banking77 \
    --model-path outputs/models/banking77_model.pkl \
    --hierarchy-file config/hierarchies/banking77_hierarchy.json \
    --intents-file config/hierarchies/banking77_intents.json \
    --save-belief-plots \
    --save-belief-logs \
    --output-dir outputs/evaluation
```

**Output:**
- `outputs/evaluation/belief_progressions/plots/` - PNG plots for each query
- `outputs/evaluation/belief_progressions/logs/` - JSON logs for each query

### User Study with Belief Tracking

```python
# run_simulated_user.py automatically tracks beliefs
python run_simulated_user.py \
    --study-queries outputs/user_study/selected_queries.csv \
    --model-path outputs/models/banking77_model.pkl \
    --hierarchy-file config/hierarchies/banking77_hierarchy.json \
    --intents-file config/hierarchies/banking77_intents.json \
    --user-id user_001 \
    --output-dir outputs/user_study/sessions
```

**Output:**
- `outputs/user_study/sessions/belief_progressions/plots/{user_id}_query_{n}_belief.png`
- `outputs/user_study/sessions/belief_progressions/logs/{user_id}_query_{n}_belief.json`

### Comparing LLM vs Human Belief Progressions

```python
# compare_simulated_vs_real.py with belief comparison
python compare_simulated_vs_real.py \
    --user-study-results outputs/user_study/sessions/user_study_results.csv \
    --belief-logs-dir outputs/belief_progressions \
    --compare-belief-progression \
    --output-dir outputs/user_study/comparison
```

**Output:**
- `outputs/user_study/comparison/belief_comparisons/query_{n}_comparison.png`

## Visualization Types

### 1. Belief Progression Bar Plot
Shows belief values for all intents across turns. Uncertainty highlighted in red.

```python
from src.utils.explainability import BeliefVisualizer

BeliefVisualizer.plot_belief_progression(
    belief_history=[(belief_dict1, "Turn 1"), (belief_dict2, "Turn 2"), ...],
    title="Belief Progression",
    save_path="belief_plot.png"
)
```

### 2. Top Intents Line Plot
Tracks top-k intents across turns using line plot.

```python
BeliefVisualizer.plot_top_intents_progression(
    belief_history=belief_history,
    top_k=5,
    title="Top 5 Intents Progression",
    save_path="top_intents.png"
)
```

### 3. Comparison Plot (LLM vs Human)
Side-by-side comparison of belief progressions.

```python
BeliefVisualizer.compare_belief_progressions(
    belief_histories=[
        (llm_belief_history, "LLM"),
        (human_belief_history, "Human")
    ],
    title="LLM vs Human Comparison",
    save_path="comparison.png"
)
```

## Belief Log Format (JSON)

```json
[
  {
    "turn": "Initial Query",
    "belief": {
      "card_payment_not_recognised": 0.45,
      "card_payment_wrong_exchange_rate": 0.35,
      "Uncertainty": 0.20
    }
  },
  {
    "turn": "Turn 2",
    "belief": {
      "card_payment_not_recognised": 0.75,
      "card_payment_wrong_exchange_rate": 0.15,
      "Uncertainty": 0.10
    }
  }
]
```

## Programmatic API

### Initialize DS Calculator with Tracking

```python
from src.models.ds_mass_function import DSMassFunction

ds_calculator = DSMassFunction(
    intent_embeddings=intent_embeddings,
    hierarchy=hierarchy,
    classifier=classifier,
    enable_belief_tracking=True  # Enable tracking
)
```

### Access Belief History

```python
# Get the tracker
tracker = ds_calculator.get_belief_tracker()

# Get full history
belief_history = tracker.get_history()
# Returns: [(belief_dict1, "Turn 1"), (belief_dict2, "Turn 2"), ...]

# Get final belief
final_belief = tracker.get_final_belief()
# Returns: {"intent1": 0.8, "intent2": 0.1, "Uncertainty": 0.1}

# Get specific turn
turn_2_belief = tracker.get_belief_at_turn(1)  # 0-indexed
```

### Save Belief Log

```python
# Save to JSON
ds_calculator.save_belief_log("belief_progression.json")

# Or via tracker
tracker = ds_calculator.get_belief_tracker()
tracker.save_to_json("belief_progression.json")
```

### Load Belief Log

```python
from src.utils.explainability import BeliefTracker

tracker = BeliefTracker()
tracker.load_from_json("belief_progression.json")
belief_history = tracker.get_history()
```

### Generate Summary Statistics

```python
from src.utils.explainability import BeliefVisualizer

summary = BeliefVisualizer.generate_belief_summary(belief_history)

print(summary)
# Output:
# {
#   "num_turns": 3,
#   "initial_top_intent": "card_payment_not_recognised",
#   "initial_top_belief": 0.45,
#   "final_top_intent": "card_payment_not_recognised",
#   "final_top_belief": 0.85,
#   "top_intents_per_turn": [...],
#   "uncertainty_progression": [...],
#   "initial_uncertainty": 0.20,
#   "final_uncertainty": 0.05
# }
```

## Integration Points

### 1. Training Pipeline (`train.py`)
No changes needed - belief tracking is for evaluation only.

### 2. Evaluation Pipeline (`evaluate.py`)
- Added `--save-belief-plots` flag
- Added `--save-belief-logs` flag
- Automatically clears history before each query
- Saves visualizations after each query

### 3. Simulated User Testing (`run_simulated_user.py`)
- Belief tracking enabled by default
- Saves per-user, per-query belief logs
- Creates visualizations automatically

### 4. Comparison Analysis (`compare_simulated_vs_real.py`)
- Added `--compare-belief-progression` flag
- Added `--belief-logs-dir` argument
- Matches LLM and Human belief logs by query index
- Creates side-by-side comparisons

## Performance Considerations

- **Memory**: Each belief state stores a dictionary with ~10-100 intent keys (depending on dataset)
- **Storage**: JSON logs are ~1-5 KB per query
- **Visualization**: PNG plots are ~50-200 KB each
- **Overhead**: Minimal - belief tracking adds <1% to evaluation time

## Disabling Belief Tracking

If you don't need explainability features:

```python
ds_calculator = DSMassFunction(
    intent_embeddings=intent_embeddings,
    hierarchy=hierarchy,
    classifier=classifier,
    enable_belief_tracking=False  # Disable tracking
)
```

Or simply don't use the `--save-belief-plots` and `--save-belief-logs` flags.

## Research Applications

### Intent Classification Explainability
- Visualize how confidence evolves during disambiguation
- Identify when uncertainty is reduced
- Understand clarification effectiveness

### LLM vs Human Comparison
- Compare belief progression patterns
- Identify where LLM diverges from human reasoning
- Validate simulation accuracy

### Model Debugging
- Identify queries where model struggles (high uncertainty)
- Analyze belief distributions for misclassified queries
- Track confidence calibration

### User Experience Research
- Study cognitive load (number of turns needed)
- Analyze user response patterns
- Optimize clarification strategies

## Example Workflow

```bash
# 1. Train model (no belief tracking)
python train.py --dataset banking77 --output-dir outputs/models

# 2. Compute optimal thresholds (no belief tracking)
python compute_thresholds.py --dataset banking77 \
    --model-path outputs/models/banking77_model.pkl \
    --hierarchy-file config/hierarchies/banking77_hierarchy.json

# 3. Evaluate with LLM and save belief progressions
python evaluate.py --dataset banking77 \
    --model-path outputs/models/banking77_model.pkl \
    --hierarchy-file config/hierarchies/banking77_hierarchy.json \
    --intents-file config/hierarchies/banking77_intents.json \
    --thresholds-file outputs/thresholds/banking77_thresholds.json \
    --use-customer-agent \
    --save-belief-plots \
    --save-belief-logs \
    --output-dir outputs/llm_evaluation

# 4. Select problematic queries for user study
python select_user_study_queries.py \
    --results-file outputs/llm_evaluation/banking77_predictions.csv \
    --output-file outputs/user_study/selected_queries.csv \
    --strategy balanced \
    --num-queries 50

# 5. Run user study with belief tracking
python run_user_study.py \
    --study-queries outputs/user_study/selected_queries.csv \
    --model-path outputs/models/banking77_model.pkl \
    --hierarchy-file config/hierarchies/banking77_hierarchy.json \
    --intents-file config/hierarchies/banking77_intents.json \
    --user-id user_001 \
    --output-dir outputs/user_study/sessions

# 6. Compare LLM vs Human with belief progressions
python compare_llm_vs_human.py \
    --user-study-results outputs/user_study/sessions/user_study_results.csv \
    --belief-logs-dir outputs \
    --compare-belief-progression \
    --output-dir outputs/user_study/comparison
```

## Troubleshooting

**Q: Belief plots are empty**
- Check that `enable_belief_tracking=True` in DSMassFunction initialization
- Verify that `clear_belief_history()` is called before each query

**Q: Comparison fails to find matching logs**
- Ensure belief logs directory structure matches expected pattern
- Check file naming conventions (user_id, query indices)

**Q: Uncertainty not showing in plots**
- Set `show_uncertainty=True` in `plot_belief_progression()`
- Verify DS Mass Function includes Uncertainty in belief calculations

## Related Files

- `src/utils/explainability.py` - Core module
- `src/models/ds_mass_function.py` - Integration with DS reasoning
- `evaluate.py` - Evaluation with belief tracking
- `run_user_study.py` - Real user study (interactive)
- `run_simulated_user.py` - Simulated user testing with belief tracking
- `compare_simulated_vs_real.py` - Comparison with belief visualization
- `DS-Model2_old/explainability/` - Original reference implementation
