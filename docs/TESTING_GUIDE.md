# System Testing Guide

## Overview
This guide explains how to test the complete DS Project system in both **LLM simulation** mode and **human user study** mode.

## System Architecture Recap

### Phase 1: Configure Optimal DS Agent (One-Time Setup)
1. Train classifier → `model.pkl`
2. Evaluate without thresholds → `banking77_beliefs.csv`
3. Compute optimal thresholds → `banking77_thresholds.json`

### Phase 2: DS Agent Deployment (Interactive Testing)
- **LLM Simulation**: Automated testing with GPT-4o-mini on test set
- **Human User Study**: Interactive sessions with real users

---

## Quick Start: Run Complete Test

### Option 1: Bash Script (Recommended)
```bash
cd DS_Project
./scripts/test_complete_system.sh
```

### Option 2: Python Script
```bash
cd DS_Project
python scripts/test_complete_system.py
```

Both scripts will:
1. ✅ Train classifier on Banking77
2. ✅ Extract belief values (100 samples)
3. ✅ Compute optimal thresholds
4. ✅ Run LLM simulation (20 queries)
5. ✅ Select queries for human study (30 queries)

**Time estimate**: ~10-15 minutes (depending on GPU/CPU)

---

## Manual Testing: Step-by-Step

### Phase 1: Configure DS Agent

#### Step 1: Train Classifier
```bash
python scripts/training/train.py \
    --dataset banking77 \
    --output-dir experiments/banking77 \
    --model-type logistic \
    --embedding-model intfloat/e5-base
```

**Output**: `experiments/banking77/model.pkl`

#### Step 2: Evaluate to Extract Beliefs
```bash
python scripts/evaluation/evaluate.py \
    --dataset banking77 \
    --model-path experiments/banking77/model.pkl \
    --hierarchy-file config/hierarchies/banking77_hierarchy.json \
    --intents-file config/hierarchies/banking77_intents.json \
    --output-dir results/banking77 \
    --max-samples 100
```

**Output**: `results/banking77/banking77_beliefs.csv` (per-intent belief values)

#### Step 3: Compute Optimal Thresholds
```bash
python scripts/evaluation/compute_thresholds.py \
    --belief-file results/banking77/banking77_beliefs.csv \
    --hierarchy-file config/hierarchies/banking77_hierarchy.json \
    --output-file config/thresholds/banking77_thresholds.json
```

**Output**: `config/thresholds/banking77_thresholds.json` (optimal thresholds per intent)

---

### Phase 2: Deploy DS Agent

#### Test Mode 1: LLM Simulation (Automated)

**Goal**: Test DS agent on entire test set using GPT-4o-mini to simulate user responses

**Prerequisites**: 
- Completed Phase 1
- Set `OPENAI_API_KEY` in `.env` file

**Steps**:

1. **Prepare test queries** (optional - can use pre-selected queries):
```python
from datasets import load_dataset
import pandas as pd

dataset = load_dataset("banking77")
test_data = dataset["test"]
intent_names = dataset["train"].features["label"].names

queries = [{
    'query': ex['text'],
    'true_intent': intent_names[ex['label']],
    'query_id': f'test_{i}'
} for i, ex in enumerate(test_data)]

pd.DataFrame(queries).to_csv("test_queries.csv", index=False)
```

2. **Run LLM simulation**:
```bash
python scripts/user_study/run_simulated_user.py \
    --study-queries test_queries.csv \
    --model-path experiments/banking77/model.pkl \
    --hierarchy-file config/hierarchies/banking77_hierarchy.json \
    --intents-file config/hierarchies/banking77_intents.json \
    --thresholds-file config/thresholds/banking77_thresholds.json \
    --output-dir outputs/user_study/llm_simulation \
    --user-id llm_test_001 \
    --max-queries 100
```

**Output**: 
- `outputs/user_study/llm_simulation/session_llm_test_001.csv` (results)
- Metrics: accuracy, avg turns, avg confidence

3. **Analyze LLM results**:
```bash
python scripts/user_study/select_user_study_queries.py \
    --results-file outputs/user_study/llm_simulation/session_llm_test_001.csv \
    --output-dir outputs/user_study \
    --max-samples 50 \
    --strategy balanced
```

**Output**: `outputs/user_study/selected_queries_for_user_study.csv`

---

#### Test Mode 2: Human User Study (Interactive)

**Goal**: Test DS agent with real human users

**Prerequisites**: 
- Completed Phase 1
- Selected queries from LLM simulation (optional)

**Steps**:

1. **Run interactive terminal**:
```bash
python scripts/user_study/run_user_study.py \
    --model-path experiments/banking77/model.pkl \
    --hierarchy-file config/hierarchies/banking77_hierarchy.json \
    --intents-file config/hierarchies/banking77_intents.json \
    --thresholds-file config/thresholds/banking77_thresholds.json
```

2. **Interact with DS agent**:
```
Your query: I lost my card

Chatbot: It seems like you're looking for something related to Card Issues. 
Could you clarify which specific thing you're interested in? 
Here are a few suggestions: (['Physical Cards', 'Virtual Cards'])

Your response: Physical card

Chatbot: It seems like you're looking for something related to Physical Cards. 
Could you clarify which specific thing you're interested in? 
Here are a few suggestions: (['lost_or_stolen_card', 'card_not_working', 'card_swallowed'])

Your response: Lost or stolen

--------------------------------------------------
Predicted Intent: lost_or_stolen_card
Confidence: 0.8732
--------------------------------------------------
```

3. **Type 'quit' to exit**

---

## Expected Results

### Phase 1 Outputs
```
experiments/banking77/
├── model.pkl                    # Trained classifier
├── intent_embeddings.pkl        # Intent embeddings
└── training_metrics.json        # Training accuracy

results/banking77/
├── banking77_predictions.csv    # Initial predictions
└── banking77_beliefs.csv        # Per-intent belief values

config/thresholds/
└── banking77_thresholds.json    # Optimal thresholds (77 intents)
```

### Phase 2 Outputs (LLM Simulation)
```
outputs/user_study/
├── llm_simulation/
│   ├── session_llm_test_001.csv           # Full session log
│   ├── session_llm_test_001_summary.json  # Metrics summary
│   └── belief_logs/                       # Per-query belief tracking
└── selected_queries_for_user_study.csv    # Curated queries for humans
```

---

## Performance Benchmarks

### Expected Metrics (Banking77)
- **Classifier Accuracy**: ~88-90%
- **DS Agent Accuracy**: ~92-95% (with optimal thresholds)
- **Average Clarification Turns**: 1.5-2.5
- **Coverage**: >95% (queries resolved within max depth)

### Timing
- **Training**: ~2-3 minutes (CPU), ~30 seconds (GPU)
- **Evaluation (100 samples)**: ~5 minutes
- **Threshold Computation**: ~1-2 minutes
- **LLM Simulation (100 queries)**: ~15-20 minutes (rate limited)

---

## Troubleshooting

### Issue: "OPENAI_API_KEY not found"
**Solution**: Create `.env` file with your API key:
```bash
echo "OPENAI_API_KEY=your-key-here" > .env
```

### Issue: "ModuleNotFoundError: No module named 'src'"
**Solution**: Run from project root and ensure `sys.path` is set:
```bash
cd DS_Project
python scripts/training/train.py ...
```

### Issue: "Config file not found"
**Solution**: Ensure hierarchy files exist:
```bash
ls config/hierarchies/banking77_hierarchy.json
ls config/hierarchies/banking77_intents.json
```

### Issue: "Model file not found"
**Solution**: Run Phase 1 training first:
```bash
python scripts/training/train.py --dataset banking77
```

---

## Next Steps

After successful testing:

1. **Run full evaluation** (entire test set ~3000 queries)
2. **Conduct human user study** with 20-50 participants
3. **Compare LLM vs human results** using `scripts/user_study/compare_simulated_vs_real.py`
4. **Analyze faithfulness** using `scripts/analysis/test_faithfulness.py`
5. **Generate ACC curves** using `scripts/analysis/analyze_acc_curves.py`

---

## Questions?

See `README.md` for full documentation or check:
- `docs/EXPLAINABILITY.md` - Belief tracking details
- `docs/FAITHFULNESS_ACC_GUIDE.md` - Research metrics
- `docs/CONTRIBUTION_ASSESSMENT.md` - Research claims validation
