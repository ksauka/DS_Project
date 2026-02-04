#  Hierarchical intent disambiguation with interactive clarifications

A production-ready, modular intent classification system using **Dempster-Shafer (DS) theory** for  the project : `What Changed Your Mind? Belief-Progression Explanations in Proactive Intent Recognition`

## Features

- **Hierarchical Intent Reasoning**: Multi-level intent classification using DS theory for belief propagation
- **Interactive Clarification**: Multi-turn dialogue for intent disambiguation
- **Dual User Modes**: 
  - Real human user study interface
  - LLM-based simulated user testing (GPT-4o-mini)
- **Explainability**: Belief progression tracking and visualization
- **Faithfulness Validation**: Verify explanations align with model behavior
- **ACC Curve Analysis**: Accuracy-Coverage-Burden tradeoffs for selective prediction
- **Multiple Datasets**: Banking77, CLINC150, SNIPS, ATIS support

---

## Project Structure

```
DS_Project/
├── src/                    # Core modular source code
│   ├── data/              # Dataset loaders and configurations
│   ├── models/            # Embeddings, Classifier, DS Mass Function
│   ├── agents/            # CustomerAgent (LLM simulation)
│   └── utils/             # Metrics, explainability, faithfulness, evaluation curves
│
├── scripts/               # Organized execution scripts
│   ├── training/          # train.py
│   ├── evaluation/        # evaluate.py, compute_thresholds.py
│   ├── user_study/        # run_user_study.py (real), run_simulated_user.py (LLM)
│   └── analysis/          # test_faithfulness.py, analyze_acc_curves.py
│
├── config/                # Configuration files
│   ├── hierarchies/       # Intent hierarchies and descriptions
│   └── thresholds/        # Optimal confidence thresholds
│
├── notebooks/             # Original Jupyter notebooks
├── docs/                  # Documentation
├── experiments/           # Training outputs (models, results)
└── outputs/              # Evaluation results and user study sessions
```

---

## System Architecture: Two-Phase Workflow

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                    PHASE 1: CONFIGURE OPTIMAL DS AGENT                          │
│                           (One-Time Setup)                                      │
└─────────────────────────────────────────────────────────────────────────────────┘

    ┌──────────────┐
    │  Banking77   │
    │   Dataset    │
    └──────┬───────┘
           │
           ▼
    ┌─────────────────────────────────────┐
    │  Step 1: Train Classifier           │
    │  scripts/training/train.py          │
    │                                     │
    │  • E5 embeddings (intfloat/e5-base)│
    │  • Logistic Regression / SVM       │
    │  • Output: trained_model.pkl       │
    └──────────────┬──────────────────────┘
                   │
                   ▼
    ┌─────────────────────────────────────┐
    │  Step 2: Evaluate (No Thresholds)  │
    │  scripts/evaluation/evaluate.py     │
    │                                     │
    │  DS Agent with DEFAULT thresholds:  │
    │  • Leaf: 0.3 | Parent: 0.5 | Root: 0.7│
    │  • Extract per-intent beliefs       │
    │  • Output: banking77_beliefs.csv    │
    └──────────────┬──────────────────────┘
                   │
                   ▼
    ┌─────────────────────────────────────┐
    │  Step 3: Compute Optimal Thresholds│
    │  scripts/evaluation/                │
    │  compute_thresholds.py              │
    │                                     │
    │  • Ancestor-aware ground truth      │
    │  • Test 101 threshold values (0-1)  │
    │  • Select best F1 per intent        │
    │  • Output: banking77_thresholds.json│
    └──────────────┬──────────────────────┘
                   │
                   │ ┌─────────────────────────────┐
                   └─► OPTIMAL DS AGENT CONFIG     │
                     │ (custom_thresholds)         │
                     └──────────────┬──────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                    PHASE 2: DS AGENT DEPLOYMENT                                 │
│                    (Interactive User Conversations)                             │
└─────────────────────────────────────────────────────────────────────────────────┘

                     ┌────────────────┐
                     │  User Query    │
                     │ "I lost my card"│
                     └───────┬────────┘
                             │
                             ▼
                ┌────────────────────────────┐
                │   DS Mass Function         │
                │   (with optimal thresholds)│
                └────────┬───────────────────┘
                         │
        ┌────────────────┼────────────────┐
        │                │                │
        ▼                ▼                ▼
┌─────────────┐  ┌─────────────┐  ┌─────────────────┐
│ Classifier  │  │ Hierarchy   │  │ Belief Tracker  │
│ Probabilities│  │ Structure   │  │ (Explainability)│
└──────┬──────┘  └──────┬──────┘  └────────┬────────┘
       │                │                   │
       └────────────────┴──────┬────────────┘
                               │
                               ▼
              ┌─────────────────────────────┐
              │  Initial Mass Function      │
              │  (from classifier probs)    │
              └──────────────┬──────────────┘
                             │
                             ▼
              ┌─────────────────────────────┐
              │  Evaluate Hierarchy         │
              │  • Compute beliefs          │
              │  • Check confidence ≥ θᵢ    │
              └──────────┬──────────────────┘
                         │
                    ┌────┴────┐
                    │         │
            Confident?    Not Confident?
                    │         │
                    │         ▼
                    │    ┌─────────────────────┐
                    │    │ Generate Clarification│
                    │    │ "Card Issues or      │
                    │    │  Payments?"          │
                    │    └──────────┬───────────┘
                    │               │
                    │               ▼
                    │    ┌─────────────────────┐
                    │    │  User Response      │
                    │    │  "It's about my card"│
                    │    └──────────┬───────────┘
                    │               │
                    │               ▼
                    │    ┌─────────────────────┐
                    │    │ Dempster's Rule     │
                    │    │ Combine masses      │
                    │    └──────────┬───────────┘
                    │               │
                    │               ▼
                    │    ┌─────────────────────┐
                    │    │ Record Belief       │
                    │    │ (Turn tracking)     │
                    │    └──────────┬───────────┘
                    │               │
                    └───────────────┘ (Loop: max 5 turns)
                         │
                         ▼
              ┌─────────────────────────────┐
              │  Final Prediction           │
              │  ("lost_or_stolen_card", 0.87)│
              └──────────────┬──────────────┘
                             │
                             ▼
              ┌─────────────────────────────┐
              │  Explainability Outputs     │
              │  • Belief progression graph │
              │  • Turn-by-turn history     │
              │  • Confidence scores        │
              └─────────────────────────────┘
```

### Phase 1: Configure Optimal DS Agent (One-Time Setup)
**Goal**: Create an optimally-configured DS agent with per-intent confidence thresholds

```bash
# Step 1: Train classifier on embeddings
python scripts/training/train.py --dataset banking77

# Step 2: Evaluate WITHOUT thresholds → Extract belief values for each query
python scripts/evaluation/evaluate.py \
    --dataset banking77 \
    --output-dir results/

# Output: banking77_beliefs.csv (per-intent belief values for threshold optimization)

# Step 3: Compute optimal thresholds using ancestor-aware ground truth
python scripts/evaluation/compute_thresholds.py \
    --belief-file results/banking77_beliefs.csv \
    --hierarchy-file config/hierarchies/banking77_hierarchy.json \
    --output-file config/thresholds/banking77_thresholds.json

# Output: banking77_thresholds.json (optimal confidence thresholds per intent)
```

### Phase 2: DS Agent Deployment (Interactive User Conversations)
**Goal**: Use the configured DS agent for interactive intent disambiguation

```python
# Load optimal thresholds
ds_calculator = DSMassFunction(
    intent_embeddings,
    hierarchy,
    classifier,
    custom_thresholds=load_thresholds("banking77_thresholds.json"),
    enable_belief_tracking=True  # ← Explainability ON
)

# For each user query
initial_mass = ds_calculator.compute_mass_function(user_query)
prediction = ds_calculator.evaluate_with_clarifications(initial_mass)
# ↑ Interactive clarification loop with belief tracking

# Access explainability
belief_history = ds_calculator.get_belief_tracker().get_history()
BeliefVisualizer.plot_belief_progression(belief_history)
```

---

## Quick Start

### 1. Setup Environment
```bash
# Clone and setup
git clone https://github.com/ksauka/DS_Project.git
cd DS_Project

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY
```

### 2. Test the System
The complete system workflow, including training, threshold computation, and interactive DS agent deployment, can be tested using:

```bash
jupyter notebook notebooks/system_workflow_demo.ipynb
```

This notebook demonstrates:
- Loading pre-trained models and optimal thresholds
- Interactive clarification with DS mass function evaluation
- LLM-based customer simulation
- Belief progression tracking and explainability

### Core Modules
- **DSMassFunction**: Dempster-Shafer reasoning engine with hierarchical belief propagation
- **IntentClassifier**: Logistic Regression or SVM classifier with E5 embeddings
- **CustomerAgent**: GPT-4o-mini powered simulated customer for automated testing
- **BeliefTracker**: Tracks and visualizes belief progression across clarification turns
- **FaithfulnessValidator**: Validates that explanations align with model predictions

### User Interaction Modes
1. **Real User Study** (`run_user_study.py`): Interactive terminal for actual human participants
2. **Simulated Testing** (`run_simulated_user.py`): Automated testing with LLM-based customer agent

### Analysis Tools
- **Faithfulness Validation**: Statistical tests for explanation reliability
- **ACC Curves**: Accuracy-Coverage-Burden analysis for selective prediction
- **Belief Visualization**: Progressive belief tracking across dialogue turns

---
```
Key dependencies (see `requirements.txt` for full list):
- `torch`, `transformers`, `sentence-transformers`
- `datasets`, `scikit-learn`, `scipy`
- `openai`, `python-dotenv`
- `pandas`, `numpy`, `matplotlib`, `seaborn`

Install:
```bash
pip install -r requirements.txt
```

---

## Configuration

- **Hierarchies**: Define intent hierarchies in `config/hierarchies/`
- **Thresholds**: Optimal per-intent thresholds in `config/thresholds/`
- **Environment**: API keys and settings in `.env` file

See `config/README.md` for detailed configuration options.

---


## Research Contributions

This system implements novel approaches in:
1. **Hierarchical DS Reasoning**: Multi-level belief propagation with adaptive thresholds
2. **Intent Disambiguation Explainability**: Belief progression tracking across clarification turns
3. **Faithfulness Validation**: Statistical tests for explanation reliability
4. **Selective Prediction Analysis**: ACC curve framework for uncertainty quantification

---

## License

This project is available for academic and research purposes.

---

## Citation

If you use this work in your research, please cite:
```bibtex
@software{ds_project_2026,
  title={Hierarchical Intent Disambiguation with Dempster-Shafer Theory},
  author={Kudzai Sauka & Krishna Manoorkar},
  year={2026},
  url={https://github.com/ksauka/DS_Project}
}
```


---

## Requirements

Dependencies are listed in `requirements.txt`. Key packages:
- `transformers`, `datasets`, `sentence-transformers`
- `spacy`, `scikit-learn`, `openai`
- `matplotlib`, `seaborn`, `tqdm`, `jsonpickle`

Install with:
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

---

## Troubleshooting

- **Dataset not loading?**
  - Confirm internet access and that `datasets` can access `contemmcm/clinc150`

- **No simulation results in evaluation?**
  - Ensure `logs/conversation_log.json` was generated and populated

- **Common fixes**:
  - Check folder paths in loaders
  - Set your OpenAI API key (`OPENAI_API_KEY`)
  - Install required dependencies and SpaCy models

---

# This is work in progress, still figuring out how to intergrate actual user  in place of simulated  customer agent
