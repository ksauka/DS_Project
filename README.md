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
    │  Step 0: Train Vanilla Baseline     │
    │  scripts/training/train.py          │
    │                                     │
    │  • E5 embeddings (intfloat/e5-base)│
    │  • Logistic Regression / SVM       │
    │  • No hierarchy, no DS reasoning   │
    │  • Output: trained_model.pkl       │
    └──────────────┬──────────────────────┘
                   │
                   ▼
    ┌─────────────────────────────────────┐
    │  Step 1: Extract Belief Values      │
    │  scripts/ds_preparation/            │
    │  extract_beliefs.py                 │
    │                                     │
    │  • DS with DEFAULT thresholds       │
    │  • Compute beliefs for all intents  │
    │  • No clarifications triggered      │
    │  • Output: banking77_beliefs.csv    │
    └──────────────┬──────────────────────┘
                   │
                   ▼
    ┌─────────────────────────────────────┐
    │  Step 2: Compute Optimal Thresholds│
    │  scripts/ds_preparation/            │
    │  compute_thresholds.py              │
    │                                     │
    │  • Ancestor-aware ground truth      │
    │  • Test multiple threshold values   │
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
# Step 0: Train vanilla baseline classifier
python scripts/training/train.py --dataset banking77

# Step 1: Extract belief values using DEFAULT thresholds
python scripts/ds_preparation/extract_beliefs.py \
    --dataset banking77 \
    --output-dir results/

# Output: banking77_beliefs.csv (per-intent belief values for threshold optimization)

# Step 2: Compute optimal thresholds using ancestor-aware ground truth
python scripts/ds_preparation/compute_thresholds.py \
    --belief-file results/banking77_beliefs.csv \
    --hierarchy-file config/hierarchies/banking77_hierarchy.json \
    --output-file config/thresholds/banking77_thresholds.json

# Output: banking77_thresholds.json (optimal confidence thresholds per intent)
```

### Phase 2: DS Agent Deployment (Interactive User Conversations)
**Goal**: Use the configured DS agent for interactive intent disambiguation

#### Option A: Streamlit Interactive Interface (Recommended)
```bash
# Start the Streamlit app
python -m streamlit run src/streamlit_app/simple_banking_assistant.py

# Opens at: http://localhost:8501
```

**Features**:
- **Auto-Starting Queries**: Each session auto-starts with a sample banking query
- **Multi-Turn Clarifications**: Chat-based dialogue for intent disambiguation
- **Real-Time Belief Tracking**: Enabled by default with `enable_belief_tracking=True`
- **Belief Visualization**: Charts show top-5 intents over clarification turns
- **Explanation on Demand**: Type `why` to see belief-based reasoning

**User Workflow**:
1. Query auto-loads (e.g., "How do I activate my card?")
2. Respond to clarification questions (if confidence is low)
3. Once resolved, type `why` to see:
   - **Final Decision**: The predicted intent
   - **Belief Score**: Confidence (0.0-1.0)
   - **Status**: ✅ Confident or ⚠️ Uncertain
   - **Belief Improvement**: How much each clarification helped
   - **Visualizations**: Charts showing belief progression

#### Option B: Programmatic API
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

# Install dependencies (core + Streamlit)
pip install -r requirements.txt
pip install -r requirements-streamlit.txt
```

### 2. Run the Interactive Streamlit App (Recommended)
```bash
# Start the interactive interface
python -m streamlit run src/streamlit_app/simple_banking_assistant.py

# Browser opens at http://localhost:8501
```

**First Time**:
- System auto-loads pre-trained model and optimal thresholds
- First query starts automatically
- Follow on-screen prompts or type `why` to see belief reasoning

**Commands**:
- Press Enter or type `next` → Move to next query
- Type `why` → See belief-based explanation with visualization
- Type any text → Respond to clarification question

### 3. Jupyter Notebook (Full Research Workflow)
```bash
jupyter notebook notebooks/system_workflow_demo.ipynb
```

Includes: Training, threshold computation, LLM simulation, query selection, and comparison analysis

---

## Core Components

- **DSMassFunction**: DS reasoning engine with hierarchical belief propagation
- **IntentClassifier**: Logistic Regression or SVM with E5 embeddings  
- **BeliefTracker**: Records belief progression across clarification turns
- **BeliefVisualizer**: Matplotlib charts for belief evolution
- **CustomerAgent**: LLM-based user simulator for testing

---

## Configuration

- **Hierarchies**: `config/hierarchies/` - Intent structures
- **Thresholds**: `config/thresholds/` - Optimal per-intent confidence values
- **Environment**: `.env` - API keys (for LLM simulation)

---

## Citation

```bibtex
@software{ds_project_2026,
  title={Hierarchical Intent Disambiguation with Dempster-Shafer Theory},
  author={Kudzai Sauka & Krishna Manoorkar},
  year={2026},
  url={https://github.com/ksauka/DS_Project}
}
```
