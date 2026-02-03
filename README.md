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

### 2. Train Model
```bash
python scripts/training/train.py \
    --dataset banking77 \
    --output-dir experiments/banking77
```

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

## Advanced Usage

### Compute Optimal Thresholds
```bash
# After evaluation generates belief values
python scripts/evaluation/compute_thresholds.py \
    --belief-file results/banking77_beliefs.csv \
    --output-file config/thresholds/banking77_thresholds.json
```

### Test Faithfulness of Explanations
```bash
python scripts/analysis/test_faithfulness.py \
    --results-file results/banking77_results.csv \
    --belief-logs results/banking77_belief_logs.json
```

### Generate ACC Curves
```bash
python scripts/analysis/analyze_acc_curves.py \
    --results-file results/banking77_results.csv \
    --output-dir outputs/acc_analysis
```

### Compare Simulated vs Real Users
```bash
python scripts/user_study/compare_simulated_vs_real.py \
    --simulated-results outputs/user_study/sessions/simulated_session.csv \
    --real-results outputs/user_study/sessions/human_session.csv
```
    --hierarchy-file config/hierarchies/banking77_hierarchy.json \
    --intents-file config/hierarchies/banking77_intents.json
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
