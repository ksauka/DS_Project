# DS_Project

A modular, extensible intent classification framework based on **Dempster-Shafer (DS) theory**, designed for robust decision-making in ambiguous or uncertain queries. This system simulates conversations between a GPT-powered customer agent and a DS-based assistant agent. The framework operates over the **Clinc150** dataset, supports multi-turn clarifications, and performs belief accumulation through a dynamic reasoning engine.

## Dempster-Shafer Intent Classification System

This project supports an end-to-end workflow from data loading, hierarchy generation, and keyword extraction to simulation, evaluation, and explainability. Built with modularity and reproducibility in mind.

---

## Project Structure

```
├── agents/                 # Agent implementations
│   ├── customer_agent.py   # GPT-powered simulated customer agent
│   └── ds_agent.py         # Dempster-Shafer assistant agent
├── config/
│   └── hierarchy_config.py # Loader for intent hierarchy and thresholds
├── conversation/           # Dialogue control and interaction
│   ├── dialog_manager.py   # Orchestrates multi-turn exchanges
│   ├── metrics.py          # Evaluation metrics collection
│   └── prompts.py          # Clarification prompt templates
├── data/
│   └── clinc_loader.py     # Dataset loading utilities for Clinc150
├── ds_core/                # Core logic for DS reasoning
│   ├── ds_mass.py          # Dempster-Shafer mass computation engine
│   ├── entity_mass.py      # Entity-based mass function
│   └── negation_handler.py # Negation-aware evidence adjustment
├── intents/                # Intent features
│   ├── embeddings.py       # Sentence-level embeddings for each intent
│   ├── keywords.py         # GPT-powered keyword extraction + boosting
│   └── matcher.py          # Entity and keyword extractor using SpaCy
├── metrics/
│   └── tracker.py          # Tracks results and logs performance
└── scripts/
    ├── evaluate_model.py   # Runs analysis of logs + metrics export
    └── run_simulation.py   # Simulation runner (multi-turn evaluation)
```

---

## Usage Examples

### 1. Run Simulations
```bash
python main.py --mode simulate --num_samples 100 --log_path logs/simulation.log
```
Simulates 100 conversations with real Clinc150 queries. Dialogue is managed by the dialog manager using DS reasoning.

### 2. Evaluate Results
```bash
python main.py --mode evaluate --log_path logs/simulation.log
```
Generates:
- `evaluation_report.json`
- `evaluation_report.csv`
- `confidence_accuracy_plot.png`

### 3. Test Data & Hierarchy Loading
```bash
python main.py --mode test --verbose
```
Use this to verify hierarchy, thresholds, embeddings, and entity mappings.

---

## Key Components

- **CustomerAgent**: Retrieves real queries from Clinc150 test set and simulates clarifications using GPT.
- **DSAgent**: Uses DS-theoretic evidence accumulation to classify intent; can ask follow-up questions.
- **DialogManager**: Controls the turn-based dialogue flow and termination criteria.
- **MetricsLogger**: Stores prediction history, clarification turns, and belief confidence per session.

---

## Configuration

Customize these files as needed:
- `config/hierarchy.py`: Loads intent hierarchy and optional thresholds, loads entity_to_intent.
- `data/clinc_loader.py`: Loads and processes the Clinc150 dataset.
- `intents/intent_generator.py`: GPT-based intent description and keyword generation.



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

# This is work in progress, I am still to figure out how to deal with intent map and intent label.
