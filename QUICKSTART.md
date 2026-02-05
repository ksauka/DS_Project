# Quick Start Guide

## ✅ System Ready!

All files are in place. Choose your testing mode below.

---

## 🚀 Complete System Test (Recommended)

Run **Phase 1** (Configuration) + **Phase 2** (LLM Simulation) in one command:

```bash
cd DS_Project
./scripts/test_complete_system.sh
```

**OR** (Python version):
```bash
python scripts/test_complete_system.py
```

**What it does**:
1. ✅ Trains classifier on Banking77 (~3 min)
2. ✅ Extracts belief values (100 samples, ~5 min)
3. ✅ Computes optimal thresholds (~2 min)
4. ✅ Runs LLM simulation (20 queries, ~5 min)
5. ✅ Selects queries for human study

**Total time**: ~15 minutes

---

## Installation (5 minutes)

### 1. Run Setup Script

```bash
chmod +x setup.sh
./setup.sh
```

This will:
- Create virtual environment
- Install all dependencies
- Create necessary directories
- Set up `.env` template

### 2. Configure Environment

Edit `.env` file:
```bash
nano .env
```

Add your OpenAI API key:
```
OPENAI_API_KEY=sk-your-actual-key-here
```

### 3. Prepare Configuration Files

You need two JSON files for your dataset:

**config/banking77_hierarchy.json** - Intent hierarchy structure
**config/banking77_intents.json** - Intent descriptions

See the `.example.json` files in `config/` for reference.

## Usage Examples

### Train a Model

```bash
# Activate environment
source venv/bin/activate

# Train
python train.py \
  --dataset banking77 \
  --classifier logistic \
  --hierarchy-file config/banking77_hierarchy.json \
  --intents-file config/banking77_intents.json
```

### Evaluate with DS Reasoning

```bash
python evaluate.py \
  --dataset banking77 \
  --model-path outputs/banking77_logistic_model.pkl \
  --hierarchy-file config/banking77_hierarchy.json \
  --intents-file config/banking77_intents.json
```

### Interactive Prediction

```bash
python run_user_study.py \
  --model-path outputs/banking77_logistic_model.pkl \
  --hierarchy-file config/banking77_hierarchy.json \
  --intents-file config/banking77_intents.json
```

## Migrating from Jupyter Notebook

### Step 1: Extract Configuration

From your notebook, save these dictionaries as JSON:

```python
# In your notebook
import json

# Save hierarchy
with open('config/banking77_hierarchy.json', 'w') as f:
    json.dump(hierarchy, f, indent=2)

# Save intent descriptions
with open('config/banking77_intents.json', 'w') as f:
    json.dump(hierarchical_intents, f, indent=2)

# Save trained model
import pickle
with open('outputs/banking77_model.pkl', 'wb') as f:
    pickle.dump(clf, f)
```

### Step 2: Use New System

```bash
# Evaluate existing model
python evaluate.py \
  --dataset banking77 \
  --model-path outputs/banking77_model.pkl \
  --hierarchy-file config/banking77_hierarchy.json \
  --intents-file config/banking77_intents.json
```

## Project Structure Overview

```
DS_Project/
├── src/              # All source code (modular)
├── config/           # Configuration files (JSON)
├── outputs/          # Trained models and results
├── experiments/      # Experiment outputs
├── train.py         # Train models
├── evaluate.py      # Evaluate with DS reasoning
├── run_user_study.py       # Real user study (interactive)
├── run_simulated_user.py   # Simulated user testing (LLM agent)
├── compare_simulated_vs_real.py  # Compare simulated vs real users
└── .env             # Environment variables (API keys)
```

## Key Features

✅ **Data-Agnostic**: Works with Banking77, CLINC150, SNIPS, ATIS
✅ **Secure**: API keys in `.env`, never committed
✅ **Modular**: Easy to extend and maintain
✅ **Production-Ready**: Follows PEP8, FAIR, DRY principles

## Getting Help

- Read `REFACTORING_SUMMARY.md` for detailed migration guide
- Check `config/README.md` for configuration file formats
- See example configs in `config/*.example.json`

## Common Issues

### API Key Not Found
```bash
# Make sure .env exists and contains:
OPENAI_API_KEY=sk-your-key-here
```

### Import Errors
```bash
# Make sure virtual environment is activated
source venv/bin/activate

# Reinstall if needed
pip install -r requirements.txt
```

### Configuration File Not Found
```bash
# Check file paths are correct
ls -la config/

# Use absolute paths if needed
python train.py --hierarchy-file /full/path/to/hierarchy.json
```

Enjoy your refactored, production-ready codebase! 🚀
