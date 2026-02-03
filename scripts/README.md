# Scripts Directory

Organized execution scripts for the DS-based intent disambiguation system.

## Directory Structure

```
scripts/
├── training/           # Model training
│   └── train.py
├── evaluation/         # Model evaluation and threshold computation
│   ├── evaluate.py
│   └── compute_thresholds.py
├── user_study/         # User studies (real and simulated)
│   ├── run_user_study.py
│   ├── run_simulated_user.py
│   ├── compare_simulated_vs_real.py
│   └── select_user_study_queries.py
└── analysis/           # Analysis and validation
    ├── test_faithfulness.py
    └── analyze_acc_curves.py
```

## Usage

### Training
```bash
python scripts/training/train.py --dataset banking77
```

### Evaluation
```bash
python scripts/evaluation/evaluate.py --dataset banking77
python scripts/evaluation/compute_thresholds.py --belief-file results/beliefs.csv --output-file config/thresholds/banking77_thresholds.json
```

### User Studies
```bash
# Real user study
python scripts/user_study/run_user_study.py --model-path experiments/banking77/model.pkl ...

# Simulated users (LLM agent)
python scripts/user_study/run_simulated_user.py --study-queries queries.csv ...

# Compare results
python scripts/user_study/compare_simulated_vs_real.py --simulated-results sim.csv --real-results real.csv
```

### Analysis
```bash
python scripts/analysis/test_faithfulness.py --results-file results.csv
python scripts/analysis/analyze_acc_curves.py --results-file results.csv
```
