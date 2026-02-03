# Optimal Thresholds

This directory contains optimal confidence thresholds computed for each intent.

## Files

- `banking77_thresholds.json` - Optimal thresholds for Banking77 dataset
- `clinc150_thresholds.json` - Optimal thresholds for CLINC150 dataset
- `snips_thresholds.json` - Optimal thresholds for SNIPS dataset
- `atis_thresholds.json` - Optimal thresholds for ATIS dataset

## How to Generate

Thresholds are computed from evaluation results:

```bash
# 1. Train model
python train.py --dataset banking77

# 2. Evaluate and save belief values
python evaluate.py --dataset banking77 --save-beliefs

# 3. Compute optimal thresholds
python compute_thresholds.py \
    --belief-file results/banking77_beliefs.csv \
    --output-file config/thresholds/banking77_thresholds.json

# 4. Use in future evaluations
python evaluate.py --dataset banking77 --custom-thresholds config/thresholds/banking77_thresholds.json
```

## Format

```json
{
  "intent_name": {
    "threshold": 0.45,
    "f1_score": 0.89
  },
  ...
}
```
