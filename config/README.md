# Configuration Files

This directory contains configuration files for hierarchical intent classification.

## File Types

### Hierarchy Files
Format: `{dataset}_hierarchy.json`

Defines the hierarchical structure of intents:
```json
{
  "ParentCategory": ["child1", "child2"],
  "child1": ["leaf_intent1", "leaf_intent2"],
  "leaf_intent1": []
}
```

### Intent Description Files
Format: `{dataset}_intents.json`

Maps intent names to human-readable descriptions:
```json
{
  "ParentCategory": "Description of this category",
  "leaf_intent1": "Specific description of this intent"
}
```

### Threshold Files
Format: `{dataset}_thresholds.json`

Optimal confidence thresholds per intent:
```json
{
  "leaf_intent1": {
    "threshold": 0.45,
    "f1_score": 0.89
  }
}
```

## Usage

Load these files using the config loaders:

```python
from config.hierarchy_loader import load_hierarchy_from_json
from config.threshold_loader import load_thresholds_from_json

hierarchy = load_hierarchy_from_json('config/banking77_hierarchy.json')
thresholds = load_thresholds_from_json('config/banking77_thresholds.json')
```
