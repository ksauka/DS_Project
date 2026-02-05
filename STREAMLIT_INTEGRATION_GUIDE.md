# Streamlit Integration Guide

This document explains how the Streamlit app integrates with the existing DS_Project codebase.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    Streamlit Web Interface                       │
│  (src/streamlit_app/)                                            │
│  ├── app_main.py (Router)                                       │
│  ├── pages/ (UI Pages)                                          │
│  ├── components/ (Reusable UI Components)                       │
│  └── utils/ (Validators, Formatters)                            │
└────────────────────┬─────────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Adapter Layer (NEW)                           │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ StreamlitDSAgent (Wraps DSMassFunction)                  │  │
│  ├─ Adds session persistence                               │  │
│  ├─ Adds progress callbacks                                │  │
│  ├─ Adds caching for models/embeddings                     │  │
│  └─ Integrates with Streamlit session state                │  │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ StreamlitTrainer (Wraps training script)                │  │
│  ├─ Real-time progress updates                            │  │
│  ├─ Training callbacks for UI                             │  │
│  └─ Model saving/loading utilities                        │  │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ StreamlitEvaluator (Wraps evaluation script)            │  │
│  ├─ Threshold computation with callbacks                  │  │
│  ├─ Real-time metric updates                             │  │
│  └─ Results export utilities                             │  │
│  └──────────────────────────────────────────────────────────┘  │
└────────────────────┬─────────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────────┐
│             Existing DS_Project Core (UNCHANGED)                │
│                                                                  │
│  src/                                                            │
│  ├── models/         (Classifiers, Embeddings)                 │
│  │   ├── classifier.py                                         │
│  │   ├── embeddings.py                                         │
│  │   └── ds_mass_function.py        ← Main logic               │
│  │                                                              │
│  ├── data/           (Dataset loaders)                          │
│  │   ├── data_loader.py                                        │
│  │   └── dataset_config.py                                     │
│  │                                                              │
│  ├── agents/         (LLM-based features)                       │
│  │   └── customer_agent.py          ← For simulation            │
│  │                                                              │
│  └── utils/          (Existing utilities)                       │
│      ├── explainability.py           ← BeliefTracker            │
│      ├── faithfulness.py                                        │
│      └── evaluation_curves.py                                   │
│                                                                  │
│  scripts/            (Command-line scripts - unchanged)         │
│  ├── training/       → Wrapped by StreamlitTrainer             │
│  ├── evaluation/     → Wrapped by StreamlitEvaluator           │
│  └── user_study/     → Integrated into User Study page         │
└─────────────────────────────────────────────────────────────────┘
```

## Data Flow

### Training Workflow

```
Streamlit UI (Train Page)
  ↓
User clicks "Start Training"
  ↓
StreamlitTrainer (Adapter)
  ├─ Progress callbacks
  └─ Real-time logging
  ↓
Existing train.py (Scripts)
  ├─ DataLoader.load()
  ├─ Embeddings generation
  ├─ Classifier.train()
  └─ Model saved to experiments/
  ↓
Update Streamlit UI with results
```

### Inference Workflow

```
Streamlit UI (User Study Page)
  ↓
User enters query
  ↓
StreamlitSessionManager
  ├─ Load model from cache
  └─ Load hierarchy from config
  ↓
DSMassFunction (Existing)
  ├─ compute_mass_function()
  ├─ compute_belief_plausibility()
  └─ generate_clarification_question()
  ↓
Streamlit UI (Belief Visualization)
  ├─ Plot belief progression
  └─ Display metrics
  ↓
Save to session
```

## Key Integration Points

### 1. Model Loading & Caching

**Streamlit Side:**
```python
@st.cache_resource
def load_ds_agent(dataset, model_path, hierarchy_file):
    # Load cached DS agent
    ds_agent = StreamlitDSAgent(...)
    return ds_agent
```

**Usage:**
```python
ds_agent = load_ds_agent(dataset, model_path, hierarchy_file)
belief = ds_agent.predict(query)
```

### 2. Session State Persistence

**Streamlit Side:**
```python
# In app_main.py
if 'ds_agent' not in st.session_state:
    st.session_state.ds_agent = None
if 'conversation_history' not in st.session_state:
    st.session_state.conversation_history = []
```

**Usage in Pages:**
```python
ds_agent = st.session_state.ds_agent
conversation = st.session_state.conversation_history
```

### 3. Progress Callbacks

**Training Example:**
```python
def training_callback(event, details):
    st.session_state.training_progress = details['progress']
    st.session_state.training_log = details['log']

trainer = StreamlitTrainer(
    dataset='banking77',
    progress_callback=training_callback
)
trainer.train()
```

### 4. Configuration Management

**Config Loading:**
```python
from src.streamlit_app.components.file_utils import list_config_files

hierarchies = list_config_files('hierarchy')
thresholds = list_config_files('thresholds')
```

**Config Saving:**
```python
from config.hierarchy_loader import load_hierarchy_from_json

hierarchy = load_hierarchy_from_json('config/banking77_hierarchy.json')
```

## Component Wrappers (Adapters)

### StreamlitDSAgent

Wraps `DSMassFunction` to work seamlessly with Streamlit:

```python
class StreamlitDSAgent:
    def __init__(self, intent_embeddings, hierarchy, classifier, ...):
        self.ds_function = DSMassFunction(...)
        self.session_manager = StreamlitSessionManager()
    
    def predict(self, query):
        belief = self.ds_function.compute_mass_function(query)
        # Cache result in session
        return belief
    
    def save_session(self):
        self.session_manager.save_session(...)
    
    def load_session(self, session_id):
        self.session_manager.load_session(session_id)
```

**To Create:** `src/streamlit_app/components/ds_agent_wrapper.py`

### StreamlitTrainer

Wraps training script with progress updates:

```python
class StreamlitTrainer:
    def __init__(self, dataset, progress_callback=None):
        self.dataset = dataset
        self.progress_callback = progress_callback
    
    def train(self):
        # Call existing train.py but with callbacks
        results = train_model(
            dataset=self.dataset,
            progress_callback=self.progress_callback
        )
        return results
```

**To Create:** `src/streamlit_app/components/trainer_wrapper.py`

### StreamlitEvaluator

Wraps evaluation script:

```python
class StreamlitEvaluator:
    def __init__(self, model_path, hierarchy_file, progress_callback=None):
        ...
    
    def evaluate(self):
        # Call existing evaluate.py with callbacks
        ...
    
    def compute_thresholds(self):
        # Call existing compute_thresholds.py with callbacks
        ...
```

**To Create:** `src/streamlit_app/components/evaluator_wrapper.py`

## File Organization

### New Files (Created for Streamlit)

```
src/streamlit_app/
├── __init__.py
├── pages/
│   ├── __init__.py
│   ├── home.py           ✅ Created
│   ├── train.py          ✅ Created
│   ├── evaluate.py       ✅ Created
│   ├── user_study.py     ✅ Created
│   ├── analysis.py       ✅ Created
│   └── settings.py       ✅ Created
├── components/
│   ├── __init__.py
│   ├── belief_viz.py     ✅ Created
│   ├── session_manager.py ✅ Created
│   ├── file_utils.py     ✅ Created
│   ├── progress.py       📝 TODO: Progress bar components
│   ├── ds_agent_wrapper.py 📝 TODO: DSMassFunction adapter
│   ├── trainer_wrapper.py  📝 TODO: Training adapter
│   └── evaluator_wrapper.py 📝 TODO: Evaluation adapter
└── utils/
    ├── __init__.py
    ├── validators.py     ✅ Created
    └── formatters.py     ✅ Created

.streamlit/
└── config.toml          ✅ Created

Root Level:
├── app_main.py          ✅ Created entry point
├── requirements-streamlit.txt ✅ Created
├── STREAMLIT_README.md  ✅ Created
└── STREAMLIT_IMPLEMENTATION_PLAN.md ✅ Created
```

### Existing Files (Unchanged)

```
src/
├── models/
│   ├── classifier.py           ✓ Use as-is
│   ├── embeddings.py           ✓ Use as-is
│   └── ds_mass_function.py     ✓ Use as-is
├── data/
│   ├── data_loader.py          ✓ Use as-is
│   └── dataset_config.py       ✓ Use as-is
├── agents/
│   └── customer_agent.py       ✓ Use as-is
└── utils/
    ├── explainability.py       ✓ Use as-is
    ├── faithfulness.py         ✓ Use as-is
    └── evaluation_curves.py    ✓ Use as-is

config/                         ✓ Use as-is
scripts/                        ✓ Use as-is
```

## Implementation Roadmap

### Phase 1: Core Structure ✅ DONE
- [x] Directory structure created
- [x] Pages created (home, train, evaluate, user_study, analysis, settings)
- [x] Components skeleton (belief_viz, session_manager, file_utils)
- [x] Utilities (validators, formatters)
- [x] Entry point (app_main.py)

### Phase 2: Adapter Wrappers 📝 TODO
- [ ] Create `StreamlitDSAgent` wrapper
- [ ] Create `StreamlitTrainer` wrapper
- [ ] Create `StreamlitEvaluator` wrapper
- [ ] Add progress tracking components
- [ ] Integrate with existing scripts

### Phase 3: Feature Implementation 📝 TODO
- [ ] Real training in Train page
- [ ] Real evaluation in Evaluate page
- [ ] Real inference in User Study page
- [ ] Interactive visualizations in Analysis page
- [ ] Session management integration

### Phase 4: Polish & Testing 📝 TODO
- [ ] UI refinement and styling
- [ ] End-to-end testing
- [ ] Error handling improvements
- [ ] Performance optimization
- [ ] Documentation updates

## Next Steps

### To implement Phase 2, create:

1. **`src/streamlit_app/components/ds_agent_wrapper.py`**
   - Wrap `DSMassFunction` for Streamlit
   - Add session persistence
   - Add caching

2. **`src/streamlit_app/components/trainer_wrapper.py`**
   - Wrap training script
   - Add progress callbacks
   - Integrate with UI

3. **`src/streamlit_app/components/evaluator_wrapper.py`**
   - Wrap evaluation script
   - Add threshold computation callbacks
   - Integrate metrics with UI

4. **`src/streamlit_app/components/progress.py`**
   - Reusable progress components
   - Training progress bar
   - Evaluation progress bar
   - Simulation progress bar

## Testing the Streamlit App

### Quick Test (Before Full Implementation)
```bash
streamlit run app_main.py
```

The UI will be fully navigable even without full backend integration. Pages will show placeholder content.

### After Phase 2 Implementation
```bash
# Should be able to:
# 1. Upload configs via Settings
# 2. Train models via Train page
# 3. Evaluate models via Evaluate page
# 4. Run inference via User Study page
```

## Notes for Developers

1. **Keep existing code unchanged** - All original `src/` code should work as-is
2. **Adapters are thin** - Wrappers only add Streamlit-specific features
3. **Caching is important** - Use `@st.cache_resource` for models/embeddings
4. **Session state is key** - All state should flow through `st.session_state`
5. **Callbacks are better than polling** - Use progress callbacks instead of checking status

## Troubleshooting Integration

### Model Not Loading
- Check `experiments/{dataset}/` directory
- Verify model path is correct
- Check file permissions

### Callbacks Not Firing
- Ensure wrapper is properly initialized
- Check callback function signature matches expected format
- Add logging to debug

### Session State Not Persisting
- Ensure state variables initialized in `app_main.py`
- Don't overwrite session_state directly, use update
- Check `.streamlit/config.toml` for session settings

## References

- [Main Documentation](README.md)
- [Implementation Plan](STREAMLIT_IMPLEMENTATION_PLAN.md)
- [Streamlit Docs](https://docs.streamlit.io/)
- [Session State API](https://docs.streamlit.io/library/api-reference/session-state)
