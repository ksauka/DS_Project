# DS Project - Streamlit User Study Interface

This Streamlit application provides an interactive web interface for **STEP 6** of the DS Project research workflow: **Real User Study with Human Participants**.

## Overview

The interface presents challenging queries (selected from STEP 5) to real human participants and tracks their interactions with the DS-based intent classification system through clarification dialogues.

## Quick Start

### 1. Launch the App

```bash
cd /home/kudzai/projects/DS_Project
streamlit run app_main.py
```

The app will open in your browser at `http://localhost:8501`

### 2. Navigate to User Study Page

Click **"User Study"** in the sidebar navigation.

### 3. Load Configuration

The configuration section will auto-expand on first visit:

- **Selected Queries CSV**: Queries from STEP 5 (default: `outputs/user_study/workflow_demo/selected_queries_for_user_study.csv`)
- **Trained Model**: Classifier from STEP 0 (default: `experiments/banking77/banking77_logistic_model.pkl`)
- **Hierarchy JSON**: Intent hierarchy (default: `config/hierarchies/banking77_hierarchy.json`)
- **Intents JSON**: Intent descriptions (default: `config/hierarchies/banking77_intents.json`)
- **Optimal Thresholds**: Thresholds from STEP 2 (default: `results/banking77/workflow_demo/banking77_optimal_thresholds.json`)
- **Output Directory**: Where to save results (default: `outputs/user_study/human_study`)

Click **"📥 Load Configuration"** to initialize the study.

### 4. Conduct User Study

**For each query:**

1. **Read the query** presented at the top
2. **Review belief progression** showing top 5 intent candidates
3. **Handle clarification** (if needed):
   - Read the chatbot's clarification question
   - Type your response as a natural language answer
   - Click **"✅ Submit Response"** to continue
4. **Validate final prediction**:
   - Review the predicted intent and confidence
   - Click **"✅ Correct Prediction"** or **"❌ Incorrect Prediction"**
   - Query automatically advances to the next one

**Optional Actions:**
- **"⏭️ Skip Clarification"**: Force prediction without answering
- **"⏩ Skip Query"**: Skip to next query without validation
- **"💾 Save Progress"**: Save current session data to disk
- **"🔄 Reset Session"**: Start over from query 1

### 5. Complete Session

When all queries are processed:
- **Session Summary** displays all results in a table
- **Download Session Data** button saves CSV with full interaction logs

## Features

### ✅ Real-Time Belief Tracking
- Top 5 intent candidates displayed with confidence scores  
- Belief values update after each clarification turn
- Color-coded metrics (High/Medium/Low confidence)

### 💬 Interactive Clarification
- Natural language responses (no multiple choice)
- Conversation history tracked for each query
- Clarification questions generated dynamically by DS agent

### 📊 Session Management
- Automatic progress tracking
- Mid-session save checkpoints
- CSV export with all interaction data

### 🔍 Transparency
- Ground truth visible (for validation)
- LLM interaction count shown (for comparison)
- Initial confidence score from STEP 3 displayed

## Output Format

Session CSV contains:

| Column | Description |
|--------|-------------|
| `query_idx` | Query number (0-based index) |
| `query` | Original user query text |
| `true_intent` | Ground truth intent label |
| `predicted_intent` | DS agent's final prediction |
| `confidence` | Final confidence score (0-1) |
| `is_correct` | Boolean: prediction matches ground truth |
| `num_clarifications` | Number of clarification turns |
| `conversation` | Full dialogue history (newline-separated) |
| `timestamp` | When validation occurred (ISO format) |

## Architecture

### Workflow Integration

```
STEP 0-2: Training & Configuration (Jupyter Notebook)
    ↓
STEP 3: LLM Simulation (Jupyter Notebook - Offline)
    ↓
STEP 5: Query Selection (Jupyter Notebook)
    ↓
STEP 6: Real User Study → STREAMLIT APP ← You are here!
    ↓
STEP 7: Comparison Analysis (Jupyter Notebook)
```

### Key Components

- **DSMassFunction**: Core belief computation and clarification logic
- **IntentEmbeddings**: E5-base embeddings for hierarchical intents
- **BeliefTracker**: Records belief progression for explainability
- **QuerySelector**: Already selected challenging queries in STEP 5

### Session State Variables

| Variable | Purpose |
|----------|---------|
| `study_queries_df` | DataFrame of selected queries |
| `current_query_idx` | Current query position (0-based) |
| `ds_agent` | Initialized DSMassFunction instance |
| `study_config` | Loaded file paths and metadata |
| `session_results` | List of completed query results |
| `current_conversation` | Dialogue history for active query |
| `current_mass` | Current belief mass distribution |

## Troubleshooting

### "Configuration file not found"
- Ensure you've run STEP 0-5 in the notebook first
- Check that file paths in configuration match your setup
- Default paths assume you're in project root: `/home/kudzai/projects/DS_Project`

### "No queries loaded"
- Run STEP 5 to generate `selected_queries_for_user_study.csv`
- Verify CSV exists at specified path
- CSV must have columns: `query`, `true_intent`

### "Model loading failed"
- Re-run STEP 0 to train the classifier
- Check model file path matches training output directory
- Model must be pickle-serialized LogisticRegression

### "Belief tracker returning None"
- Ensure `enable_belief_tracking=True` in DSMassFunction init
- This is set automatically by the app

## Development

### File Structure

```
src/streamlit_app/
├── pages/
│   ├── home.py              # Welcome page
│   ├── train.py             # Training interface (future)
│   ├── evaluate.py          # Evaluation interface (future)
│   ├── user_study.py        # ← Real user study (STEP 6)
│   ├── analysis.py          # Results visualization (future)
│   └── settings.py          # Configuration management (future)
├── components/
│   ├── belief_viz.py        # Plotly visualization functions
│   ├── session_manager.py  # Session persistence
│   └── file_utils.py        # File upload/download helpers
├── utils/
│   ├── validators.py        # Input validation
│   └── formatters.py        # Pretty-print helpers
└── README.md                # This file
```

### Extending the Interface

**To add belief progression charts:**
1. Use `components/belief_viz.py`
2. Call `plot_belief_progression(belief_tracker.get_history())`
3. Render with `st.plotly_chart(fig)`

**To customize clarification UI:**
1. Edit `user_study.py` around line 270
2. Replace `st.text_input()` with custom widgets
3. Conversation format in `st.session_state.current_conversation`

## References

- **Notebook Workflow**: [notebooks/system_workflow_demo.ipynb](../../notebooks/system_workflow_demo.ipynb)
- **QUICKSTART Guide**: [QUICKSTART.md](../../QUICKSTART.md)
- **DS Mass Function**: [src/models/ds_mass_function.py](../../src/models/ds_mass_function.py)
- **Query Selector**: [src/utils/query_selector.py](../../src/utils/query_selector.py)

## Citation

If you use this interface in your research, please cite our work on Dempster-Shafer based intent clarification with hierarchical reasoning.
