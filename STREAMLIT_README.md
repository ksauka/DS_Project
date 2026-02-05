# DS_Project Streamlit Application

A modern, interactive web interface for hierarchical intent classification using Dempster-Shafer Theory.

## Quick Start

### 1. Install Dependencies

```bash
# Install all requirements including Streamlit
pip install -r requirements.txt
pip install -r requirements-streamlit.txt

# OR with Poetry (if using esd_platform style)
pip install streamlit>=1.28.0 plotly>=5.17.0 altair st-aggrid
```

### 2. Configure Environment

```bash
# Copy example environment file
cp .env.example .env

# Edit .env with your OpenAI API key (for LLM simulated users)
nano .env
```

Add:
```
OPENAI_API_KEY=sk-your-key-here
```

### 3. Run the App

```bash
# Start Streamlit app
streamlit run app_main.py

# App will open at http://localhost:8501
```

## Features

### 🏠 **Home Page**
- Project overview and introduction
- Quick start guide with step-by-step instructions
- Session status dashboard
- Example workflows

### ⚙️ **Settings Page**
- Dataset configuration (Banking77, CLINC150, SNIPS, ATIS, TOPv2)
- Model parameters (classifier type, embedding model)
- Threshold configuration
- Hierarchy upload and management
- API configuration for LLM features

### 🎯 **Train Page**
- Model training interface
- Classifier selection (Logistic Regression, SVM)
- Training progress tracking
- Real-time logging
- Model artifact management

### 📊 **Evaluate Page**
- Evaluate trained models
- Compute optimal confidence thresholds
- Threshold optimization strategies
- Performance metrics computation
- Download results

### 💬 **User Study Page**
- **Real User Mode**: Manual query input with real-time belief visualization
- **Simulated User Mode**: LLM-based user agent for batch testing
- **Batch Processing**: Upload CSV of queries for evaluation
- Conversation history tracking
- Session persistence

### 📈 **Analysis Page**
- Overall performance metrics dashboard
- ACC (Accuracy-Coverage-Burden) curve visualization
- Per-intent performance analysis
- Belief progression trends over dialogue turns
- Results export (JSON, CSV, PDF)

## Page Structure

```
app_main.py (Entry Point)
├── 🏠 Home (home.py)
├── 🎯 Train (train.py)
├── 📊 Evaluate (evaluate.py)
├── 💬 User Study (user_study.py)
├── 📈 Analysis (analysis.py)
└── ⚙️ Settings (settings.py)
```

## Components

### Visualization Components (`components/belief_viz.py`)
- `plot_belief_progression()` - Interactive belief curve over dialogue turns
- `plot_threshold_visualization()` - F1 scores vs threshold values
- `plot_belief_comparison()` - Before/after clarification comparison
- `plot_acc_curves()` - Accuracy-Coverage-Burden tradeoff
- `render_conversation_ui()` - Chat-like conversation display

### Session Management (`components/session_manager.py`)
- `StreamlitSessionManager` - Persistent session storage
- `save_current_session()` - Auto-save to disk
- `load_session()` - Resume previous sessions
- Session list and recovery

### File Utilities (`components/file_utils.py`)
- `save_uploaded_hierarchy()` - Upload hierarchy JSON
- `save_uploaded_config()` - Upload config files
- `load_json_file()` - Parse JSON files
- `list_config_files()` - List available configs
- `get_model_paths()` - Find trained models

### Validators (`utils/validators.py`)
- `validate_hierarchy()` - Check hierarchy structure
- `validate_thresholds()` - Validate threshold values
- `validate_json_file()` - Parse and validate JSON
- `validate_dataset_name()` - Check dataset validity
- `validate_query()` - Validate user input

### Formatters (`utils/formatters.py`)
- `format_belief_for_display()` - Pretty-print beliefs
- `format_confidence()` - Confidence level indicator
- `format_uncertainty()` - Uncertainty indicator
- `format_metrics_table()` - Metrics as DataFrame
- `format_conversation_history()` - Readable conversation

## Configuration

### Streamlit Config (``.streamlit/config.toml`)
- Color theme customization
- Server settings (port, max upload size)
- Logger configuration

### Application Settings
- Dataset selection
- Model parameters
- Confidence thresholds (leaf, parent, root)
- Advanced options (belief tracking, max turns)

## Workflow Examples

### Example 1: Complete Pipeline (25 minutes)

```
1. Settings       → Select Banking77 dataset
2. Train          → Train new model (10 min)
3. Evaluate       → Compute thresholds (5 min)
4. User Study     → Run 20 test queries (10 min)
5. Analysis       → View results and curves
```

### Example 2: Quick Evaluation (5 minutes)

```
1. Settings       → Use existing model
2. Evaluate       → Check thresholds
3. User Study     → Test 1 query interactively
4. View belief progression in real-time
```

### Example 3: Batch Simulation (15 minutes)

```
1. Settings       → Configure LLM agent
2. User Study     → Launch 50-query simulation
3. Analysis       → View ACC curves and trends
```

## Session Management

### Save Session
- Click "Save Session" button in User Study page
- Auto-saves to `outputs/sessions/{session_id}.json`
- Includes all conversation history and beliefs

### Load Session
- Go to User Study → Session Management
- Select previous session to resume
- Continue where you left off

### Export Results
- Download as JSON (full session data)
- Download as CSV (tabular results)
- Download as PDF (formatted report)

## Advanced Usage

### Custom Hierarchy Upload
1. Create JSON file with intent hierarchy
2. Go to Settings → Hierarchy
3. Upload your `hierarchy.json`
4. System validates and saves

### Custom Thresholds
1. Run Evaluate page to compute optimal thresholds
2. Go to Settings → Model → Threshold Configuration
3. "Use custom thresholds per intent" checkbox
4. Upload computed `thresholds.json`

### LLM Simulation Configuration
1. Settings → Advanced → API Configuration
2. Enter OpenAI API key
3. Select model (gpt-4o-mini, gpt-4, etc.)
4. Adjust temperature for response randomness
5. User Study page now shows LLM agent option

## File Structure

```
outputs/
├── sessions/              # Saved sessions (JSON)
│   └── abc12345.json
├── user_study/
│   └── results.csv
└── evaluation/
    └── metrics.json

experiments/
└── banking77/
    ├── model.pkl          # Trained classifier
    ├── embeddings.pkl     # Cached embeddings
    └── metadata.json      # Training info

config/
├── hierarchies/
│   ├── banking77_hierarchy.json
│   └── banking77_intents.json
└── thresholds/
    └── banking77_thresholds.json
```

## Troubleshooting

### App Won't Start
```bash
# Clear cache
rm -rf ~/.streamlit/

# Check Python version (>=3.8 required)
python --version

# Reinstall dependencies
pip install --upgrade -r requirements-streamlit.txt
```

### Models Not Found
- Ensure you've run Training page first
- Check `experiments/{dataset}/` directory exists
- Training might still be in progress

### LLM Features Not Working
- Verify `OPENAI_API_KEY` in `.env`
- Check OpenAI account has credits
- Try with fewer queries first

### Threshold Computation Takes Too Long
- Reduce number of test samples in Evaluate page
- Reduce threshold values to test (21 instead of 101)
- Use Logistic Regression instead of SVM

### Session Won't Load
- Check `outputs/sessions/` directory exists
- Verify session JSON file is valid
- Try with fresh session

## Performance Tips

1. **Faster Training**: Use Logistic Regression + fewer samples
2. **Faster Evaluation**: Reduce test samples + fewer thresholds
3. **Faster Simulation**: Use fewer queries, fewer turns
4. **Memory Usage**: Don't keep too many sessions, clear old ones

## Development

### Adding New Pages
1. Create new file in `src/streamlit_app/pages/`
2. Implement `run()` function
3. Add to navigation in `app_main.py`

### Adding Visualizations
1. Create functions in `src/streamlit_app/components/belief_viz.py`
2. Return Plotly Figure objects
3. Use `st.plotly_chart()` to display

### Extending Validators
1. Add validation function to `src/streamlit_app/utils/validators.py`
2. Return `(is_valid, message)` tuple
3. Call in relevant page

## API Reference

### Session Manager
```python
from src.streamlit_app.components.session_manager import StreamlitSessionManager

mgr = StreamlitSessionManager()
mgr.save_session("session_id", data_dict)
session_data = mgr.load_session("session_id")
```

### Belief Visualization
```python
from src.streamlit_app.components.belief_viz import plot_belief_progression

fig = plot_belief_progression(belief_history)
st.plotly_chart(fig, use_container_width=True)
```

### Validators
```python
from src.streamlit_app.utils.validators import validate_query

is_valid, message = validate_query(user_input)
if not is_valid:
    st.error(message)
```

## Resources

- [Streamlit Documentation](https://docs.streamlit.io/)
- [Plotly Python Documentation](https://docs.plotly.com/python/)
- [Project README](README.md)
- [Implementation Plan](STREAMLIT_IMPLEMENTATION_PLAN.md)

## License

Same as main DS_Project

## Authors

Original DS_Project + Streamlit Interface Integration
