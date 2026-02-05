# DS_Project Streamlit Implementation Plan

## Executive Summary

**Goal:** Transform DS_Project from a CLI-based system into a modern, interactive Streamlit application with real-time visualizations, progress tracking, and an improved user study interface.

**Timeline:** 3 phases
- Phase 1: Core Streamlit structure & multi-page layout
- Phase 2: Real-time visualizations & belief progression UI
- Phase 3: Integrated user study with session management

---

## 📋 Current State Analysis

### ✅ Strengths
- **Well-architected source code**: Modular design (data, models, agents, utils)
- **Comprehensive belief tracking**: BeliefTracker + BeliefVisualizer already exist
- **Dual user modes**: Real human + LLM-simulated testing
- **Complete workflows**: Training → Evaluation → Threshold computation → Deployment
- **Type hints & logging**: Production-ready code quality

### 🔴 Weaknesses (High Priority)

#### 1. **CLI Limitations**
- Sequential user input, no visual feedback
- Results scattered across `outputs/` directory
- No progress bars for long operations
- Difficult to track multi-turn clarification progress

#### 2. **Missing Real-Time Feedback**
```
Current: User types query → Model processes → Text output (no belief visualization)
Needed: Interactive belief progression with sliders/metrics
```

#### 3. **Configuration Management Issues**
```
Current workflow: Manual JSON file creation → Path arguments in CLI
Issues:
  - Easy to lose track of threshold configs
  - Can't switch datasets mid-session
  - Thresholds computed but hard to visualize/compare
```

#### 4. **Session Management**
- No persistent session storage
- User study results scattered in CSV files
- Hard to download/share results

#### 5. **Error Handling**
- Minimal validation of user inputs
- Poor error messages for missing configs
- Silent failures in evaluation pipeline

---

## 🏗️ Proposed Architecture

### Multi-Page Streamlit App Structure

```
DS_Project/
├── src/
│   ├── streamlit_app/             # NEW: Streamlit-specific modules
│   │   ├── __init__.py
│   │   ├── app.py                 # Main Streamlit entry point
│   │   ├── pages/
│   │   │   ├── __init__.py
│   │   │   ├── 01_train.py        # Model training page
│   │   │   ├── 02_evaluate.py     # Evaluation & threshold computation
│   │   │   ├── 03_user_study.py   # Interactive user study
│   │   │   ├── 04_analysis.py     # Results analysis & visualization
│   │   │   └── 05_settings.py     # Configuration management
│   │   ├── components/
│   │   │   ├── __init__.py
│   │   │   ├── belief_viz.py      # Belief visualization components
│   │   │   ├── progress.py        # Progress tracking components
│   │   │   ├── session_manager.py # Session state management
│   │   │   └── file_utils.py      # File upload/download utilities
│   │   └── utils/
│   │       ├── __init__.py
│   │       ├── validators.py      # Input validation
│   │       └── formatters.py      # Output formatting
│   ├── models/                    # Existing modelsUnchanged
│   ├── data/                      # Existing data loaders (Unchanged)
│   ├── agents/                    # Existing agents (Unchanged)
│   └── utils/                     # Existing utils (Enhanced)
│       ├── explainability.py      # Keep existing
│       ├── faithfulness.py        # Keep existing
│       └── evaluation_curves.py   # Keep existing
│
├── requirements-streamlit.txt      # NEW: Streamlit-specific dependencies
├── streamlit_config.toml           # NEW: Streamlit configuration
├── app_main.py                    # NEW: Entry point (python -m streamlit run app_main.py)
└── .streamlit/
    └── secrets.toml               # NEW: Secrets management (gitignored)
```

### Page Workflow

```
┌─────────────────────────────────┐
│  🏠 Home Page (Settings)        │
│  • Dataset selection             │
│  • Model configuration           │
│  • Quick start buttons           │
└────────────┬────────────────────┘
             │
     ┌───────┴──────┬──────────────┬───────────────┐
     │              │              │               │
     ▼              ▼              ▼               ▼
┌─────────┐  ┌──────────┐  ┌──────────────┐  ┌──────────┐
│1️⃣ Train │  │2️⃣ Evaluate│ │3️⃣ User Study │  │4️⃣ Analysis│
│         │  │          │  │              │  │          │
│• Select │  │• Run eval│  │• Real user   │  │• ACC     │
│  dataset│  │• Compute │  │• Simulated   │  │  curves  │
│• Train  │  │  optimal │  │• Intent &    │  │• Belief  │
│  model  │  │  threshold│ │  confidence  │  │  trends  │
│• Progress│ │• Visualize│ │  tracking    │  │• Export  │
└─────────┘  │  thresholds│ │• Real-time   │  │  results │
             │• Download │  │  belief viz  │  │          │
             │  results  │  │• Save session│  └──────────┘
             └──────────┘  │              │
                          └──────────────┘
```

---

## 📦 Streamlit Dependencies (New)

```  file="requirements-streamlit.txt"
streamlit>=1.28.0
streamlit-plotly-events>=0.0.6        # Interactive plot handling
streamlit-lottie>=0.0.5               # Animations
plotly>=5.17.0                        # Interactive visualizations
altair>=5.0.0                         # Declarative visualization
st-aggrid>=1.0.0                      # Enhanced data grid

# Keep all existing dependencies from requirements.txt
```

---

## 🎨 Key UI Components

### 1. **Belief Visualization Component**
```python
def plot_belief_progression(belief_history, show_uncertainty=True):
    """
    Interactive belief progression plot showing:
    - Main intent belief curves over turns
    - Uncertainty bands (Pl - Bel)
    - Confidence thresholds as horizontal lines
    - Hover information
    """
    # Returns Plotly figure ready for st.plotly_chart()
```

### 2. **Threshold Visualization & Optimization**
```python
def plot_threshold_optimization(beliefs_df, intent_name):
    """
    Shows:
    - Histogram of beliefs for single intent
    - F1 scores at different threshold values
    - Recommended optimal threshold (marked)
    """
```

### 3. **Multi-Turn Conversation UI**
```python
def render_conversation(messages: List[Dict]):
    """
    Displays messages in chat-like format:
    - User queries on right
    - Agent responses with belief state on left
    - Clarification suggestions as buttons
    """
```

### 4. **Session Management**
```python
def manage_session():
    """
    Streamlit session state handler:
    - Persist DS agent across page reloads
    - Track conversation history
    - Save results to outputs/ automatically
    """
```

---

## 🔧 Implementation Steps

### Phase 1: Core Structure (2-3 days)

**Step 1.1**: Create Streamlit app skeleton
- `app_main.py` with page routing
- `pages/01_train.py` - Training interface
- `pages/05_settings.py` - Configuration management
- Session state initialization

**Step 1.2**: Refactor DS agent for Streamlit reuse
- Create `StreamlitDSAgent` wrapper (inherits from `DSMassFunction`)
- Add `.get_session_state()` method
- Add `.save_session(path)` / `.load_session(path)` methods

**Step 1.3**: Add file upload/download utilities
- Config file upload (hierarchy, intents, thresholds)
- Download trained models & results
- Session export/import

---

### Phase 2: Visualizations (3-4 days)

**Step 2.1**: Implement belief visualization
- Enhanced `BeliefVisualizer` for Streamlit (returns Plotly figures)
- Real-time belief progression during clarification
- Use `st.plotly_chart(use_container_width=True)`

**Step 2.2**: Create threshold analysis page
- ACC curve visualization
- Threshold comparison across datasets
- Optimal threshold recommendation

**Step 2.3**: Build progress tracking
- Training progress bars
- Evaluation progress with ETA
- Query processing progress in user study

---

### Phase 3: Interactive User Study (2-3 days)

**Step 3.1**: Real-time user interface (`pages/03_user_study.py`)
```
┌────────────────────────────────┐
│ Current Query (Turn 1)          │
│ ─────────────────────────────   │
│ "I need to activate my card"    │
│                                 │
│ Predicted Intent:               │
│ • activate_card (Bel: 0.75)     │
│ • lost_card (Bel: 0.10)         │
│                                 │
│ [View Belief Progression ▼]    │
│                                 │
│ Confidence: 0.75 (High ✓)      │
│                                 │
│ Clarification Generated:        │
│ "Are you having trouble with    │
│  a new card or existing card?"  │
│                                 │
│ ┌────────────────────────────┐  │
│ │ New Card | Existing Card ½│  │
│ └────────────────────────────┘  │
│                                 │
│ 📊 [View Full Conversation ▼] │
└────────────────────────────────┘
```

**Step 3.2**: Session persistence
- Auto-save session state every N queries
- Pause/resume functionality
- Export session as downloadable JSON

**Step 3.3**: LLM simulated user integration
```python
def run_simulated_session(num_queries, num_turns):
    """
    Run LLM agent as simulated user
    - Select queries from dataset
    - Show as table with progress
    - Real-time belief updates
    - Export results
    """
```

---

## 🐛 Code Quality Improvements

### Priority 1: Bug Fixes (Do First)
1. **DSMassFunction.get_threshold()** bug
   - Currently returns different defaults than documented
   - Fix: Use consistent thresholds or clarify logic

2. **Embedding consistency** issue
   - Training vs inference prefix mismatch in old code
   - Status: Currently fixed in main code

3. **Missing validation**
   - No error handling for missing config files
   - Add: Graceful fallback + user-friendly error messages

### Priority 2: Refactoring (Extract to `src/streamlit_app/`)
1. Create `StreamlitDSAgent` class
2. Extract visualization logic to components
3. Add `SessionManager` for state persistence
4. Add comprehensive input validators

### Priority 3: Performance (Soon)
1. Cache embedding computations (use `@st.cache_resource`)
2. Lazy load models only when needed
3. Batch evaluate multiple queries efficiently

---

## 📊 Implementation Checklist

- [ ] **Phase 1: Core Structure**
  - [ ] Create `src/streamlit_app/` directory structure
  - [ ] Implement `app_main.py` with routing
  - [ ] Create `StreamlitDSAgent` wrapper class
  - [ ] Build settings/config page
  - [ ] Add file upload/download utilities
  - [ ] Test basic navigation

- [ ] **Phase 2: Visualizations**
  - [ ] Enhance `BeliefVisualizer` for Streamlit
  - [ ] Create threshold analysis page
  - [ ] Add real-time belief progression view
  - [ ] Build progress tracking components
  - [ ] Test visualization responsiveness

- [ ] **Phase 3: User Study**
  - [ ] Build interactive user study page
  - [ ] Implement session persistence
  - [ ] Integrate LLM simulated user
  - [ ] Add results export functionality
  - [ ] Full end-to-end testing

- [ ] **Quality Assurance**
  - [ ] Fix known bugs in DSMassFunction
  - [ ] Add comprehensive input validation
  - [ ] Improve error messages
  - [ ] Performance optimization (caching)
  - [ ] CSS/styling polish

---

## 🚀 Deployment Considerations

### Local Development
```bash
streamlit run app_main.py --logger.level=debug
```

### Production (Streamlit Cloud / Docker)
```dockerfile
FROM python:3.10-slim
WORKDIR /app
COPY requirements*.txt .
RUN pip install -r requirements-streamlit.txt
COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "app_main.py"]
```

### Environment Variables (`.streamlit/secrets.toml`)
```toml
[api]
openai_api_key = "sk-..."
anthropic_api_key = "..."  # Optional future use

[deployment]
github_token = "ghp_..."  # For saving results to GitHub
max_upload_size = 200    # MB
```

---

## 📈 Expected Improvements

| Metric | Current | After Streamlit |
|--------|---------|-----------------|
| **User Feedback Loop** | Manual input | Real-time buttons & sliders |
| **Progress Visibility** | None | Live progress bars |
| **Belief Visualization** | Static plots | Interactive, real-time charts |
| **Session Management** | Manual file tracking | Persistent, recoverable sessions |
| **Configuration Complexity** | CLI args + JSON files | UI forms |
| **Error Handling** | Silent failures | User-friendly notifications |
| **Result Export** | CSV only | JSON, CSV, PDF, Session replay |
| **Time to User Study** | 15 minutes setup | 2 minutes drag-and-drop |

---

## 🎯 Success Criteria

✅ **Phase 1 Complete When:**
- All pages load without errors
- Can upload config files via UI
- Session state persists across page changes

✅ **Phase 2 Complete When:**
- Belief progression shows in real-time
- Threshold analysis is interactive
- Visualizations are responsive

✅ **Phase 3 Complete When:**
- Full multi-turn user study functions
- Results can be downloaded
- LLM simulated user works reliably
- Can compare real vs simulated results

---

## 📚 References

- **HicXAI Pattern**: `/home/kudzai/projects/HicXAI_agent/src/app.py` (session management, feedback)
- **Streamlit Docs**: https://docs.streamlit.io/
- **Plotly Interactive Charts**: https://docs.plotly.com/python/
- **Session State**: https://docs.streamlit.io/library/api-reference/session-state

---

## 💡 Future Enhancements (Post-MVP)

1. **Multi-User Support**: JWT authentication, user profiles
2. **Collaborative Annotation**: Real-time collaborative labeling
3. **Model Comparison**: Side-by-side A/B testing
4. **Dataset Analytics**: Difficulty analysis, confusion matrices by intent
5. **Advanced Metrics**: Per-intent ROC curves, calibration plots
6. **Integration**: GitHub Actions for CI/CD, pre-commit hook validation
