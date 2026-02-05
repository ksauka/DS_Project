# DS Project Architecture: LLM Simulation → Real User Validation

This document explains the complete workflow architecture, clarifying where automation (LLM) ends and human validation (Streamlit) begins.

## 🏗️ Two-Phase Architecture

### Phase 1: Automated Configuration & Testing (Jupyter Notebook)
**Purpose:** Train, configure, and identify problematic queries automatically

### Phase 2: Human Validation (Streamlit Interface)
**Purpose:** Validate system performance with real users on challenging queries

---

## 📊 Complete Workflow

```
┌─────────────────────────────────────────────────────────────────────┐
│                    JUPYTER NOTEBOOK (Automated)                      │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  STEP 0: Vanilla Baseline                                           │
│  ├─ Train LogisticRegression + E5 embeddings                        │
│  ├─ Evaluate on 3,080 test queries                                  │
│  └─ Accuracy: 87.66%, F1: 87.28%                                    │
│                                                                      │
│  STEP 1: Belief Extraction                                          │
│  ├─ Compute DS beliefs for all intents                              │
│  ├─ No threshold filtering yet                                      │
│  └─ Output: beliefs.csv (96 intents × 3,080 queries)                │
│                                                                      │
│  STEP 2: Threshold Computation                                      │
│  ├─ Analyze belief distributions per intent                         │
│  ├─ Find optimal threshold (maximize F1)                            │
│  └─ Output: 96 optimal thresholds                                   │
│                                                                      │
│  STEP 3: LLM Simulation (AUTOMATED)                                 │
│  ├─ Process ALL 3,080 test queries                                  │
│  ├─ CustomerAgent (GPT-4o-mini) responds to clarifications          │
│  ├─ Full dialogue automation                                        │
│  ├─ Metrics:                                                        │
│  │   • Accuracy with clarifications                                 │
│  │   • Avg clarification turns                                      │
│  │   • Low/medium/high confidence queries                           │
│  └─ Output: ds_evaluation/banking77_predictions.csv                 │
│                                                                      │
│  STEP 5: Query Selection (INTELLIGENT FILTERING)                    │
│  ├─ Analyze STEP 3 results                                          │
│  ├─ Identify problematic queries:                                   │
│  │   • High interactions: 205 queries (≥2 clarifications)           │
│  │   • Low confidence: 2,741 queries (≤0.7 confidence)              │
│  │   • Incorrect: 330 queries (wrong prediction)                    │
│  │   • Problematic: 142 queries (high interaction + incorrect)      │
│  │   • Uncertain: 272 queries (low confidence + incorrect)          │
│  ├─ Select balanced sample: 91 challenging queries                  │
│  │   • Avg clarifications: 2.07                                     │
│  │   • Avg confidence: 0.3897                                       │
│  │   • LLM accuracy: 24.18% (very challenging!)                     │
│  └─ Output: selected_queries_for_user_study.csv                     │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
                                   ↓
                    ┌──────────────────────────────┐
                    │   HAND-OFF TO REAL USERS     │
                    └──────────────────────────────┘
                                   ↓
┌─────────────────────────────────────────────────────────────────────┐
│                    STREAMLIT APP (Human Interface)                   │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  STEP 6: Real User Study (MANUAL VALIDATION)                        │
│  ├─ Load 91 selected queries from STEP 5                            │
│  ├─ Human participant replaces LLM CustomerAgent                    │
│  ├─ Query-by-query interface:                                       │
│  │   1. Display query                                               │
│  │   2. Show belief progression (top 5 intents)                     │
│  │   3. Ask clarification (if needed)                               │
│  │   4. Human types natural language response                       │
│  │   5. Update beliefs with human response                          │
│  │   6. Repeat until confident                                      │
│  │   7. Human validates: correct/incorrect                          │
│  ├─ Session tracking:                                               │
│  │   • Conversation history per query                               │
│  │   • Belief progression visualization                             │
│  │   • Progress: completed/remaining                                │
│  │   • Save checkpoints                                             │
│  └─ Output: human_study/session_YYYYMMDD_HHMMSS.csv                 │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
                                   ↓
┌─────────────────────────────────────────────────────────────────────┐
│                    JUPYTER NOTEBOOK (Analysis)                       │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  STEP 7: Comparison Analysis                                        │
│  ├─ Compare three approaches:                                       │
│  │   • Vanilla baseline (STEP 0)                                    │
│  │   • LLM simulation (STEP 3) on challenging queries               │
│  │   • Real human users (STEP 6) on same queries                    │
│  ├─ Metrics:                                                        │
│  │   • Accuracy improvement                                         │
│  │   • Clarification effectiveness                                  │
│  │   • Human vs LLM performance gap                                 │
│  └─ Output: Comparison reports and visualizations                   │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 🔑 Key Design Decisions

### Why LLM Simulation First? (STEP 3)

1. **Scale**: Process all 3,080 queries automatically
2. **Speed**: Complete in ~2 hours vs weeks with humans
3. **Cost**: ~$5-10 in API costs vs $500+ for human annotators
4. **Identification**: Automatically find edge cases
5. **Baseline**: Establish upper bound for automation

### Why Real Users After? (STEP 6)

1. **Validation**: Verify LLM results on hard cases
2. **Ground truth**: Humans are gold standard for ambiguous queries
3. **Efficiency**: Only 91 queries need human validation (3% of test set)
4. **Focus**: Human time spent on most valuable queries
5. **Research**: Compare human vs LLM clarification strategies

### Why Separate Interfaces?

| Aspect | Jupyter Notebook | Streamlit App |
|--------|------------------|---------------|
| **Audience** | Researchers, developers | Study participants, end users |
| **Purpose** | Experimentation, analysis | Data collection, interaction |
| **Workflow** | Sequential, documented | Interactive, user-friendly |
| **Output** | Metrics, plots, CSVs | Session logs, timestamps |
| **Reproducibility** | Code + markdown | Web interface, shareable URL |

---

## 📈 Results Interpretation

### STEP 3 LLM Results (Example from your run)

```
Total queries: 3,080
Queries with clarifications: varies by threshold
Average clarifications: ~0.5-2.0 turns
Accuracy: typically 85-92%
```

**LLM identifies:**
- 205 high-interaction queries (≥2 turns)
- 2,741 low-confidence queries (≤0.7)
- 330 incorrect predictions

### STEP 5 Selection (Your actual output)

```
Selected: 91 challenging queries
Avg clarifications: 2.07
Avg confidence: 0.3897
LLM accuracy: 24.18%  ← Very challenging!

Category distribution:
  high_interaction: 20
  low_confidence: 20
  incorrect: 19
  problematic: 18
  uncertain: 14
```

**Interpretation:** These 91 queries represent the **hardest 3% of the dataset** where:
- System is most uncertain
- Multiple clarifications are needed
- LLM fails most often (76% error rate!)

### STEP 6 Human Validation (Expected)

**Research questions:**
1. Do humans perform better than LLM on these hard queries?
2. How many clarifications do humans need vs LLM?
3. Are human responses more concise/informative?
4. Does belief progression differ with human input?

**Typical findings:**
- Human accuracy: 60-80% (vs LLM 24%)
- Human clarifications: 1.5-2.0 turns (similar to LLM)
- Human response quality: Higher information density
- System confidence: Increases faster with human input

---

## 🚀 Deployment Strategy

### Local Development

```bash
# Phase 1: Configure and identify queries
jupyter notebook notebooks/system_workflow_demo.ipynb
# Run cells 3, 5, 7, 9, 11, 14 (STEP 0-5)

# Phase 2: Collect human data
streamlit run app_main.py
# Navigate to "User Study" page
```

### Production Deployment (Streamlit Cloud)

```bash
# 1. Push to GitHub
git add .
git commit -m "Add user study interface"
git push origin main

# 2. Deploy on Streamlit Cloud
# - Visit https://share.streamlit.io
# - Connect GitHub repo
# - Set main file: app_main.py
# - Deploy

# 3. Share study URL with participants
# https://your-app-name.streamlit.app
```

### Research Reproducibility

**To replicate this study:**

1. **Code**: Clone GitHub repo
2. **Config**: Use provided hierarchy/intents files
3. **Data**: Download Banking77 from HuggingFace
4. **STEP 0-5**: Run Jupyter notebook cells
5. **STEP 6**: Launch Streamlit app locally OR use our public deployment
6. **STEP 7**: Analyze results in notebook

**Citation:**
```bibtex
@article{yourname2026ds,
  title={Hierarchical Intent Classification with Dempster-Shafer Theory},
  author={Your Name},
  journal={Conference/Journal},
  year={2026},
  note={Code: github.com/your-repo, Demo: your-app.streamlit.app}
}
```

---

## 🔄 Iterative Workflow

### First Iteration
1. Run STEP 0-5 with default thresholds
2. Collect data from 5-10 pilot participants (STEP 6)
3. Analyze results (STEP 7)
4. Adjust thresholds if needed

### Subsequent Iterations
1. Modify thresholds in STEP 2
2. Re-run STEP 3 with new thresholds
3. Select different query set in STEP 5
4. Collect more human data (STEP 6)
5. Compare iterations in STEP 7

### Hyperparameter Tuning
- **Threshold values**: Leaf (0.3), Parent (0.5), Root (0.7)
- **Selection criteria**: min_interactions=2, max_confidence=0.7
- **Query count**: 50-150 queries balance quality/quantity

---

## 💡 Best Practices

### For Researchers

1. **Always run STEP 3 first**: LLM simulation is cheap and fast
2. **Validate on subset**: Don't waste human time on easy queries
3. **Document everything**: Jupyter notebook is living documentation
4. **Version control**: Tag releases, track config changes
5. **Share deployments**: Streamlit Cloud for reproducibility

### For Study Participants

1. **Natural responses**: Type as you would in real conversation
2. **First instinct**: Don't overthink clarifications
3. **Save progress**: Use checkpoints for long sessions
4. **Report issues**: Use feedback form if interface breaks

### For Developers

1. **Keep interfaces separate**: Jupyter for research, Streamlit for users
2. **Maintain consistency**: Same model, thresholds, hierarchy across phases
3. **Test locally first**: Validate workflow before deploying
4. **Monitor costs**: LLM API calls can add up with large datasets

---

## 📞 Support

- **Notebook issues**: See [notebooks/system_workflow_demo.ipynb](notebooks/system_workflow_demo.ipynb)
- **Streamlit issues**: See [src/streamlit_app/README.md](src/streamlit_app/README.md)
- **User study guide**: See [STEP6_USER_STUDY_GUIDE.md](STEP6_USER_STUDY_GUIDE.md)
- **Quick start**: See [QUICKSTART.md](QUICKSTART.md)
