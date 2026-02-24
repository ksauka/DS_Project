# Qualtrics/Prolific Integration Guide

## Overview

The Streamlit app now supports seamless integration with Qualtrics surveys and Prolific participant tracking. This enables:

1. **Per-query feedback collection** - Users rate clarity and confidence after each query
2. **Final survey** - Overall experience rating at study completion
3. **Automatic redirect** - Return to Qualtrics/Prolific after completion
4. **Participant tracking** - Track participant IDs and experimental conditions

## URL Parameters

The app accepts the following URL parameters:

- `pid` - Participant ID (custom identifier)
- `PROLIFIC_PID` - Prolific participant ID (standard Prolific parameter)
- `cond` - Experimental condition (e.g., "control", "experimental", "A", "B")
- `return` - URL-encoded Qualtrics return URL

## Example URL

```
https://your-app.streamlit.app/?pid=P123&cond=experimental&return=https%3A%2F%2Fsurvey.qualtrics.com%2Fjfe%2Fform%2FSV_ABC123%3Fpid%3DP123%26done%3D1
```

## URL Parameter Breakdown

- `pid=P123` - Participant identifier
- `cond=experimental` - Experimental condition
- `return=...` - URL-encoded Qualtrics return URL
  - Decoded: `https://survey.qualtrics.com/jfe/form/SV_ABC123?pid=P123&done=1`

## Qualtrics Survey Setup

### Step 1: Create Survey Flow

1. In Qualtrics, go to **Survey Flow**
2. Add **Embedded Data** at the top:
   - `pid` (participant ID)
   - `cond` (condition)
   - `done` (completion flag, default: 0)

### Step 2: Add Web Service Element

Add a **Web Service** element with Streamlit URL:

```
https://your-app.streamlit.app/?pid=${e://Field/pid}&cond=${e://Field/cond}&return=${e://Field/Q_URL}
```

### Step 3: Set Return URL

The app will automatically append `done=1` to the return URL when complete.

## Prolific Integration

When recruiting via Prolific:

1. Use the standard completion URL: `https://app.prolific.com/submissions/complete?cc=XXXXXX`
2. Prolific automatically adds `PROLIFIC_PID` parameter
3. The app captures and stores this ID in feedback data

### Example Prolific Flow

**Study Link in Prolific:**
```
https://survey.qualtrics.com/jfe/form/SV_ABC123?pid={{%PROLIFIC_PID%}}&cond=A
```

**Qualtrics redirects to Streamlit with:**
```
https://your-app.streamlit.app/?pid=5f1234567890abc&PROLIFIC_PID=5f1234567890abc&cond=A&return=...
```

## Data Collection

### Per-Query Feedback

After each query resolution, users provide:
- **Clarity rating** (1-5 stars): How clear was the assistant's response?
- **Confidence rating** (1-5 stars): How confident are you in the result?
- **Optional comment**: Any concerns or feedback

Stored in CSV with columns:
- `feedback_clarity`
- `feedback_confidence`
- `feedback_comment`
- `feedback_submitted`

### Final Survey

After completing all queries, users provide:
- **Overall rating** (1-5): Poor to Excellent
- **Trust** (1-5 stars): How much do you trust the system?
- **Ease of use** (1-5 stars): How easy was it to use?
- **Recommendation** (5 options): Would recommend? (Definitely to Definitely Not)
- **Additional comments** (text): Open feedback

### Saved Files

All data saved to `outputs/user_study/feedback/`:

1. **`session_{session_id}_final.json`** - Final survey responses
2. **`session_{session_id}_complete.json`** - Combined data:
   - Final feedback
   - All query results with per-query feedback
3. **`human_session_{session_id}_{timestamp}.csv`** - Downloadable CSV

### Data Fields

Each query result includes:
```json
{
  "session_id": "abc123",
  "query_index": 0,
  "query_text": "I want to transfer money",
  "true_intent": "transfer",
  "predicted_intent": "transfer",
  "confidence": 0.87,
  "num_clarification_turns": 1,
  "is_correct": true,
  "interaction_time_seconds": 23.5,
  "conversation_transcript": "User: I want to transfer money\nAssistant: ...",
  "timestamp": "2026-02-23T10:30:00",
  "participant_id": "P123",
  "condition": "experimental",
  "prolific_pid": "5f1234567890abc",
  "feedback_clarity": 5,
  "feedback_confidence": 4,
  "feedback_comment": "Very clear explanation",
  "feedback_submitted": true,
  "llm_predicted_intent": "transfer",
  "llm_num_interactions": 2,
  "llm_confidence": 0.85,
  "llm_was_correct": true
}
```

## Testing Locally

### Without Qualtrics (Local Development)

Simply run:
```bash
streamlit run src/streamlit_app/simple_banking_assistant.py
```

Users can complete the study and download CSV. No redirect button appears.

### With Simulated Qualtrics Parameters

```bash
streamlit run src/streamlit_app/simple_banking_assistant.py -- \
  --server.port 8501 \
  --server.address localhost
```

Then visit:
```
http://localhost:8501/?pid=TEST123&cond=test&return=https%3A%2F%2Fexample.com%2Fsurvey
```

## Deployment to Streamlit Cloud

1. Push code to GitHub
2. Deploy on Streamlit Cloud: https://share.streamlit.io/
3. Note your app URL: `https://your-app.streamlit.app/`
4. Use this URL in Qualtrics Web Service

### Environment Variables (Optional)

Set in Streamlit Cloud → Settings → Secrets:

```toml
OPENAI_API_KEY = "sk-..."
OPENAI_MODEL = "gpt-4o-mini"
TEMPERATURE = "0.6"
MAX_TOKENS = "400"
```

## User Study Workflow

1. **Participant clicks Qualtrics link** (from Prolific or email)
2. **Qualtrics captures participant ID** and assigns condition
3. **Qualtrics redirects to Streamlit app** with parameters
4. **User completes 100 banking queries** with per-query feedback
5. **User completes final survey** (overall experience)
6. **App redirects back to Qualtrics** with `done=1`
7. **Qualtrics records completion** and redirects to Prolific (if applicable)

## Troubleshooting

### "Return link missing or invalid"

**Cause:** No `return` parameter or non-Qualtrics domain

**Solution:** Ensure return URL includes `qualtrics.com` in domain

### Feedback form not advancing

**Cause:** Form not submitted or validation error

**Solution:** Check all fields are filled (comments are optional)

### Participant ID not captured

**Cause:** URL parameter misspelled or not passed

**Solution:** Use `pid` (custom) or `PROLIFIC_PID` (Prolific standard)

## Best Practices

1. **Test end-to-end flow** before launching study
2. **Pilot with 3-5 participants** to catch issues
3. **Monitor feedback files** during data collection
4. **Back up data regularly** from `outputs/user_study/feedback/`
5. **Use meaningful condition names** (e.g., "control_v1", "experimental_v2")

## Data Analysis

Load and analyze results:

```python
import pandas as pd
import json
from pathlib import Path

# Load all session data
feedback_dir = Path("outputs/user_study/feedback")
sessions = []

for file in feedback_dir.glob("*_complete.json"):
    with open(file) as f:
        data = json.load(f)
        sessions.append(data)

# Extract query-level data
all_queries = []
for session in sessions:
    for result in session['query_results']:
        all_queries.append(result)

df = pd.DataFrame(all_queries)

# Analysis examples
print(f"Total queries: {len(df)}")
print(f"Accuracy: {df['is_correct'].mean():.2%}")
print(f"Avg clarity: {df['feedback_clarity'].mean():.2f}/5")
print(f"Avg confidence: {df['feedback_confidence'].mean():.2f}/5")
print(f"Avg interactions: {df['num_clarification_turns'].mean():.2f}")

# By condition
condition_stats = df.groupby('condition').agg({
    'is_correct': 'mean',
    'feedback_clarity': 'mean',
    'num_clarification_turns': 'mean'
})
print(condition_stats)
```

## Support

For issues or questions:
- Check `outputs/user_study/feedback/` for saved data
- Review Streamlit logs for errors
- Test locally with simulated parameters first
