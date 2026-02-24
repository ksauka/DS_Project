# GitHub Data Logging Configuration Guide

## Overview

The DS_Project now supports automatic saving of user study data to a private GitHub repository. This enables:
- Centralized data storage for all participant sessions
- Automatic backup to prevent data loss
- Easy access for analysis from anywhere
- Version-controlled data tracking

## Setup Steps

### 1. Use Existing Private GitHub Repository

This project uses the existing **hicxai-data-private** repository for data logging:

```bash
# Repository: https://github.com/ksauka/hicxai-data-private
# Status: Already created and configured
# Visibility: PRIVATE
# Note: Same repository used by hicxai-research project
```

### 2. Generate GitHub Personal Access Token

1. Go to GitHub Settings → Developer settings → Personal access tokens → Tokens (classic)
   - URL: https://github.com/settings/tokens
2. Click "Generate new token (classic)"
3. Set:
   - **Note**: "DS Project Data Logging"
   - **Expiration**: 90 days (or custom)
   - **Scopes**: Check `repo` (Full control of private repositories)
4. Click "Generate token"
5. **IMPORTANT**: Copy the token immediately (you won't see it again!)

### 3. Configure Environment Variables

#### For Local Development:

Add to `.env` file (create if doesn't exist):

```bash
# GitHub Data Logging
GITHUB_DATA_REPO="ksauka/hicxai-data-private"
GITHUB_DATA_TOKEN="ghp_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
```

#### For Streamlit Cloud Deployment:

1. Go to your app settings on Streamlit Cloud
2. Select "Secrets" section
3. Add:

```toml
# GitHub Data Logging
GITHUB_DATA_REPO = "ksauka/hicxai-data-private"
GITHUB_DATA_TOKEN = "ghp_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
```

### 4. Test Connection

Run the test script:

```bash
cd /home/kudzai/projects/DS_Project
python -c "
from src.utils.github_saver import test_github_connection
import os

repo = os.getenv('GITHUB_DATA_REPO')
token = os.getenv('GITHUB_DATA_TOKEN')

success, message = test_github_connection(token, repo)
print(message)
"
```

Expected output:
```
✅ Connected to ksauka/hicxai-data-private
```

## Data Structure

### Repository Organization

```
hicxai-data-private/
├── sessions/
│   ├── 2026-02-23/
│   │   ├── P123_experimental_20260223_103045.json
│   │   ├── P124_control_20260223_110520.json
│   │   └── ...
│   ├── 2026-02-24/
│   │   └── ...
│   └── ...
└── README.md
```

### Session File Structure

Each JSON file contains:

```json
{
  "metadata": {
    "participant_id": "P123",
    "condition": "experimental",
    "session_id": "abc123",
    "session_start": "2026-02-23T10:30:00",
    "session_end": "2026-02-23T11:15:00",
    "duration_seconds": 2700,
    "dataset": "banking77",
    "system": "DS_hierarchical_intent_classification"
  },
  "summary_statistics": {
    "total_queries": 100,
    "queries_correct": 92,
    "queries_incorrect": 8,
    "accuracy": 0.92,
    "total_clarifications": 130,
    "avg_clarifications_per_query": 1.3,
    "total_why_questions": 15,
    "total_interaction_time_seconds": 2570,
    "avg_time_per_query_seconds": 25.7,
    "avg_feedback_clarity": 4.2,
    "avg_feedback_confidence": 4.1
  },
  "query_results": [
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
      "conversation_transcript": "User: I want to transfer money\n...",
      "timestamp": "2026-02-23T10:30:00",
      "feedback_clarity": 5,
      "feedback_confidence": 4,
      "feedback_comment": "Very clear",
      "feedback_submitted": true
    }
  ],
  "final_feedback": {
    "overall_rating": 4,
    "trust": 4,
    "ease_of_use": 5,
    "would_recommend": "Probably",
    "additional_comments": "Great system!"
  },
  "behavior_metrics": {
    "total_queries": 100,
    "total_clarifications": 130,
    "total_why_questions": 15,
    "total_interaction_time": 2570.0,
    "queries_correct": 92,
    "queries_incorrect": 8
  }
}
```

## How It Works

1. **Session Start**: Data logger initialized with participant ID, condition, session ID
2. **During Study**: Each query result is logged (prediction, feedback, timing)
3. **"Why" Questions**: Tracked separately to measure explainability usage
4. **Study Completion**: Final feedback collected and entire session saved to GitHub
5. **Fallback**: If GitHub save fails, data is saved locally to `data/sessions/`

## Data Analysis

### Download All Session Data

Clone your private repository:

```bash
git clone https://github.com/ksauka/hicxai-data-private.git
cd hicxai-data-private
```

### Analyze with Python

```python
import json
import pandas as pd
from pathlib import Path

# Load all session files
sessions = []
for json_file in Path("sessions").rglob("*.json"):
    with open(json_file) as f:
        sessions.append(json.load(f))

print(f"Total sessions: {len(sessions)}")

# Convert to DataFrame for analysis
summary_data = [s['summary_statistics'] for s in sessions]
metadata = [s['metadata'] for s in sessions]

df_summary = pd.DataFrame(summary_data)
df_metadata = pd.DataFrame(metadata)

# Merge
df = pd.concat([df_metadata, df_summary], axis=1)

# Analyze by condition
print("\nAccuracy by Condition:")
print(df.groupby('condition')['accuracy'].mean())

print("\nAvg Clarifications by Condition:")
print(df.groupby('condition')['avg_clarifications_per_query'].mean())

# Extract all query-level data
all_queries = []
for session in sessions:
    for result in session['query_results']:
        result['participant_id'] = session['metadata']['participant_id']
        result['condition'] = session['metadata']['condition']
        all_queries.append(result)

df_queries = pd.DataFrame(all_queries)
print(f"\nTotal queries analyzed: {len(df_queries)}")
print(f"Overall accuracy: {df_queries['is_correct'].mean():.2%}")
```

## Security Considerations

1. **Never commit tokens to code**: Always use environment variables or Streamlit secrets
2. **Use private repository**: Participant data must be protected
3. **Token expiration**: Set reasonable expiration dates and rotate tokens
4. **Access control**: Only grant access to team members who need it
5. **Local fallback**: System saves locally if GitHub fails (prevents data loss)

## Troubleshooting

### "GitHub save failed"

**Possible causes:**
1. Invalid token → Regenerate token with `repo` scope
2. Wrong repository name → Check `GITHUB_DATA_REPO` format: `username/repo`
3. Repository doesn't exist → Create private repo on GitHub
4. Network issues → Data saved locally as fallback

### "Repository not found or no access"

**Solution:** Ensure repository is created and token has `repo` permissions

### "Local save failed"

**Solution:** Check write permissions for `data/sessions/` directory

## Best Practices

1. **Test before deployment**: Run test script to verify connection
2. **Monitor during study**: Check GitHub repo after first few participants
3. **Backup locally**: Keep local copies in addition to GitHub
4. **Regular commits**: GitHub API automatically commits each session
5. **Document conditions**: Use clear condition names (e.g., "control_v1", "experimental_v2")

## Privacy & Ethics

- Participant IDs should be anonymized (e.g., P001, P002) not real names
- Store only necessary data for research analysis
- Follow IRB guidelines and data retention policies
- Inform participants about data collection and storage
- Provide data deletion option if requested

## Support

For issues:
1. Check Streamlit logs for error messages
2. Verify environment variables are set correctly
3. Test GitHub connection with test script
4. Check local fallback files in `data/sessions/`
5. Verify repository permissions on GitHub
