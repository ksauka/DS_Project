# GitHub Data Logging - Quick Start

## What is this?

Automatically saves all user study sessions to a private GitHub repository for:
- Centralized data storage
- Easy access from anywhere
- Automatic backups
- Version control

## 5-Minute Setup

### 1. Use Existing Private Repository

This project uses the existing **hicxai-data-private** repository:
- **Repository**: `ksauka/hicxai-data-private`
- **URL**: https://github.com/ksauka/hicxai-data-private
- **Status**: Already created ✓
- **Note**: Same repo used by hicxai-research project

### 2. Generate Token

Go to https://github.com/settings/tokens and:
1. Click "Generate new token (classic)"
2. Name it "DS Project Data"
3. Check `✓ repo` (Full control of private repositories)
4. Generate and **COPY THE TOKEN**

### 3. Configure

Add to `.env`:
```bash
GITHUB_DATA_REPO="ksauka/hicxai-data-private"
GITHUB_DATA_TOKEN="ghp_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
```

### 4. Test

```bash
python test_github_logging.py
```

Expected output:
```
✅ Connected to ksauka/hicxai-data-private
✅ TEST SESSION SAVED SUCCESSFULLY!
```

## How It Works

1. Participant completes study in Streamlit app
2. After final feedback is submitted, data is automatically saved to GitHub
3. Files organized as: `sessions/2026-02-23/P123_experimental_20260223_103045.json`
4. If GitHub fails, data saves locally to `data/sessions/` as backup

## Streamlit Cloud Deployment

Add to your app's secrets (Settings → Secrets):
```toml
GITHUB_DATA_REPO = "ksauka/hicxai-data-private"
GITHUB_DATA_TOKEN = "ghp_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
```

## Troubleshooting

**"Repository not found"**
- Ensure repository exists and is spelled correctly
- Check token has `repo` scope

**"Local save only"**
- Check internet connection
- Verify token is valid and not expired

**No errors but want to verify**
- Check your GitHub repo: `https://github.com/ksauka/hicxai-data-private`
- Look for `sessions/` folder with JSON files

## Full Documentation

See [GITHUB_LOGGING_SETUP.md](GITHUB_LOGGING_SETUP.md) for complete details.
