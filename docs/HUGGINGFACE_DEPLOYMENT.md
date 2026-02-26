# Deploying DS_Project to Hugging Face Spaces

This guide walks you through deploying the DS Banking Assistant to Hugging Face Spaces for user studies.

## Prerequisites

1. **Hugging Face Account**: Create at [huggingface.co/join](https://huggingface.co/join)
2. **HF CLI**: Install Hugging Face CLI
   ```bash
   pip install huggingface_hub
   ```
3. **HF Token**: Get your token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
   - Needs `write` permission
4. **Git LFS**: Required for large model files
   ```bash
   # Ubuntu/Debian
   sudo apt-get install git-lfs
   
   # macOS
   brew install git-lfs
   ```

## Quick Start

### 1. Login to Hugging Face

```bash
huggingface-cli login
# Enter your token when prompted
```

### 2. Create Space on Hugging Face

Go to [huggingface.co/new-space](https://huggingface.co/new-space) and create:
- **Name**: `ds-banking-assistant` (or your preferred name)
- **SDK**: Docker
- **License**: MIT
- **Visibility**: Private (for anonymous research) or Public

### 3. Deploy

```bash
cd /home/kudzai/projects/DS_Project
bash scripts/deployment/deploy_to_huggingface.sh YOUR_USERNAME ds-banking-assistant
```

**For organization deployment:**
```bash
bash scripts/deployment/deploy_to_huggingface.sh your-org ds-banking-assistant
```

### 4. Add Secrets to Space

Go to your Space Settings → Variables and secrets:

**Required Secrets:**
- `OPENAI_API_KEY` - Your OpenAI API key
- `OPENAI_MODEL` - `gpt-4o-mini` or `gpt-4`
- `GITHUB_TOKEN` - GitHub PAT for session logging
- `GITHUB_REPO` - `ksauka/hicxai-data-private`
- `DROPBOX_APP_KEY` - Dropbox app key
- `DROPBOX_APP_SECRET` - Dropbox app secret
- `DROPBOX_REFRESH_TOKEN` - Dropbox refresh token

**Optional Secrets:**
- `QUALTRICS_URL` - Qualtrics survey URL for redirects

### 5. Upload Models to Dropbox

The app downloads models from Dropbox on first run:

```bash
# Upload your trained models
# Example: banking77_logistic_model.pkl → /ds_project_models/banking77_logistic_model.pkl
```

See [Dropbox setup instructions](../GITHUB_LOGGING_SETUP.md#dropbox-setup) for details.

## Deployment Architecture

```
DS_Project (Local)
    ↓
Deployment Script
    ↓
Hugging Face Space (Docker)
    ├── app_main_hicxai.py (entry point)
    ├── src/ (application code)
    ├── config/ (hierarchies, thresholds)
    ├── experiments/ (local models, backed up)
    └── .streamlit/config.toml
    
Runtime:
    ├── Download models from Dropbox (if missing)
    ├── Load classifier + DS agent
    └── Log sessions to GitHub
```

## What Gets Deployed

### ✅ **Included:**
- `src/` - All application code
- `config/` - Hierarchies, thresholds, intent descriptions
- `app_main_hicxai.py` - Main Streamlit entry point
- `experiments/` - Pretrained models (via Git LFS)
- `sample_study_queries.csv` - Test queries
- `.streamlit/config.toml` - Streamlit configuration

### ❌ **Excluded:**
- `venv/`, `.venv/` - Virtual environments
- `.env` - Environment variables (use Space Secrets instead)
- `test_*.py` - Test files
- `outputs/`, `results/` - Generated outputs
- `notebooks/` - Jupyter notebooks
- `__pycache__/` - Python cache

## Updating Deployment

After making code changes:

```bash
bash scripts/deployment/deploy_to_huggingface.sh YOUR_USERNAME ds-banking-assistant
```

The script will:
1. Pull latest from HF Space
2. Copy updated source code
3. Commit and push changes
4. HF will automatically rebuild

## Multiple Conditions (A/B Testing)

To deploy multiple experimental conditions:

```bash
# Condition 1: No explanations
bash scripts/deployment/deploy_to_huggingface.sh YOUR_USERNAME ds-assistant-condition-1

# Condition 2: With explanations
bash scripts/deployment/deploy_to_huggingface.sh YOUR_USERNAME ds-assistant-condition-2
```

Modify the deployment script to copy different app files:
```bash
# In deploy_to_huggingface.sh, change:
cp "$PROJECT_DIR/app_main_hicxai.py" ./app_main_hicxai.py

# To use condition-specific file:
cp "$PROJECT_DIR/app_condition_1.py" ./app_main_hicxai.py
```

## Anonymous Deployment (Research Studies)

For anonymous user studies:

1. **Use Organization Account:**
   ```bash
   # Create org at: huggingface.co/organizations/new
   bash scripts/deployment/deploy_to_huggingface.sh research-org ds-assistant
   ```

2. **Make Space Private:**
   - Settings → Visibility → Private
   - Share embed URL: `https://research-org-ds-assistant.hf.space`

3. **Disable Discussions:**
   - Settings → Discussions → Disable
   - Removes community features

4. **Professional Appearance:**
   - Use neutral space name
   - No personal branding in README
   - Generic title/description

## Monitoring

### Check Space Logs
1. Go to your Space page
2. Click "Logs" tab
3. Look for errors during:
   - Docker build
   - Model download from Dropbox
   - Session logging to GitHub

### Common Issues

**Build Fails:**
- Check requirements.txt for version conflicts
- Verify Dockerfile syntax
- Check logs for missing system dependencies

**Models Not Loading:**
- Verify Dropbox secrets are set
- Check model files exist in `/ds_project_models/`
- Check logs for download errors

**Session Logging Fails:**
- Verify GitHub token has `repo` permission
- Check repo name is correct: `ksauka/hicxai-data-private`
- Verify repo exists and token has access

## Testing Deployment

Before sharing with participants:

1. **Open Space URL** (allow 2-3 minutes for initial build)
2. **Test Dropbox Model Download:**
   - First load should download from Dropbox
   - Check logs for "✅ Downloaded successfully"
3. **Test Query Processing:**
   - Enter sample banking query
   - Verify intent classification works
   - Check clarification questions if uncertain
4. **Test Session Logging:**
   - Complete a test session
   - Check GitHub repo for session file
   - Verify data structure is correct

## Production Checklist

Before launching user study:

- [ ] Space builds without errors
- [ ] All secrets configured correctly
- [ ] Models download from Dropbox successfully
- [ ] Session logging to GitHub works
- [ ] Qualtrics redirect works (if used)
- [ ] Test with 2-3 sample participants
- [ ] Check GitHub repo receives session data
- [ ] Space is private (for anonymous studies)
- [ ] Discussions disabled (for professional look)
- [ ] Backup deployment strategy ready

## Troubleshooting

### Docker Build Times Out
- Large models in Git LFS can be slow
- Consider keeping only essential models
- Use Dropbox for all large files

### Out of Memory
- Hugging Face Spaces have limited RAM
- Use smaller models or quantized versions
- Consider CPU-only PyTorch if not using GPU

### Secrets Not Loading
- Verify secret names match exactly (case-sensitive)
- Check for typos in secret values
- Restart space after adding secrets

## Alternative: Streamlit Cloud

If Hugging Face Spaces doesn't work, deploy to Streamlit Cloud:

```bash
# Push to GitHub repo
git push origin main

# Deploy at: share.streamlit.io
# Connect your GitHub repo
# App file: app_main_hicxai.py
```

Streamlit Cloud Secrets: Same format as `.streamlit/secrets.toml`

## Support

For deployment issues:
- Check [Hugging Face Spaces Docs](https://huggingface.co/docs/hub/spaces)
- Ask in Hugging Face Discord
- Check deployment script logs for errors

For application issues:
- See main [README.md](../../README.md)
- Check [QUICKSTART.md](../../QUICKSTART.md)
