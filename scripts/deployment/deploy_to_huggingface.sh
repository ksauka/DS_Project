#!/bin/bash
# Deploy DS_Project to Hugging Face Spaces
# Usage: ./deploy_to_huggingface.sh YOUR_HF_USERNAME [space-name]

set -e

# Check if logged in
if ! command -v huggingface-cli &> /dev/null; then
    echo "⚠️  huggingface-cli not found. Installing..."
    pip install huggingface_hub
fi

# Check authentication
if ! huggingface-cli whoami &> /dev/null; then
    echo "❌ Not logged in to Hugging Face"
    echo ""
    echo "Please login first:"
    echo "  huggingface-cli login"
    echo ""
    echo "Get your token at: https://huggingface.co/settings/tokens"
    exit 1
fi

if [ -z "$1" ]; then
    echo "Usage: ./deploy_to_huggingface.sh YOUR_HF_USERNAME [space-name]"
    echo ""
    echo "Examples:"
    echo "  ./deploy_to_huggingface.sh myusername"
    echo "  ./deploy_to_huggingface.sh myusername ds-banking-assistant"
    echo "  ./deploy_to_huggingface.sh my-org ds-banking-assistant"
    exit 1
fi

HF_USERNAME="$1"
SPACE_NAME="${2:-ds-banking-assistant}"
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DEPLOY_DIR="/tmp/hf_deploy_ds"

CURRENT_USER=$(huggingface-cli whoami | grep 'username:' | awk '{print $2}')

echo "🚀 Deploying DS_Project to Hugging Face Spaces"
echo "Logged in as: $CURRENT_USER"
echo "Deploying to: $HF_USERNAME/$SPACE_NAME"
echo "Project directory: $PROJECT_DIR"
echo ""
echo "💡 Deployment Tips:"
echo "   • For anonymous deployment, use organization: ./deploy_to_huggingface.sh your-org"
echo "   • Make space private after deployment if needed"
echo "   • Add secrets via Space Settings → Variables and secrets"
echo ""

mkdir -p "$DEPLOY_DIR"

SPACE_DIR="$DEPLOY_DIR/$SPACE_NAME"

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📦 Deploying Customer Service Assistant with Hierarchical Intent Classification"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Clone or pull space (using HF token for authentication)
HF_TOKEN=$(cat ~/.cache/huggingface/token 2>/dev/null || cat ~/.huggingface/token 2>/dev/null)

if [ -d "$SPACE_DIR" ]; then
    echo "   Updating existing space..."
    cd "$SPACE_DIR"
    git pull
else
    echo "   Cloning space repository..."
    cd "$DEPLOY_DIR"
    git clone "https://$HF_USERNAME:$HF_TOKEN@huggingface.co/spaces/$HF_USERNAME/$SPACE_NAME" || {
        echo "❌ Failed to clone. Make sure the space exists at:"
        echo "   https://huggingface.co/spaces/$HF_USERNAME/$SPACE_NAME"
        echo ""
        echo "Create it first at: https://huggingface.co/new-space"
        echo "   • Name: $SPACE_NAME"
        echo "   • SDK: Docker"
        echo "   • License: MIT"
        exit 1
    }
    cd "$SPACE_NAME"
fi

echo "   Copying application files..."

# Copy source code (excluding __pycache__, test files, etc.)
rsync -av --delete \
    --exclude='__pycache__/' \
    --exclude='*.pyc' \
    --exclude='test_*.py' \
    --exclude='.pytest_cache/' \
    --exclude='venv/' \
    --exclude='.venv/' \
    --exclude='.env' \
    --exclude='outputs/' \
    --exclude='results/' \
    --exclude='notebooks/' \
    "$PROJECT_DIR/src/" ./src/

# Copy main app file
cp "$PROJECT_DIR/app_main_hicxai.py" ./app_main_hicxai.py

# Copy config files
rsync -av --delete "$PROJECT_DIR/config/" ./config/

# Copy experiments directory (for pretrained models) - will use Git LFS
if [ -d "$PROJECT_DIR/experiments" ]; then
    echo "   Copying experiments directory (models will use Git LFS)..."
    rsync -av --delete \
        --exclude='*.log' \
        --exclude='*.csv' \
        "$PROJECT_DIR/experiments/" ./experiments/
fi

# Copy sample queries for testing
if [ -f "$PROJECT_DIR/sample_study_queries.csv" ]; then
    cp "$PROJECT_DIR/sample_study_queries.csv" ./
fi

# Combine requirements files
echo "   Creating combined requirements.txt..."
cat > requirements.txt << 'EOF'
# Core Dependencies
streamlit>=1.28.0
torch>=2.0.0
transformers>=4.30.0
sentence-transformers>=2.2.0

# Machine Learning
scikit-learn>=1.3.0
numpy>=1.24.0
scipy>=1.10.0

# Data Processing
pandas>=2.0.0
datasets>=2.14.0

# API and Storage
openai>=1.0.0
python-dotenv>=1.0.0
PyGithub>=2.1.0
dropbox>=11.36.0
requests>=2.31.0

# Visualization
plotly>=5.17.0
matplotlib>=3.7.0
seaborn>=0.12.0
altair>=5.0.0

# Utilities
tqdm>=4.65.0
pyyaml>=6.0
EOF

# Copy .streamlit directory (template only, not secrets)
mkdir -p .streamlit
cat > .streamlit/config.toml << 'CONFIGEOF'
[server]
port = 7860
headless = true
address = "0.0.0.0"
maxUploadSize = 200

[browser]
gatherUsageStats = false

[theme]
primaryColor = "#1f77b4"
backgroundColor = "#ffffff"
secondaryBackgroundColor = "#f0f2f6"
textColor = "#262730"
font = "sans serif"
CONFIGEOF

# Create secrets template
cat > .streamlit/secrets.toml.template << 'SECRETSEOF'
# OpenAI Configuration (required)
OPENAI_API_KEY = "sk-..."
OPENAI_MODEL = "gpt-4o-mini"

# GitHub Data Logging (required for user studies)
GITHUB_TOKEN = "github_pat_..."
GITHUB_REPO = "ksauka/hicxai-data-private"

# Dropbox Model Storage (required for large models)
DROPBOX_APP_KEY = "..."
DROPBOX_APP_SECRET = "..."
DROPBOX_REFRESH_TOKEN = "..."

# Qualtrics Integration (optional)
QUALTRICS_URL = "https://survey.qualtrics.com/jfe/form/SV_..."
SECRETSEOF

# Create Dockerfile for Hugging Face Spaces
cat > Dockerfile << 'DOCKEREOF'
FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python packages
COPY requirements.txt ./
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy application files
COPY src/ ./src/
COPY config/ ./config/
COPY app_main_hicxai.py ./
COPY .streamlit/ ./.streamlit/

# Copy experiments (models) if present
COPY experiments/ ./experiments/

# Copy sample queries if present
COPY sample_study_queries.csv ./sample_study_queries.csv 2>/dev/null || true

# Expose port 7860 for Hugging Face Spaces
EXPOSE 7860

# Set environment variables
ENV STREAMLIT_SERVER_PORT=7860
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0
ENV STREAMLIT_SERVER_HEADLESS=true

# Run Streamlit
CMD ["streamlit", "run", "app_main_hicxai.py", "--server.port=7860", "--server.address=0.0.0.0"]
DOCKEREOF

echo "   Creating README.md..."

# Create Hugging Face README with metadata
cat > README.md << 'EOF'
---
title: Customer Service Assistant
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
license: mit
app_port: 7860
---

# Customer Service Assistant

**Interactive intent classification system using hierarchical Dempster-Shafer theory for uncertain query resolution.**

## Overview

This application demonstrates a novel approach to intent classification in customer service:

- **Hierarchical Reasoning**: Uses Dempster-Shafer theory to handle uncertainty across parent-child intent hierarchies
- **Explainable AI**: Provides transparent belief updates and clarification strategies
- **Interactive Dialogue**: Engages users in multi-turn conversations to resolve ambiguous queries
- **Banking77 Dataset**: Trained on 77 real-world banking intents organized in a 3-level hierarchy

## Features

### 🎯 **Intent Classification**
- Support for 77 banking intents across domains: Account Management, Transactions, Card Services, etc.
- Hierarchical classification with parent-child relationships
- Confidence thresholds for each hierarchy level

### 🤔 **Uncertainty Management**
- Dempster-Shafer belief propagation
- Ancestor-aware confidence scoring
- Intelligent clarification questions when belief is uncertain

### 📊 **Explainability**
- Real-time belief progression visualization
- Top-5 intent probabilities display
- Transparent decision-making process

### 📈 **Research Integration**
- Session logging to private GitHub repository
- Qualtrics/Prolific integration for user studies
- Dropbox model storage for large ML models
- Comprehensive interaction tracking

## Technical Architecture

### Core Components
- **Classifier**: Logistic Regression with sentence-transformers embeddings (intfloat/e5-base)
- **DS Reasoning**: Hierarchical belief propagation with configurable thresholds
- **Clarification**: Context-aware question generation using GPT-4
- **Data Logging**: GitHub API integration for session persistence

### Dataset
- **Banking77**: 77 intents across 13,083 queries
- **Hierarchy**: 3 levels (root → parent categories → leaf intents)
- **Split**: 80% train, 10% dev, 10% test

## Usage

### For Participants
1. Enter your Prolific ID (if coming from a study)
2. Interact with banking queries
3. Answer clarification questions when needed
4. View intent predictions with confidence scores
5. Complete post-study survey

### For Researchers
Configure secrets in Space Settings:
- `OPENAI_API_KEY`: For clarification generation
- `GITHUB_TOKEN`: For session logging
- `DROPBOX_*`: For model storage

## Research Context

Part of the HicXAI research project investigating explainable AI methods in hierarchical intent classification for customer service automation.

**Note**: This is a research prototype. Not intended for production banking systems.

## Citation

If you use this system in your research, please cite:

```bibtex
@inproceedings{ds-intent-classification-2026,
  title={Hierarchical Intent Classification with Dempster-Shafer Reasoning},
  author={TBD},
  year={2026}
}
```

## License

MIT License - See LICENSE file for details
EOF

echo "   Setting up Git LFS for large model files..."
git lfs install 2>/dev/null || echo "Git LFS already installed"
cat > .gitattributes << 'LFSEOF'
# Model files
*.pkl filter=lfs diff=lfs merge=lfs -text
*.pth filter=lfs diff=lfs merge=lfs -text
*.pt filter=lfs diff=lfs merge=lfs -text
*.bin filter=lfs diff=lfs merge=lfs -text
*.h5 filter=lfs diff=lfs merge=lfs -text

# Data files
*.csv filter=lfs diff=lfs merge=lfs -text
*.json filter=lfs diff=lfs merge=lfs -text

# Images
*.png filter=lfs diff=lfs merge=lfs -text
*.jpg filter=lfs diff=lfs merge=lfs -text
*.jpeg filter=lfs diff=lfs merge=lfs -text
LFSEOF

echo "   Committing changes..."
git add .gitattributes
git add .

if git diff --staged --quiet; then
    echo "   ℹ️  No changes to commit"
else
    COMMIT_MSG="Update DS Banking Assistant - $(date +%Y-%m-%d)"
    git commit -m "$COMMIT_MSG"
    echo "   Pushing to Hugging Face..."
    git push
    echo "   ✅ Deployment successful!"
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🎉 Deployment complete!"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "Your app is available at:"
echo "  🌐 https://huggingface.co/spaces/$HF_USERNAME/$SPACE_NAME"
echo "  🔗 Direct embed: https://$HF_USERNAME-$SPACE_NAME.hf.space"
echo ""
echo "⚠️  IMPORTANT: Next steps:"
echo ""
echo "1️⃣  Add Secrets (Settings → Variables and secrets):"
echo "   • OPENAI_API_KEY = your-openai-key"
echo "   • OPENAI_MODEL = gpt-4o-mini"
echo "   • GITHUB_TOKEN = your-github-pat"
echo "   • GITHUB_REPO = ksauka/hicxai-data-private"
echo "   • DROPBOX_APP_KEY = your-dropbox-key"
echo "   • DROPBOX_APP_SECRET = your-dropbox-secret"
echo "   • DROPBOX_REFRESH_TOKEN = your-dropbox-token"
echo ""
echo "2️⃣  Upload Models to Dropbox:"
echo "   • Upload trained models (.pkl files) to /ds_project_models/"
echo "   • banking77_logistic_model.pkl"
echo "   • Other dataset models as needed"
echo ""
echo "3️⃣  For Anonymous Research:"
echo "   • Settings → Visibility → Private"
echo "   • Share embed URL with participants"
echo "   • Disable Discussions for professional look"
echo ""
echo "4️⃣  Monitor Space:"
echo "   • Check Logs tab for errors"
echo "   • Verify model downloads work"
echo "   • Test with sample queries"
echo ""
