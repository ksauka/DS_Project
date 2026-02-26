#!/bin/bash
# Setup script for DS_Project using conda environment

echo "=========================================="
echo "DS_Project Setup Script (Conda)"
echo "=========================================="
echo ""

# Check if in conda environment
if [ -z "$CONDA_DEFAULT_ENV" ]; then
    echo "Error: No conda environment detected"
    echo "Please activate your conda environment first:"
    echo "  conda activate dsproject"
    exit 1
fi

echo "Using conda environment: $CONDA_DEFAULT_ENV"
echo ""

# Check Python version
echo "Checking Python version..."
python --version

# Upgrade pip in conda environment
echo ""
echo "Upgrading pip..."
python -m pip install --upgrade pip

# Install dependencies using conda's pip
echo ""
echo "Installing dependencies..."
python -m pip install -r requirements.txt

# Setup .env file
echo ""
if [ ! -f ".env" ]; then
    echo "Creating .env file from template..."
    cp .env.example .env
    echo "⚠ IMPORTANT: Edit .env and add your OPENAI_API_KEY"
    echo "✓ .env file created"
else
    echo "✓ .env file already exists"
fi

# Check for API key
echo ""
if grep -q "your_openai_api_key_here" .env 2>/dev/null; then
    echo "⚠ WARNING: Please update your OpenAI API key in .env file"
    echo "   Edit .env and replace 'your_openai_api_key_here' with your actual key"
fi

# Summary
echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Make sure you're in the conda environment:"
echo "   conda activate dsproject"
echo ""
echo "2. Update .env with your OpenAI API key"
echo ""
echo "3. Train your first model:"
echo "   python scripts/training/train.py --dataset banking77"
echo ""
echo "See README.md for more details."
echo ""
