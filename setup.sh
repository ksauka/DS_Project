#!/bin/bash
# Setup script for DS_Project

echo "=========================================="
echo "DS_Project Setup Script"
echo "=========================================="
echo ""

# Check Python version
echo "Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "Found Python $python_version"

# Create virtual environment
echo ""
echo "Creating virtual environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo "✓ Virtual environment created"
else
    echo "✓ Virtual environment already exists"
fi

# Activate virtual environment
echo ""
echo "Activating virtual environment..."
source venv/bin/activate || {
    echo "Error: Could not activate virtual environment"
    exit 1
}

# Upgrade pip
echo ""
echo "Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo ""
echo "Installing dependencies..."
pip install -r requirements.txt

# # Create necessary directories
# echo ""
# echo "Creating project directories..."
# mkdir -p outputs/models
# mkdir -p outputs/evaluation
# mkdir -p experiments
# mkdir -p logs
# echo "✓ Directories created"

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
echo "1. Activate virtual environment:"
echo "   source venv/bin/activate"
echo ""
echo "2. Update .env with your OpenAI API key"
echo ""
echo "3. Create configuration files in config/:"
echo "   - banking77_hierarchy.json"
echo "   - banking77_intents.json"
echo ""
echo "4. Train your first model:"
echo "   python train.py --dataset banking77 \\"
echo "     --hierarchy-file config/banking77_hierarchy.json \\"
echo "     --intents-file config/banking77_intents.json"
echo ""
echo "See REFACTORING_SUMMARY.md for more details."
echo ""
