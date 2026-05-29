#!/bin/bash
# SynthID Remover - Quick Start Script
# Run this to set up and launch the app

echo "=================================="
echo "SynthID Remover - Quick Start"
echo "=================================="
echo ""

# Check Python version
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "Python version: $python_version"

# Create venv if not exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate venv
echo "Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# Check models
echo ""
echo "Checking models..."
python download_models.py --check

if [ $? -ne 0 ]; then
    echo ""
    echo "Models missing. Download now? (y/n)"
    read -r response
    if [ "$response" = "y" ]; then
        python download_models.py
    fi
fi

# Launch
echo ""
echo "Launching SynthID Remover..."
python main.py
