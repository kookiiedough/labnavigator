#!/bin/bash

# LabNavigator AI MVP Startup Script
# This script starts the LabNavigator AI dashboard application

echo "Starting LabNavigator AI MVP..."
echo "------------------------------"

# Check if Python and required packages are installed
echo "Checking dependencies..."
python3 -c "import numpy, pandas, matplotlib, flask, skopt" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "Installing required packages..."
    pip3 install numpy pandas matplotlib flask scikit-optimize
fi

# Ensure the knowledge base exists
if [ ! -f "/home/ubuntu/knowledge_base.csv" ]; then
    echo "Error: Knowledge base file not found!"
    exit 1
fi

# Start the dashboard application
echo "Starting LabNavigator AI dashboard..."
cd /home/ubuntu/dashboard
python3 app.py
