#!/bin/bash

echo "========================================"
echo "   Vehicle Heatmap Dashboard Launcher"
echo "========================================"
echo

# Change to script directory
cd "$(dirname "$0")"

# Check if Python is available
if ! command -v python3 &> /dev/null && ! command -v python &> /dev/null; then
    echo "ERROR: Python is not installed or not in PATH"
    echo "Please install Python 3.x"
    exit 1
fi

# Use python3 if available, otherwise python
if command -v python3 &> /dev/null; then
    PYTHON_CMD="python3"
    PIP_CMD="pip3"
else
    PYTHON_CMD="python"
    PIP_CMD="pip"
fi

echo "Using Python: $PYTHON_CMD"

# Check if required packages are installed
echo "Checking required packages..."
$PYTHON_CMD -c "import streamlit, pandas, folium, streamlit_folium" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "ERROR: Required packages not found"
    echo "Installing required packages..."
    echo
    $PIP_CMD install streamlit pandas folium streamlit-folium numpy branca
    if [ $? -ne 0 ]; then
        echo "ERROR: Failed to install packages"
        exit 1
    fi
fi

# Check if data file exists
if [ ! -f "Data/cleaned_Kolkta_2_spots.csv" ]; then
    echo "ERROR: Data file not found at Data/cleaned_Kolkta_2_spots.csv"
    echo "Please ensure the data file is in the correct location"
    exit 1
fi

echo
echo "Launching Streamlit dashboard..."
echo "The dashboard will open in your default web browser"
echo "Press Ctrl+C to stop the application"
echo

# Make sure the script is executable
chmod +x "$0"

# Run Streamlit
$PYTHON_CMD -m streamlit run streamlit_heatmap_app.py

echo "Dashboard stopped."