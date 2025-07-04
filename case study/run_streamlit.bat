@echo off
echo ========================================
echo    Vehicle Heatmap Dashboard Launcher
echo ========================================
echo.
echo Starting Streamlit application...
echo.

REM Change to script directory
cd /d %~dp0

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python or add it to your system PATH
    echo.
    pause
    exit /b 1
)

REM Check if required packages are installed
echo Checking required packages...
python -c "import streamlit, pandas, folium, streamlit_folium" >nul 2>&1
if errorlevel 1 (
    echo ERROR: Required packages not found
    echo Installing required packages...
    echo.
    pip install streamlit pandas folium streamlit-folium numpy branca
    if errorlevel 1 (
        echo ERROR: Failed to install packages
        pause
        exit /b 1
    )
)

REM Check if data file exists
if not exist "Data\cleaned_Kolkta_2_spots.csv" (
    echo ERROR: Data file not found at Data\cleaned_Kolkta_2_spots.csv
    echo Please ensure the data file is in the correct location
    echo.
    pause
    exit /b 1
)

echo.
echo Launching Streamlit dashboard...
echo The dashboard will open in your default web browser
echo Press Ctrl+C to stop the application
echo.

REM Run Streamlit
python -m streamlit run streamlit_heatmap_app.py

pause
