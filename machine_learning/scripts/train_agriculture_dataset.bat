@echo off
echo ========================================
echo AGRICULTURE DATASET ML TRAINING STARTER
echo ========================================
echo This will train your agriculture dataset using multiple approaches
echo ========================================

cd /d "%~dp0\.."

echo Starting comprehensive agriculture dataset training...
python dairy_module\agriculture_ml_pipeline.py

echo.
echo Training complete! Check the models, reports, and visualizations directories.
pause