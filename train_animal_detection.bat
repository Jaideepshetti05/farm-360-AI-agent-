@echo off
echo ========================================
echo ANIMAL DETECTION TRAINING STARTER
echo ========================================
echo This will train your animal detection dataset using multiple approaches
echo ========================================

cd /d "C:\Users\Jaideep\OneDrive\Desktop\ml models"

echo Starting comprehensive animal detection training...
python animal_module\train_all_animal_approaches.py

echo.
echo Training complete! Check the models, reports, and visualizations directories.
pause