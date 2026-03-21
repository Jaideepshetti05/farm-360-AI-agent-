@echo off
echo.
echo =====================================================
echo  New Animal Image Dataset Training System
echo =====================================================
echo.
echo Installing required packages (if needed)...
pip install tensorflow scikit-learn matplotlib seaborn pandas numpy opencv-python

echo.
echo Starting new animal image classification training...
python train_new_animal_dataset.py

echo.
echo Training completed! Check the models directory for saved models.
pause