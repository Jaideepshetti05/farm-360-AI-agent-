@echo off
echo.
echo =====================================================
echo  Animal Image Classification Training System
echo =====================================================
echo.
echo Installing required packages...
pip install -r requirements.txt
echo.
echo Starting animal image classification training...
python animal_image_classifier.py
echo.
echo Training completed!
pause