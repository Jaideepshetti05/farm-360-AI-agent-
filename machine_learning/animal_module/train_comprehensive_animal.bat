@echo off
echo.
echo =====================================================
echo  Comprehensive Animal Image Classification Training
echo =====================================================
echo.
echo Installing required packages (if needed)...
pip install tensorflow scikit-learn matplotlib seaborn pandas numpy opencv-python

echo.
echo Starting comprehensive animal image classification training...
echo This will train multiple models using various approaches:
echo - Simple CNN
echo - Advanced CNN with residual connections  
echo - Transfer Learning with ResNet50, EfficientNetB0, MobileNetV2, VGG16, DenseNet121
echo - Fine-tuned versions of transfer learning models
echo - Attention mechanism model
echo - Focal loss model for class imbalance
echo.
echo This process will take several hours depending on your hardware.
echo.
pause

python train_comprehensive_animal.py

echo.
echo Training completed! Check the models directory for saved models.
pause