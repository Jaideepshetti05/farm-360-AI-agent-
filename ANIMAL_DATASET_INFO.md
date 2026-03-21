# New Animal Image Dataset Information

## Dataset Overview
- **Location**: `data/animal_images/animalsdata/raw-img/`
- **Number of Classes**: 10
- **Total Images**: Approximately 27,000+ images
- **Classes**: dog, horse, elephant, butterfly, chicken, cat, cow, sheep, spider, squirrel (in Italian: cane, cavallo, elefante, farfalla, gallina, gatto, mucca, pecora, ragno, scoiattolo)

## Dataset Structure
The dataset is organized in the following structure:
```
data/animal_images/animalsdata/raw-img/
├── cane/          (dogs) - ~4,863 images
├── cavallo/       (horses) - ~2,623 images
├── elefante/      (elephants) - ~1,446 images
├── farfalla/      (butterflies) - ~2,112 images
├── gallina/       (chickens) - ~3,098 images
├── gatto/         (cats) - ~1,668 images
├── mucca/         (cows) - ~1,866 images
├── pecora/        (sheep) - ~1,820 images
├── ragno/         (spiders) - ~4,821 images
├── scoiattolo/    (squirrels) - ~1,862 images
```

## Dataset Format
- **Image Format**: JPEG
- **Image Sizes**: Various sizes (resized during training to 224x224)
- **Labeling**: Folder-based labeling (each folder represents a class)

## Training Instructions
To train a model on this dataset:

1. **Using the Python script directly**:
   ```bash
   python animal_module/train_new_animal_dataset.py
   ```

2. **Using the batch file**:
   ```bash
   animal_module\train_new_animal_dataset.bat
   ```

## Training Features
- Transfer learning using pre-trained models (ResNet50, EfficientNetB0, MobileNetV2)
- Data augmentation (rotation, shifting, flipping, brightness adjustment)
- Fine-tuning of pre-trained models
- Model checkpointing to save best models
- Validation split for monitoring training progress

## Expected Output
- Trained models saved in the `models/` directory
- Training history and metrics
- Best performing model saved as `.h5` files

## Model Architecture
The system uses transfer learning with:
- Pre-trained base models (loaded with ImageNet weights)
- Custom classifier head with dense layers
- Batch normalization and dropout for regularization
- Global average pooling for feature aggregation

## Notes
- The dataset has a good variety of animals from domestic pets to wild animals
- Some classes have a large number of images which helps with training
- The dataset includes insects (butterflies, spiders) which adds diversity
- All images are in Italian folder names, but the training script translates them for clarity