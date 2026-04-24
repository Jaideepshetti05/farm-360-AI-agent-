"""
Animal Image Classification for New Dataset
===========================================

Script to train animal image classification model on the new animalsdata dataset.
This script loads the dataset from data/animal_images/animalsdata/raw-img and trains
multiple models using transfer learning approaches.
"""

import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import (
    ResNet50, VGG16, InceptionV3, EfficientNetB0, MobileNetV2,
    DenseNet121, Xception
)
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import warnings
warnings.filterwarnings('ignore')

class NewAnimalImageClassifier:
    """Class to handle animal image classification for the new dataset."""
    
    def __init__(self, data_dir: str = "data/animal_images/animalsdata/raw-img"):
        self.data_dir = Path(data_dir)
        self.animal_classes = [d.name for d in self.data_dir.iterdir() if d.is_dir()]
        self.num_classes = len(self.animal_classes)
        self.input_shape = (224, 224, 3)  # Standard size for most models
        self.models = {}
        self.history = {}
        self.results = {}
        self.class_labels = {}
        
        # Create class label mapping
        for i, class_name in enumerate(self.animal_classes):
            # Translate Italian names to English using the translation dict
            translation_map = {
                "cane": "dog", "cavallo": "horse", "elefante": "elephant", 
                "farfalla": "butterfly", "gallina": "chicken", "gatto": "cat", 
                "mucca": "cow", "pecora": "sheep", "scoiattolo": "squirrel",
                "ragno": "spider"
            }
            translated_name = translation_map.get(class_name, class_name)
            self.class_labels[i] = translated_name
        
        print(f"Found {self.num_classes} animal classes: {self.animal_classes}")
        print(f"Translated class labels: {self.class_labels}")
        print(f"Total classes: {self.num_classes}")
        
    def create_data_generators(self, batch_size: int = 32, validation_split: float = 0.2):
        """Create data generators with augmentation for training."""
        
        # Training data generator with augmentation
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=30,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            brightness_range=[0.8, 1.2],
            validation_split=validation_split,
            fill_mode='nearest'
        )
        
        # Validation/test data generator (only rescaling)
        val_datagen = ImageDataGenerator(
            rescale=1./255,
            validation_split=validation_split
        )
        
        # Training generator
        train_generator = train_datagen.flow_from_directory(
            self.data_dir,
            target_size=self.input_shape[:2],
            batch_size=batch_size,
            class_mode='categorical',
            subset='training',
            seed=42
        )
        
        # Validation generator
        val_generator = val_datagen.flow_from_directory(
            self.data_dir,
            target_size=self.input_shape[:2],
            batch_size=batch_size,
            class_mode='categorical',
            subset='validation',
            seed=42
        )
        
        return train_generator, val_generator
    
    def create_transfer_learning_model(self, base_model_name='ResNet50'):
        """Create a transfer learning model using pre-trained architectures."""
        
        # Select base model
        if base_model_name == 'ResNet50':
            base_model = ResNet50(weights='imagenet', include_top=False, input_shape=self.input_shape)
        elif base_model_name == 'VGG16':
            base_model = VGG16(weights='imagenet', include_top=False, input_shape=self.input_shape)
        elif base_model_name == 'InceptionV3':
            base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=self.input_shape)
        elif base_model_name == 'EfficientNetB0':
            base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=self.input_shape)
        elif base_model_name == 'MobileNetV2':
            base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=self.input_shape)
        elif base_model_name == 'DenseNet121':
            base_model = DenseNet121(weights='imagenet', include_top=False, input_shape=self.input_shape)
        elif base_model_name == 'Xception':
            base_model = Xception(weights='imagenet', include_top=False, input_shape=self.input_shape)
        else:
            raise ValueError(f"Unknown base model: {base_model_name}")
        
        # Freeze base model layers
        base_model.trainable = False
        
        # Add custom classifier on top
        model = keras.Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(512, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.2),
            layers.Dense(self.num_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.0001),
            loss='categorical_crossentropy',
            metrics=['accuracy', 'top_3_accuracy']
        )
        
        return model

    def train_model(self, model, model_name, train_gen, val_gen, epochs=50):
        """Train a model with callbacks."""
        
        # Create models directory if it doesn't exist
        os.makedirs('models', exist_ok=True)
        
        # Callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_accuracy',
                patience=10,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.2,
                patience=5,
                min_lr=1e-7,
                verbose=1
            ),
            ModelCheckpoint(
                f'models/{model_name}_best.h5',
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            )
        ]
        
        # Train the model
        history = model.fit(
            train_gen,
            epochs=epochs,
            validation_data=val_gen,
            callbacks=callbacks,
            verbose=1
        )
        
        return history
    
    def fine_tune_transfer_model(self, model, model_name, train_gen, val_gen, epochs=20):
        """Fine-tune a transfer learning model by unfreezing some layers."""
        
        # Unfreeze top layers for fine-tuning
        model.layers[0].trainable = True
        
        # Fine-tune from this layer onwards
        fine_tune_at = len(model.layers[0].layers) // 2
        
        # Freeze all the layers before fine_tune_at
        for layer in model.layers[0].layers[:fine_tune_at]:
            layer.trainable = False
        for layer in model.layers[0].layers[fine_tune_at:]:
            layer.trainable = True
        
        # Use a lower learning rate for fine-tuning
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.0001/10),
            loss='categorical_crossentropy',
            metrics=['accuracy', 'top_3_accuracy']
        )
        
        # Fine-tuning callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_accuracy',
                patience=5,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.2,
                patience=3,
                min_lr=1e-8,
                verbose=1
            ),
            ModelCheckpoint(
                f'models/{model_name}_fine_tuned_best.h5',
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            )
        ]
        
        # Continue training
        history = model.fit(
            train_gen,
            epochs=epochs,
            validation_data=val_gen,
            callbacks=callbacks,
            verbose=1
        )
        
        return history

    def evaluate_model(self, model, val_gen):
        """Evaluate a trained model."""
        results = model.evaluate(val_gen, verbose=0)
        return {
            'loss': float(results[0]),
            'accuracy': float(results[1]),
            'top_3_accuracy': float(results[2]) if len(results) > 2 else 0
        }

    def train_single_model(self, model_name, train_gen, val_gen, epochs=25):
        """Train a single model with the given name."""
        print(f"\nTraining {model_name}...")
        
        # Create model
        model = self.create_transfer_learning_model(model_name)
        self.models[model_name] = model
        
        # Train initial model
        self.history[f'{model_name}'] = self.train_model(
            model, f'{model_name.lower()}_initial', train_gen, val_gen, epochs=epochs
        )
        
        # Fine-tune the model
        print(f"\nFine-tuning {model_name}...")
        self.history[f'{model_name}_finetuned'] = self.fine_tune_transfer_model(
            model, f'{model_name.lower()}', train_gen, val_gen, epochs=max(epochs//2, 10)
        )
        
        # Evaluate the model
        evaluation = self.evaluate_model(model, val_gen)
        self.results[model_name] = evaluation
        
        print(f"{model_name} - Accuracy: {evaluation['accuracy']:.4f}, "
              f"Top-3 Accuracy: {evaluation['top_3_accuracy']:.4f}")
        
        return model

    def run_training(self):
        """Run the complete training pipeline."""
        print("=" * 80)
        print("ANIMAL IMAGE CLASSIFICATION FOR NEW DATASET")
        print("=" * 80)
        print(f"Dataset: {self.data_dir}")
        print(f"Number of classes: {self.num_classes}")
        print(f"Classes: {list(self.class_labels.values())}")
        print("=" * 80)
        
        # Create data generators
        print("Creating data generators...")
        train_gen, val_gen = self.create_data_generators(batch_size=16)  # Reduced batch size for memory
        
        # Train multiple models
        models_to_train = ['ResNet50', 'EfficientNetB0', 'MobileNetV2']
        
        for model_name in models_to_train:
            try:
                self.train_single_model(model_name, train_gen, val_gen, epochs=25)
            except Exception as e:
                print(f"Error training {model_name}: {e}")
                continue
        
        # Print final results
        print("\n" + "=" * 80)
        print("TRAINING COMPLETED!")
        print("=" * 80)
        print("Final Results:")
        for model_name, metrics in self.results.items():
            print(f"  {model_name}: Accuracy = {metrics['accuracy']:.4f}, "
                  f"Top-3 Accuracy = {metrics['top_3_accuracy']:.4f}")
        
        # Find best model
        if self.results:
            best_model = max(self.results.items(), key=lambda x: x[1]['accuracy'])
            print(f"\nBest performing model: {best_model[0]} "
                  f"(Accuracy: {best_model[1]['accuracy']:.4f})")
        
        return self.results


def main():
    """Main function to run the animal image classification training."""
    print("🐾 Starting Animal Image Classification Training 🐾")
    
    # Initialize the classifier with the new dataset
    classifier = NewAnimalImageClassifier()
    
    # Run the complete pipeline
    results = classifier.run_training()
    
    print("\n🎉 Training completed successfully!")
    print("Models have been saved to the 'models/' directory")
    print("You can now use these models for animal image classification.")


if __name__ == "__main__":
    main()