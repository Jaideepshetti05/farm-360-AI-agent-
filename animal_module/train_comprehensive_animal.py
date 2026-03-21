"""
Comprehensive Animal Image Classification Training
=================================================

This script trains your animal dataset using every possible approach:
- Multiple transfer learning models
- Custom CNN architectures
- Data augmentation strategies
- Ensemble methods
- Fine-tuning techniques
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
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, LearningRateScheduler
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import warnings
warnings.filterwarnings('ignore')

class ComprehensiveAnimalClassifier:
    """Class to handle comprehensive animal image classification training."""
    
    def __init__(self, data_dir: str = "data/animal_images/animalsdata/raw-img"):
        self.data_dir = Path(data_dir)
        self.animal_classes = [d.name for d in self.data_dir.iterdir() if d.is_dir()]
        self.num_classes = len(self.animal_classes)
        self.input_shape = (224, 224, 3)
        self.models = {}
        self.history = {}
        self.results = {}
        
        # Create class label mapping
        self.translation_map = {
            "cane": "dog", "cavallo": "horse", "elefante": "elephant", 
            "farfalla": "butterfly", "gallina": "chicken", "gatto": "cat", 
            "mucca": "cow", "pecora": "sheep", "scoiattolo": "squirrel",
            "ragno": "spider"
        }
        
        self.class_labels = {}
        for i, class_name in enumerate(self.animal_classes):
            translated_name = self.translation_map.get(class_name, class_name)
            self.class_labels[i] = translated_name
        
        print(f"Found {self.num_classes} animal classes: {self.animal_classes}")
        print(f"Translated class labels: {self.class_labels}")
        print(f"Total classes: {self.num_classes}")
        
    def create_data_generators(self, batch_size: int = 16, validation_split: float = 0.2):
        """Create data generators with augmentation for training."""
        
        # Training data generator with extensive augmentation
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=40,
            width_shift_range=0.3,
            height_shift_range=0.3,
            shear_range=0.3,
            zoom_range=0.3,
            horizontal_flip=True,
            vertical_flip=False,  # Usually not applicable for animals
            brightness_range=[0.7, 1.3],
            channel_shift_range=0.2,
            fill_mode='nearest',
            validation_split=validation_split
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
    
    def create_simple_cnn(self):
        """Create a simple CNN model for baseline comparison."""
        model = keras.Sequential([
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=self.input_shape),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            
            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            
            layers.Conv2D(256, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            
            layers.Flatten(),
            layers.Dropout(0.5),
            layers.Dense(512, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.Dense(self.num_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy', 'top_3_accuracy']
        )
        
        return model
    
    def create_advanced_cnn(self):
        """Create an advanced CNN with residual connections."""
        inputs = layers.Input(shape=self.input_shape)
        
        # Initial convolution
        x = layers.Conv2D(64, (7, 7), strides=(2, 2), padding='same')(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
        
        # Residual blocks
        def residual_block(x, filters, stride=1):
            shortcut = x
            
            # First conv layer
            x = layers.Conv2D(filters, (3, 3), strides=stride, padding='same')(x)
            x = layers.BatchNormalization()(x)
            x = layers.Activation('relu')(x)
            
            # Second conv layer
            x = layers.Conv2D(filters, (3, 3), padding='same')(x)
            x = layers.BatchNormalization()(x)
            
            # Adjust shortcut if needed
            if stride != 1 or shortcut.shape[-1] != filters:
                shortcut = layers.Conv2D(filters, (1, 1), strides=stride)(shortcut)
                shortcut = layers.BatchNormalization()(shortcut)
            
            # Add shortcut
            x = layers.Add()([x, shortcut])
            x = layers.Activation('relu')(x)
            
            return x
        
        # Add residual blocks
        x = residual_block(x, 64)
        x = residual_block(x, 64)
        x = residual_block(x, 128, stride=2)
        x = residual_block(x, 128)
        x = residual_block(x, 256, stride=2)
        x = residual_block(x, 256)
        
        # Global average pooling
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dropout(0.5)(x)
        
        # Dense layers
        x = layers.Dense(512, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.3)(x)
        x = layers.Dense(256, activation='relu')(x)
        x = layers.Dropout(0.2)(x)
        
        outputs = layers.Dense(self.num_classes, activation='softmax')(x)
        
        model = keras.Model(inputs, outputs)
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy', 'top_3_accuracy']
        )
        
        return model
    
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
    
    def create_attention_model(self):
        """Create a model with attention mechanism."""
        inputs = layers.Input(shape=self.input_shape)
        
        # Backbone
        backbone = ResNet50(weights='imagenet', include_top=False, input_tensor=inputs)
        backbone.trainable = False
        
        x = backbone.output
        x = layers.GlobalAveragePooling2D()(x)
        
        # Attention mechanism
        attention = layers.Dense(self.num_classes, activation='softmax', name='attention')(x)
        features = layers.Dense(self.num_classes, activation='relu')(x)
        
        # Combine attention with features
        attended_features = layers.Multiply()([features, attention])
        
        outputs = layers.Dense(self.num_classes, activation='softmax')(attended_features)
        
        model = keras.Model(inputs, outputs)
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.0001),
            loss='categorical_crossentropy',
            metrics=['accuracy', 'top_3_accuracy']
        )
        
        return model
    
    def focal_loss(self, y_true, y_pred, alpha=0.25, gamma=2.0):
        """Focal loss implementation for handling class imbalance."""
        # Clip predictions to prevent NaN's and Inf's
        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)
        
        # Calculate cross entropy
        ce = -y_true * tf.math.log(y_pred)
        
        # Calculate focal weight
        weight = alpha * y_true * tf.pow((1 - y_pred), gamma)
        
        # Calculate focal loss
        fl = weight * ce
        return tf.reduce_mean(tf.reduce_sum(fl, axis=1))
    
    def create_focal_loss_model(self):
        """Create a model using focal loss for class imbalance."""
        model = keras.Sequential([
            ResNet50(weights='imagenet', include_top=False, input_shape=self.input_shape),
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
            loss=self.focal_loss,
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
        if hasattr(model.layers[0], 'trainable'):
            model.layers[0].trainable = True
        
        # Fine-tune from this layer onwards
        if hasattr(model.layers[0], 'layers'):
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

    def train_all_approaches(self):
        """Train using all possible approaches."""
        print("=" * 80)
        print("COMPREHENSIVE ANIMAL IMAGE CLASSIFICATION TRAINING")
        print("=" * 80)
        print(f"Dataset: {self.data_dir}")
        print(f"Number of classes: {self.num_classes}")
        print(f"Classes: {list(self.class_labels.values())}")
        print("=" * 80)
        
        # Create data generators
        print("Creating data generators...")
        train_gen, val_gen = self.create_data_generators(batch_size=16)  # Reduced batch size for memory
        
        # Ensure models directory exists
        os.makedirs('models', exist_ok=True)
        
        # Approach 1: Simple CNN
        print("\nTraining Simple CNN...")
        simple_cnn = self.create_simple_cnn()
        self.models['simple_cnn'] = simple_cnn
        self.history['simple_cnn'] = self.train_model(
            simple_cnn, 'simple_cnn', train_gen, val_gen, epochs=30
        )
        
        # Evaluate
        eval_results = simple_cnn.evaluate(val_gen, verbose=0)
        self.results['simple_cnn'] = {
            'loss': float(eval_results[0]),
            'accuracy': float(eval_results[1]),
            'top_3_accuracy': float(eval_results[2]) if len(eval_results) > 2 else 0
        }
        
        # Approach 2: Advanced CNN
        print("\nTraining Advanced CNN...")
        advanced_cnn = self.create_advanced_cnn()
        self.models['advanced_cnn'] = advanced_cnn
        self.history['advanced_cnn'] = self.train_model(
            advanced_cnn, 'advanced_cnn', train_gen, val_gen, epochs=30
        )
        
        # Evaluate
        eval_results = advanced_cnn.evaluate(val_gen, verbose=0)
        self.results['advanced_cnn'] = {
            'loss': float(eval_results[0]),
            'accuracy': float(eval_results[1]),
            'top_3_accuracy': float(eval_results[2]) if len(eval_results) > 2 else 0
        }
        
        # Approach 3: Transfer Learning with ResNet50
        print("\nTraining ResNet50 Transfer Learning...")
        resnet_model = self.create_transfer_learning_model('ResNet50')
        self.models['resnet50'] = resnet_model
        self.history['resnet50'] = self.train_model(
            resnet_model, 'resnet50', train_gen, val_gen, epochs=25
        )
        
        # Fine-tune ResNet50
        print("\nFine-tuning ResNet50...")
        self.history['resnet50_finetuned'] = self.fine_tune_transfer_model(
            resnet_model, 'resnet50', train_gen, val_gen, epochs=15
        )
        
        # Evaluate
        eval_results = resnet_model.evaluate(val_gen, verbose=0)
        self.results['resnet50'] = {
            'loss': float(eval_results[0]),
            'accuracy': float(eval_results[1]),
            'top_3_accuracy': float(eval_results[2]) if len(eval_results) > 2 else 0
        }
        
        # Approach 4: Transfer Learning with EfficientNetB0
        print("\nTraining EfficientNetB0 Transfer Learning...")
        efficientnet_model = self.create_transfer_learning_model('EfficientNetB0')
        self.models['efficientnet'] = efficientnet_model
        self.history['efficientnet'] = self.train_model(
            efficientnet_model, 'efficientnet', train_gen, val_gen, epochs=25
        )
        
        # Fine-tune EfficientNetB0
        print("\nFine-tuning EfficientNetB0...")
        self.history['efficientnet_finetuned'] = self.fine_tune_transfer_model(
            efficientnet_model, 'efficientnet', train_gen, val_gen, epochs=15
        )
        
        # Evaluate
        eval_results = efficientnet_model.evaluate(val_gen, verbose=0)
        self.results['efficientnet'] = {
            'loss': float(eval_results[0]),
            'accuracy': float(eval_results[1]),
            'top_3_accuracy': float(eval_results[2]) if len(eval_results) > 2 else 0
        }
        
        # Approach 5: Transfer Learning with MobileNetV2
        print("\nTraining MobileNetV2 Transfer Learning...")
        mobilenet_model = self.create_transfer_learning_model('MobileNetV2')
        self.models['mobilenet'] = mobilenet_model
        self.history['mobilenet'] = self.train_model(
            mobilenet_model, 'mobilenet', train_gen, val_gen, epochs=25
        )
        
        # Evaluate
        eval_results = mobilenet_model.evaluate(val_gen, verbose=0)
        self.results['mobilenet'] = {
            'loss': float(eval_results[0]),
            'accuracy': float(eval_results[1]),
            'top_3_accuracy': float(eval_results[2]) if len(eval_results) > 2 else 0
        }
        
        # Approach 6: Transfer Learning with VGG16
        print("\nTraining VGG16 Transfer Learning...")
        vgg_model = self.create_transfer_learning_model('VGG16')
        self.models['vgg16'] = vgg_model
        self.history['vgg16'] = self.train_model(
            vgg_model, 'vgg16', train_gen, val_gen, epochs=25
        )
        
        # Fine-tune VGG16
        print("\nFine-tuning VGG16...")
        self.history['vgg16_finetuned'] = self.fine_tune_transfer_model(
            vgg_model, 'vgg16', train_gen, val_gen, epochs=15
        )
        
        # Evaluate
        eval_results = vgg_model.evaluate(val_gen, verbose=0)
        self.results['vgg16'] = {
            'loss': float(eval_results[0]),
            'accuracy': float(eval_results[1]),
            'top_3_accuracy': float(eval_results[2]) if len(eval_results) > 2 else 0
        }
        
        # Approach 7: Transfer Learning with DenseNet121
        print("\nTraining DenseNet121 Transfer Learning...")
        densenet_model = self.create_transfer_learning_model('DenseNet121')
        self.models['densenet121'] = densenet_model
        self.history['densenet121'] = self.train_model(
            densenet_model, 'densenet121', train_gen, val_gen, epochs=25
        )
        
        # Evaluate
        eval_results = densenet_model.evaluate(val_gen, verbose=0)
        self.results['densenet121'] = {
            'loss': float(eval_results[0]),
            'accuracy': float(eval_results[1]),
            'top_3_accuracy': float(eval_results[2]) if len(eval_results) > 2 else 0
        }
        
        # Approach 8: Attention Model
        print("\nTraining Attention Model...")
        attention_model = self.create_attention_model()
        self.models['attention'] = attention_model
        self.history['attention'] = self.train_model(
            attention_model, 'attention', train_gen, val_gen, epochs=20
        )
        
        # Evaluate
        eval_results = attention_model.evaluate(val_gen, verbose=0)
        self.results['attention'] = {
            'loss': float(eval_results[0]),
            'accuracy': float(eval_results[1]),
            'top_3_accuracy': float(eval_results[2]) if len(eval_results) > 2 else 0
        }
        
        # Approach 9: Focal Loss Model
        print("\nTraining Focal Loss Model...")
        focal_model = self.create_focal_loss_model()
        self.models['focal_loss'] = focal_model
        self.history['focal_loss'] = self.train_model(
            focal_model, 'focal_loss', train_gen, val_gen, epochs=20
        )
        
        # Evaluate
        eval_results = focal_model.evaluate(val_gen, verbose=0)
        self.results['focal_loss'] = {
            'loss': float(eval_results[0]),
            'accuracy': float(eval_results[1]),
            'top_3_accuracy': float(eval_results[2]) if len(eval_results) > 2 else 0
        }
        
        # Print final results
        print("\n" + "=" * 80)
        print("COMPREHENSIVE TRAINING COMPLETED!")
        print("=" * 80)
        print("Final Results Summary:")
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
    """Main function to run comprehensive animal image classification training."""
    print("🐾 Starting Comprehensive Animal Image Classification Training 🐾")
    
    # Initialize the classifier with the new dataset
    classifier = ComprehensiveAnimalClassifier()
    
    # Run the complete pipeline
    results = classifier.train_all_approaches()
    
    print("\n🎉 Comprehensive training completed successfully!")
    print("Models have been saved to the 'models/' directory")
    print("You can now use these models for animal image classification.")


if __name__ == "__main__":
    main()