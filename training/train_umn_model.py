"""
Train UMN abnormal behavior detection model using Keras/TensorFlow.

This script loads CLAHE-preprocessed images from the UMN dataset
(labeled as 0=normal, 1=abnormal) and trains a CNN model.

Expected directory structure:
    train_data/
        0/ (normal images)
        1/ (abnormal images)
    test_data/
        0/
        1/
"""

import argparse
import os
import cv2
import numpy as np
from pathlib import Path
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import pickle


def count_images_in_dir(image_dir):
    """Count images in directory structure: dir/0/ and dir/1/"""
    count = 0
    for class_name in ['0', '1']:
        class_dir = os.path.join(image_dir, class_name)
        if not os.path.exists(class_dir):
            continue
        for img_file in os.listdir(class_dir):
            if img_file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                count += 1
    return count


def build_model(input_shape=(224, 224, 3)):
    """Build a CNN model for abnormal behavior detection"""
    model = Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(256, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(1, activation='sigmoid')  # Binary classification (normal=0, abnormal=1)
    ])
    
    return model


def main(args):
    train_dir = args.train_dir
    test_dir = args.test_dir
    output_model = args.output_model
    epochs = args.epochs
    batch_size = args.batch_size
    
    # Count images
    train_count = count_images_in_dir(train_dir)
    if train_count == 0:
        raise ValueError(f"No training images found in {train_dir}")
    
    print(f"Found {train_count} training images")
    
    # Use ImageDataGenerator for memory-efficient batch loading
    print("Setting up data generators...")
    train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255,
        horizontal_flip=True,
        vertical_flip=True,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        zoom_range=0.2,
        validation_split=0.2  # 80% train, 20% validation
    )
    
    # Load training data with generator (memory efficient)
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(224, 224),
        batch_size=batch_size,
        class_mode='binary',
        subset='training',
        classes={'0': 0, '1': 1}
    )
    
    val_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(224, 224),
        batch_size=batch_size,
        class_mode='binary',
        subset='validation',
        classes={'0': 0, '1': 1}
    )
    
    # Test generator (no augmentation for test data)
    test_count = 0
    test_generator = None
    if test_dir and os.path.exists(test_dir):
        test_count = count_images_in_dir(test_dir)
        if test_count > 0:
            print(f"Found {test_count} test images")
            test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
            test_generator = test_datagen.flow_from_directory(
                test_dir,
                target_size=(224, 224),
                batch_size=batch_size,
                class_mode='binary',
                shuffle=False,
                classes={'0': 0, '1': 1}
            )
    
    # Build model
    print("Building model...")
    model = build_model()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    print(model.summary())
    
    # Callbacks
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint(output_model, monitor='val_loss', save_best_only=True, verbose=1)
    ]
    
    # Train with generators
    print("Training model...")
    history = model.fit(
        train_generator,
        epochs=epochs,
        validation_data=val_generator,
        callbacks=callbacks,
        verbose=1
    )
    
    # Evaluate on test set if available
    if test_generator is not None:
        print("\nEvaluating on test set...")
        test_loss, test_accuracy = model.evaluate(test_generator, verbose=0)
        print(f"Test Loss: {test_loss:.4f}")
        print(f"Test Accuracy: {test_accuracy:.4f}")
    
    # Save model
    print(f"Saving model to {output_model}")
    model.save(output_model)
    
    # Save training history
    history_file = output_model.replace('.h5', '_history.pkl')
    with open(history_file, 'wb') as f:
        pickle.dump(history.history, f)
    print(f"Training history saved to {history_file}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train UMN abnormal behavior detection model")
    parser.add_argument("--train-dir", type=str, 
                        default=r"C:\Users\ASUS\Downloads\Crowd-Detection-and-Stampede-prevention-System-main\Crowd-Detection-and-Stampede-prevention-System-main\CrowdAnomalyDetection_DeepLearning-master\UMN\Final UMN dataset\UMN_CHALE_5159\train_UMN_clahe",
                        help="Path to training images directory (should contain 0/ and 1/ subdirs)")
    parser.add_argument("--test-dir", type=str, 
                        default=r"C:\Users\ASUS\Downloads\Crowd-Detection-and-Stampede-prevention-System-main\Crowd-Detection-and-Stampede-prevention-System-main\CrowdAnomalyDetection_DeepLearning-master\UMN\Final UMN dataset\UMN_CHALE_5159\train_UMN_clahe",
                        help="Path to test images directory (optional)")
    parser.add_argument("--output-model", type=str, default="umn_abnormal_model.h5",
                        help="Output path for trained model")
    parser.add_argument("--epochs", type=int, default=30,
                        help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Batch size for training")
    parser.add_argument("--validation-split", type=float, default=0.2,
                        help="Validation split ratio (if no test set provided)")
    args = parser.parse_args()
    
    main(args)
