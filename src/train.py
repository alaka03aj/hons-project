import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def create_data_generators(train_dir, test_dir, img_height=224, img_width=224, batch_size=16):
    # Set random seeds for reproducibility
    np.random.seed(42)
    tf.random.set_seed(42)
    
    # Training data augmentation
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        validation_split=0.2  # 20% validation
    )
    
    # Test data only rescaling
    test_datagen = ImageDataGenerator(rescale=1./255)
    
    # Create generators with explicit subset
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='categorical',
        subset='training',
        shuffle=True,
        seed=42
    )
    
    validation_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation',
        shuffle=True,
        seed=42
    )
    
    # Test data generator
    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False
    )
    
    return train_generator, validation_generator, test_generator

def create_callbacks(model_save_path):
    # Early stopping
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True,
        verbose=1
    )
    
    # Learning rate reduction
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-6,
        verbose=1
    )
    
    # Model checkpointing
    model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath=model_save_path,
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    )
    
    return [early_stopping, reduce_lr, model_checkpoint]

def train_model(train_dir, test_dir, model_save_path, img_height=224, img_width=224, batch_size=16, epochs=50):
    # Configure GPU memory growth
    try:
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
    except Exception as e:
        print(f"GPU memory growth configuration failed: {e}")
    
    # Create data generators
    train_generator, validation_generator, test_generator = create_data_generators(
        train_dir, test_dir, img_height, img_width, batch_size
    )
    
    # Get number of classes
    num_classes = len(train_generator.class_indices)
    
    # Import model creation function here to avoid circular import
    from .model import create_waste_classification_model
    
    # Create model
    model = create_waste_classification_model(
        input_shape=(img_height, img_width, 3), 
        num_classes=num_classes
    )
    
    # Create callbacks
    callbacks = create_callbacks(model_save_path)
    
    # Calculate steps
    steps_per_epoch = train_generator.samples // batch_size
    validation_steps = validation_generator.samples // batch_size
    
    try:
        # Train the model
        history = model.fit(
            train_generator,
            steps_per_epoch=steps_per_epoch,
            epochs=epochs,
            validation_data=validation_generator,
            validation_steps=validation_steps,
            callbacks=callbacks,
            verbose=1
        )
        
        # Evaluate on test set
        test_loss, test_accuracy = model.evaluate(test_generator)
        print(f"Test Accuracy: {test_accuracy}")
        
        return model, history
    except Exception as e:
        print(f"Training failed: {e}")
        import traceback
        traceback.print_exc()
        return None, None
