import os
import sys
import tensorflow as tf

# Add project root to Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from src.train import train_model
from src.predict import WasteClassifier

# Project configuration
TRAIN_DIR = os.path.join(project_root, 'data', 'train')
TEST_DIR = os.path.join(project_root, 'data', 'test')
MODEL_SAVE_PATH = os.path.join(project_root, 'models', 'best_model.h5')

# Ensure models directory exists
os.makedirs(os.path.join(project_root, 'models'), exist_ok=True)

# Waste classification metadata
CLASS_NAMES = [
    'E-waste', 'Automobile Wastes', 'Battery Waste', 
    'Glass Waste', 'Light Bulbs', 'Metal Waste', 
    'Organic Waste', 'Paper Waste', 'Plastic Waste'
]

BIODEGRADABILITY = {
    'E-waste': False,
    'Automobile Wastes': False,
    'Battery Waste': False,
    'Glass Waste': False,
    'Light Bulbs': False,
    'Metal Waste': False,
    'Organic Waste': True,
    'Paper Waste': True,
    'Plastic Waste': False
}

def main():
    # Print system and TensorFlow information
    print("Python Version:", sys.version)
    print("TensorFlow Version:", tf.__version__)
    print("GPU Available:", tf.config.list_physical_devices('GPU'))
    
    # Train the model
    model, history = train_model(
        TRAIN_DIR, 
        TEST_DIR, 
        MODEL_SAVE_PATH, 
        img_height=224, 
        img_width=224, 
        batch_size=16,  # Reduced batch size 
        epochs=50
    )
    
    # Check if model training was successful
    if model is None:
        print("Model training failed.")
        return
    
    # Example prediction
    try:
        classifier = WasteClassifier(MODEL_SAVE_PATH)
        classifier.set_class_info(CLASS_NAMES, BIODEGRADABILITY)
        
        # Predict on a sample image (if available)
        sample_images = [
            os.path.join(TEST_DIR, class_name, os.listdir(os.path.join(TEST_DIR, class_name))[0])
            for class_name in os.listdir(TEST_DIR)
            if os.path.isdir(os.path.join(TEST_DIR, class_name))
        ]
        
        if sample_images:
            sample_image_path = sample_images[0]
            prediction = classifier.predict(sample_image_path)
            print("Prediction:", prediction)
        else:
            print("No sample images found for prediction.")
    
    except Exception as e:
        print(f"Prediction failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()