from src.predict import WasteClassifier
import os
import sys

project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)
MODEL_SAVE_PATH = os.path.join(project_root, 'models', 'best_model.h5')
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
try:
    classifier = WasteClassifier(MODEL_SAVE_PATH)
    classifier.set_class_info(CLASS_NAMES, BIODEGRADABILITY)
        
    
    sample_image_path = 'data/test/paper waste/paper waste (1).jpg'
    prediction = classifier.predict(sample_image_path)
    print("Prediction:", prediction)
    
except Exception as e:
        print(f"Prediction failed: {e}")
        import traceback
        traceback.print_exc()