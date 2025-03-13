import numpy as np
import tensorflow as tf
import cv2

class WasteClassifier:
    def __init__(self, model_path):
        self.model = tf.keras.models.load_model(model_path)
        self.class_names = None
        self.biodegradability = None
    
    def set_class_info(self, class_names, biodegradability):
        self.class_names = class_names
        self.biodegradability = biodegradability
    
    def predict(self, image_path, img_size=(224, 224)):
        # Load and preprocess image
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, img_size)
        img_array = img / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        # Make prediction
        prediction = self.model.predict(img_array)[0]
        predicted_class_idx = np.argmax(prediction)
        confidence = prediction[predicted_class_idx] * 100
        
        # Get class name and biodegradability
        predicted_class = self.class_names[predicted_class_idx]
        is_biodegradable = self.biodegradability.get(predicted_class, False)
        
        return {
            'waste_type': predicted_class,
            'confidence': confidence,
            'is_biodegradable': is_biodegradable
        }