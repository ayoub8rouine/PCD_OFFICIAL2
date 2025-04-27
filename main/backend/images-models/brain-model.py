import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# === MODEL ===
def load_brain_model(model_path='main\backend\models-weight\braintumor_binary.h5'):
    """Load the brain tumor detection model."""
    model = load_model(model_path)
    return model

# === TRANSFORM ===
def preprocess_brain_image(image_input, image_size=(128, 128)):
    """Preprocess the image for prediction."""
    if isinstance(image_input, str):
        img = load_img(image_input, target_size=image_size)
    else:
        img = image_input.resize(image_size)
    
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# === PREDICTION FUNCTION ===
def predict_brain_image(model_path, image_input):
   
    model = load_brain_model(model_path)
    
    img_array = preprocess_brain_image(image_input)
    
    pred_prob = model.predict(img_array)[0][0]
    pred_class = int(np.round(pred_prob))
    
    return pred_class, pred_prob 


'''if __name__ == "__main__":
    model_path = 'braintumor_binary.h5'
    image_path = 'sampleimages\yestumor.jpg'  # <-- your image here
    
    pred_class, prob = predict_brain_image(model_path, image_path)
    
    label = 'Yes Tumor' if pred_class == 1 else 'No Tumor'
    print(f"Prediction: {label} (Confidence: {prob:.4f})")'''
