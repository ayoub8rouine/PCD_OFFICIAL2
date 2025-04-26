import joblib
import numpy as np

# Load models and label encoders (make sure you saved LabelEncoders for both models)
group_classifier = joblib.load('main\backend\models-weights\symptom-classifier-weight-model.pkl')
model_100plus = joblib.load('main\backend\models-weights\group-10to99-weight-model-.pkl')
model_10to99 = joblib.load('main\backend\models-weights\group-10to99-weight-model-.pkl')

le_100plus = joblib.load('main\backend\models-weights\group-10to99-weight-model-.pkl')  # LabelEncoder for 100+ diseases
le_10to99 = joblib.load('main\backend\models-weights\group-10to99-weight-model-.pkl')    # LabelEncoder for 10–99 diseases

def predict_disease_from_symptoms(symptom_vector):
    """
    symptom_vector: list or np.array, shape = (num_features,)
    
    Returns a dictionary with predicted group and disease name.
    """
    input_array = np.array(symptom_vector).reshape(1, -1)
    
    # Step 1: Predict group (0 = 10-99, 1 = 100+)
    group = group_classifier.predict(input_array)[0]
    
    # Step 2: Predict disease within that group
    if group == 1:
        disease_encoded = model_100plus.predict(input_array)[0]
        disease = le_100plus.inverse_transform([disease_encoded])[0]
        group_name = "100+ samples"
    else:
        disease_encoded = model_10to99.predict(input_array)[0]
        disease = le_10to99.inverse_transform([disease_encoded])[0]
        group_name = "10-99 samples"
    
    return {
        "group": group_name,
        "predicted_disease": disease
    }

