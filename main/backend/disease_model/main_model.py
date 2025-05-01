import joblib
import numpy as np

class DiseasePredictor:
    def __init__(self, group_model_path, model_100plus_path, model_10to99_path, le_100plus_path, le_10to99_path):
        # Load models and label encoders
        self.group_classifier = joblib.load(group_model_path)
        self.model_100plus = joblib.load(model_100plus_path)
        self.model_10to99 = joblib.load(model_10to99_path)

        self.le_100plus = joblib.load(le_100plus_path)  # LabelEncoder for 100+ diseases
        self.le_10to99 = joblib.load(le_10to99_path)    # LabelEncoder for 10–99 diseases

    def predict_disease_from_symptoms(self, symptom_vector):
        """
        symptom_vector: list or np.array, shape = (num_features,)
        
        Returns a dictionary with predicted group and disease name.
        """
        input_array = np.array(symptom_vector).reshape(1, -1)

        # Step 1: Predict group (0 = 10-99, 1 = 100+)
        group = self.group_classifier.predict(input_array)[0]

        # Step 2: Predict disease within that group
        if group == 1:
            disease_encoded = self.model_100plus.predict(input_array)[0]
            disease = self.le_100plus.inverse_transform([disease_encoded])[0]
            group_name = "100+ samples"
        else:
            disease_encoded = self.model_10to99.predict(input_array)[0]
            disease = self.le_10to99.inverse_transform([disease_encoded])[0]
            group_name = "10-99 samples"

        return {
            "group": group_name,
            "predicted_disease": disease
        }

# Example Usage
# Initialize with paths to models and label encoders
# predictor = DiseasePredictor(
#     group_model_path='main/backend/models-weights/symptom-classifier-weight-model.pkl',
#     model_100plus_path='main/backend/models-weights/group-100plus-weight-model.pkl',
#     model_10to99_path='main/backend/models-weights/group-10to99-weight-model-.pkl',
#     le_100plus_path='main/backend/models-weights/le_100plus.pkl',
#     le_10to99_path='main/backend/models-weights/le_10to99.pkl'
# )

# # Example symptom vector
# symptom_vector = [0, 1, 0, 1, 1]  # example input

# # Get prediction
# result = predictor.predict_disease_from_symptoms(symptom_vector)
# print(result)
