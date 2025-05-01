import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

class BrainModel:
    def __init__(self, model_path='main/backend/models-weight/braintumor_binary.h5', image_size=(128, 128)):
        """
        Initializes the brain tumor detection model.
        :param model_path: Path to the Keras model (.h5)
        :param image_size: Target image size for the model input
        """
        self.model_path = model_path
        self.image_size = image_size
        self.model = self._load_model()

    def _load_model(self):
        """Load the brain tumor detection model from file."""
        return load_model(self.model_path)

    def preprocess_image(self, image_input):
        """
        Preprocesses an image file path or PIL image for model prediction.
        :param image_input: file path (str) or PIL.Image
        :return: Preprocessed image as numpy array
        """
        if isinstance(image_input, str):
            img = load_img(image_input, target_size=self.image_size)
        else:
            img = image_input.resize(self.image_size)

        img_array = img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        return img_array

    def predict(self, image_input):
        """
        Predicts whether a brain tumor is present.
        :param image_input: file path (str) or PIL.Image
        :return: (predicted_class, confidence_score, class_name)
        """
        img_array = self.preprocess_image(image_input)
        pred_prob = self.model.predict(img_array)[0][0]
        pred_class = int(np.round(pred_prob))
        class_name = 'Yes Tumor' if pred_class == 1 else 'No Tumor'
        return pred_class, pred_prob, class_name


# How to use 
# brain_model = BrainModel(model_path=r'main\backend\models-weight\brain-weight-model.h5')
# image_path = r'C:\Users\USER\Downloads\data\imageclassifier\train\brain\Te-no_0010.jpg'

# pred_class, prob, class_name = brain_model.predict(image_path)

# print(f"Prediction: {class_name} (Confidence: {prob:.4f})")
