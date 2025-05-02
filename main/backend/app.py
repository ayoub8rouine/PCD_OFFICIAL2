from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_jwt_extended import JWTManager, create_access_token
from sqlalchemy import inspect
import uuid
import os
import tempfile
from PIL import Image
from images_models.model_classifier import ModelClassifier
from images_models.blood_model import BloodModel
from images_models.skin_model import SkinModelClassifier
from images_models.brain_model import BrainModel
from  images_models.chest_model import ChestClassifier

from config import Config
from models import db, bcrypt, User

app = Flask(__name__)
app.config.from_object(Config)
CORS(app)

# Initialize extensions with app
db.init_app(app)
bcrypt.init_app(app)
jwt = JWTManager(app)

TEMP_IMAGES_DIR = os.path.join(tempfile.gettempdir(), "pcd_images")
os.makedirs(TEMP_IMAGES_DIR, exist_ok=True)

@app.route('/post', methods=['POST'])
def post_data():
    input_text = request.form.get('text')
    image_file = request.files.get('image')
    audio_file = request.files.get('audio')

    domain_result = "unknown"
    probs = []
    
    if image_file:
        try:
            # Create secure temporary file
            with tempfile.NamedTemporaryFile(
                dir=TEMP_IMAGES_DIR,
                suffix=".jpg",
                delete=False
            ) as temp_file:
                image_file.save(temp_file.name)
                temp_path = temp_file.name

            # Define model paths for each domain
            image_classifier_model_path = os.path.join(
                app.root_path,  # Flask app's root directory
                'models-weight',  # models folder
                'image-classifier-weight-mpdel.pth'  # model file
            )

            blood_model_path = os.path.join(
                app.root_path,  # Flask app's root directory
                'models-weight',  # models folder
                'blood-weight-model.pth'  # model file
            )

            skin_model_path = os.path.join(
                app.root_path,  # Flask app's root directory
                'models-weight',  # models folder
                'skin-weight-model.pth'  # model file
            )

            brain_model_path = os.path.join(
                app.root_path,  # Flask app's root directory
                'models-weight',  # models folder
                'brain-weight-model.h5'  # model file
            )

            yolo_model_path = os.path.join(
                app.root_path,  # Flask app's root directory
                'models-weight',  # models folder
                'yolo-weight-model.pt'  # model file
            )
            chest_model_path=os.path.join(
                app.root_path,
                'models-weight',
                'chest-wright-model-model'
                ''   
            )

            # Initialize the image classifier with the correct model path
            classifier = ModelClassifier(model_path=image_classifier_model_path)
            
            # Predict the domain from the image
            pred_class_name, probs = classifier.predict_image(temp_path)
            domain_result = pred_class_name

            # Handle each domain prediction separately using "if" and "not elif"
            if domain_result == "blood":
                model = BloodModel(
                    yolo_model_path=yolo_model_path,
                    patch_classifier_model_path=blood_model_path,
                    num_classes=4
                )
                pred_class, class_probs = model.predict(temp_path)

            if domain_result == "skin":
                classifier = SkinModelClassifier(skin_model_path, num_classes=8)
                pred_class , class_probs= classifier.predict(temp_path)

            if domain_result == "brain":
                brain_model = BrainModel(model_path=brain_model_path)
                pred, class_probs, pred_class = brain_model.predict(temp_path)

            # still in progress
            if domain_result=="chest":
                chest_model=ChestClassifier(chest_model_path,4)
                chest_model.predict(temp_path)
                
                
        finally:
            # Cleanup temp file
            if 'temp_path' in locals():
                try:
                    os.unlink(temp_path)
                except Exception as e:
                    app.logger.error(f"Error deleting temp file: {str(e)}")

    return jsonify({
        'message': 'Processed',
        'domain': domain_result,
        'probabilities': probs.tolist() if 'probs' in locals() else [],
        'type': pred_class,
        'prob':  class_probs.tolist() if 'probs' in locals() else []
    })



@app.route('/signup', methods=['POST'])
def signup():
    data = request.get_json()
    name = data.get('name')
    email = data.get('email')
    password = data.get('password')
    role = data.get('role', 'user')

    if User.query.filter_by(email=email).first():
        return jsonify({"message": "Email already registered"}), 409

    new_user = User(
        name=name,
        email=email,
        role=role
    )
    new_user.set_password(password)

    db.session.add(new_user)
    db.session.commit()

    return jsonify({"message": "User registered successfully", "user_id": new_user.id}), 201



@app.route('/signin', methods=['POST'])
def signin():
    data = request.get_json()
    email = data.get('email')
    password = data.get('password')

    if not email or not password:
        return jsonify({"error": "Email and password are required"}), 400

    user = User.query.filter_by(email=email).first()

    if user and user.check_password(password):
        return jsonify({"message": "Signin successful", "user_id": user.id})
    else:
        return jsonify({"error": "Invalid email or password"}), 401

def create_tables_if_not_exist():
    with app.app_context():
        inspector = inspect(db.engine)
        if not inspector.has_table("user"):
            db.create_all()
            print("Tables created.")
        else:
            print("Tables already exist.")

if __name__ == '__main__':
    create_tables_if_not_exist()
    print(f"App root path: {app.root_path}")

    app.run(debug=True)
