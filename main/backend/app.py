from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_jwt_extended import JWTManager, create_access_token
from sqlalchemy import inspect
import openai
import uuid
from main.backend.disease_model.audioconverter import AudioToTextConverter
from main.backend.images_models.model_classifier import ModelClassifier
from main.backend.disease_model.main_model import DiseasePredictor
from config import Config
from models import db, bcrypt, User

app = Flask(__name__)
app.config.from_object(Config)
CORS(app)

# Initialize extensions with app
db.init_app(app)
bcrypt.init_app(app)
jwt = JWTManager(app)

SYMPTOM_LIST = ['fever', 'headache', 'cough', 'fatigue', 'nausea', 'rash', 'vomiting', 'diarrhea']  # Example order

openai.api_key = os.getenv("OPENAI_API_KEY")

@app.route('/post', methods=['POST'])
def post_data():
    audio_text = ""
    input_text = request.form.get('text')
    image_file = request.files.get('image')
    audio_file = request.files.get('audio')

    # Step 1: Audio to text - Initialize only if audio is provided
    if audio_file:
        audio_converter = AudioToTextConverter(language='en')
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio:
            audio_file.save(temp_audio.name)
            audio_text = audio_converter.convert(temp_audio.name)
            os.remove(temp_audio.name)

    # Step 2: Combine text sources
    combined_text = " ".join(filter(None, [input_text, audio_text]))

    # Step 3: Image domain classification - Initialize only if image is provided
    domain_result = "unknown"
    if image_file:
        image_classifier = ModelClassifier(model_path='path/to/domain_classifier.pth')
        with tempfile.TemporaryDirectory() as temp_dir:
            domain_path = os.path.join(temp_dir, "domain")
            os.makedirs(domain_path, exist_ok=True)
            class_dir = os.path.join(domain_path, "0")
            os.makedirs(class_dir, exist_ok=True)
            image_path = os.path.join(class_dir, "input.jpg")
            image = Image.open(image_file).convert("RGB")
            image.save(image_path)
            preds = image_classifier.predict_directory(domain_path)
            if preds:
                domain_result = preds[0]

    # Step 4: Extract symptoms from OpenAI - Initialize and call only if combined text is available
    response_text = ""
    symptom_vector = [0] * len(SYMPTOM_LIST)
    if combined_text:
        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "Extract relevant symptoms from the user's input. Output the symptoms separated by #."},
                    {"role": "user", "content": combined_text}
                ]
            )
            response_text = response.choices[0].message['content']
            found_symptoms = set(map(str.strip, response_text.lower().split("#")))
            symptom_vector = [1 if sym.lower() in found_symptoms else 0 for sym in SYMPTOM_LIST]
        except Exception as e:
            response_text = f"[ERROR] OpenAI failed: {e}"

    # Step 5: Predict disease group and disease - Initialize only if symptom vector is not empty
    prediction_result = {}
    if any(symptom_vector):
        disease_predictor = DiseasePredictor(
            group_model_path='path/to/group_model.pkl',
            model_100plus_path='path/to/100plus_model.pkl',
            model_10to99_path='path/to/10to99_model.pkl',
            le_100plus_path='path/to/le_100plus.pkl',
            le_10to99_path='path/to/le_10to99.pkl'
        )
        prediction_result = disease_predictor.predict_disease_from_symptoms(symptom_vector)

    # Return the response
    return jsonify({
        "domain": domain_result,
        "audio_text": audio_text,
        "input_text": input_text,
        "combined_text": combined_text,
        "openai_response": response_text,
        "symptom_vector": symptom_vector,
        "prediction_result": prediction_result
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
    app.run(debug=True)
