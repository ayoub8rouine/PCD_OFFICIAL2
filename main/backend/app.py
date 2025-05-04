from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_jwt_extended import JWTManager, create_access_token
from sqlalchemy import inspect
import uuid
import os
import numpy as np
import tempfile
from PIL import Image
import time
from images_models.model_classifier import ModelClassifier
from images_models.blood_model import BloodModel
from images_models.skin_model import SkinModelClassifier
from images_models.brain_model import BrainModel
from images_models.chest_model import ChestClassifier
from disease_model.openAI_api import AzureAssistant
from disease_model.configopenAI import AzureOpenAIConfig
from disease_model.main_model import DiseasePredictor
from  disease_model.audioconverter import AudioToTextConverter

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

@app.route('/uploads', methods=['POST'])
def post_data():
    start_time = time.time() 
    input_text = request.form.get('text')
    image_file = request.files.get('image')
    audio_file = request.files.get('audio')

    domain_result = "unknown"
    probs = []
    final_output=""
    client_pred=""
    result_llm=""
    response=""
    imput_for_llm=""
    list_vect_symp=list()

    config = AzureOpenAIConfig()
    ai = AzureAssistant(config)
    if audio_file:
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                audio_file.save(temp_file.name)
                temp_path = temp_file.name

            transcriber =  AudioToTextConverter(temp_path)
            transcriber.transcribe()
            input_text = transcriber.get_text()

        finally:
            if 'temp_path' in locals():
                try:
                    os.unlink(temp_path)
                except Exception as e:
                    app.logger.error(f"Error deleting temp file: {str(e)}")
    if input_text:
        imput_for_llm="extract symptoms from this phrase "+input_text
        response = ai.ask(imput_for_llm)
        list_reponse=response.split("#")
        symtoms=['anxiety and nervousness', 'depression', 'shortness of breath', 'depressive or psychotic symptoms', 'sharp chest pain', 'dizziness', 'insomnia', 'abnormal involuntary movements', 'chest tightness', 'palpitations', 'irregular heartbeat', 'breathing fast', 'hoarse voice', 'sore throat', 'difficulty speaking', 'cough', 'nasal congestion', 'throat swelling', 'diminished hearing', 'lump in throat', 'throat feels tight', 'difficulty in swallowing', 'skin swelling', 'retention of urine', 'groin mass', 'leg pain', 'hip pain', 'suprapubic pain', 'blood in stool', 'lack of growth', 'emotional symptoms', 'elbow weakness', 'back weakness', 'symptoms of the scrotum and testes', 'swelling of scrotum', 'pain in testicles', 'flatulence', 'pus draining from ear', 'jaundice', 'mass in scrotum', 'white discharge from eye', 'irritable infant', 'abusing alcohol', 'fainting', 'hostile behavior', 'drug abuse', 'sharp abdominal pain', 'feeling ill', 'vomiting', 'headache', 'nausea', 'diarrhea', 'vaginal itching', 'vaginal dryness', 'painful urination', 'involuntary urination', 'pain during intercourse', 'frequent urination', 'lower abdominal pain', 'vaginal discharge', 'blood in urine', 'hot flashes', 'intermenstrual bleeding', 'hand or finger pain', 'wrist pain', 'hand or finger swelling', 'arm pain', 'wrist swelling', 'arm stiffness or tightness', 'arm swelling', 'hand or finger stiffness or tightness', 'wrist stiffness or tightness', 'lip swelling', 'toothache', 'abnormal appearing skin', 'skin lesion', 'acne or pimples', 'dry lips', 'facial pain', 'mouth ulcer', 'skin growth', 'eye deviation', 'diminished vision', 'double vision', 'cross-eyed', 'symptoms of eye', 'pain in eye', 'eye moves abnormally', 'abnormal movement of eyelid', 'foreign body sensation in eye', 'irregular appearing scalp', 'swollen lymph nodes', 'back pain', 'neck pain', 'low back pain', 'pain of the anus', 'pain during pregnancy', 'pelvic pain', 'impotence', 'infant spitting up', 'vomiting blood', 'regurgitation', 'burning abdominal pain', 'restlessness', 'symptoms of infants', 'wheezing', 'peripheral edema', 'neck mass', 'ear pain', 'jaw swelling', 'mouth dryness', 'neck swelling', 'knee pain', 'foot or toe pain', 'bowlegged or knock-kneed', 'ankle pain', 'bones are painful', 'knee weakness', 'elbow pain', 'knee swelling', 'skin moles', 'knee lump or mass', 'weight gain', 'problems with movement', 'knee stiffness or tightness', 'leg swelling', 'foot or toe swelling', 'heartburn', 'smoking problems', 'muscle pain', 'infant feeding problem', 'recent weight loss', 'problems with shape or size of breast', 'difficulty eating', 'scanty menstrual flow', 'vaginal pain', 'vaginal redness', 'vulvar irritation', 'weakness', 'decreased heart rate', 'increased heart rate', 'bleeding or discharge from nipple', 'ringing in ear', 'plugged feeling in ear', 'itchy ear(s)', 'frontal headache', 'fluid in ear', 'neck stiffness or tightness', 'spots or clouds in vision', 'eye redness', 'lacrimation', 'itchiness of eye', 'blindness', 'eye burns or stings', 'itchy eyelid', 'feeling cold', 'decreased appetite', 'excessive appetite', 'excessive anger', 'loss of sensation', 'focal weakness', 'slurring words', 'symptoms of the face', 'disturbance of memory', 'paresthesia', 'side pain', 'fever', 'shoulder pain', 'shoulder stiffness or tightness', 'shoulder weakness', 'shoulder swelling', 'tongue lesions', 'leg cramps or spasms', 'ache all over', 'lower body pain', 'problems during pregnancy', 'spotting or bleeding during pregnancy', 'cramps and spasms', 'upper abdominal pain', 'stomach bloating', 'changes in stool appearance', 'unusual color or odor to urine', 'kidney mass', 'swollen abdomen', 'symptoms of prostate', 'leg stiffness or tightness', 'difficulty breathing', 'rib pain', 'joint pain', 'muscle stiffness or tightness', 'hand or finger lump or mass', 'chills', 'groin pain', 'fatigue', 'abdominal distention', 'regurgitation.1', 'symptoms of the kidneys', 'melena', 'flushing', 'coughing up sputum', 'seizures', 'delusions or hallucinations', 'pain or soreness of breast', 'excessive urination at night', 'bleeding from eye', 'rectal bleeding', 'constipation', 'temper problems', 'coryza', 'wrist weakness', 'hemoptysis', 'lymphedema', 'skin on leg or foot looks infected', 'allergic reaction', 'congestion in chest', 'muscle swelling', 'low back weakness', 'sleepiness', 'apnea', 'abnormal breathing sounds', 'excessive growth', 'blood clots during menstrual periods', 'absence of menstruation', 'pulling at ears', 'gum pain', 'redness in ear', 'fluid retention', 'flu-like syndrome', 'sinus congestion', 'painful sinuses', 'fears and phobias', 'recent pregnancy', 'uterine contractions', 'burning chest pain', 'back cramps or spasms', 'stiffness all over', 'muscle cramps, contractures, or spasms', 'low back cramps or spasms', 'back mass or lump', 'nosebleed', 'long menstrual periods', 'heavy menstrual flow', 'unpredictable menstruation', 'painful menstruation', 'infertility', 'frequent menstruation', 'sweating', 'mass on eyelid', 'swollen eye', 'eyelid swelling', 'eyelid lesion or rash', 'unwanted hair', 'symptoms of bladder', 'irregular appearing nails', 'itching of skin', 'hurts to breath', 'skin dryness, peeling, scaliness, or roughness', 'skin on arm or hand looks infected', 'skin irritation', 'itchy scalp', 'incontinence of stool', 'warts', 'bumps on penis', 'too little hair', 'foot or toe lump or mass', 'skin rash', 'mass or swelling around the anus', 'ankle swelling', 'drainage in throat', 'dry or flaky scalp', 'premenstrual tension or irritability', 'feeling hot', 'foot or toe stiffness or tightness', 'pelvic pressure', 'elbow swelling', 'early or late onset of menopause', 'bleeding from ear', 'hand or finger weakness', 'low self-esteem', 'itching of the anus', 'swollen or red tonsils', 'irregular belly button', 'lip sore', 'vulvar sore', 'hip stiffness or tightness', 'mouth pain', 'arm weakness', 'leg lump or mass', 'penis pain', 'loss of sex drive', 'obsessions and compulsions', 'antisocial behavior', 'neck cramps or spasms', 'poor circulation', 'thirst', 'sneezing', 'bladder mass', 'premature ejaculation', 'leg weakness', 'penis redness', 'penile discharge', 'shoulder lump or mass', 'cloudy eye', 'hysterical behavior', 'arm lump or mass', 'nightmares', 'bleeding gums', 'pain in gums', 'bedwetting', 'diaper rash', 'lump or mass of breast', 'vaginal bleeding after menopause', 'itching of scrotum', 'postpartum problems of the breast', 'hesitancy', 'muscle weakness', 'throat redness', 'joint swelling', 'redness in or around nose', 'wrinkles on skin', 'foot or toe weakness', 'hand or finger cramps or spasms', 'back stiffness or tightness', 'wrist lump or mass', 'skin pain', 'low urine output', 'sore in nose', 'ankle weakness']
        for i in range(len(symtoms)):
            if symtoms[i] in list_reponse:
                list_vect_symp.append(1)
            else:
                list_vect_symp.append(0)
        print("AI Response:", response)
        # Define model paths for the symptom classifier and group models
        symptom_classifier_model_path = os.path.join(
            app.root_path,  # Flask app's root directory
            'models-weight',  # models folder
            'symptom-classifier-weight-model.pkl'  # model file
        )

        group_100plus_model_path = os.path.join(
            app.root_path,  # Flask app's root directory
            'models-weight',  # models folder
            'group-100plus-weight-model.pkl'  # model file
        )

        group_10to99_model_path = os.path.join(
            app.root_path,  # Flask app's root directory
            'models-weight',  # models folder
            'group-10to99-weight-model.pkl'  # model file
        )

        le_100plus_path = os.path.join(
            app.root_path,  # Flask app's root directory
            'model-LabelEncoder',  # Label encoder folder
            'le_100plus.pkl'  # label encoder file for 100+ group
        )

        le_10to99_path = os.path.join(
            app.root_path,  # Flask app's root directory
            'model-LabelEncoder',  # Label encoder folder
            'le_10to99.pkl'  # label encoder file for 10-99 group
        )

        # Initialize the DiseasePredictor with the correct model paths
        predictor = DiseasePredictor(
            group_model_path=symptom_classifier_model_path,
            model_100plus_path=group_100plus_model_path,
            model_10to99_path=group_10to99_model_path,
            le_100plus_path=le_100plus_path,
            le_10to99_path=le_10to99_path
        )
        result_llm = predictor.predict_disease_from_symptoms(list_vect_symp)


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
                'chest-wright-model-model.pth'   
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
                pred_class=chest_model.predict(temp_path)
                
            probs = probs.tolist() if isinstance(probs, np.ndarray) else probs
            class_probs = class_probs.tolist() if isinstance(class_probs, np.ndarray) else class_probs
                
                
        finally:
            # Cleanup temp file
            if 'temp_path' in locals():
                try:
                    os.unlink(temp_path)
                except Exception as e:
                    app.logger.error(f"Error deleting temp file: {str(e)}")
    final_output=""
    if input_text:
        final_output="this is the discriptyion of the patient : "+input_text+" this the prediction from my model from the patient discription : "+result_llm["predicted_disease"]+"."
    if image_file:
        final_output=final_output+" this the prediction from the image that my model gave the domain is : "+domain_result+" and the disease is : "+pred_class

    client_pred=ai.ask(final_output)
    end_time = time.time()  # End timing

    elapsed_time = end_time - start_time

    return jsonify({
        'status': 'success',
        'text': client_pred
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
