from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_jwt_extended import JWTManager, create_access_token
from sqlalchemy import inspect
import uuid

from config import Config
from models import db, bcrypt, User

app = Flask(__name__)
app.config.from_object(Config)
CORS(app)

# Initialize extensions with app
db.init_app(app)
bcrypt.init_app(app)
jwt = JWTManager(app)

@app.route('/post', methods=['POST'])
def post_data():
    audio_text = ""
    input_text = request.form.get('text')
    image_file = request.files.get('image')
    audio_file = request.files.get('audio')


    print(f"Received text: {input_text}")
    print(f"Received image: {image_file.filename if image_file else 'No image'}")
    print(f"Received audio: {audio_file.filename if audio_file else 'No audio'}")
    
    # Your existing processing logic here
    return 'Processed'

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
