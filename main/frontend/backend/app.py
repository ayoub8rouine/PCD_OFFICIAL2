from flask import Flask, request, jsonify, send_from_directory, url_for
from flask_cors import CORS
from werkzeug.utils import secure_filename
import os
import time

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Configure upload folder
UPLOAD_FOLDER = './uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload

@app.route('/api/chat', methods=['POST'])
def chat():
    """
    Handle chat API requests with multimodal inputs
    """
    text = request.form.get('text', '')
    image = request.files.get('image')
    audio = request.files.get('audio')
    
    time.sleep(1)
    
    response = {}
    
    if text:
        response['text'] = f"You said: {text}"
    
    if image:
        filename = secure_filename(image.filename)
        filename = f"{int(time.time())}_{filename}"
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        image.save(image_path)
        image_url = url_for('uploaded_file', filename=filename, _external=True)
        response['image'] = image_url

        if not text:
            response['text'] = "I received your image. Here's what it looks like."
    
    if audio:
        filename = secure_filename(audio.filename)
        filename = f"{int(time.time())}_{filename}"
        audio_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        audio.save(audio_path)
        audio_url = url_for('uploaded_file', filename=filename, _external=True)
        response['audio'] = audio_url

        if not text and not image:
            response['text'] = "I received your audio recording. Here it is played back."
    
    if not text and not image and not audio:
        response['text'] = "Please send a message, image, or audio recording."
    
    return jsonify(response)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    """
    Serve uploaded files
    """
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    print("Starting Flask server on http://localhost:8001")
    app.run(host='0.0.0.0', port=8001, debug=True)
