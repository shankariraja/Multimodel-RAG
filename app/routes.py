# app/routes.py
from flask import Blueprint,Flask, request, jsonify
import os
import requests

routes = Blueprint('routes', __name__)

ALLOWED_EXTENSIONS = {'pdf'}
UPLOAD_FOLDER = 'uploads'
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16 MB

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@routes.route('/')
def index():
    return "Welcome to the Multimodel RAG API!"

@routes.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files or 'query' not in request.form:
        return jsonify({'response': 'No file or query provided'}), 400
    
    file = request.files['file']
    query = request.form['query']
    
    if file.filename == '':
        return jsonify({'response': 'No selected file'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'response': 'File type not allowed'}), 400
    
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(file_path)
    
    # Forward the file and query to your backend
    files = {'file': open(file_path, 'rb')}
    data = {'query': query}
    
    try:
        backend_url = f'http://127.0.0.1:{os.getenv("PORT", "5001")}/process'  # Replace with your actual backend URL
        response = requests.post(backend_url, files=files, data=data)
        
        if response.status_code == 200:
            # Backend returned a successful response
            response_text = response.text
            return jsonify({'response': response_text}), 200
        else:
            # Backend returned an error response
            return jsonify({'response': 'Error processing your request'}), 500
    
    except requests.exceptions.RequestException as e:
        print(f"Error forwarding request to backend: {e}")
        return jsonify({'response': 'Error processing your request'}), 500
