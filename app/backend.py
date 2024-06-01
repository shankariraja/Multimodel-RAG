# app/backend.py
from flask import Flask, request, jsonify
import os

app = Flask(__name__)

@app.route('/process', methods=['POST'])
def process_file_and_query():
    if 'file' not in request.files or 'query' not in request.form:
        return jsonify({'response': 'No file or query provided'}), 400
    
    file = request.files['file']
    query = request.form['query']
    
    # Process the file and query here (dummy processing for demonstration)
    response = f'Processed query: {query}, with file: {file.filename}'
    
    return jsonify({'response': response})

if __name__ == '__main__':
    app.run(port=5001)
