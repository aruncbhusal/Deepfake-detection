import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
import logging
from backend.preprocessing import extract_frames
from backend.inference import run_inference
from backend.config import UPLOAD_FOLDER

logging.basicConfig(level=logging.INFO)

# Define constants
ALLOWED_EXTENSIONS = {"mp4", "avi", "mov", "mkv"}

# Ensure the upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Define Flask app
app = Flask(__name__)
CORS(app)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


# File validation
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Prediction endpoint
@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]

    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    if not allowed_file(file.filename):
        return jsonify({"error": "Invalid file format"}), 400
    
    logging.info(f"Received file: {file.filename}")
    
    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)

    try:
        frames = extract_frames(file_path)
        prediction = run_inference(frames)

        result = "Fake" if prediction == 1 else "Real"
        logging.info(f"Prediction result: {result}")

        return jsonify({"prediction": result})

    except Exception as e:
        logging.error(f"Inference failed: {str(e)}")
        return jsonify({"error": str(e)}), 500

    finally:
        if os.path.exists(file_path):
            os.remove(file_path)

def run_app():
    app.run(host="0.0.0.0", port=5000, debug=True)
