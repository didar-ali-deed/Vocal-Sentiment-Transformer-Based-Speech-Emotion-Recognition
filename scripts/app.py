from flask import Flask, request, render_template, jsonify
from pathlib import Path
import os
import librosa
import torch
import torch.nn.functional as F
import numpy as np
from transformers import Wav2Vec2Processor, Wav2Vec2Model
import torch.nn as nn
import pandas as pd
import json
import time
import logging

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Paths
BASE_DIR = Path(__file__).parent.parent  # Project root
MODEL_PATH = BASE_DIR / "models" / "emotion_classifier" / "final_emotion_classifier_transformer.pth"
CONFIG_PATH = BASE_DIR / "models" / "emotion_classifier" / "model_config.json"
UPLOAD_FOLDER = BASE_DIR / "deployment" / "uploads"
PREDICTIONS_FILE = BASE_DIR / "results" / "inference_predictions.json"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024  # 10MB max file size

# Ensure upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(BASE_DIR / "results", exist_ok=True)

# Check if model file exists
if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model file not found at: {MODEL_PATH}")

# Load Wav2Vec2 processor and model
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
wav2vec_model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")

# Load the trained Transformer model
class EmotionTransformer(nn.Module):
    def __init__(self, input_dim, d_model, num_heads, num_layers, num_classes):
        super(EmotionTransformer, self).__init__()
        self.d_model = d_model
        
        self.input_projection = nn.Linear(input_dim, d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=num_heads, 
            dim_feedforward=512,  # Match training setup
            dropout=0.2, 
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, x):
        x = self.input_projection(x)  # [batch_size, d_model]
        x = x.unsqueeze(1)  # [batch_size, 1, d_model] for transformer
        x = self.transformer_encoder(x)  # [batch_size, 1, d_model]
        x = x.squeeze(1)  # [batch_size, d_model]
        return self.classifier(x)

# Load model configuration
with open(CONFIG_PATH, 'r') as f:
    config = json.load(f)

# Load label names (from training data)
features_df = pd.read_csv(BASE_DIR / "Extracted Features" / "combined_wav2vec_features.csv")
label_names = pd.factorize(features_df['label'])[1]

# Initialize the model
D_MODEL = config['d_model']
NUM_HEADS = config['num_heads']
NUM_LAYERS = config['num_layers']
feature_dim = 768  # Wav2Vec2 feature dimension
model = EmotionTransformer(
    input_dim=feature_dim,
    d_model=D_MODEL,
    num_heads=NUM_HEADS,
    num_layers=NUM_LAYERS,
    num_classes=len(label_names)
)
model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
model.eval()

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
wav2vec_model = wav2vec_model.to(device)

def extract_wav2vec_features(file_path):
    try:
        audio, sr = librosa.load(file_path, sr=16000)
        if len(audio) < 16000:  # Ensure audio is at least 1 second long
            raise ValueError("Audio must be at least 1 second long")
        inputs = processor(audio, return_tensors="pt", sampling_rate=16000, padding=True)
        inputs = {key: value.to(device) for key, value in inputs.items()}
        with torch.no_grad():
            features = wav2vec_model(**inputs).last_hidden_state
        features = features.mean(dim=1)  # Mean pool across time
        return features.squeeze().cpu().numpy()
    except Exception as e:
        logger.error(f"Feature extraction failed: {str(e)}")
        raise Exception(f"Failed to process audio: {str(e)}")

def predict_emotion(features):
    try:
        features = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            outputs = model(features)
            probs = F.softmax(outputs, dim=1).cpu().numpy()[0]
            pred = torch.argmax(outputs, dim=1).cpu().numpy()[0]
        emotion = label_names[pred] if pred < len(label_names) else "unknown"
        confidence = float(probs[pred] if pred < len(probs) else 0.0)
        probabilities = {label_names[i] if i < len(label_names) else "unknown": float(probs[i] if i < len(probs) else 0.0) for i in range(len(probs))}
        return emotion, probabilities, confidence
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        raise Exception(f"Prediction error: {str(e)}")

def save_prediction(audio_file, predicted_emotion, probabilities, confidence):
    prediction = {
        "audio_file": audio_file,
        "predicted_emotion": predicted_emotion,
        "confidence": confidence,
        "probabilities": probabilities,
        "timestamp": int(time.time())
    }
    predictions = []
    if PREDICTIONS_FILE.exists():
        with open(PREDICTIONS_FILE, 'r') as f:
            predictions = json.load(f)
    predictions.append(prediction)
    with open(PREDICTIONS_FILE, 'w') as f:
        json.dump(predictions, f, indent=4)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        logger.warning("No file part in request")
        return jsonify({"status": "error", "message": "No file provided"}), 400
    file = request.files['file']
    if file.filename == '':
        logger.warning("No selected file")
        return jsonify({"status": "error", "message": "No file selected"}), 400
    
    allowed_extensions = {'.wav', '.mp3', '.flac'}
    file_ext = os.path.splitext(file.filename)[1].lower()
    if file_ext not in allowed_extensions:
        logger.warning(f"Invalid file format: {file_ext}")
        return jsonify({"status": "error", "message": f"Invalid file format. Allowed: {', '.join(allowed_extensions)}"}), 400

    file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(file_path)
    
    try:
        features = extract_wav2vec_features(file_path)
        emotion, probabilities, confidence = predict_emotion(features)
        save_prediction(file.filename, emotion, probabilities, confidence)
        logger.info(f"Prediction successful for {file.filename}: {emotion} ({confidence:.2%})")
        return jsonify({
            "status": "success",
            "data": {
                "predicted_emotion": emotion,
                "probabilities": probabilities,
                "confidence": confidence
            }
        })
    except Exception as e:
        logger.error(f"Error processing {file.filename}: {str(e)}")
        return jsonify({"status": "error", "message": str(e)}), 500
    finally:
        if os.path.exists(file_path):
            os.remove(file_path)

@app.route('/predictions')
def get_predictions():
    try:
        if PREDICTIONS_FILE.exists():
            with open(PREDICTIONS_FILE, 'r') as f:
                return jsonify({"status": "success", "data": json.load(f)})
        return jsonify({"status": "success", "data": []})
    except Exception as e:
        logger.error(f"Error loading predictions: {str(e)}")
        return jsonify({"status": "error", "message": "Failed to load predictions"}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)