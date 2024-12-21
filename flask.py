from flask import Flask, request, jsonify
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import os

app = Flask(__name__)

# Paths to model
MODEL_PATH = 'cnn_model.h5'

# Load the trained model
try:
    model = load_model(MODEL_PATH)
    print("Model loaded successfully.")
except Exception as e:
    raise RuntimeError(f"Error loading model: {e}")

# Function to preprocess a frame or image
def preprocess_frame(frame):
    try:
        frame_resized = cv2.resize(frame, (128, 128))
        frame_normalized = img_to_array(frame_resized) / 255.0
        return np.expand_dims(frame_normalized, axis=0)
    except Exception as e:
        raise ValueError(f"Error preprocessing frame: {e}")

# Analyze predictions
def analyze_predictions(predictions):
    predictions_np = np.array(predictions)
    best_prediction = max(predictions_np)
    classification = 'Real' if best_prediction > 0.4 else 'Fake'
    return predictions_np, best_prediction, classification

# Process video and classify frames
def classify_video(video_path, model, num_frames=20):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video file: {video_path}")

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_interval = max(1, frame_count // num_frames)
    predictions = []

    frame_idx = 0
    while cap.isOpened() and len(predictions) < num_frames:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % frame_interval == 0:
            try:
                preprocessed_frame = preprocess_frame(frame)
                prediction = model.predict(preprocessed_frame, verbose=0)[0][0]
                predictions.append(prediction)
            except Exception as e:
                print(f"Error predicting frame at index {frame_idx}: {e}")
        frame_idx += 1

    cap.release()
    return predictions

# Process photo and classify
def classify_photo(photo_path, model):
    try:
        frame = cv2.imread(photo_path)
        if frame is None:
            raise FileNotFoundError(f"Unable to read image: {photo_path}")
        preprocessed_frame = preprocess_frame(frame)
        prediction = model.predict(preprocessed_frame, verbose=0)[0][0]
        return [prediction]
    except Exception as e:
        raise RuntimeError(f"Error processing image: {e}")

@app.route('/classify', methods=['POST'])
def classify():
    file = request.files.get('file')
    if not file:
        return jsonify({"error": "No file uploaded"}), 400

    file_path = f"temp_{file.filename}"
    file.save(file_path)

    try:
        if file.filename.lower().endswith(('.mp4', '.avi', '.mov')):
            predictions = classify_video(file_path, model)
        elif file.filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            predictions = classify_photo(file_path, model)
        else:
            return jsonify({"error": "Unsupported file format"}), 400

        predictions_np, best_prediction, classification = analyze_predictions(predictions)
        os.remove(file_path)  # Clean up the temporary file

        return jsonify({
            "predictions": predictions_np.tolist(),
            "best_prediction": best_prediction,
            "classification": classification
        })
    except Exception as e:
        os.remove(file_path)
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
