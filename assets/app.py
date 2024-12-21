import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import os

# Paths to model and media
MODEL_PATH = 'cnn_model.h5'
MEDIA_PATH = './assets/sample.mp4'  # Change this to your file

# Validate paths
def validate_file_path(path, file_type):
    if not os.path.exists(path):
        raise FileNotFoundError(f"{file_type} not found at {path}")

validate_file_path(MODEL_PATH, "Model file")
validate_file_path(MEDIA_PATH, "Media file")

# Load the trained model
try:
    model = load_model(MODEL_PATH)
    print("Model loaded successfully.")
except Exception as e:
    raise RuntimeError(f"Error loading model: {e}")

# Function to preprocess a frame or image
def preprocess_frame(frame):
    try:
        frame_resized = cv2.resize(frame, (128, 128))  # Resize to match model input
        frame_normalized = img_to_array(frame_resized) / 255.0  # Normalize pixel values
        return np.expand_dims(frame_normalized, axis=0)  # Add batch dimension
    except Exception as e:
        raise ValueError(f"Error preprocessing frame: {e}")

# Process video and classify frames
def classify_video(video_path, model, num_frames=20):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video file: {video_path}")

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_interval = max(1, frame_count // num_frames)  # Interval to extract `num_frames` frames
    predictions = []

    print("Processing video...")
    frame_idx = 0
    while cap.isOpened() and len(predictions) < num_frames:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % frame_interval == 0:  # Extract frame at intervals
            try:
                preprocessed_frame = preprocess_frame(frame)
                prediction = model.predict(preprocessed_frame, verbose=0)[0][0]  # Get prediction
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

# Analyze predictions
def analyze_predictions(predictions):
    predictions_np = np.array(predictions)
    best_prediction = max(predictions_np)
    classification = 'Real' if best_prediction > 0.4 else 'Fake'
    return predictions_np, best_prediction, classification

# Main logic
try:
    if MEDIA_PATH.lower().endswith(('.mp4', '.avi', '.mov')):  # Video file extensions
        predictions = classify_video(MEDIA_PATH, model)
    elif MEDIA_PATH.lower().endswith(('.jpg', '.jpeg', '.png')):  # Image file extensions
        predictions = classify_photo(MEDIA_PATH, model)
    else:
        raise ValueError("Unsupported file format. Please use a valid image or video file.")

    predictions_np, best_prediction, classification = analyze_predictions(predictions)

    print(f"Predictions: {predictions_np}")
    print(f"Best Prediction: {best_prediction:.4f} ({classification})")
except Exception as e:
    print(f"An error occurred: {e}")
