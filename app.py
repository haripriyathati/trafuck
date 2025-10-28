from __future__ import division, print_function
import sys
import os
import glob
import re
import numpy as np
import tensorflow as tf
import cv2

# Import necessary Keras/TF components
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Import Flask components
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename

# Define the location for file uploads
# Store uploads inside 'static/uploads' so they can be accessed via url_for('static', ...)
UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), 'static', 'uploads')
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Initialize Flask App
app = Flask(__name__)
# Set the upload folder for Flask to recognize
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Model loading (Must be outside the route function)
MODEL_PATH = 'model.h5'

try:
    # Ensure model loading is robust
    model = load_model(MODEL_PATH)
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}", file=sys.stderr)
    # If the model fails to load, the app might not function correctly.

# --- Image Preprocessing Functions ---

def grayscale(img):
    """Converts image to grayscale."""
    if len(img.shape) == 3 and img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img

def equalize(img):
    """Applies histogram equalization to improve contrast."""
    img = cv2.equalizeHist(img)
    return img

def preprocessing(img):
    """Combines all preprocessing steps."""
    img = grayscale(img)
    img = equalize(img)
    img = img / 255.0  # Normalize
    return img

# --- Class Name Mapping Function ---

def getClassName(classNo):
    """Maps class index to human-readable traffic sign name."""
    if classNo == 0: return 'Speed Limit 20 km/h'
    elif classNo == 1: return 'Speed Limit 30 km/h'
    elif classNo == 2: return 'Speed Limit 50 km/h'
    elif classNo == 3: return 'Speed Limit 60 km/h'
    elif classNo == 4: return 'Speed Limit 70 km/h'
    elif classNo == 5: return 'Speed Limit 80 km/h'
    elif classNo == 6: return 'End of Speed Limit 80 km/h'
    elif classNo == 7: return 'Speed Limit 100 km/h'
    elif classNo == 8: return 'Speed Limit 120 km/h'
    elif classNo == 9: return 'No passing'
    elif classNo == 10: return 'No passing for vehicles over 3.5 metric tons'
    elif classNo == 11: return 'Right-of-way at the next intersection'
    elif classNo == 12: return 'Priority road'
    elif classNo == 13: return 'Yield'
    elif classNo == 14: return 'Stop'
    elif classNo == 15: return 'No vehicles'
    elif classNo == 16: return 'Vehicles over 3.5 metric tons prohibited'
    elif classNo == 17: return 'No entry'
    elif classNo == 18: return 'General caution'
    elif classNo == 19: return 'Dangerous curve to the left'
    elif classNo == 20: return 'Dangerous curve to the right'
    elif classNo == 21: return 'Double curve'
    elif classNo == 22: return 'Bumpy road'
    elif classNo == 23: return 'Slippery road'
    elif classNo == 24: return 'Road narrows on the right'
    elif classNo == 25: return 'Road work'
    elif classNo == 26: return 'Traffic signals'
    elif classNo == 27: return 'Pedestrians'
    elif classNo == 28: return 'Children crossing'
    elif classNo == 29: return 'Bicycles crossing'
    elif classNo == 30: return 'Beware of ice/snow'
    elif classNo == 31: return 'Wild animals crossing'
    elif classNo == 32: return 'End of all speed and passing limits'
    elif classNo == 33: return 'Turn right ahead'
    elif classNo == 34: return 'Turn left ahead'
    elif classNo == 35: return 'Ahead only'
    elif classNo == 36: return 'Go straight or right'
    elif classNo == 37: return 'Go straight or left'
    elif classNo == 38: return 'Keep right'
    elif classNo == 39: return 'Keep left'
    elif classNo == 40: return 'Roundabout mandatory'
    elif classNo == 41: return 'End of no passing'
    elif classNo == 42: return 'End of no passing by vehicles over 3.5 metric tons'
    else: return 'Unknown Sign'

# --- Prediction Function ---

def model_predict(img_path, model):
    """
    Loads, preprocesses, predicts an image, and returns the class name and confidence.
    """
    img = cv2.imread(img_path, cv2.IMREAD_COLOR) 
    
    if img is None:
        raise ValueError("Image could not be loaded by OpenCV. Check file type or corruption.")

    img = cv2.resize(img, (32, 32)) 
    img = preprocessing(img)
    
    # Reshape for Prediction (Batch size 1, 32 height, 32 width, 1 channel)
    img = img[np.newaxis, ..., np.newaxis] 
    
    # PREDICT IMAGE
    predictions = model.predict(img)[0]
    
    # Get class index and confidence
    classIndex = np.argmax(predictions)
    confidence = np.max(predictions) * 100  # Convert max probability to percentage
    
    preds = getClassName(classIndex)
    
    return preds, f"{confidence:.2f}%"

# --- FLASK ROUTES ---

@app.route('/', methods=['GET', 'POST']) 
def index():
    # Handle POST request (Form Submission / File Upload)
    if request.method == 'POST':
        # Check if the file is in the request
        if 'file' not in request.files:
            return render_template('index.html', error_message="No file part in the request.")
        
        f = request.files['file']

        # Check if the filename is empty
        if f.filename == '':
            return render_template('index.html', error_message="No selected file.")

        # Process the file
        if f:
            # Securely save the file
            file_name = secure_filename(f.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], file_name)
            f.save(file_path)

            # Perform prediction
            try:
                predicted_class, confidence = model_predict(file_path, model)
            except Exception as e:
                print(f"Prediction error: {e}", file=sys.stderr)
                return render_template('index.html', error_message="Error during prediction. Check server logs.")

            # Create a URL path for the image to be displayed in the HTML
            uploaded_image_url = url_for('static', filename=f'uploads/{file_name}')
            
            # Render the page with results
            return render_template('index.html', 
                                   predicted_class=predicted_class, 
                                   confidence=confidence, 
                                   uploaded_image_url=uploaded_image_url) 

    # Handle GET request (Initial page load)
    return render_template('index.html')


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)
