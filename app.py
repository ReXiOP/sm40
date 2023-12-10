from flask import Flask, render_template, request, jsonify
import urllib.request
import cv2
import numpy as np
from PIL import Image
from io import BytesIO
import torch
import ssl
import certifi
from datetime import datetime
import json

app = Flask(__name__)

# Load YOLOv5 model and set device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = torch.hub.load('ultralytics/yolov5:v6.0', 'yolov5s', pretrained=True).to(device).eval()

def detect_objects(image, class_indices):
    # Preprocess the image
    img = Image.open(BytesIO(image)).convert('RGB')
    img = np.array(img)

    # Run YOLOv5 inference
    results = model(img)

    # Check if objects of the specified class indices are detected
    for det in results.pred[0]:
        if det[-1].item() in class_indices:
            return True

    return False

def detect_human(image):
    # Class index for person
    return detect_objects(image, [0])

def detect_animal(image):
    # Class indices for cat and dog
    animal_class_indices = [14, 15, 16, 17, 18, 19, 20, 21, 22, 77]
    return detect_objects(image, animal_class_indices)

def save_to_log(url, result):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = {"timestamp": timestamp, "url": url, "result": result}

    try:
        with open("log.json", "a") as log_file:
            json.dump(log_entry, log_file)
            log_file.write('\n')
    except Exception as e:
        print(f"Error saving to log: {e}")

@app.route('/')
def index():
    return render_template('upload_form.html')

@app.route('/result', methods=['POST'])
def result():
    try:
        # Get the image URL from the request
        image_url = request.form['image_url']

        # Download the image using requests
        response = urllib.request.urlopen(image_url, context=ssl.create_default_context(cafile=certifi.where()))
        image = response.read()

        # Detect humans and animals
        face_detected = detect_human(image)
        animal_detected = detect_animal(image)

        # Return result in JSON format
        result = {'face_detected': face_detected, 'animal_detected': animal_detected}

        # Save the result to the log
        save_to_log(image_url, result)

        return jsonify(result)

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True, port=5001)
