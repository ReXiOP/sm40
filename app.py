from flask import Flask, render_template, request, jsonify
import urllib.request
import cv2
import numpy as np
from PIL import Image
from io import BytesIO
import torch

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
    animal_class_indices = [14,15, 16, 17, 18, 19, 20, 21, 22, 77]

    return detect_objects(image, animal_class_indices)

@app.route('/')
def index():
    return render_template('upload_form.html')

@app.route('/result', methods=['POST'])
def result():
    try:
        # Get the image URL from the request
        image_url = request.form['image_url']

        # Download the image
        response = urllib.request.urlopen(image_url)
        image = response.read()

        # Detect humans and animals
        face_detected = detect_human(image)
        animal_detected = detect_animal(image)

        # Return result in JSON format
        result = {'face_detected': face_detected, 'animal_detected': animal_detected}
        return jsonify(result)

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True, port=5001)
