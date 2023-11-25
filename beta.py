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

def detect_human(image):
    # Preprocess the image
    img = Image.open(BytesIO(image)).convert('RGB')
    img = np.array(img)
    
    # Run YOLOv5 inference
    results = model(img)

    # Check if humans are detected
    for det in results.pred[0]:
        if det[-1].item() == 0:  # Class index for person
            return True

    return False
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

        # Detect humans
        true = detect_human(image)

        # Return result in JSON format
        result = {'face_detected': true}
        return jsonify(result)

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True,port=5001)
