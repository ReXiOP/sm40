from flask import Flask, request, jsonify, send_file
import os
from PIL import Image
import argparse
import time
from pathlib import Path
import urllib.request
import math
import numpy as np
import cv2
import torch
from numpy import random

def download_image(url, save_path):
    urllib.request.urlretrieve(url, save_path)

def detect():
    # YOLOv5 detection code
    pass  # Placeholder, replace with your actual YOLOv5 detection code

app = Flask(__name__)

@app.route('/detect', methods=['GET'])
def detect_image():
    try:
        # Get the image URL from the query parameters
        image_url = request.args.get('image_url')

        if not image_url:
            return jsonify({'error': 'Missing image_url parameter'}), 400

        # Download the image
        image_path = "downloaded_image.jpg"
        download_image(image_url, image_path)

        # Run YOLOv5 detection
        with torch.no_grad():
            detect()

        # Return the processed image
        processed_image_path = "runs/detect/exp/downloaded_image.jpg"
        return send_file(processed_image_path, mimetype='image/jpeg')

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # You can specify the host and port as needed
    app.run(host='0.0.0.0', port=80, debug=True)
