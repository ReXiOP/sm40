from flask import Flask, render_template, request, redirect, url_for, jsonify
import cv2
import numpy as np
import os
import datetime
import random
import string
import urllib.request

# Load the pre-trained Haarcascades for various detections
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')
cat_face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalcatface.xml')
cat_face_ext_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalcatface_extended.xml')
full_body_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_fullbody.xml')
lower_body_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_lowerbody.xml')
profile_face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_profileface.xml')
upper_body_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_upperbody.xml')

app = Flask(__name__)

def generate_random_string(length=8):
    """Generate a random string of lowercase letters and digits."""
    return ''.join(random.choices(string.ascii_lowercase + string.digits, k=length))


def get_image_from_url(url):
    try:
        # Download the image
        req = urllib.request.urlopen(url)
        arr = np.asarray(bytearray(req.read()), dtype=np.uint8)
        imge = cv2.imdecode(arr, -1)
        return imge
    except Exception as e:
        print(f"Error downloading image from URL: {e}")
        return None

def download_and_check_human(url):
    """Download an image from an HTTP URL and check if there are human features."""
    img = get_image_from_url(url)

    # Check if the image is successfully loaded
    if img is None:
        print("Error: Could not read the image.")
        return img

    # Convert the image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect faces in the image
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Detect eyes in the image
    eyes = eye_cascade.detectMultiScale(gray)

    # Detect smiles in the image
    smiles = smile_cascade.detectMultiScale(gray, scaleFactor=1.8, minNeighbors=20, minSize=(25, 25))

    # Detect cat faces in the image
    cat_faces = cat_face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    cat_faces_ext = cat_face_ext_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Detect full body, lower body, profile faces, and upper body
    full_bodies = full_body_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    lower_bodies = lower_body_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    profile_faces = profile_face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    upper_bodies = upper_body_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Ensure blur_amount is a positive odd number
    blur_amount = 351  # You can adjust this value based on your preference

    # Blur the entire image if any features are detected
    if  len(faces) > 0 :
        print("face=true")
        return img, True
    else:
        print("face=false")
        return img, False

    return img

    

def blur_if_features_detected(image_path):
    # Load the image
    img = cv2.imread(image_path)

    # Check if the image is successfully loaded
    if img is None:
        print("Error: Could not read the image.")
        return img

    # Convert the image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect faces in the image
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Detect eyes in the image
    eyes = eye_cascade.detectMultiScale(gray)

    # Detect smiles in the image
    smiles = smile_cascade.detectMultiScale(gray, scaleFactor=1.8, minNeighbors=20, minSize=(25, 25))

    # Detect cat faces in the image
    cat_faces = cat_face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    cat_faces_ext = cat_face_ext_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Detect full body, lower body, profile faces, and upper body
    full_bodies = full_body_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    lower_bodies = lower_body_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    profile_faces = profile_face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    upper_bodies = upper_body_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Ensure blur_amount is a positive odd number
    blur_amount = 351  # You can adjust this value based on your preference

    # Blur the entire image if any features are detected
    if len(faces) > 0 :
        img = cv2.GaussianBlur(img, (blur_amount, blur_amount), 0)

        # Add "not allowed" text in the middle of the image
        font = cv2.FONT_HERSHEY_SIMPLEX
        text = "Human Photo Are Not Allowed"
        text_size = 0
        text_thickness = 3
        text_color = (0, 0, 255)  # Red color
        (text_width, text_height), _ = cv2.getTextSize(text, font, text_size, text_thickness)
        text_position = ((img.shape[1] - text_width) // 2, (img.shape[0] + text_height) // 2)
        cv2.putText(img, text, text_position, font, text_size, text_color, text_thickness, cv2.LINE_AA)
        print("face=true")
        return img, True
    else:
        print("face=false")
        return img, False

    return img

@app.route('/', methods=['GET', 'POST'])
def index():
    error = None
    original_image = None
    blurred_image = None
    face_detected = None

    if request.method == 'POST':
        if 'file' in request.files:
            # File upload case
            file = request.files['file']
            if file.filename == '':
                error = 'No selected file'
            elif file and file.content_type.startswith('image'):
                timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
                random_string = generate_random_string()
                original_image_path = f'static/original_image_{timestamp}_{random_string}{os.path.splitext(file.filename)[1]}'
                file.save(original_image_path)
                blurred_image, face_detected = blur_if_features_detected(original_image_path)
                blurred_image_path = f'static/blurred_image_{timestamp}_{random_string}{os.path.splitext(file.filename)[1]}'
                if blurred_image is not None and not np.all(blurred_image == 0):
                    blurred_image_uint8 = blurred_image.astype(np.uint8)
                    cv2.imwrite(blurred_image_path, cv2.cvtColor(blurred_image_uint8, cv2.COLOR_BGR2RGB))
                    original_image = original_image_path
                    blurred_image = blurred_image_path
                    return redirect(url_for('result', original_image=original_image, blurred_image=blurred_image, face_detected=face_detected))
                else:
                    print("Error: Blurred image is empty. Could not write.")
        elif 'url' in request.form:
    # Image URL case
            url = request.form['url']
            if url:
                blurred_image, face_detected = download_and_check_human(url)
                if blurred_image is not None:
                    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
                    random_string = generate_random_string()
                    original_image_path = f'static/original_image_{timestamp}_{random_string}.png'
                    original_image = original_image_path
                    cv2.imwrite(original_image_path, cv2.cvtColor(blurred_image, cv2.COLOR_BGR2RGB))
                    blurred_image_path = f'static/blurred_image_{timestamp}_{random_string}.png'
                    cv2.imwrite(blurred_image_path, cv2.cvtColor(blurred_image, cv2.COLOR_BGR2RGB))
                    blurred_image = blurred_image_path
                    # Display the result only if face_detected is True
                    if face_detected:
                        return redirect(url_for('result', original_image=original_image, blurred_image=blurred_image, face_detected=face_detected))
                    else:
                        return redirect(url_for('result', blurred_image=blurred_image, face_detected=face_detected))
                else:
                    # Handle the case where there is an error processing the image
                    error = "Error processing the image."
                    return render_template('index.html', error=error, original_image=None, blurred_image=None, face_detected=None)
    return render_template('index.html', error=error, original_image=original_image, blurred_image=blurred_image, face_detected=face_detected)


@app.route('/result')
def result():
    blurred_image = request.args.get('blurred_image')
    face_detected = request.args.get('face_detected')
    
    result_data = {
        'data': {
            'blurred_image': blurred_image,
            'face_detected': face_detected,
        }
    }

    return jsonify(result_data)


# if __name__ == '__main__':
#     app.run(debug=True, port=8888)
