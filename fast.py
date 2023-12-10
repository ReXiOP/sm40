from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.templating import Jinja2Templates
import aiohttp
import torch
import numpy as np
from PIL import Image
from io import BytesIO
import json
from datetime import datetime
from aiohttp import ClientSession

app = FastAPI()
templates = Jinja2Templates(directory="templates")

# Load YOLOv5 model and set device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = torch.hub.load('ultralytics/yolov5:v6.0', 'yolov5s', pretrained=True).to(device).eval()

async def detect_objects(image, class_indices):
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

async def detect_human(image):
    # Class index for person
    return await detect_objects(image, [0])

async def detect_animal(image):
    # Class indices for cat and dog
    animal_class_indices = [14, 15, 16, 17, 18, 19, 20, 21, 22, 77]

    return await detect_objects(image, animal_class_indices)

def save_to_log(url, result):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = {"timestamp": timestamp, "url": url, "result": result}

    try:
        with open("log.json", "a") as log_file:
            json.dump(log_entry, log_file)
            log_file.write('\n')
    except Exception as e:
        print(f"Error saving to log: {e}")

@app.get("/")
def index(request: Request):
    return templates.TemplateResponse("upload_form.html", {"request": request})

@app.get("/result")
async def result(
    request: Request,
    image_url: str = Query(..., description="URL of the image to be analyzed")
):
    try:
        # Download the image with SSL verification disabled
        async with ClientSession(connector=aiohttp.TCPConnector(ssl=False)) as session:
            async with session.get(image_url) as response:
                image = await response.read()

        # Detect humans and animals
        face_detected = await detect_human(image)
        animal_detected = await detect_animal(image)

        # Save to log
        save_to_log(image_url, {"face_detected": face_detected, "animal_detected": animal_detected})

        # Return result in JSON format
        result = {'face_detected': face_detected, 'animal_detected': animal_detected}
        return result

    except Exception as e:
        # Save error to log
        save_to_log(image_url, {"error": str(e)})
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
