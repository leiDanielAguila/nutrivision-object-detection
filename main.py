from fastapi import FastAPI, File, UploadFile, HTTPException
from ultralytics import YOLO
from PIL import Image
import gc
import io
import os
import requests
import uvicorn

app = FastAPI()

MODEL_URL = "https://huggingface.co/leiDanielAguila/nutrivision/resolve/main/yolov8_nutrivision.pt"

def load_model():
    model_path = "yolov8_nutrivision.pt"
    response = requests.get(MODEL_URL, stream=True)

    if response.status_code == 200:
        with open(model_path, "wb") as f:
            f.write(response.content)
        print("Model downloaded from Hugging Face")
    else:
        raise Exception("Failed to download model from Hugging Face")

    return YOLO(model_path).to("cpu").half()  # Load model in half precision to save RAM


# Load the YOLO model
model = load_model()


def preprocess_image(image):
    return image.resize((640, 640))  # Resize image to 640x640

@app.get("/greet")
async def hello_world():
    return {"status": "working properly"}


@app.post("/detect")
async def detect_fruits(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes))
        image = preprocess_image(image)

        # Perform detection
        results = model.predict(image, conf=0.6)
        class_names = model.names

        detected_objects = results[0].boxes.cls if results else []
        object_count = {}

        for cls_id in detected_objects:
            class_name = class_names[int(cls_id)]
            object_count[class_name] = object_count.get(class_name, 0) + 1

        del image_bytes, image, results
        gc.collect()


        if not object_count:
            return {"message": "No fruits detected"}

        return {"detections": object_count}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))  # Default to 8000 if PORT is not set
    uvicorn.run(app, host="0.0.0.0", port=port)