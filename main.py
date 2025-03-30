from fastapi import FastAPI, File, UploadFile, HTTPException
from ultralytics import YOLO
from typing import List
from PIL import Image
import gc
import io
import os
import requests
import uvicorn

app = FastAPI()

apple_calories = 95
apple_sodium = 0.0018
apple_sugar = 19

mango_calories = 202
mango_sodium = 0.0034
mango_sugar = 46

orange_calories = 69
orange_sodium = 0.0014
orange_sugar = 12

MODEL_URL = "https://huggingface.co/leiDanielAguila/nutrivision/resolve/main/nutrivision_model.pt"

def load_model():
    model_path = "model.pt"
    response = requests.get(MODEL_URL, stream=True)

    if response.status_code == 200:
        with open(model_path, "wb") as f:
            f.write(response.content)
        print("Model downloaded from Hugging Face")
    else:
        raise Exception("Failed to download model from Hugging Face")

    return YOLO(model_path)

model = load_model()

@app.get("/greet")
async def hello_world():
    return {"status": "working properly"}


@app.post("/detect")
async def detect_fruits(files: List[UploadFile] = File(...)):
    try:
        total_object_count = {}

        for file in files:
            image_bytes = await file.read()
            image = Image.open(io.BytesIO(image_bytes))

            results = model.predict(image, conf=0.7)

            if not results or len(results[0].boxes.cls) == 0:
                continue

            class_names = model.names
            detected_objects = results[0].boxes.cls

            for cls_id in detected_objects:
                class_name = class_names[int(cls_id)]
                total_object_count[class_name] = total_object_count.get(class_name, 0) + 1

            del image_bytes, image, results
            gc.collect()

        if not total_object_count:
            return {"message": "No fruits detected"}

        return total_object_count

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))  # Default to 8000 if PORT is not set
    uvicorn.run(app, host="0.0.0.0", port=port)