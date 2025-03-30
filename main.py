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
async def detect_fruits(
        # user_age: int = File(...),
        # user_height: int = File(...),
        # user_weight: int = File(...),
        files: List[UploadFile] = File(...)
):
    try:

        result_list = []

        for file in files:
            image_bytes = await file.read()
            image = Image.open(io.BytesIO(image_bytes))
            results = model.predict(image, conf=0.7)
            class_names = model.names
            detected_objects = results[0].boxes.cls if results else []
            object_count = {}

            for cls_id in detected_objects:
                class_name = class_names[int(cls_id)]
                object_count[class_name] = object_count.get(class_name, 0) + 1

            del image_bytes, image, results
            gc.collect()
            result_list.append({
                "filename": file.filename,
                "detections": object_count if object_count else "no fruits detected."
            })
        if not result_list:
            print("❌ No objects detected")
            return {"message": "No fruits detected"}

        print(f"✅ Detection output: {result_list}")
        return {"detections": result_list}

    except Exception as e:
        print(f"❌ Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))  # Default to 8000 if PORT is not set
    uvicorn.run(app, host="0.0.0.0", port=port)