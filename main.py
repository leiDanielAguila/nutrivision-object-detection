from fastapi import FastAPI, File, UploadFile, HTTPException
from ultralytics import YOLO
from PIL import Image
import io
import uvicorn

app = FastAPI()

# Load the YOLO model
model = YOLO("yolov8_nutrivision.pt")


@app.post("/detect")
async def detect_fruits(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes))

        # Perform detection
        results = model.predict(image, conf=0.6)
        class_names = model.names

        detected_objects = results[0].boxes.cls if results else []
        object_count = {}

        for cls_id in detected_objects:
            class_name = class_names[int(cls_id)]
            object_count[class_name] = object_count.get(class_name, 0) + 1

        if not object_count:
            return {"message": "No fruits detected"}

        return {"detections": object_count}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

