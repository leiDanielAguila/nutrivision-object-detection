from fastapi import FastAPI, File, UploadFile, HTTPException
from ultralytics import YOLO
from typing import List
from PIL import Image
import gc
import io
import os
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Nutrition values per fruit
apple_carbs = 25
apple_sodium = 0.0018
apple_protein = 0.5

mango_carbs = 50
mango_sodium = 0.0034
mango_protein = 2.8

orange_carbs = 18
orange_sodium = 0.0014
orange_protein = 1.3

# Load model once at startup
model = YOLO('app/nutrivision_v3.pt')

@app.get("/greet")
async def hello_world():
    return {"status": "working properly"}


@app.post("/detect")
async def detect_fruits(files: List[UploadFile] = File(...)):
    try:
        if len(files) > 5:
            raise HTTPException(status_code=400, detail="You can only upload up to 5 images at a time.")

        total_object_count = {}
        carbs = 0
        protein = 0
        sodium = 0

        images = []

        for file in files:
            image_bytes = await file.read()
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

            # Resize for performance
            image = image.resize((640, 640))
            images.append(image)

        # Batched prediction
        results = model.predict(images, conf=0.7)

        class_names = model.names

        for result in results:
            detected_objects = result.boxes.cls

            for cls_id in detected_objects:
                class_name = class_names[int(cls_id)]
                total_object_count[class_name] = total_object_count.get(class_name, 0) + 1

        # Clean up memory
        del images, results
        gc.collect()

        if not total_object_count:
            return {"message": "No fruits detected"}

        # Nutrition calculation
        if "mango" in total_object_count:
            carbs += total_object_count["mango"] * mango_carbs
            sodium += total_object_count["mango"] * mango_sodium
            protein += total_object_count["mango"] * mango_protein
        if "apple" in total_object_count:
            carbs += total_object_count["apple"] * apple_carbs
            sodium += total_object_count["apple"] * apple_sodium
            protein += total_object_count["apple"] * apple_protein
        if "orange" in total_object_count:
            carbs += total_object_count["orange"] * orange_carbs
            sodium += total_object_count["orange"] * orange_sodium
            protein += total_object_count["orange"] * orange_protein

        return {
            "Fruits_Detected": total_object_count,
            "total_carbs": carbs,
            "total_protein": round(protein, 5),
            "total_sodium": round(sodium, 5),
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
