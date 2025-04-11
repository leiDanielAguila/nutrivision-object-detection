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
    allow_origins=["*"],  # You can restrict this to your app domain later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

apple_carbs = 25
apple_sodium = 0.0018
apple_protein = 0.5

mango_carbs = 50
mango_sodium = 0.0034
mango_protein = 2.8

orange_carbs = 18
orange_sodium = 0.0014
orange_protein = 1.3

# new comment

model = YOLO('app/nutrivision_v3.pt')

@app.get("/greet")
async def hello_world():
    return {"status": "working properly"}


@app.post("/detect")
async def detect_fruits(files: List[UploadFile] = File(...)):
    try:
        total_object_count = {}
        carbs = 0
        protein = 0
        sodium = 0

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

        # return total_object_count

        if "mango" in total_object_count.keys():
            carbs += (total_object_count["mango"] * mango_carbs)
            sodium += (total_object_count["mango"] * mango_sodium)
            protein += (total_object_count["mango"] * mango_protein)
        if "apple" in total_object_count.keys():
            carbs += (total_object_count["apple"] * apple_carbs)
            sodium += (total_object_count["apple"] * apple_sodium)
            protein += (total_object_count["apple"] * apple_protein)
        if "orange" in total_object_count.keys():
            carbs += (total_object_count["orange"] * orange_carbs)
            sodium += (total_object_count["orange"] * orange_sodium)
            protein += (total_object_count["orange"] * orange_protein)

        return {
            "Fruits_Detected": total_object_count,
            "total_carbs": carbs,
            "total_protein": round(protein, 5),
            "Total_sodium": round(sodium, 5 ),
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))  # Default to 8000 if PORT is not set
    uvicorn.run(app, host="0.0.0.0", port=port)