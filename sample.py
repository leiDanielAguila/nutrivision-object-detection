from ultralytics import YOLO
from PIL import Image

model = YOLO("model/nutrivision_model.pt")
print(model.names)

# image = Image.open("test_pictures/mango1.jpg")
# results = model.predict(image, conf=0.7)

# fruits = {
#     "Orange": 4,
#     "Mango": 15
# }
#
# if "Orange" in fruits.keys():
#     print(fruits["Orange"] * 2)

# for result in results:
#     result.show()