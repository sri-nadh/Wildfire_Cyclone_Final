import io
from fastapi import FastAPI
import cv2
from fastapi.responses import StreamingResponse
import numpy as np
from ultralytics import YOLO
import gradio as gr
from gradio_ui import ui
app = FastAPI()

from fastapi import UploadFile
model = YOLO("Detection_best_2.pt")

@app.post("/detect/")
async def detect_objects(file: UploadFile):
  # Process the uploaded image for object detection
  image_bytes = await file.read()
  image = np.frombuffer(image_bytes, dtype=np.uint8)
  image = cv2.imdecode(image, cv2.IMREAD_COLOR)
  detections = model.predict(image)
  confidence_value = detections[0].boxes[0].conf.item()
  # return confidence_value
  for r in detections:
      boxes = r.boxes
      labels = []
      for box in boxes:
          c = box.cls
          l = model.names[int(c)]
          labels.append(l)
  frame = detections[0].plot()
  
  print(file)

  # Convert the original image array to bytes
  _, img_encoded = cv2.imencode('.jpg', frame)
  img_bytes = img_encoded.tobytes()
  if img_bytes== 0:
      return 

  # Send the original image back as a streaming response
  return StreamingResponse(io.BytesIO(img_bytes), media_type="image/jpeg")

app=gr.mount_gradio_app(app,ui,path='')
