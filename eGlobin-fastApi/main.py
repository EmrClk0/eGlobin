from typing import Union
from fastapi import FastAPI, File,UploadFile, HTTPException, status,Query
from fastapi.responses import FileResponse
import uuid
from ultralytics import YOLO
import cv2

from pydantic import BaseModel
import numpy as np
from keras.models import load_model
import os

from YOLOV8predict import extraxtROI


app = FastAPI()

IMAGE_DIR = "images/"
CROPPED_DIR = "cropped/"
accepted_file_types = ["image/png", "image/jpeg", "image/jpg", "png", "jpeg", "jpg"] 

weightPath="ptweights/best200.pt"
confidence=0.8

model = load_model("anemia_model_new2.h5")
class PredictionRequest(BaseModel):
    image_path: str
    age: int
    gender: str

class PredictionResponse(BaseModel):
    predicted_hblevel: float

def predict_hblevel(model, image_path, age, gender):
    try:
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError("Image upload ERROR")
        image = cv2.resize(image, (64, 64))
        image = image / 255.0
        gender_numeric = 0 if gender.lower() == 'male' else 1
        predicted_hblevel = model.predict([np.expand_dims(image, axis=0), np.array([age]), np.array([gender_numeric])])[0][0]
        return predicted_hblevel
    except Exception as e:
        print("ERROR predict:", e)
        return None


@app.post("/upload/")
async def uploadImage(file: UploadFile=File(...)):
    content_type = file.content_type

    if content_type in accepted_file_types:
        file.filename=f"{uuid.uuid4()}{file.filename}"
        contents = await file.read()
    
        #saving file
        with open(f"{IMAGE_DIR}{file.filename}","wb") as f:
            f.write(contents)
        
        model = YOLO(weightPath)
        imageSource=IMAGE_DIR+file.filename
        image = cv2.imread(imageSource)
        result = extraxtROI(model,confidence,image,CROPPED_DIR)
        
        if(result != None):
            segment, cropped = result
            return {"segmentedImagePATH":segment,
                    "croppedImagePATH":cropped}
        else:
            raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                            detail="ROI DOES NOT EXİST")

        return {"filename":file.filename}
        #end of if
    else:
        raise HTTPException(status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                            detail=f"{content_type} is not supported format")
    
    
@app.get("/getImage")  #get ımage
def getImageByPath(imagePath:str):
    return FileResponse(imagePath)


@app.get("/predict", response_model=PredictionResponse)
async def predict_hblevel_endpoint(
    image_name: str = Query(..., description="Name of the image file in the cropped directory"),
    age: int = Query(..., description="Age of the person"),
    gender: str = Query(..., description="Gender of the person (male/female)")
):
    """
    Predict Hemoglobin Level
    ---
    parameters:
      - name: image_name
        in: query
        type: string
        required: true
      - name: age
        in: query
        type: integer
        required: true
      - name: gender
        in: query
        type: string
        required: true
    responses:
      200:
        description: The predicted hemoglobin level
    """
    try:
        # `cropped dosyasından gelen path buraya yüklenecek
        full_image_path = image_name
        
        # Tahmin işlemini gerçekleştirin
        prediction = predict_hblevel(model, full_image_path, age, gender)
        
        if prediction is None:
            raise HTTPException(status_code=500, detail="Prediction failed")
        
        return {"predicted_hblevel": prediction}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))