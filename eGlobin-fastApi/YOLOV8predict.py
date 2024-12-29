
import numpy as np
import pathlib
import matplotlib.pyplot as plt
from IPython import display
#display.clear_output()
import ultralytics
import uuid
from PIL import Image, ImageDraw

from IPython.display import display, Image

from ultralytics import YOLO
from PIL import Image
import cv2

YELLOW = '\033[93m'
GREEN = '\033[92m'
RESET = '\033[0m'
RED = '\033[91m'

#GÖZ TESPİTİ EKLENEBİLİRS


def extraxtROI(model: YOLO,confidence,image,saveDir):
    print(YELLOW + "bilgilendirme conf-> ", confidence, RESET)
    minConfidence = 0.4
    reduction=0.05
    #image = cv2.imread(imageSource)   #eğer bunu kullanırsan Image.open(imgPATH) ---->   img = cv2.imread(imgPATH) kırpma işlemi için yapmalısın
    results = model.predict(source=image,retina_masks=True, conf=confidence,save=True)
    result = results[0]
    
    predictCount = len(result.boxes)
    if(predictCount>=1):
        pairs = result.masks[0].xy
        points = np.array(pairs,dtype=np.int32)

        height = image.shape[0]
        width = image.shape[1]
        mask = np.zeros((height, width), dtype=np.uint8)
        cv2.fillPoly(mask, points, (255))

        res = cv2.bitwise_and(image,image,mask = mask)
        rect = cv2.boundingRect(points) # returns (x,y,w,h) of the rect
        cropped = res[rect[1]: rect[1] + rect[3], rect[0]: rect[0] + rect[2]]
        uniqueName = str(uuid.uuid4())

        cv2.imwrite(saveDir+uniqueName+".png", cropped)
        print(GREEN + uniqueName, "başarıyla kayıt edildi.." + RESET)
        segmentedImagePATH = result.save_dir +"\\" +result.path
        segmentedImagePATH =segmentedImagePATH.replace("\\", "/")
        return segmentedImagePATH,saveDir+uniqueName+".png" #diğer algoritma için segmented resim ve kırpılmış resim adı
    else:
        print(RED +"roı bulunamadı.. tekrar bakılıyor..." + RESET)
        confidence -= reduction #confidence azalt ve tekrar çalıştır
        if(confidence>=minConfidence):
            #recursive
            return extraxtROI(model,confidence,image,saveDir)
            
        else:
            print(RED +"!!!!!!!!!!!!!!! ROI YOK EXİTİNG..!!!!!!!!!!!" + RESET)
            return None #if not none      if none try again
        
