import cv2
import os
import numpy as np
from PIL import Image

# Yüz tanıma için LBPH sınıflandırıcısını oluştur
taniyici = cv2.face.LBPHFaceRecognizer_create()

# Yüz dedektörünü yükle
dedektor = cv2.CascadeClassifier("D:/python repos/haarcascade_frontalface_default (1).xml")

def getImagesAndLabels(path):
    imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
    images = []
    labels = []
    for imagePath in imagePaths:
        # Resmi aç ve gri tonlamaya çevir
        PIL_img = Image.open(imagePath).convert('L')
        img_numpy = np.array(PIL_img, 'uint8')
        # Etiket ID'sini al
        id = int(os.path.split(imagePath)[-1].split(".")[1])
        print(id)
        
        # Yüzleri tespit et
        yuzler = dedektor.detectMultiScale(img_numpy)
        
        for (x, y, w, h) in yuzler:
            images.append(img_numpy[y:y+h, x:x+w])
            labels.append(id)
    
    return images, labels

yuzler,labels=getImagesAndLabels('veri')
taniyici.train(yuzler, np.array(labels))
taniyici.save('D:/python repos/deneme.yml')

