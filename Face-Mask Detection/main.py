import cv2 as cv
import numpy as np
from NeuNet import nn

labels={1:'No mask',0:'Mask'}
colors={1:(0,0,255),0:(0,255,0)}

cam = cv.VideoCapture(0)

clfr = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_frontalface_default.xml')

while True:
    (rval, img) = cam.read()

    img=cv.flip(img,1,1)
    gray=cv.cvtColor(img,cv.COLOR_BGR2GRAY) 

    face_rec = clfr.detectMultiScale(gray)

    for x, y, l, w in face_rec:  
        face_img = gray[y:y+w, x:x+l]
        resized=cv.resize(face_img,(100,100))
        normalized=resized/255.0
        reshaped=np.reshape(normalized,(1,100,100,1))
        
        result=nn.predict(reshaped)
        
        label=np.argmax(result,axis=1)[0]
        
        cv.rectangle(img,(x,y),(x+l,y+w),colors[label],2)
        cv.rectangle(img,(x,y-40),(x+l,y),colors[label],-1)
        cv.putText(img, labels[label], (x, y-10),cv.FONT_HERSHEY_PLAIN,0.8,(255,255,255),2)

    cv.imshow('LIVE',img)
    key = cv.waitKey(1)
    
    if key == 27:
        break

cv.destroyAllWindows()