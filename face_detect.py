import cv2 as cv

img = cv.imread('Photos/group 1.jpg')
cv.imshow('Lady', img)

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

haar_Cascade = cv.CascadeClassifier('haar_face.xml')

faces_rec = haar_Cascade.detectMultiScale(gray,
                                          scaleFactor=1.04,
                                          minNeighbors=3)

for (x,y,w,h) in faces_rec:
    cv.rectangle(img, (x,y), (x+w,y+h), (0,255,0), thickness=2)

cv.imshow('Detected Faces', img)

cv.waitKey(0)