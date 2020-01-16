import cv2
import os
import numpy as np
import faceRecognition as fr

test_img = cv2.imread('C:/Users/Dell/Desktop/Projects/Face Recognition/TestImages/SR2.jpg')

faces_detected, gray_img = fr.faceDetection(test_img)
print("Face detected : ", faces_detected)

# for (x,y,w,h) in faces_detected:
# 	cv2.rectangle(test_img,(x,y),(x+w,y+h),(255,0,0),thickness=5)


"""While training for the first time."""
# faces, faceID = fr.labels_for_training_data('C:/Users/Dell/Desktop/Projects/Face Recognition/TrainImages')
# face_recognizer = fr.train_classifier(faces, faceID)
# face_recognizer.save("trainingData.yml") #Saves the trained model


"""These two lines are used instead of prev 3 lines to avoid training again and again."""
face_recognizer = cv2.face.LBPHFaceRecognizer_create()
face_recognizer.read('C:/Users/Dell/Desktop/Projects/Face Recognition/trainingData.yml')

name = {0:"RDJ",1:"Sagar",2:"Romil"}


for face in faces_detected:
	print(face)
	(x,y,w,h) = face
	roi_gray = gray_img[y:y+w,x:x+h]
	label, confidence = face_recognizer.predict(roi_gray)
	print("Label : ",label)
	print("Confidence : ",confidence)
	fr.draw_rect(test_img,face)
	predicted_name = name[label]
	print(predicted_name)
	if confidence > 50:
		continue
	fr.put_text(test_img,predicted_name,x,y)

resized_img = cv2.resize(test_img,(700,1000))
cv2.imshow('Face Detection',resized_img)

cv2.waitKey(0)
cv2.destroyAllWindows()