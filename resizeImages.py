import cv2
import os
import numpy as np

count = 0
for path, subdirnames, filenames in os.walk('C:/Users/Dell/Desktop/Projects/Face Recognition/TrainImages'):
	if subdirnames:
		for subdir in subdirnames:
			os.makedirs("C:/Users/Dell/Desktop/Projects/Face Recognition/resizedTrainingImages/"+subdir)
	for filename in filenames:
		if filename.startswith("."):
			print("Skipping file : ", filename)
				continue

		id = os.path.basename(path)
		img_path = os.path.join(path,filename)
		print("Img path:",img_path)
		print("Id",id)

		img = cv2.imread(img_path)

		if test_img.any() == None:
			print("Image not loaded properly")
			continue

		resized_img = cv2.resize(img,(100,100))
		new_path = "resizedTrainingImages"+"/"+str(id)
		print("Desired path : ",os.path.join(new_path,"frame%d.jpg" %count))
		cv2.imwrite(os.path.join(new_path,"frame%d.jpg" %count), resized_img)
		count += 1