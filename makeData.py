import cv2

cap = cv2.VideoCapture(0)

count = 0
while True:
	ret, test_img = cap.read()
	if not ret:
		continue

	cv2.imwrite("frame%d.jpg" %count, test_img)
	count += 1
	resized_img = cv2.resize(test_img,(700,1000))
	cv2.imshow("Face Detection", resized_img) 
	if cv2.waitKey(10) == ord('q'):
		break

cap.release()
cv2.destroyAllWindows