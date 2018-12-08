import cv2
import numpy as np

cap = cv2.VideoCapture(0)

face_cascade = cv2.CascadeClassifier("/home/icts/cool_custom_filters/haarcascade_frontalface_default.xml")

crown = cv2.imread("crown.jpg")
crown_mask = crown[:,:,2]
crown_inv_mask = cv2.bitwise_not(crown_mask)

while True:
	ret, frame = cap.read()
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	faces = face_cascade.detectMultiScale(gray, 1.3, 5)

	for x,y,w,h in faces:
		cv2.rectangle(frame, (x,y), (x+w, y+h), (0, 255, 0), 2)

		x1 = x
		y1 = y-80
		x2 = x+w
		y2 = y+h
		cv2.rectangle(frame, (x1,y1), (x2, y2), (0, 255, 0), 2)

		crown = cv2.resize(crown, (x2-x1, y2-y1), interpolation=cv2.INTER_AREA)
		crown_mask = cv2.resize(crown_mask, (x2-x1, y2-y1), interpolation=cv2.INTER_AREA)
		crown_inv_mask = cv2.resize(crown_inv_mask, (x2-x1, y2-y1), interpolation=cv2.INTER_AREA)

		roi = frame[y1:y2, x1:x2]


		roi_bg = cv2.bitwise_and(roi, roi, mask=crown_inv_mask)
		roi_fg = cv2.bitwise_and(crown, crown, mask=crown_mask)

		merged_roi = cv2.add(roi_bg, roi_fg)
		frame[y1:y2, x1:x2] = merged_roi

		break
		
	cv2.imshow('frame_with_rect', frame)

	if cv2.waitKey(1) &0xFF == ord('z'):
		break

cap.release()
cv2.destroyAllWindows()
