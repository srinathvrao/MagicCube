import cv2
import os
import csv
import numpy as np

imgnames = os.listdir("images/")

for imgname in imgnames[2:]:
	image = cv2.imread("images/test480.jpg")#+imgname)
	im = image.copy()
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 
	blur = cv2.GaussianBlur(gray, (5,5), 0)
	thresh = cv2.adaptiveThreshold(blur, 255, 1, 1, 11, 2)
	edged = cv2.Canny(gray, 30, 200) 
	contours, hierarchy = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

	print("Number of Contours found = " + str(len(contours[0]))) 
	mask = np.zeros((gray.shape),np.uint8)

	max_area = 0
	c = 0
	best_cnt = 0
	for i in contours:
		area = cv2.contourArea(i)
		if area > 100:
			if area>max_area:
				max_area = area
				best_cnt = i
				image = cv2.drawContours(image, contours, c, (0, 255, 0), 3)
		c+=1
	cv2.drawContours(mask,[best_cnt],0,255,-1)
	cv2.drawContours(mask,[best_cnt],0,0,2)
	out = np.zeros_like(gray)
	out[mask == 255] = gray[mask == 255]
	blur = cv2.GaussianBlur(out, (5,5), 0)
	thresh = cv2.adaptiveThreshold(blur, 255, 1, 1, 11, 2)
	contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
	c = 0
	z=0
	for i in contours:
		area = cv2.contourArea(i)
		if area > 0:
			cv2.drawContours(image, contours, c, (0, 255, 0), 1)
			z+=1
		c+=1
	print(z)
	cv2.imshow("Original Image", im)
	cv2.imshow("Grid Detection", image)
	cv2.waitKey(0)
	break
# best_cnt = i
                    # image = cv2.drawContours(image, contours, c, (0, 255, 0), 3)