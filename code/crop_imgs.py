import cv2
import os
imgnames = os.listdir("images/")

for imgname in imgnames:
	img = cv2.imread("images/"+imgname)
	crop_img = img[0:420, 0:500]
	cv2.imwrite("images/"+imgname,crop_img)
