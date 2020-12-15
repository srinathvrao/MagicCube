import cv2
import os
import numpy as np
from math import sqrt

imgnames = os.listdir("images2/")

# for imgname in imgnames:
# 	img = cv2.imread("images2/"+imgname)
# 	crop_img = img[0:420, 0:500]
# 	cv2.imwrite("images2/"+imgname,crop_img)

# image = cv2.imread("images/test0.jpg")
# cv2.circle(image, (248,209), 163, (0, 0, 0), 1)
# cv2.imshow("frame",image)
# cv2.waitKey(0)

# l1 = [1.74024822, -3.5865901, -3.33245539, -1.73281525]
# l1_hat = l1 / np.linalg.norm(l1)
# print(l1_hat)
a = [0,0,230]
b = [14,249,16]
# (10, 115, 115) (-4, -129, 99)
ax = np.cross(a,b,axis=0)
sina = np.linalg.norm(ax) / (np.linalg.norm(a)*np.linalg.norm(b))
print(sqrt(1- (sina ** 2) ))
print(sina, ax / np.linalg.norm(ax))