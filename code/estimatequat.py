import cv2
import os
import csv
import imutils
import numpy as np
from math import sqrt
import math

imgnames = os.listdir("images/")
# test250.jpg	1.74024822	-3.5865901	-3.33245539	-1.73281525

def crossprod(a, b):
	c = np.cross(a,b,axis=0)
	return c

def color(pix):
	b, g, r = int(pix[0]), int(pix[1]), int(pix[2])
	if abs(b-g)<10 and abs(b-r)<10 and abs(b-235)<50:
		return 0 # white
	elif b<10 and g<10 and r>180:
		return 1 # red
	elif b<25 and g>150 and r<10:
		return 2 # green
	elif b<10 and g>100 and g<120 and r>200:
		return 3 # orange
	elif b>100 and g<10 and r<10:
		return 4 # blue
	else:
		return 5 # yellow

def dist(a,b):
	return sum([(x1 - x2)**2 for x1,x2 in zip(a,b)]) ** 0.5

def abs(x):
	if x<0:
		return -x
	else:
		return x

for imgname in imgnames[2:]:
	image = cv2.imread("images2/test330.jpg")#+imgname) # images2/330
	im = image.copy()
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 
	blur = cv2.GaussianBlur(gray, (3,3), 0)
	thresh = cv2.adaptiveThreshold(blur, 255, 1, 1, 11, 2)
	edged = cv2.Canny(gray, 30, 30) 
	contours, hierarchy = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
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
				# image = cv2.drawContours(image, contours, c, (0, 255, 0), 3)
		c+=1
	cv2.drawContours(mask,[best_cnt],0,255,-1)
	cv2.drawContours(mask,[best_cnt],0,0,2)
	out = np.zeros_like(gray)
	out[mask == 255] = gray[mask == 255]
	blur = cv2.GaussianBlur(out, (5,5), 0)
	thresh = cv2.adaptiveThreshold(blur, 255, 1, 1, 11, 2)
	contours,_ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
	c = 0
	z=0
	areas = []
	for i in contours:
		area = cv2.contourArea(i)
		areas.append(int(area))
	sareas = sorted(areas)
	colors = [[],[],[],[],[],[]]
	contourcenters = []
	for i in contours:
		area = int(cv2.contourArea(i))
		if area == sareas[-1]:
			pass
		elif area>150:
			z+=1
			M = cv2.moments(contours[c])
			cX = int(M["m10"] / M["m00"])
			cY = int(M["m01"] / M["m00"])
			pix = image[cY][cX]
			col = color(pix)
			# print(pix, col, cX,cY)
			'''
			0 - white
			1 - red
			2 - green
			3 - orange
			4 - blue
			5 - yellow
			'''
			colors[col].append([cX,cY])
			contourcenters.append([cX,cY])
			cv2.drawContours(image, contours, c, (0, 255, 0), 1)
			cv2.circle(image, (cX, cY), 2, (0, 0, 0), -1)
		c+=1
	# if abs(z-18)<=2: # 2 colors seen
	# 	print(colors[0])
	ncolors = []
	cols = []
	for zz in range(6):
		le = len(colors[zz])
		ncolors.append(le)
		if le>1:
			cols.append(zz)
	a,b = cols[0], cols[1]
	mindist = 10000
	mini, minj= 0,0
	dists, ij = [],[]

	for i in colors[a]:
		for j in colors[b]:
			dists.append(dist(i,j))
			ij.append([i,j])
			# if d<mindist:
			# 	mindist = d
			# 	mini = i
			# 	minj = j
	sortedpoints = np.argsort(dists)[:3]
	aa = 0
	ncents = len(contourcenters)
	dists , injn = [],[]
	xs = []
	for i in sortedpoints:
		x,y = ij[i][0], ij[i][1]
		xs.append(x)
	center = 0
	if abs( dist(xs[0],xs[1]) - dist(xs[0],xs[2]) ) <2:
		center = 0
		pass # x[0] is center
	elif abs( dist(xs[0],xs[1]) - dist(xs[1],xs[2]) ) <2:
		center = 1
		pass # x[1] is center
	elif abs( dist(xs[0],xs[2]) - dist(xs[1],xs[2]) ) <2:
		center = 2
		pass # x[2] is center
	# print(center,"--- center",xs[center])
	points = []
	centerc = []
	for i in sortedpoints:
		x,y = ij[i][0], ij[i][1]
		# image = cv2.line(image, (x[0],x[1]), (y[0],y[1]), (255,0,0), 2)
		# if aa%2 == 0:
		dists , injn = [],[]
		# find nearest point for a in colors[a]
		for pt in colors[a]:
			dists.append(dist(x,pt))
			injn.append([x,pt])
		res = min([idx for idx, val in enumerate(sorted(dists)) if val != 0])
		spts = np.argsort(dists)[res:res+1]
		# print(sorted(dists),spts)
		# print("---")
		xa = injn[spts[0]][1]
		# image = cv2.line(image, (x[0],x[1]), (xa[0],xa[1]), (255,0,0), 2)
		if x[0] != xs[center][0] and x[1] != xs[center][1]:
			# x, xs[center] and xa - edge/center
			l1x = ( (3*x[0] - xa[0]) // 2, (3*x[1] - xa[1]) // 2)
			points.append(l1x)
			# l2x = ( (3*x[0] - xs[center][0]) // 2, (3*x[1] - xs[center][1])//2)
			cv2.circle(image, l1x, 4, (255, 0, 0), -1)
			# cv2.circle(image, (l2x[0], l2x[1]), 2, (255, 0, 0), -1)
		else:
			centerc = ( (3*x[0] - xa[0]) // 2, (3*x[1] - xa[1]) // 2)
		# aa+=1

	p1 = points[0]
	p2 = points[1]
	point2 = ( (5*p1[0] - p2[0]) // 4, ((5*p1[1] - p2[1]) // 4 )-2 )
	point1 = ( (5*p2[0] - p1[0]) // 4, ((5*p2[1] - p1[1]) // 4 )+3 )
	cv2.circle(image, point1, 4, (0, 255, 0), 2) # top corner
	cv2.circle(image, point2, 4, (255, 0, 0), 2)
	image = cv2.line(image, point1, point2, (255,0,255), 2)
	z0 = int(sqrt(26569 - ( (point1[0]-248)**2 + (point1[1]-209)**2 )))
	z1 = int(sqrt(26569 - ( (point2[0]-248)**2 + (point2[1]-209)**2 )))
	print(point1, point2)	
	rwgcorner = (point1[0],point1[1],z0)
	rwbcorner = (point2[0],point2[1],z1)

	# test330.jpg	-0.25867456	0.67844045	0.6308267	0.27361231

	v1 = [115,-115,115]
	# v1 = v1 / np.linalg.norm(v1)
	v2 = [ rwgcorner[i] for i in range(3)  ]
	# v2 = v2 / np.linalg.norm(v2)
	print(v1,v2,"hiiiiiii")
	xyz = crossprod(v2,v1)
	print(xyz,"AXIS OF ROT:",xyz / np.linalg.norm(xyz))
	xyz = crossprod(v1,v2)
	w = (len(v1)*len(v2) ) + np.dot(v1, v2)
	w = w%360
	cosw = math.cos(w/2)
	s = math.sin(w/2)
	xyz = [s*i for i in xyz]
	xyz_hat = xyz / np.linalg.norm(xyz)
	print(cosw,xyz_hat,"-------FINAL ANSWER?")
	rwg = (rwgcorner[0]-248, 209-rwgcorner[1], rwgcorner[2])
	rwb = (rwbcorner[0]-248, 209-rwbcorner[1], rwbcorner[2])
	print(rwg,rwb,"------------------")

	# quat = [int(w%360), int(xyz[0]), int(xyz[1]), int(xyz[2])]
	# print(quat_norm)

	# print(quat_norm," ---------- FINAL ANSWER")
	print(rwgcorner, rwbcorner,"-----",rwg,rwb)


	# midpt = ( ( (p1[0]+p2[0]) // 2 ) - 248 , ( (p1[1]+p2[1]) // 2 ) - 209, ( (rwgcorner[2]+rwbcorner[2]) // 2 ) )
	# (x,y,z) -> (x-248, 209-y, z)
	midpt = ( ( (p1[0]+p2[0]) // 2 ), ( (p1[1]+p2[1]) // 2 ), ( (rwgcorner[2]+rwbcorner[2]) // 2 ) )
	rwmid = ( ( (rwgcorner[0]+midpt[0]) // 2 ) , ( (rwgcorner[1]+midpt[1]) // 2 ) , ( (rwgcorner[2]+midpt[2]) // 2 ) )
	# print(rwgcorner, rwbcorner, midpt, rwmid)
	# cv2.circle(image, (midpt[0],midpt[1]), 4, (255, 0 , 0), 2)
	# cv2.circle(image, (rwmid[0],rwmid[1]), 4, (255, 0, 0), 2)

	# rwg = (rwgcorner[0]-248, 209-rwgcorner[1], rwgcorner[2])
	# rwmidmid = (rwmid[0]-248, 209-rwmid[1], rwmid[2])
	# midpt = (midpt[0]-248, 209-midpt[1], midpt[2])
	# rwb = (rwbcorner[0]-248, 209-rwbcorner[1], rwbcorner[2])

	# print(rwg,rwmidmid,midpt,rwb)



	print("Number of centroids:",z)
	cv2.imshow("Original Image", im)
	cv2.circle(image, (248,209), 163, (0, 0, 0), 1)
	cv2.imshow("Grid Detection", image)
	cv2.waitKey(0)
	break

'''
    x =  q1.x * q2.w + q1.y * q2.z - q1.z * q2.y + q1.w * q2.x
    y = -q1.x * q2.z + q1.y * q2.w + q1.z * q2.x + q1.w * q2.y
    z =  q1.x * q2.y - q1.y * q2.x + q1.z * q2.w + q1.w * q2.z
    w = -q1.x * q2.x - q1.y * q2.y - q1.z * q2.z + q1.w * q2.w
'''

'''
	vector a = crossproduct(v1, v2);
	q.xyz = a;
	q.w = sqrt((v1.Length ^ 2) * (v2.Length ^ 2)) + dotproduct(v1, v2)
'''

	# print(v1_hat,"----------",v2_hat)
	# print(np.linalg.norm(v1_hat), np.linalg.norm(v2_hat) )
	# sina = np.linalg.norm(xyz) / ( (np.linalg.norm(v1))*(np.linalg.norm(v2)) )
	# cosa = np.dot(v1,v2) / ( (np.linalg.norm(v1))*(np.linalg.norm(v2)) )
	# print(cosa,sina*xyz[0], sina*xyz[1], sina*xyz[2],"--- is this it??")
