from pyimagesearch.shapedetector import ShapeDetector
import numpy as np
import argparse
import imutils
import cv2

f = open("list.txt", "w+")
element_unsorted = []


def sort_contours(cnts, method="left-to-right"):
	# initialize the reverse flag and sort index
	reverse = False
	i = 0
 
	# handle if we need to sort in reverse
	if method == "right-to-left" or method == "bottom-to-top":
		reverse = True
 
	# handle if we are sorting against the y-coordinate rather than
	# the x-coordinate of the bounding box
	if method == "top-to-bottom" or method == "bottom-to-top":
		i = 1
 
	# construct the list of bounding boxes and sort them from top to
	# bottom
	boundingBoxes = [cv2.boundingRect(c) for c in cnts]
	(cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),
		key=lambda b:b[1][i], reverse=reverse))
 
	# return the list of sorted contours and bounding boxes
	return (cnts, boundingBoxes)

im = cv2.imread('rlph.jpg')
resized = imutils.resize(im, width=600)
ratio = im.shape[0] / float(resized.shape[0])

kernel = np.ones((2,2),np.uint16)
erosion = cv2.dilate(resized,kernel,iterations = 1)
imgray = cv2.cvtColor(erosion,cv2.COLOR_BGR2GRAY)
cv2.imshow("SImage", imgray)

adjusted = cv2.Canny(imgray, 50, 200)
#ret, adjusted = cv2.threshold(imgray, 60, 255, 0)
#adjusted = ~adjusted

cv2.imshow("SIEmage", adjusted)

image, cnts, hierarchy = cv2.findContours(adjusted, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_KCOS)

sd = ShapeDetector()

#img = cv2.drawContours(im, cnts, -1, (0,0,0), 2)
#cv2.imshow("copy", img)
#cv2.waitKey(0)

#cntsort = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]
 
# sort the contours according to the provided method
#(cntsort, boundingBoxes) = sort_contours(cnts, "left-to-right")
 
# loop over the (now sorted) contours and draw them
#for (i, c) in enumerate(cntsort):
#	cv2.drawContours(resized, [c], -1, (0, 255, 0), 1)
 
# show the output image
#cv2.imshow("Sorted", adjusted)
#cv2.waitKey(0)


# loop over the contours
for c in cnts:

	#Calculate the shape centroid positions
	cX = 0
	cY = 0
	for p in c:
		cX += p[0][0]
    	cY += p[0][1]
	cX = int(cX/len(c))
	cY = int(cY/len(c))
	
		
	if(cv2.contourArea(c) <= 500 or cv2.contourArea(c) >= 30000):
		print("possible invalid shape")
		
	elif(cv2.contourArea(c) <= 1000):
		#print("arrow found")
		lmax = tuple(c[c[:,:,0].argmin()][0])
		rmax = tuple(c[c[:,:,0].argmax()][0])
		lravgx = (lmax[0] + rmax[0])/2
		lravgy = (lmax[1] + rmax[1])/2
		wid = abs(rmax[0] - lmax[0])
		
		umax = tuple(c[c[:,:,1].argmin()][0])
		dmax = tuple(c[c[:,:,1].argmax()][0])
		udavgx = (umax[0] + dmax[0])/2
		udavgy = (umax[1] + dmax[1])/2
		hgt = abs(umax[1] - dmax[1])
		

		if(wid > hgt):
			cv2.drawContours(resized, [c], -1, (0, 0, 255), 2)
			if(lravgx > udavgx):
						element_unsorted.append(("larrow,",cX,cY))
			else:
						element_unsorted.append(("rarrow,",cX,cY))
		else:
			cv2.drawContours(im, [c], -1, (0, 0, 255), 1)
			if(lravgy > udavgy):
						element_unsorted.append(("darrow,",cX,cY))	
			else:
						element_unsorted.append(("uarrow,",cX,cY))				
	else:
		shape = sd.detect(c)
		shape += ","
		element_unsorted.append((shape,cX,cY))
	
	
	for i in element_unsorted:
		f.write(i[0])
	
	

