from pyimagesearch.shapedetector import ShapeDetector
import numpy as np
import argparse
import imutils
import cv2

im = cv2.imread('irlphoto.png')
resized = imutils.resize(im, width=600)
ratio = im.shape[0] / float(resized.shape[0])

kernel = np.ones((3,3),np.uint8)
erosion = cv2.dilate(resized,kernel,iterations = 1)
imgray = cv2.cvtColor(erosion,cv2.COLOR_BGR2GRAY)
cv2.imshow("SImage", imgray)

#canny_output = cv2.Canny(imgray, 50, 200)
ret, adjusted = cv2.threshold(imgray, 125, 255, 0)
cv2.imshow("SIEmage", adjusted)

image, cnts, hierarchy = cv2.findContours(adjusted, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

sd = ShapeDetector()

#img = cv2.drawContours(im, cnts, -1, (0,0,0), 2)
#cv2.imshow("copy", img)
#cv2.waitKey(0)


# loop over the contours
for c in cnts:
	# compute the center of the contour, then detect the name of the
	# shape using only the contour
	if(cv2.contourArea(c) <= 1000):
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
						cv2.putText(resized, "Left Arrow", lmax, cv2.FONT_HERSHEY_SIMPLEX,
							0.5, (0, 0, 255), 1)
			else:
						cv2.putText(resized, "Right Arrow", lmax, cv2.FONT_HERSHEY_SIMPLEX,
							0.5, (0, 0, 255), 1)
		else:
			cv2.drawContours(im, [c], -1, (0, 0, 255), 1)
			if(lravgy > udavgy):
						cv2.putText(resized, "Down Arrow", lmax, cv2.FONT_HERSHEY_SIMPLEX,
							0.5, (0, 0, 255), 1)		
			else:
						cv2.putText(resized, "Up Arrow", lmax, cv2.FONT_HERSHEY_SIMPLEX,
							0.5, (0, 0, 255), 1)					
	else:
		M = cv2.moments(c)
		cX = int((M["m10"] / M["m00"]))
		cY = int((M["m01"] / M["m00"]))
		shape = sd.detect(c)
	
		# multiply the contour (x, y)-coordinates by the resize ratio,
		# then draw the contours and the name of the shape on the image
		cv2.drawContours(resized, [c], -1, (0, 255, 0), 1)
		cv2.putText(resized, shape, (cX, cY), cv2.FONT_HERSHEY_SIMPLEX,
			0.5, (255, 255, 255), 1)
	
		# show the output image
		cv2.imshow("Image", resized)
		cv2.waitKey(0)
	

