from pyimagesearch.shapedetector import ShapeDetector
import numpy as np
import argparse
import imutils
import cv2

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	help="path to the input image")
args = vars(ap.parse_args())

# load the image and resize it to a smaller factor so that
# the shapes can be approximated better
im = cv2.imread(args["image"])
resized = imutils.resize(im, width=600)
ratio = im.shape[0] / float(resized.shape[0])

kernel = np.ones((3,3),np.uint8)
erosion = cv2.dilate(resized,kernel,iterations = 1)
#imgray = cv2.cvtColor(erosion,cv2.COLOR_BGR2GRAY)

adjusted = cv2.Canny(erosion, 50, 200)

image, cnts, hierarchy = cv2.findContours(adjusted, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

sd = ShapeDetector()

# loop over the contours
for c in cnts:
	# compute the center of the contour, then detect the name of the
	# shape using only the contour
	
	#print(cv2.contourArea(c))
	
	#Calculate the shape centroid positions
	cX = 0
	cY = 0
	for p in c:
		cX += p[0][0]
    	cY += p[0][1]
	cX = int(cX/len(c))
	cY = int(cY/len(c))
	
	if(cv2.contourArea(c) <= 15 or cv2.contourArea(c) >= 20000):
		cv2.drawContours(resized, [c], -1, (255, 0, 255), 1)
		#cv2.putText(resized, "What", cX, cv2.FONT_HERSHEY_SIMPLEX, 
		#	0.5, (255, 0, 255), 1)
		
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
			cv2.drawContours(resized, [c], -1, (0, 0, 255), 1)
			if(lravgx > udavgx):
						string = "LArrow "
						string += str(cv2.contourArea(c))
						cv2.putText(resized, string, lmax, cv2.FONT_HERSHEY_SIMPLEX,
							0.5, (0, 0, 255), 1)
			else:
						string = "RArrow "
						string += str(cv2.contourArea(c))
						cv2.putText(resized, string, lmax, cv2.FONT_HERSHEY_SIMPLEX,
							0.5, (0, 0, 255), 1)
		else:
			cv2.drawContours(im, [c], -1, (0, 0, 255), 1)
			if(lravgy > udavgy):
						string = "DArrow "
						string += str(cv2.contourArea(c))
						cv2.putText(resized, string, lmax, cv2.FONT_HERSHEY_SIMPLEX,
							0.5, (0, 0, 255), 1)
			else:
						string = "UArrow "
						string += str(cv2.contourArea(c))
						cv2.putText(resized, string, lmax, cv2.FONT_HERSHEY_SIMPLEX,
							0.5, (0, 0, 255), 1)		
	else:
		M = cv2.moments(c)
		cX = int((M["m10"] / M["m00"]))
		cY = int((M["m01"] / M["m00"]))
		shape = sd.detect(c)
	
		# multiply the contour (x, y)-coordinates by the resize ratio,
		# then draw the contours and the name of the shape on the image
		cv2.drawContours(resized, [c], -1, (0, 255, 0), 1)
		string = shape
		string += str(cv2.contourArea(c))
		cv2.putText(resized, string, (cX, cY), cv2.FONT_HERSHEY_SIMPLEX,
			0.5, (255, 255, 255), 1)
	
		# show the output image
		cv2.imshow("Image", resized)
		cv2.waitKey(0)
	
