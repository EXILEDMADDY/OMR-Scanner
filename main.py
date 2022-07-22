import cv2
from cv2 import boundingRect
import imutils
from imutils import contours
import numpy

# Setup SimpleBlobDetector parameters.
params = cv2.SimpleBlobDetector_Params()

params.filterByColor = 1
params.blobColor = 255

# Change thresholds
params.minThreshold = 0
params.maxThreshold = 200

# Filter by Area.
params.filterByArea = True
params.minArea = 5
params.maxArea = 5000000

# Filter by Circularity
params.filterByCircularity = True
params.minCircularity = 0

# Filter by Convexity
params.filterByConvexity = True
params.minConvexity = 0

# Filter by Inertia
params.filterByInertia = True
params.minInertiaRatio = 0.5


params.minDistBetweenBlobs = 5

# Create a detector with the parameters
detector = cv2.SimpleBlobDetector_create(params)

def get_points(keypoints):
    
    points = []
    
    # Loop over all the keypoints
    for keyPoint in keypoints:
        x = int(keyPoint.pt[0])
        y = int(keyPoint.pt[1])
        # saving the points
        points.append((x,y))
    
    return points

# checking if point is in range of rectangle
def inrange(rec,point): 
    if point[0]>=rec[0] and point[0]<=(rec[0]+rec[2]) and point[1]>= rec[1] and point[1]<=rec[1]+rec[3]:
        return True
    else:
        return False

# input file path
input_file = "OMR-CTET-SHEET-Sample.jpeg"
img = cv2.imread(input_file)
img2 = img.copy()

# changing image to grayscale
gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
thresh = cv2.threshold(gray_image, 0, 255,
	cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1] # binarizing image
thresh = cv2.blur(thresh,(3,3))
cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, # finding contours in image
	cv2.CHAIN_APPROX_SIMPLE)

# threshold on grayscale image for detecting selected answers
low = numpy.array([0])
high = numpy.array([100])
thresh2 = cv2.inRange(gray_image, low, high)  
	
# using blib detector to detect keypoints
keypoints = detector.detect(thresh2)
keypoints2 = get_points(keypoints)

# looping over contours to select the ones of required size
cnts = imutils.grab_contours(cnts)
questionCnts = []
rects = []
for c in cnts:
	(x, y, w, h) = cv2.boundingRect(c)
	selected = False
	ar = w / float(h) # aspect ratio

	if w <= 50 and h <= 50 and ar<=1.3 and ar>=0.7 and w >= 13 and h >= 13:
		questionCnts.append(c)
		rects.append((x,y,w,h))

		for k in keypoints2:
			if inrange((x,y,w,h),k):
				selected = True
				break
		if selected:
			cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2) # marking selected bubbles in blue in img
		else:
			cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2) # marking unselected bubbles in green in img
		cv2.rectangle(img2,(x,y),(x+w,y+h),(0,255,0),2) # marking all bubbles in green in img2

questionCnts = contours.sort_contours(questionCnts,
	method="left-to-right")[0]

# drawing keypoints for threshholded image
im_with_keypoints = cv2.drawKeypoints(thresh2, keypoints, numpy.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

# displaying sample images
cv2.imshow("All bubbles", img2)
cv2.waitKey(0)
cv2.imshow("Selected answers", im_with_keypoints)
cv2.waitKey(0)
cv2.imshow("Selected bubbles", img)
cv2.waitKey(0)