import numpy as np
import cv2

# white balancing
def simplest_cb(img, percent=10):
    out_channels = []
    cumstops = (
        img.shape[0] * img.shape[1] * percent / 200.0,
        img.shape[0] * img.shape[1] * (1 - percent / 200.0)
    )
    for channel in cv2.split(img):
        cumhist = np.cumsum(cv2.calcHist([channel], [0], None, [256], (0,256)))
        low_cut, high_cut = np.searchsorted(cumhist, cumstops)
        lut = np.concatenate((
            np.zeros(low_cut),
            np.around(np.linspace(0, 255, high_cut - low_cut + 1)),
            255 * np.ones(255 - high_cut)
        ))
        out_channels.append(cv2.LUT(channel, lut.astype('uint8')))
    return cv2.merge(out_channels)

# read image
img = cv2.imread(r'C:\Users\artsy\OneDrive\Desktop\RubiksCubeProject\ccmTest\darkCube.jpg', cv2.IMREAD_COLOR)

# scale picture to fit in screen
scale_percent = 50
width = int(img.shape[1] * scale_percent / 100)
height = int(img.shape[0] * scale_percent / 100)
dim = (width, height)
resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)

# white balance the image
resized = simplest_cb(resized)

# convert image to gray
grayImg = cv2.cvtColor(resized,cv2.COLOR_BGR2GRAY)

# filter preserves edges more (blur = cv2.bilateralFilter(grayImg,10,100,100))
blur = cv2.medianBlur(grayImg,3)

#somewhat color dependent edge detection (use parameters from picture to compute?)
edge = cv2.Canny(blur, 32, 50)

# use a kernel sharpening matrix to sharpen image
sharpen_kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
sharp = cv2.filter2D(edge, -1, sharpen_kernel)

# extrapolate and morphological transformations to make image fuller
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
dilation = cv2.dilate(sharp,kernel,iterations = 1)
close = cv2.morphologyEx(sharp, cv2.MORPH_CLOSE, kernel, iterations=2)

#find all contours
image, contours, hierarchy = cv2.findContours(close, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
#print("Total Contours = " + str(len(contours)))

# sort through contours to find rectangles
#1500
area = 5000
rectContours = []
for contour in contours:
    # approximate contour using perimeter (1-5% of original perimeter)
    approx = cv2.approxPolyDP(contour,0.03 * cv2.arcLength(contour,True), True)
    # if len(approx) == 4 and cv2.contourArea(approx) > area and cv2.contourArea(approx) <400000 20000
    if len(approx) == 4 and cv2.contourArea(approx) > area and cv2.contourArea(approx) < 400000:
        print('Area = ' + str(cv2.contourArea(approx)))
        x = approx.ravel()[0]
        y = approx.ravel()[1]
        copy = False
        for i in rectContours:
            if abs(i.ravel()[0] - x) <= 20 and abs(i.ravel()[1] - y) <= 20:
                copy = True
        if copy == False:
            rectContours.append(approx)

squareCont = []
for rect in rectContours:
    squareCont.append(cv2.boundingRect(rect))


# print rectangles
topLeft = []
# make these two average later
widthMeasure = []
heightMeasure = []

print("Total Rectangles = " + str(len(squareCont)))
for index,cont in enumerate(squareCont, 1):
    x,y,w,h = cont
    widthMeasure.append(w)
    heightMeasure.append(h)
    """
      cv2.rectangle(resized, (x, y), (x + w, y + h), (255, 0, 0), 1, 30)
    cv2.putText(resized, str(index), (x, y), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 0, 0))
    contX = x + w/2
    contY = y + h/2
    cv2.circle(resized, (int(contX), int(contY)), 5, (0, 0, 0), 10)
    """
    topLeft.append((x,y))

smallestX = topLeft[0][0]
largestX = 0
smallestY = topLeft[0][1]
largestY = 0
for index,coord in enumerate(topLeft, 0):
    if topLeft[index][0] < smallestX:
        smallestX = topLeft[index][0]
    if topLeft[index][0] > largestX:
        largestX = topLeft[index][0]
    if topLeft[index][1] < smallestY:
        smallestY = topLeft[index][1]
    if topLeft[index][1] > largestY:
        largestY = topLeft[index][1]

largeX = largestX + widthMeasure[0]
largeY = largestY + heightMeasure[0]
cube = ((smallestX, largeY) , (largeX, smallestY))

#cv2.rectangle(resized, (smallestX, largeY) , (largeX, smallestY) , (0,255,0), 8)
averageX = ((smallestX + largestX)/ 2)
averageY = ((smallestY + largestY) /2)
centerX= int(averageX)
centerY = int(averageY)

averageWidth = widthMeasure[0]
"""
# average width of contour
totalWidth = 0
for w in widthMeasure:
    totalWidth = totalWidth + w
averageWidth = int (totalWidth/len(widthMeasure))
"""


# first row (left -> right)
cv2.rectangle(resized, (smallestX, smallestY) , (smallestX + averageWidth, smallestY + heightMeasure[0]), (0,255,0) , 5)
cv2.rectangle(resized, (centerX, smallestY) , (centerX + averageWidth ,smallestY + heightMeasure[0]), (0,255,0) , 5)
cv2.rectangle(resized, (largeX, smallestY) , (largeX - averageWidth, smallestY + heightMeasure[0]), (0,255,0) , 5)
# second row (left -> right)
cv2.rectangle(resized, (smallestX, centerY), (smallestX + averageWidth, centerY + heightMeasure[0]), (0,255,0), 5)
cv2.rectangle(resized, (centerX, centerY) , (centerX + averageWidth ,centerY + heightMeasure[0]), (0,255,0) , 5)
cv2.rectangle(resized, (largeX, centerY) , (largeX - averageWidth, centerY + heightMeasure[0]), (0,255,0) , 5)
# third row  (left -> right)
cv2.rectangle(resized, (smallestX, largestY), (smallestX + averageWidth, largestY + heightMeasure[0]), (0,255,0), 5)
cv2.rectangle(resized, (centerX, largestY) , (centerX + averageWidth ,largestY + heightMeasure[0]), (0,255,0) , 5)
cv2.rectangle(resized, (largeX, largestY) , (largeX - averageWidth, largestY + heightMeasure[0]), (0,255,0) , 5)

"""
averageX = ((smallestX + largestX)/ 2)
averageY = ((smallestY + largestY) /2)
centerX= int(averageX)
centerY = int(averageY)
cv2.circle(resized, (centerX, centerY), 5, (255,255,255), 10)
cv2.rectangle(resized, (smallestX, largestY) , (largestX, smallestY) , (0,255,0), 8)
"""


# show image
cv2.imshow('image', resized)
cv2.imshow('edge', edge)
cv2.imshow('close', close)
cv2.waitKey(0)
cv2.destroyAllWindows()

