import numpy as np
import cv2

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
# red is sketchy -> keeps getting mixed up with orange
# white is sketchy -> if light enough it is found if too blue/dark it is completely missed

correct = cv2.imread(r'C:\Users\artsy\OneDrive\Desktop\RubiksCubeProject\correctedTest.png', cv2.IMREAD_COLOR)
old = cv2.imread(r'C:\Users\artsy\OneDrive\Desktop\RubiksCubeProject\ccmTest\lightCube.jpg', cv2.IMREAD_COLOR)
# scale picture to fit in screen
scale_percent = 50
width = int(correct.shape[1] * scale_percent / 100)
height = int(correct.shape[0] * scale_percent / 100)
dim = (width, height)
correct = cv2.resize(correct, dim, interpolation=cv2.INTER_AREA)

correct = simplest_cb(correct)

# scale picture to fit in screen
scale_percent = 20
width = int(old.shape[1] * scale_percent / 100)
height = int(old.shape[0] * scale_percent / 100)
dim = (width, height)
old = cv2.resize(old, dim, interpolation=cv2.INTER_AREA)

correctHSV = cv2.cvtColor(correct,cv2.COLOR_BGR2HSV)
oldHSV = cv2.cvtColor(old,cv2.COLOR_BGR2HSV)

lower_blue = np.array([100, 100, 100])
upper_blue = np.array([140, 255, 255])

lower_green = np.array([40, 100, 100])
upper_green = np.array([80, 255, 255])

lower_orange = np.array([0, 100, 100])
upper_orange = np.array([20, 255, 255])

lower_yellow = np.array([20, 100, 100])
upper_yellow = np.array([40, 255, 255])

lower_white = np.array([0, 0, 200])
upper_white = np.array([180, 50, 255])

image = cv2.bitwise_not(correct)
imageHSV = cv2.cvtColor(correct,cv2.COLOR_BGR2HSV)
lower_red_0 = np.array([0, 100, 100])
upper_red_0 = np.array([15, 255, 255])
lower_red_1 = np.array([160, 100, 100])
upper_red_1 = np.array([180, 255, 255])

redC = cv2.inRange(correctHSV, (80,100,100) ,(100,255,255))
red0 = cv2.inRange(oldHSV, lower_red_1 , upper_red_1 )

greenC = cv2.inRange(correctHSV, lower_green, upper_green)
greenO = cv2.inRange(oldHSV, lower_green, upper_green)

yellowC = cv2.inRange(correctHSV, lower_yellow, upper_yellow)
yellowO = cv2.inRange(oldHSV, lower_yellow, upper_yellow)

whiteC = cv2.inRange(correctHSV, lower_white, upper_white)
whiteO = cv2.inRange(oldHSV, lower_white, upper_white)

orangeC = cv2.inRange(correctHSV, lower_orange, upper_orange)
orangeO = cv2.inRange(oldHSV, lower_orange, upper_orange)

blueC = cv2.inRange(correctHSV, lower_blue, upper_blue)
blueO = cv2.inRange(oldHSV, lower_blue, upper_blue)


"""
cv2.imshow('redC',redC)
cv2.imshow('redO',redO)

cv2.imshow('orangeC',orangeC)
cv2.imshow('orangeO',orangeO)

cv2.imshow('yellowC',yellowC)
cv2.imshow('yellowO',yellowO)

cv2.imshow('greenC',greenC)
cv2.imshow('greenO',greenO)


cv2.imshow('whiteC', whiteC)
cv2.imshow('whiteO',whiteO)
"""

cv2.imshow('blueC',blueC)
cv2.imshow('redC',redC)
cv2.imshow('greenC',greenC)
cv2.imshow('yellowC',yellowC)
cv2.imshow('whiteC',whiteC)
cv2.imshow('orangeC',orangeC)
cv2.imshow('image', image)
cv2.imshow('imageHSV', imageHSV)
#cv2.imshow('blueO',blueO)
#cv2.imshow('old', old)
cv2.imshow('correct', correct)
cv2.imshow('hsv',correctHSV)
cv2.waitKey(0)
cv2.destroyAllWindows()