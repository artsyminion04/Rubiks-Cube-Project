import numpy as np
import cv2
from copy import copy, deepcopy

standardValues = [[53,53,53],
[84,84,84],
[120,121,122],
[159,160,159],
[199,200,200],
[236,241,242],
[163,134,13],
[144,83,177],
[56,199,234],
[58,48,167],
[76,148,86],
[149,62,52],
[58,162,223],
[74,186,162],
[105,62,89],
[97,85,184],
[162,91,76],
[49,121,207],
[170,189,115],
[173,128,129],
[69,108,91],
[154,122,98],
[130,149,189],
[67,81,111]]

def best_fit(actualBGRvalues, standard, channel):
    actualBGR = []
    standardChannelValues = []
    # create C array with x values
    for i in range(24):
        b = int(actualBGRvalues[i][0])
        g = int(actualBGRvalues[i][1])
        r = int(actualBGRvalues[i][2])
        actualBGR.append([1,b,g,r, b*r, b*g, g*r,b**2, g**2,r**2])
    # create array of standard values
    # first blue values
    for i in range(24):
        standardChannelValues.append(standard[i][channel])

    # get C transpose
    actualBGR_transpose = np.transpose(actualBGR)

    matrix = actualBGR_transpose.dot(actualBGR)

    coefficients = np.linalg.inv(matrix).dot(actualBGR_transpose).dot(standardChannelValues)

    return coefficients
# collect actual values
collectedValues = []

#img = cv2.imread(r'C:\Users\artsy\OneDrive\Desktop\RubiksCubeProject\9-30 test\test1Chart.jpg', cv2.IMREAD_COLOR)
img = cv2.imread(r'C:\Users\artsy\OneDrive\Desktop\RubiksCubeProject\bestFit\lindImg.jpg', cv2.IMREAD_COLOR)

scale_percent = 20
width = int(img.shape[1] * scale_percent / 100)
height = int(img.shape[0] * scale_percent / 100)
dim = (width, height)
img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)

r = cv2.selectROI(img, fromCenter=False)

crop = img[int(r[1]):int(r[1] + r[3]), int(r[0]):int(r[0] + r[2])]
h, w, c = crop.shape

constantW = 0.03125
constantH = 0.048
constantS = 0.212
constantB = 0.15

smallWidth = int(constantW * w)
smallHeight = int(constantH * h)
boxConstant = int(constantB * w)

startX = 2 * smallWidth
startY = smallHeight + 10
for y in range (4):
    for x in range (6):
        roi = crop[startY: startY - smallHeight + boxConstant, startX:startX - smallWidth + boxConstant]
        cv2.rectangle(crop, (startX, startY), (startX - smallWidth + boxConstant, startY - smallHeight + boxConstant),(255, 255, 0), 2)
        # change from average to dominant color? more accurate
        avg_color_per_row = np.average(roi, axis=0)
        avg_color = np.average(avg_color_per_row, axis=0)
        collectedValues.append(avg_color)
        startX += 5*smallWidth
    startY += int(5 * smallHeight)
    startX = 2 * smallWidth

print(collectedValues)

# calculate transform
coefficients = []
#blue
coefficients.append(best_fit(collectedValues, standardValues, 0))
#green
coefficients.append(best_fit(collectedValues, standardValues, 1))
#red
coefficients.append(best_fit(collectedValues, standardValues, 2))

print(coefficients)

# apply transform
cube = cv2.imread(r'C:\Users\artsy\OneDrive\Desktop\RubiksCubeProject\bestFit\cube.jpg', cv2.IMREAD_COLOR)

# scale picture to fit in screen
scale_percent = 20
width = int(cube.shape[1] * scale_percent / 100)
height = int(cube.shape[0] * scale_percent / 100)
dim = (width, height)
cube = cv2.resize(cube, dim, interpolation=cv2.INTER_AREA)

resized = deepcopy(cube)

w = resized.shape[0]
h = resized.shape[1]
#print('width = ' + str(w) + 'height = ' + str(h))

for x in range(w):
    for y in range(h):
        pixel = resized[x][y]
        b = pixel[0]
        g = pixel[1]
        r = pixel[2]
        correctB = coefficients[0][0] + coefficients[0][1] * b + coefficients[0][2] * g + coefficients[0][3] * r + coefficients[0][4]* b*r + coefficients[0][5]* b*g + coefficients[0][6] * g*r + coefficients[0][7] * b**2 + coefficients[0][8] * g**2 + coefficients[0][9] * r**2
        correctG = coefficients[1][0] + coefficients[1][1] * b + coefficients[1][2] * g + coefficients[1][3] * r + coefficients[1][4]* b*r + coefficients[1][5]* b*g + coefficients[1][6] * g*r + coefficients[1][7] * b**2 + coefficients[1][8] * g**2 + coefficients[1][9] * r**2
        correctR = coefficients[2][0] + coefficients[2][1] * b + coefficients[2][2] * g + coefficients[2][3] * r + coefficients[2][4] * b*r + coefficients[2][5] * b*g + coefficients[2][6] * g*r + coefficients[2][7] * b**2 + coefficients[2][8] * g**2 + coefficients[2][9] * r**2
        resized[x][y] = (correctB, correctG, correctR)

cv2.imwrite(r'C:\Users\artsy\OneDrive\Desktop\RubiksCubeProject\bestFit\test.jpg', resized, [cv2.IMWRITE_JPEG_QUALITY, 100])

# detect colors
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

correct = cv2.imread(r'C:\Users\artsy\OneDrive\Desktop\RubiksCubeProject\bestFit\test.jpg', cv2.IMREAD_COLOR)

# scale picture to fit in screen
scale_percent = 120
width = int(correct.shape[1] * scale_percent / 100)
height = int(correct.shape[0] * scale_percent / 100)
dim = (width, height)
correct = cv2.resize(correct, dim, interpolation=cv2.INTER_AREA)

correct = simplest_cb(correct)

correctHSV = cv2.cvtColor(correct,cv2.COLOR_BGR2HSV)

lower_green = np.array([40, 100, 100])
upper_green = np.array([80, 255, 255])

lower_blue = np.array([100, 100, 100])
upper_blue = np.array([140, 255, 255])

lower_orange = np.array([0, 100, 100])
upper_orange = np.array([20, 255, 255])

lower_yellow = np.array([20, 100, 100])
upper_yellow = np.array([40, 255, 255])

lower_white = np.array([0, 0, 200])
upper_white = np.array([180, 50, 255])

lower_red_0 = np.array([0, 100, 100])
upper_red_0 = np.array([15, 255, 255])
lower_red_1 = np.array([160, 100, 100])
upper_red_1 = np.array([180, 255, 255])

redC = cv2.inRange(correctHSV, lower_red_0 , upper_red_0)
blueC = cv2.inRange(correctHSV, lower_blue, upper_blue)
greenC = cv2.inRange(correctHSV, lower_green, upper_green)
yellowC = cv2.inRange(correctHSV, lower_yellow, upper_yellow)
whiteC = cv2.inRange(correctHSV, lower_white, upper_white)
orangeC = cv2.inRange(correctHSV, lower_orange, upper_orange)


cv2.imshow('redC',redC)
cv2.imshow('orangeC',orangeC)
cv2.imshow('yellowC',yellowC)
cv2.imshow('greenC',greenC)
cv2.imshow('blueC',blueC)
cv2.imshow('whiteC', whiteC)

cv2.imshow('correctHSV', correctHSV)
cv2.imshow('correct', correct)
cv2.imshow('original', cube)
cv2.imshow('image', img)
cv2.imshow('resized', resized)
cv2.waitKey(0)
cv2.destroyAllWindows()