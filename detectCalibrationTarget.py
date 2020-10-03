import numpy as np
import cv2

actualBGRvalues = [[112.96002591, 106.07069492,  70.03201629],
                   [110.53755267,  97.05532149, 108.05779447],
                   [50.92223584, 75.52898482, 81.38297672],
                   [111.013,   85.7357, 68.0361],
                   [109.97234481, 135.38982054, 183.84907326],
                   [39.59304384, 48.29084661, 96.05165053],
                   [ 50.22377764, 118.69141814, 168.32501522],
                   [ 60.35108025, 139.88078704, 147.5392554 ],
                    [53.92069001, 37.96049811, 79.24089007],
                    [ 49.70998446,  56.75806138, 158.52447552],
                    [99.92403199, 58.46369949, 35.16172138],
                    [ 39.36402062,  85.04123711, 172.14969072],
                    [94.55254717, 51.02471698, 16.27783019],
                    [ 56.15056977,  51.61591631, 145.76246964],
                    [ 42.6925905,  149.95145173, 179.06108597],
                    [ 25.14450629,  24.37250052, 143.34147598],
                    [69.39310972, 97.68862396, 56.18771469],
                    [79.29449838, 41.11427994, 20.88167476],
                    [25.59975181, 18.5083047,  24.06347843],
                    [51.23871025, 51.29669547, 67.20806983],
                    [ 83.2449113,   89.94668534, 110.92343604],
                    [120.0058463,  134.19717115, 153.25421971],
                    [149.50114242, 172.26408987, 189.9480198 ],
                    [166.76805152, 194.53610305, 206.56068578]]

collectedValues = []


img = cv2.imread(r'C:\Users\artsy\OneDrive\Desktop\RubiksCubeProject\bestFit\lindImg.jpg', cv2.IMREAD_COLOR)

scale_percent = 20
width = int(img.shape[1] * scale_percent / 100)
height = int(img.shape[0] * scale_percent / 100)
dim = (width, height)
img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)

r = cv2.selectROI(img, fromCenter=False)

crop = img[int(r[1]):int(r[1] + r[3]), int(r[0]):int(r[0] + r[2])]

h, w, c = crop.shape

print('height = ' + str(h) + 'width = ' + str(width))

constantW = 0.03125
constantH = 0.048
constantS = 0.212
constantB = 0.15

smallWidth = int(constantW * w)
smallHeight = int(constantH * h)
boxConstant = int(constantB * w)

# change to circles
#cv2.rectangle(crop, (smallWidth ,smallHeight) , (w - smallWidth , h - smallHeight), (0,255,0) , 5)
startX = 2 * smallWidth
startY = smallHeight + 10
for y in range (4):
    for x in range (6):
        cv2.rectangle(crop, (startX, startY) , (startX - smallWidth + boxConstant, startY - smallHeight + boxConstant), (255,255,0) , 2)
        roi = crop[startY: startY - smallHeight + boxConstant, startX:startX - smallWidth + boxConstant]
        # change from average to dominant color? more accurate
        avg_color_per_row = np.average(roi, axis=0)
        avg_color = np.average(avg_color_per_row, axis=0)
        #print(avg_color)
        collectedValues.append(avg_color)
        startX += 5*smallWidth
    startY += int(5 * smallHeight)
    startX = 2 * smallWidth

print(collectedValues)
cv2.imshow("cropped", crop)
cv2.waitKey(0)
cv2.destroyAllWindows()
