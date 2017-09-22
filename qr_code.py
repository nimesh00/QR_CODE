import cv2
import numpy as np

areas = []

lower_black = [0, 0, 0]
upper_black = [5, 0, 0]
img = cv2.imread('qr.png')
cv2.imshow('starting', img)
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
lower = np.array(lower_black)
upper = np.array(upper_black)
mask = cv2.inRange(hsv, lower, upper)
res = cv2.bitwise_and(img, img, mask = mask)
gray = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(gray, 127, 255, 0)
cv2.imshow('res', res)
cv2.imshow('mask', mask)
cv2.imshow('thresh', thresh)
image, contours, h = cv2.findContours(mask, 1, 2)
required_contours = []
for cnt in contours:
	epsilon = 0.15 * cv2.arcLength(cnt, True)
	area = cv2.contourArea(cnt)
	if area > 400:
		areas = areas + [cv2.contourArea(cnt)]
		np.sort(areas)
		if len(areas) < 2:
			continue
		areas.reverse()
		approx = cv2.approxPolyDP(cnt, epsilon, True)
		
				

for cnt in required_contours:
    cv2.drawContours(img, [cnt], 0, (0, 0, 255), 1)
    print 'contour', cnt[0]
cv2.imshow('img', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
