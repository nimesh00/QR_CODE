import cv2
import numpy as np
import math

cam = cv2.VideoCapture(0)

lower_black = [0, 0, 0]
upper_black = [180, 255, 80]
def detect_all_squares(image):
	squares = []
	#clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
	hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
	#hsv[:, :, 2] = clahe.apply(hsv[:, :, 2])
	#hsv[:, :, 2] = cv2.equalizeHist(hsv[:, :, 2])
	lower = np.array(lower_black)
	upper = np.array(upper_black)
	mask = cv2.inRange(hsv, lower, upper)

	cv2.imshow('mask', mask)
	img, contours,  h = cv2.findContours(mask, 1, 2)
	for cnt in contours:
		epsilon = 0.15 * cv2.arcLength(cnt, True)
		approx = cv2.approxPolyDP(cnt, epsilon, True)
		area = cv2.contourArea(cnt)
		if area > 200:
			if len(approx) == 4:
				cv2.drawContours(image, [cnt], 0, (255, 0, 0), 1)
				squares = squares + [cnt]
	image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
	cv2.imshow('img', image)
	return squares

def sorted_areas(contours):
	areas = []
	required_areas = []
	required_contour = []
	no_of_contours = 0
	for cnt in contours:
		no_of_contours += 1
		if no_of_contours == 1:
			largest_area = cv2.contourArea(cnt)
			requried_contour = required_contour + [cnt]
		area = cv2.contourArea(cnt)
		if area >= largest_area:
			largest_area = area
			required_areas = [area] + required_areas
			required_contour = [cnt] + required_contour

		else:
			no_of_contour_here = 0
			for cnt1 in required_contour:
				no_of_contour_here += 1
				if no_of_contour_here == len(required_contour):
					required_contour = required_contour + [cnt]
					break
				elif area > cv2.contourArea(cnt1):
					required_contour[no_of_contour_here:] = [cnt] + required_contour[no_of_contour_here:]
					break
				elif area <= cv2.contourArea(cnt1):
					required_contour[: no_of_contour_here + 1] = required_contour[: no_of_contour_here + 1] + [cnt]
					break
		required_contour = required_contour[:9]
	return required_contour

def angle(xp,yp,xb,yb):
    dx=float(xp-xb)
    dy=float(yp-yb)
    if dx == 0:
        return 90
    global mtan
    if(dx > 0 and dy > 0):
        mtan=math.degrees(math.atan(float(dy/dx)))
    elif(dy>0 and dx<0):
        mtan=180 + math.degrees(math.atan(float(dy/dx)))
    elif(dy <0 and dx<0):
        mtan=180+math.degrees(math.atan(float(dy/dx)))
    else:
        mtan=360+math.degrees(math.atan(float(dy/dx)))
    print mtan
    return mtan

def positioned(pt1, pt2, pt3):
	angle1 = angle(pt2[0], pt2[1], pt1[0], pt1[1])
	angle2 = angle(pt1[0], pt1[1], pt3[0], pt3[1])
	print angle1, angle2
	if angle1 - angle2 != 90:
		return False
	return True

def check_points(pos_sq):
	swapper = []
	angles = []
	ulc = pos_sq[1][0][0]
	blc = pos_sq[2][1][0]
	urc = pos_sq[0][3][0]
	if ulc[0] < urc[0] and ulc[1] > blc[1]:
		swapper = pos_sq[1]
		pos_sq[1] = pos_sq[2]
		pos_sq[2] = swapper
	x1 = [ulc[0], urc[0], blc[0]]
	y1 = [ulc[1], urc[1], blc[1]]
	angles = angle(x1[:], y1[:], x1[:], y1[:])
	sort(angles)
	
	return pos_sq


def main():
	positioning_squares = []
	swapper = []
	font = cv2.FONT_HERSHEY_SIMPLEX
	while True:
		cv2.waitKey(30)
		ret, image = cam.read()
		cv2.imshow('blue', image[:, :, 0])
		cv2.imshow('green', image[:, :, 1])
		cv2.imshow('red', image[:, :, 2])
		if not ret:
			image = cv2.imread("qr.png")
		image = cv2.GaussianBlur(image, (3,3), 0)
		'''
		for c in range(0,3):
			image[:, :, c] = cv2.equalizeHist(image[:, :, c])
			'''
		square_contours = detect_all_squares(image)
		positioning_squares = sorted_areas(square_contours)
		for coor in positioning_squares :
			cv2.drawContours(image, [coor], 0, (0, 0, 255), 1)
		cv2.imshow('image', image);
		print "trying to get the corner squares......"
		if (len(positioning_squares) > 2):
			positioning_squares = check_points(positioning_squares)
			upper_left_corner = positioning_squares[1][0][0]
			bottom_left_corner = positioning_squares[2][1][0]
			upper_right_corner = positioning_squares[0][3][0]
		else:
			continue
			
		if not positioned(upper_left_corner, bottom_left_corner, upper_right_corner):
			cv2.circle(image, (upper_left_corner[0], upper_left_corner[1]), 10, (255, 255, 0), -1)
			cv2.putText( image, "upper_left_corner", (upper_left_corner[0], upper_left_corner[1]), font, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
			cv2.circle(image, (bottom_left_corner[0], bottom_left_corner[1]), 10, (255, 255, 0), -1)
			cv2.putText( image, "bottom_left_corner", (bottom_left_corner[0], bottom_left_corner[1]), font, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
			cv2.circle(image, (upper_right_corner[0], upper_right_corner[1]), 10, (255, 255, 0), -1)
			cv2.putText( image, "upper_right_corner", (upper_right_corner[0], upper_right_corner[1]), font, 0.8, (0, 255, 0), 2, cv2.LINE_AA)

			cv2.imshow('markers', image)
			continue
		
		print positioning_squares
		cv2.imwrite('snap_img.png', image)
		break
		k = cv2.waitKey(30)
		if k == 27:
			break
	cv2.destroyAllWindows()
	cam.release()


if __name__ == "__main__":
	main()
