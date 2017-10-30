import cv2
import numpy as np
import math

lower_black = [0, 0, 0]
upper_black = [0, 0, 10]
def detect_all_squares(image):
	squares = []
	hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
	lower = np.array(lower_black)
	upper = np.array(upper_black)
	mask = cv2.inRange(hsv, lower, upper)
	cv2.waitKey(30)
	img, contours,  h = cv2.findContours(mask, 1, 2)
	for cnt in contours:
		epsilon = 0.15 * cv2.arcLength(cnt, True)
		approx = cv2.approxPolyDP(cnt, epsilon, True)
		area = cv2.contourArea(cnt)
		if area > 400:
			if len(approx) == 4:
				squares = squares + [cnt]
	return squares

def sorted_areas(contours):
	areas = []
	print "total contours: ", len(contours)
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
		print len(required_contour)
	
	return required_contour

def make_grid(pos_sq, imag):
	upper_left = pos_sq[0][0][0]
	bottom_left = pos_sq[2][1][0]
	upper_right = pos_sq[1][3][0]
	wx = (abs(upper_right[0] - upper_left[0]) / 21)
	wy = (abs(bottom_left[1] - upper_left[1]) / 21)
	x_iter = upper_left[0]
	y_iter = upper_left[1]
	print "wx: ", wx, "wy: ", wy
	while x_iter < upper_right[0]:
		cv2.line(imag, (x_iter, y_iter), (x_iter, bottom_left[1]), (0, 255, 255), 1)
		x_iter += wx
	x_iter = upper_left[0]
	while y_iter < bottom_left[1]:
		cv2.line(imag, (x_iter, y_iter), (upper_right[0], y_iter), (0, 255, 255), 1)
		y_iter += wy
	cv2.imshow('grided', imag)
	return imag

def mark_data_squares(image):
	squares = []
	hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
	lower = np.array(lower_black)
	upper = np.array(upper_black)
	mask = cv2.inRange(hsv, lower, upper)
	img, contours,  h = cv2.findContours(mask, 1, 2)
	for cnt in contours:
		epsilon = 0.005 * cv2.arcLength(cnt, True)
		approx = cv2.approxPolyDP(cnt, epsilon, True)
		area = cv2.contourArea(cnt)
		if len(approx) == 4:
			cv2.drawContours(image, [cnt], 0, (0, 0, 255), 1)
			squares = squares + [cnt]
	return squares
'''
def generate_array(img, ul, ur, bl):
	array = [[0 for i in range(21)] for j in range(21)]
	qr = img[ul[1]:bl[1], ul[0]:ur[0]]
	qr = cv2.resize(qr, (21, 21), interpolation = cv2.INTER_AREA)
	blue_channel = qr[:, :, 0]
	for i in range(21):
		for j in range(21):
			if blue_channel[i, j] < 127:
				array[i][j] = 1
			else:
				array[i][j] = 0
	for k in range(21):
		print array[k]
	cv2.imshow('clipped', qr)
'''
def distance(x1, y1, x2, y2):
	return int(np.sqrt((x1 - x2)**2 + (y1 - y2)**2))

def rotate(img, angle):
	height, width = img.shape[:2]
	M = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1)
	rot = cv2.warpAffine(img, M, (height, width))
	return rot

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


def get_small_area_image(img, p1, p2, p0):
	x_coor = []
	y_coor = []
	max_dis_bw = []
	points = [p0, p1, p2]
	for i in range(3):
		for cor in points[i]:
			x_coor.append(cor[0][0])
			y_coor.append(cor[0][1])
	xmin = np.amin(x_coor)
	xmax = np.amax(x_coor)
	ymin = np.amin(y_coor)
	ymax = np.amax(y_coor)
	distance1 = distance(p0[0][0][0], p0[0][0][1], p1[0][0][0], p1[0][0][1])
	max_dis_bw = [p0, p1]
	distance2 = distance(p0[0][0][0], p0[0][0][1], p2[0][0][0], p2[0][0][1])
	distance3 = distance(p1[0][0][0], p1[0][0][1], p2[0][0][0], p2[0][0][1])
	if distance2 > distance1 and distance1 > distance3:
		max_dis_bw = [p0, p2]
	if distance3 > distance1 and distance3 > distance2:
		max_dis_bw = [p1, p2]
	for p in points:
		if p in max_dis_bw:
			continue
		else:
			rotation_angle = angle(p[0][0][0], p[0][0][1], p1[0][0][0], p1[0][0][1])
	y_correction = distance(xmin, ymin, xmax, ymax) - (ymax - ymin)
	x_correction = distance(xmin, ymin, xmax, ymax) - (xmax - xmin)
	small_img = img[ymin - y_correction:ymax + y_correction, xmin - x_correction:xmax + x_correction]
	
	small_img = rotate(small_img, rotation_angle - 90)
	
	cv2.imshow('small_one', small_img)
	return small_img

def generate_array(img, ul, ur, bl):
	array = [[0 for i in range(21)] for j in range(21)]
	qr = img[ul[1]:bl[1], ul[0]:ur[0]]
	qr = cv2.resize(qr, (21, 21), interpolation = cv2.INTER_AREA)
	blue_channel = qr[:, :, 0]
	for i in range(21):
		for j in range(21):
			if blue_channel[i, j] < 127:
				if ((i < 7 or i > 13) and j < 7)  or ((j < 7 or j > 13) and i < 7):
					array[i][j] = 2
				else:
					array[i][j] = 1
			else:
				array[i][j] = 0
	for k in range(21):
		print array[k]
	cv2.imshow('clipped', qr)
	return array

def get_corner_points(p0, p1, p2):
	x_coor = []
	y_coor = []
	points = [p0, p1, p2]
	for i in range(3):
		for cor in points[i]:
			x_coor.append(cor[0][0])
			y_coor.append(cor[0][1])
	xmin = np.amin(x_coor)
	xmax = np.amax(x_coor)
	ymin = np.amin(y_coor)
	ymax = np.amax(y_coor)
	return [xmin, ymin], [xmax, ymin], [xmin, ymax], [xmax, ymax]

def main():
	positioning_squares = []
	font = cv2.FONT_HERSHEY_SIMPLEX
	image = cv2.imread("qr_dev.png")
	
	ret, image[:, :, :] = cv2.threshold(image[:, :, :], 110, 255, cv2.THRESH_BINARY)
	#image = cv2.GaussianBlur(image, (5,5), 0)
	image2 = cv2.imread("qr_dev.png")
	cv2.imshow('thresholding', image)
	square_contours = detect_all_squares(image)
	positioning_squares = sorted_areas(square_contours)
	#gridImage = make_grid(positioning_squares, image2)
	#data_squares = mark_data_squares(gridImage)
	print "contours returned: ", len(positioning_squares)
	i = 0
	#cv2.drawContours(image2, [positioning_squares[3]], 0, (0, 255, 0), 1)
	#cv2.imshow('image', image)
	for coor in positioning_squares:
		print "coordinate",i, "area", cv2.contourArea(coor),": ", coor
		i += 1
		#cv2.drawContours(image2, [coor], 0, (0, 0, 255), 1)
	
	small_image = get_small_area_image(image2, positioning_squares[1], positioning_squares[2], positioning_squares[0])
	small_image = cv2.GaussianBlur(small_image, (3, 3), 0)
	small_image = cv2.cvtColor(small_image, cv2.COLOR_BGR2GRAY)
	ret, small_image = cv2.threshold(small_image, 100, 255, cv2.THRESH_TOZERO)
	small_image = cv2.GaussianBlur(small_image, (3, 3), 0)
	ret, small_image = cv2.threshold(small_image, 100, 255, cv2.THRESH_BINARY)
	small_image = cv2.cvtColor(small_image, cv2.COLOR_GRAY2BGR)
	cv2.imshow('small_with_corner', small_image)
	cv2.waitKey(0)
	cv2.imwrite('small_qr.png', small_image)
	small2 = cv2.imread('small_qr.png')
	print 'blue channel: ', small_image[1, 1, 0]
	#small_image[:, :, :1] = [0, 0, 0]
	#ret, small_image[:, :, :] = cv2.threshold(small_image[:, :, :], 127, 255, cv2.THRESH_BINARY)
	'''
	height, width = small_image.shape[:2]
	print height, width
	for i in range(height):
		for j in range(width):
			if small_image[i, j, 0] + small_image[i, j, 1] + small_image[i, j, 2] > 200:
				for k in range(3):
					small_image[i, j, k] = 255
			else:
				for k in range(3):
					small_image[i, j, k] = 0
	'''
	cv2.imshow('small_thresh', small_image)
	#small_image = cv2.GaussianBlur(small_image, (5,5), 0)
	square_contours = detect_all_squares(small_image)
	positioning_squares = sorted_areas(square_contours)
	#cv2.drawContours(small2, [positioning_squares[3]], 0, (0, 255, 0), 1)
	for coor in positioning_squares:
		#print "coordinate",i, "area", cv2.contourArea(coor),": ", coor
		i += 1
		cv2.drawContours(small2, [coor], 0, (0, 0, 255), 1)
	'''
	upper_left_corner = positioning_squares[1][0][0]
	bottom_left_corner = positioning_squares[2][1][0]
	upper_right_corner = positioning_squares[0][3][0]
	'''
	upper_left_corner, upper_right_corner, bottom_left_corner, bottom_right_corner = get_corner_points(positioning_squares[1], positioning_squares[2], positioning_squares[0])
	qr_array = generate_array(cv2.imread('small_qr.png'), upper_left_corner, upper_right_corner, bottom_left_corner)
	
	
	cv2.circle(small2, (upper_left_corner[0], upper_left_corner[1]), 10, (255, 255, 0), -1)
	cv2.putText(small2, "upper_left_corner", (upper_left_corner[0], upper_left_corner[1]), font, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
	cv2.circle(small2, (bottom_left_corner[0], bottom_left_corner[1]), 10, (255, 255, 0), -1)
	cv2.putText(small2, "bottom_left_corner", (bottom_left_corner[0], bottom_left_corner[1]), font, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
	cv2.circle(small2, (upper_right_corner[0], upper_right_corner[1]), 10, (255, 255, 0), -1)
	cv2.putText(small2, "upper_right_corner", (upper_right_corner[0], upper_right_corner[1]), font, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
	cv2.imshow('final contours', small2)
	
	
	image_to_show = cv2.imread('qr_dev.png')
	height, width = small2.shape[:2]
	for i in range(height):
		for j in range(width):
			image_to_show[i, j] = small2[i, j]
	cv2.imshow('original with final contours', image_to_show)
	
	cv2.waitKey(0)
	cv2.destroyAllWindows()



if __name__ == "__main__":
	main()
