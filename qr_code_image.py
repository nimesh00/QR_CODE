import cv2
import numpy as np

lower_black = [0, 0, 0]
upper_black = [0, 0, 10]
def detect_all_squares(image):
	squares = []
	hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
	lower = np.array(lower_black)
	upper = np.array(upper_black)
	mask = cv2.inRange(hsv, lower, upper)
	cv2.imshow('mask', mask)
	img, contours,  h = cv2.findContours(mask, 1, 2)
	for cnt in contours:
		epsilon = 0.15 * cv2.arcLength(cnt, True)
		approx = cv2.approxPolyDP(cnt, epsilon, True)
		area = cv2.contourArea(cnt)
		if area > 400:
			if len(approx) == 4:
				cv2.drawContours(image, [cnt], 0, (255, 0, 0), 1)
				squares = squares + [cnt]
	cv2.imshow('img', image)
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
				
		print len(required_contour)
	
	return required_contour


def main():
	positioning_squares = []
	image = cv2.imread("qr.png")
	square_contours = detect_all_squares(image)
	positioning_squares = sorted_areas(square_contours)
	print "contours returned: ", len(positioning_squares)
	i = 0
	cv2.drawContours(image, [positioning_squares[3]], 0, (0, 255, 0), 1)
	cv2.imshow('image', image)
	for coor in positioning_squares:
		print "coordinate",i, "area", cv2.contourArea(coor),": ", coor
		i += 1
		cv2.drawContours(image, [coor], 0, (0, 0, 255), 1)
	cv2.imshow('image', image);
	k = cv2.waitKey(0)
	cv2.destroyAllWindows()



if __name__ == "__main__":
	main()
