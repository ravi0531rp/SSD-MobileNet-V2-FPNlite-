import numpy as np
import cv2
import json
import math
import imghdr
import base64
from PIL import Image, ImageDraw, ImageColor
from io import BytesIO
import requests
import os

class Helpers:

	def __init__(self):
		self.BlurrKernel = (11,11)
		self.BigBlurrKernel = (21,21)
		self.onesK = np.ones(self.BlurrKernel,dtype=np.uint8)
		self.BigonesK = np.ones(self.BlurrKernel,dtype=np.uint8)
		pass

	def get_img_from_array(self,arry):
		pil_img = Image.fromarray(arry)
		buff = BytesIO()
		pil_img.save(buff, format="JPEG")
		img = base64.b64encode(buff.getvalue()).decode("utf-8")
		return img

	def read_image_from_url(self,url):
		open_cv_image, pilimage = None, None
		try:
		    response = requests.get(url)
		    pilimage = Image.open(BytesIO(response.content)).convert('RGB')
		    open_cv_image = np.ascontiguousarray(pilimage)
		except:
		    return None, None

		return open_cv_image, pilimage

	def validate_image(self,stream):
		header = stream.read(512)  # 512 bytes should be enough for a header check
		stream.seek(0)  # reset stream pointer
		format = imghdr.what(None, header)
		if not format:
		    return None
		return '.' + (format if format != 'jpeg' else 'jpg')

	def getNextImage(self,images,cnt):
		if cnt<0:
			return None,"Below Zero Index"
		elif cnt < len(images):
			imagePath = images[cnt]
			original_img = cv2.imread(imagePath)
			return original_img,imagePath
		else:
			print("Counter exceeded number of images")
			return None,"Above size Index"

	def make_dir_if_not(self,dir_name):
		if not os.path.exists(dir_name):
			os.makedirs(dir_name)

	def get_angle(self,p1, p2):
		dx = p2[0] - p1[0]
		dy = p2[1] - p1[1]
		return math.atan2(dx, dy) / math.pi * 180

	def dist(self,a, b):
		return np.sqrt(np.power(b[0] - a[0], 2) + np.power(b[1] - a[1], 2))

	def is_point_inside_image(self,img, pt):
		width, height = img.shape[:2]
		x = pt[0]
		y = pt[1]
		if 0 < x < height and 0 < y < width:
			return True
		else:
			return False

	#  new format of lines for better calculation
	def get_line(self,point1, point2):
		A = (point1[1] - point2[1])
		B = (point2[0] - point1[0])
		C = (point1[0]*point2[1] - point2[0]*point1[1])
		return A, B, -C

	# finds intersection point of two lines
	def get_intersection(self,Line1, Line2):
		D  = Line1[0] * Line2[1] - Line1[1] * Line2[0]
		Dx = Line1[2] * Line2[1] - Line1[1] * Line2[2]
		Dy = Line1[0] * Line2[2] - Line1[2] * Line2[0]
		if D != 0:
			x = Dx / D
			y = Dy / D
			return (True, (x,y))
		else:
			return (False, (0,0))


	def find_intersection_points(self,lines):
		intersection_points=[]
		new_lines=[]
		for line in lines:
			new_lines.append(self.get_line(line[0], line[1]))
		for line_index1 in range(len(new_lines)):
			for line_index2 in range(len(new_lines)):
				if(line_index1==line_index2):
					continue
				else:
					if_intersect, intersect_point = self.get_intersection(new_lines[line_index1], new_lines[line_index2])
					if (if_intersect):
						intersection_points.append(intersect_point)
		return intersection_points

	def find_intersection_point(self,line1, line2):
		line1 = self.get_line(line1[0], line1[1])
		line2 = self.get_line(line2[0], line2[1])
		if_intersect, intersection_point = self.get_intersection(line1, line2)
		if if_intersect:
			x_coordinate = int(intersection_point[0])
			y_coordinate = int(intersection_point[1])
			intersection_point = (x_coordinate, y_coordinate)
			return intersection_point

	# lines in array of line. A Line is of format ([start_x,start_y], [end_x,end_y])
	def avg_intersection_point(self,lines):
		intersection_points = self.find_intersection_points(lines)
		if(len(intersection_points)==0):
			print("All lines are Parallel!!!")
			return
		x_sum=0
		y_sum=0
		for point in intersection_points:
			x_sum+= point[0]
			y_sum+= point[1]
		x_avg= x_sum/len(intersection_points)
		y_avg= y_sum/len(intersection_points)
		return (x_avg, y_avg)


	def draw_bounding_box(self,img,anomaly_points):
		image = np.zeros((img.shape[0],img.shape[1],3), np.uint8)
		for i in anomaly_points:
			c,d = i
			image = cv2.circle(image, (c,d), 10, (255,255,255), -1)
		gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		gray = cv2.morphologyEx(gray, cv2.MORPH_ERODE, self.onesK,iterations = 2)
		gray = cv2.morphologyEx(gray, cv2.MORPH_DILATE, self.BigonesK,iterations = 2)
		blurred = cv2.GaussianBlur(gray, (3, 3), 0)
		canny = cv2.Canny(blurred, 120, 255, 1)

		# Find contours
		cnts = cv2.findContours(canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
		cnts = cnts[0] if len(cnts) == 2 else cnts[1]

		# Iterate thorugh contours and draw rectangles around contours
		for c in cnts:
			x,y,w,h = cv2.boundingRect(c)
			cv2.rectangle(img, (x, y), (x + w, y + h), (36,255,12), 2)

		return img
	def get_points(self):
		point1 = ()
		point2 = ()
		rect_point=[]

		with open('points.txt', 'r') as f:
			data = json.load(f)
			first = data['point1']
			second = data['point2']
			point1 = (first[0], first[1])
			point2 = (second[0], second[1])
			rect_point.append(point1)
			rect_point.append(point2)
		return rect_point


	def makeFolderStructure(self, rootPath):

		pathList = {"upload_clean_o": rootPath+"/CleanContainers/original/",
					"upload_clean_d": rootPath+"/CleanContainers/detections/",
					"upload_clean_p": rootPath+"/CleanContainers/perspective/",
					"upload_ordinary_o": rootPath+"/OrdinaryCleaning/original/",
					"upload_ordinary_d": rootPath+"/OrdinaryCleaning/detections/",
					"upload_ordinary_p": rootPath+"/OrdinaryCleaning/perspective/",
					"upload_excessive_o": rootPath+"/ExcessiveCleaning/original/",
					"upload_excessive_d": rootPath+"/ExcessiveCleaning/detections/",
					"upload_excessive_p": rootPath+"/ExcessiveCleaning/perspective/",

					"validated_excessive_o": rootPath+"/Validated/ExcessiveCleaning/original/",
					"validated_excessive_p": rootPath+"/Validated/ExcessiveCleaning/perspective/",
					"validated_ordinary_o": rootPath+"/Validated/OrdinaryCleaning/original/",
					"validated_ordinary_p": rootPath+"/Validated/OrdinaryCleaning/perspective/",
					"validated_clean_o": rootPath+"/Validated/CleanContainers/original/",
					"validated_clean_p": rootPath+"/Validated/CleanContainers/perspective/"}

		for key,val in pathList.items():
			self.make_dir_if_not(val) 


		val_root_path = {"validated_root_path": rootPath+"/Validated"}
		pathList.update(val_root_path)


		return pathList

	# resizes an image with an aspect ratio
	def resize_with_aspect_ratio(self, image_low_res, width=None, height=None, inter=cv2.INTER_AREA):
		dim = None
		(h, w) = image_low_res.shape[:2]
		if width is None and height is None:
			return image_low_res
		if width is None:
			r = height / float(h)
			dim = (int(w * r), height)
		else:
			r = width / float(w)
			dim = (width, int(h * r))

		return cv2.resize(image_low_res, dim, interpolation=inter)

	# adds a white border to azure ocr image before being sent for reading
	def add_border_ocr(self, image):
		h, w, c = image.shape
		bordersize = int(w / 2)
		border = cv2.copyMakeBorder(
			image,
			top=bordersize,
			bottom=bordersize,
			left=bordersize,
			right=bordersize,
			borderType=cv2.BORDER_CONSTANT,
			value=[255, 255, 255]
		)
		return border

	# Create new image(numpy array) filled with certain color in RGB
	def create_blank(self, width, height, rgb_color):
		# Create black blank image
		image = np.zeros((height, width, 3), np.uint8)
		# Since OpenCV uses BGR, convert the color first
		color = tuple(reversed(rgb_color))
		# Fill image with color
		image[:] = color

		return image

	# rotates image
	def rotate_image(self, image, angle):
		image_center = tuple(np.array(image.shape[1::-1]) / 2)
		rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
		result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR,
								borderMode=cv2.BORDER_CONSTANT,
								borderValue=(0, 0, 0))
		return result

	# get a rotation matrix for a particular angle of rotation
	def getRotationMatrix2D(self, image, angle):
		image_center = tuple(np.array(image.shape[1::-1]) / 2)
		rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
		return rot_mat

	# rotates image without cropping
	def rotate_image_without_cut(self, image, point_top, point_down):
		h, w = image.shape[:2]
		x1, y1 = point_top[0], point_top[1]
		x2, y2 = point_down[0], point_down[1]
		angle_with_x_axis = math.degrees(math.atan((y2 - y1) / (x2 - x1)))

		# get angle formed with with y axis
		if angle_with_x_axis >= 0:
			angle_with_y_axis = 90 - angle_with_x_axis
		else:
			angle_with_y_axis = -1 * (90 + angle_with_x_axis)
		# get rotation matrix
		rot_mat = self.getRotationMatrix2D(image, -1 * angle_with_y_axis)

		# form a border so rotated image is not cut
		# get point1 location after rotation
		top_left = np.array([(0, 0)], np.int32)
		top_left = np.reshape(top_left, (1, 1, 2))
		top_left = cv2.transform(top_left, rot_mat)
		# extract values
		top_left_x = top_left[0][0][0]
		top_left_y = top_left[0][0][1]
		# get border to be added
		w_added = abs(top_left_x)
		h_added = abs(top_left_y)

		img_border = self.create_blank(w + 2 * w_added, h + 2 * h_added, (0, 0, 0))
		img_border[h_added:h + h_added, w_added: w + w_added] = image

		rot_img = self.rotate_image(img_border, -1 * angle_with_y_axis)
		return rot_img

	

		


