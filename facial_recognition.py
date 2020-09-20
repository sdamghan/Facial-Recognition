######################################################
##################### IMPORTS ########################
######################################################
import cv2
import argparse

import pickle

import face_recognition

######################################################
#################### FUNCTIONS #######################
######################################################
def args_function():
	# construct the argument parser and parse the arguments
	ap = argparse.ArgumentParser()
	ap.add_argument("-e", "--embeddings", required=True,
		help="path to serialized db of facial encodings")
	ap.add_argument("-i", "--image", required=True,
		help="path to input image")
	ap.add_argument("-d", "--detection-method", type=str, default="hog",
		help="face detection model to use: either `hog` or `cnn`")
	return vars(ap.parse_args())

######################################################
####################### MAIN #########################
######################################################
args = args_function()

# load faces and embeddings
data = pickle.loads(open(args["embeddings"], "rb").read())

# Change OpenCV ordering to dlib ordering
img = cv2.imread(args["image"])
rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# detect a face in the each input image the return the coordinate of its central point
central_points = face_recognition.face_locations(rgb, model=args["detection_method"])
# compute the facial embedding for the face
encode_values = face_recognition.face_encodings(rgb, central_points)

# the list of recognized names in input image
names = []
# loop over the facial embeddings
for encode_value in encode_values:
	# match recognized face with our db
	name = "Unknown"
	matches = face_recognition.compare_faces(data["encodings"], encode_value)
	# create an array to store indexes of matched faces
	matched_indexes=[]
	# go through matched faces
	for (i, match) in enumerate(matches):
		if match == True:
			matched_indexes.append(i)

	if True in matches:
		# create a dict to store the occurence number of each face matching
		counts = {}
		for i in matched_indexes:
			name = data["names"][i]
			counts[name] = counts.get(name, 0) + 1

		# detect the recognized face with largest number of occurence and update the list
		name = max(counts, key=counts.get)
	names.append(name)

# loop over the recognized faces
color = (0, 255, 0)
for ((top, right, bottom, left), name) in zip(central_points, names):
	# draw the predicted face name on the image
	cv2.rectangle(img, (left, top), (right, bottom), color, 4)
	text_pos = top - 15 if top - 15 > 15 else top + 15
	cv2.putText(img, name, (left, text_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)

# show the output image
cv2.imshow("Image", img)
cv2.waitKey(0)
