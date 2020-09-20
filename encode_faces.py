######################################################
##################### IMPORTS ########################
######################################################
import cv2
import os
import argparse

import pickle
from imutils import paths

import face_recognition

######################################################
#################### FUNCTIONS #######################
######################################################
def args_function():
	# construct the argument parser and parse the arguments
	ap = argparse.ArgumentParser()
	ap.add_argument("-i", "--dataset", required=True,
		help="path to input directory of faces + images")
	ap.add_argument("-e", "--embeddings", required=True,
		help="path to serialized db of facial encodings")
	ap.add_argument("-d", "--detection-method", type=str, default="hog",
		help="face detection model to use: either `hog` or `cnn`")
	return vars(ap.parse_args())

######################################################
####################### MAIN #########################
######################################################
args = args_function()
# select the paths to the input images
img_paths = list(paths.list_images(args["dataset"]))

#list of encodings and names
encodings = []
names = []
for (i, img_path) in enumerate(img_paths):
	print("processing image " + str(i + 1) + " from " + str(len(img_paths)))
	name = img_path.split(os.path.sep)[-2]
	# Change OpenCV ordering to dlib ordering
	img = cv2.imread(img_path)
	rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
	# detect a face in the each input image the return the coordinate of its central point
	central_points = face_recognition.face_locations(rgb, model=args["detection_method"])
	# compute the facial embedding for the face
	encode_values = face_recognition.face_encodings(rgb, central_points)

	# loop over the encodings
	for encode_value in encode_values:
		# store each encode_value and corresponding name
		encodings.append(encode_value)
		names.append(name)

# write down the embeddings values on disk
data = {"encodings": encodings, "names": names}
f = open(args["embeddings"], "wb")
f.write(pickle.dumps(data))
f.close()
