#!/usr/bin/env python
import os
import cv2
import argparse
import numpy as np
import mediapipe as mp
import sys
import os
import itertools
import random
from bh.bansi import *

parser = argparse.ArgumentParser(description='Create face mask for input image(s).')
parser.add_argument('images', metavar='src', nargs='+', help='Input image file(s)')
parser.add_argument('-o', '--output', metavar='output', help=f'Output file or directory')
parser.add_argument('-m', '--maskonly', action='store_true', help='Write grayscale mask instead of masked image')
parser.add_argument('-u', '--upscale', type=int, help='Upscale this many times', default=1)
parser.add_argument('-d', '--dilate', type=int, help='Dilate mask # pixels (automatically * upscale for you)')
parser.add_argument('-M', '--nomask', action='store_true', help="Don't mask colored image either (use with -l -t, etc.)")
parser.add_argument('-i', '--invert', action='store_true', help='Invert the output')
parser.add_argument('-l', '--landmarks', action='store_true', help='Generate image with landmark indices')
parser.add_argument('-L', '--alllandmarks', action='store_true', help='Generate image with ALL landmark indices (otherwise obey -f nose, for instance)')
parser.add_argument('-t', '--tesselation', action='store_true',  help='Draw tesselation')
parser.add_argument('-c', '--contours', action='store_true',  help='Draw contours')
parser.add_argument('-f', '--feature', metavar='feature', default='face', choices=['face', 'nose'], help='Facial feature to isolate')
parser.add_argument('-K', '--keeperrors', action='store_true', help='Copy detection failures to output for review too')
parser.add_argument("-v", "--verbose", action="count",
		                    default=0, help="Increase output verbosity")
args = parser.parse_args()

# Load the FaceMesh model
mp_face_mesh = mp.solutions.face_mesh
#face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1)
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1,
                                  refine_landmarks=True,
                                  )

# For drawing contours and stuff
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

output_file = None

# Outlines are drawn clockwise from the viewer's perspective,
# and generally beginning at the left-most point.
landmark_indices = {
    'nose_bridge': list(range(27, 31)),
    'perinasal_outline': [100, 47, 114, 188, 122, 6, 351, 412, 
                 343, 277, 355, 371, 423,
                 391, 393, 164, 167, 165, 206, 205, 101,
                 # Added for bridge
                 232, 233, 245, 193, 168, 417, 465, 357
                 ],
    'nostril_r': list(range(98, 100)),
    'nostril_l': list(range(105, 107)),

    'eye_r_top': [33, 246, 161, 160, 159, 158, 157, 173, 133],
    'eye_r_bottom': [33, 155, 154, 153, 145, 144, 163, 7],
    'eye_r': [33, 246, 161, 160, 159, 158, 157, 173, 133] + \
             [33, 155, 154, 153, 145, 144, 163, 7][1::][::-1],
    'eye_l_top': [362, 398, 384, 385, 386, 387, 388, 466, 263],
    'eye_l_bottom': [362, 382, 381, 380, 374, 373, 390, 249, 263],
    'eye_l': [362, 398, 384, 385, 386, 387, 388, 466, 263] + \
             [362, 382, 381, 380, 374, 373, 390, 249, 263][1::][::-1],

    'upper_lip': list(range(312, 327)),
    'lower_lip': list(range(375, 390)),

    'eyebrow_l_midline': [285, 295, 282, 283, 276, 383],
    'eyebrow_l_top': [285, 336, 296, 334, 293, 300],
    'eyebrow_l_outline': [285, 336, 296, 334, 293, 300] + \
                         [285, 336, 296, 334, 293, 300][1::][::-1],
    'eyebrow_r_midline': [156, 46, 53, 52, 65, 55],
    'eyebrow_r_top': [156, 70, 63, 105, 66, 107],
    'eyebrow_r_outline': [156, 46, 53, 52, 65, 55] + \
                         [70, 63, 105, 66, 107][1::][::-1],
}

feature_sets = {
	'face': [landmark_indices.keys()],
	'nose': ['perinasal_outline'],
	# 'nose': ['nose_bridge', 'perinasal_outline', 'nostril_r', 'nostril_l'],
	# 'nose': ['perinasal_outline'],
}

def get_all_features():
	return list(range(0,468))

def get_generalized_feature_landmarks(genfeat):
# Get the list of feature names for the given generalized feature
	feature_names = feature_sets[genfeat]
# Get the list of landmark indices for each feature
	feature_landmarks = [landmark_indices[featname]
		for featname in feature_names]
# Flatten the list of lists into a single list
	flattened_landmarks = list(itertools.chain.from_iterable(
		feature_landmarks))
	return flattened_landmarks

def pv(vlevel, *aa, **kwargs):
	if vlevel >= args.verbose: print(*aa, **kwargs)

def pe(rc, *args, **kwargs):  # Print error (and leave)
	print(*args, file=sys.stderr, **kwargs)
	if rc is not None: sys.exit(rc)

# Define output directory or file
if not args.output:
	pe(1, 'Error: -o Output file/dir is required')
elif len(args.images) > 1:
	if not os.path.isdir(args.output):
		pe(1, 'Error: -o Output must be a directory when processing multiple input files.')
elif len(args.images) < 1:
	pe(1, "No input files specified?")
elif len(args.images) == 1 and not os.path.isdir(args.output):
	output_file = args.output
# else:
# 	pe(1, "Error: Bug. I don't know why we're here. We were handling args.images.")

def rand_rgb():
	return (random.randint(0,256),
	        random.randint(0,256),
	        random.randint(0,256))

def dilate(img, amt):
	kernel = np.ones((amt, amt), np.uint8)
	return cv2.dilate(img, kernel, iterations=1)

# Process each input image

for image_path in args.images:
	fnplain = os.path.split(image_path)[1]
	if output_file is None:
		output_file = os.path.join(args.output, fnplain)
	# Load the input image
	image = cv2.imread(image_path)

	# Extract the facial landmarks for the selected feature
	results = face_mesh.process(image)
	if results.multi_face_landmarks is None:
		pe(None, f"{bred}Undetected features in image: {image_path}{rst}")
		if not args.keeperrors:
			pv(0, f'  {bred}Skipping (no -K specified) {output_file}{rst}')
		else:
			pv(0, f'  {bred}Writing anyway (to {output_file}){rst}')
			cv2.imwrite(output_file, image)
		output_file = None
		continue
	all_indices = get_all_features()

	if args.feature == 'face':
		feat_indices = all_indices
	elif args.feature == 'nose':
		try:
			pv(1, f'Landmarks cnt {len(results.multi_face_landmarks[0].landmark)}')
		except:
			import ipdb; ipdb.set_trace()
		feat_indices = get_generalized_feature_landmarks('nose')
		if not args.alllandmarks:
			all_indices = feat_indices
	else:
		pe(1, "Unrecognized feature {args.feature}")
	feat_landmarks = [results.multi_face_landmarks[0].landmark[i]
								for i in feat_indices]
	all_landmarks = [results.multi_face_landmarks[0].landmark[i]
								for i in all_indices]

	if args.upscale < 1:
		pe(1, f"Downscaling not supported (upscale value given as {args.upscale})")
	if args.upscale > 1:
		image = cv2.resize(image, (0,0), fx=args.upscale, fy=args.upscale)

	######### Mask
	# Create a binary mask that isolates just the feature
	mask = np.zeros(image.shape[:2], dtype=np.uint8)
	points = np.array([[int(l.x * image.shape[1]), int(l.y * image.shape[0])]
	                      for l in feat_landmarks])
	hull = cv2.convexHull(points)
	cv2.drawContours(mask, [hull], -1, 255, -1)
	if args.dilate:
		mask = dilate(mask, args.dilate * args.upscale)

	if args.invert:
		pv(1, "Using inverted mask")
		mask = cv2.bitwise_not(mask)
	if args.maskonly:
		pv(1, "Writing mask only")
		image = mask
	elif not args.nomask: # Normal mode we mask rgb image
		pv(1, "Writing masked image")
		alpha = .5
		alpha = 1.0
		mask = (mask * alpha).astype(np.uint8)
		image = cv2.bitwise_and(image, image, mask=mask)

    # Add text labels to the image
	if args.tesselation:
		pv(1, "Drawing tessas")
		for face_landmarks in results.multi_face_landmarks:
			mp_drawing.draw_landmarks(
				image=image,
				landmark_list=face_landmarks,
				connections=mp_face_mesh.FACEMESH_TESSELATION,
				landmark_drawing_spec=None,
				connection_drawing_spec=mp_drawing_styles
				.get_default_face_mesh_tesselation_style())
	if args.contours:
		pv(1, "Drawing contours")
		for face_landmarks in results.multi_face_landmarks:
			mp_drawing.draw_landmarks(
				image=image,
				landmark_list=face_landmarks,
				connections=mp_face_mesh.FACEMESH_CONTOURS,
				landmark_drawing_spec=None,
				connection_drawing_spec=mp_drawing_styles
				.get_default_face_mesh_contours_style())

	if args.landmarks:
		pv(1, "Generating landmarks labels")
		for i, landmark in enumerate(all_landmarks):
			x, y = int(landmark.x * image.shape[1]), \
			       int(landmark.y * image.shape[0])
			if i in feat_indices:
				fg=(150,230,255)
				bg=(0,0,0)
			else:
				fg = rand_rgb()
				bg = (255,255,255)
			cv2.putText(image, str(i), (x+1, y+1),
				cv2.FONT_HERSHEY_SIMPLEX, 1,
				bg, 3, cv2.LINE_AA)
			cv2.putText(image, str(i), (x, y),
				cv2.FONT_HERSHEY_SIMPLEX, 1,
				fg, 1, cv2.LINE_AA)

	# Write the mask or masked image to the output file
	pv(0, f'Writing to {output_file}')
	cv2.imwrite(output_file, image)
	output_file = None
