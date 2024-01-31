#!/usr/bin/env python

# Requirements:
#   Run:
#     pip install imutils getch

# import the necessary packages
from imutils import paths
import argparse
import cv2
import matplotlib
matplotlib.use('GTK3Agg')
import matplotlib.pyplot as plt
import getch
import imageio
import ipdb as pdb
import os
import numpy as np
import math
import sys
import shutil

opt_blur=False
opt_preview=False

image_list=None
blur_log_fn="blur.log"
blur_log_bak_fn="blur.log.bak"
# shutil.copy()
print(f"We'd like to create a backup log, but we're not: {blur_log_bak_fn}", file=sys.stderr)
blurf = open(blur_log_fn, "w")

blurar = []

# construct the argument parse and parse the arguments
argpa = argparse.ArgumentParser()
argpa.add_argument("-f", "--file", required=True, help="video filename, OR a directory of IMAGES")
argpa.add_argument("-v", "--verbose", action="count", default=1,
                   help="increase output verbosity")
argpa.add_argument("-s", "--show", action='store_true', help="show each image")
argpa.add_argument("-g", "--grid", action='store_true', help="show grid at end (not available without bh.imgutils)")
argpa.add_argument("-c", "--crop", type=float, default=100, help="crop to center percentage")
argpa.add_argument("-t", "--threshold", type=float, default=10.0,
	help="focus measures that fall below this value will be considered 'blurry'")
argpa.add_argument("-x", "--max", type=int, default=-1, help="max frames to process")
args = argpa.parse_args()
imgidx = 0 # Starting up! (this is reset each video for frames in videos)
quit = False
pyplot_frame = None
if args.grid: # Preload so as not to waste time
	import bh.imgutils as imgutils

# import the necessary packages

def det_blur_variance_of_laplacian(image):
	# compute the Laplacian of the image and then return the focus
	# measure, which is simply the variance of the Laplacian
	return cv2.Laplacian(image, cv2.CV_64F).var()

def det_blur_fft(image, size=60, vis=True):
	# https://pyimagesearch.com/2020/06/15/opencv-fast-fourier-transform-fft-for-blur-detection-in-images-and-video-streams/
	(h,w) = image.shape
	(cX, cY) = (int(w/2.0), int(h/2.0))
	fft = np.fft.fft2(image)
	fftShift = np.fft.fftshift(fft)
	# check to see if we are visualizing our output
	if vis:
		# compute the magnitude spectrum of the transform
		magnitude_fft = 20 * np.log(np.abs(fftShift))
	fftShift[cY - size:cY + size, cX - size:cX + size] = 0
	fftShift = np.fft.ifftshift(fftShift)
	recon = np.fft.ifft2(fftShift)
	
	# compute the magnitude spectrum of the reconstructed image,
	# then compute the mean of the magnitude values
	magnitude = 20 * np.log(np.abs(recon))
	mean = np.mean(magnitude)
	# the image will be considered "blurry" if the mean value of the
	# magnitudes is less than the threshold value

	if vis:
		# display the original input image
		(fig, ax) = plt.subplots(1, 3, )
		ax[0].imshow(image, cmap="gray")
		ax[0].set_title("Input")
		ax[0].set_xticks([])
		ax[0].set_yticks([])
		# display the magnitude image
		ax[1].imshow(magnitude_fft, cmap="gray")
		ax[1].set_title("Magnitude Spectrum")
		ax[1].set_xticks([])
		ax[1].set_yticks([])
		# display the magnitude image
		ax[2].imshow(magnitude, cmap="gray")
		ax[2].set_title(f"Recon (mean:{mean})")
		ax[2].set_xticks([])
		ax[2].set_yticks([])
		# show our plots
		plt.ioff()
		plt.show()
		plt.pause(0.01)
	
	# print(mean)
	return mean

def main():
#print("Looping...")
#print("Images only:", paths.list_images(args["images"]))
# loop over the input images
	if os.path.isdir(args.file):
		image_list = list(sorted(paths.list_images(args.file)))
	elif os.path.isfile(args.file):
		image_list = [args.file]
	else:
		raise(ValueError(f"Don't know what type of thing '{args.file}' is"))

	for filio in image_list:
		if args.max > -1 and len(blurar) >= args.max:
			break
		handle_file(filio)
		if quit: break

import mimetypes
mimetypes.init()

def handle_file(filio):
	global imgidx
	global quit
	global pyplot_frame
	is_vid = False
	mimestart = mimetypes.guess_type(filio)[0]
	if mimestart is None:
		raise(ValueError(f"mimetypes couldn't determine file type for {filio}"))
	else:
		mimestart = mimestart.split('/')[0]
	if mimestart == 'image':
		image = cv2.imread(filio)
	elif mimestart == 'video':
		is_vid = True
		vid = imageio.get_reader(filio, 'ffmpeg')
		imgidx = 0 # Restart frame count for videos
		#last_frame = vid.get_length()-1  # returns inf on image.. eh
		last_frame = vid.count_frames()-1
	else:
		raise(ValueError(f"mimetypes thinks this is file type '{mimestart}', which we don't know how to handle (image and video only)"))
	while True:
		if args.max > -1 and len(blurar) >= args.max:
			break
		# load the image, convert it to grayscale, and compute the
		# focus measure of the image using the Variance of Laplacian
		# method
		# print("Imagepath:", filio)
		# image = cv2.imread(filio)
		if is_vid:
			image = vid.get_data(imgidx)
		# print("Image:", image)
		#gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		# if opt_blur: image = cv2.GaussianBlur(image, (5,5), 0)
		if args.crop < 100:
			sy,sx = image.shape[0:2]
			wx = int(sx*args.crop / 100)
			wy = int(sy*args.crop / 100)
			skipx = int((sx - wx)/2)
			skipy = int((sy - wy)/2)
			# print("      size:", sx, sy)
			# print("crop width:", wx, wy)
			# print("skip size :", skipx, skipy)
			image = image[skipy:skipy+wy, skipx:skipx+wx]
			# print("      Crop:", image.shape[1], image.shape[0])
			# print("")

		gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		#fm = det_blur_variance_of_laplacian(gray)
		fm = det_blur_fft(gray, vis=False)
		blurar.append({"file":filio, "frame":imgidx, "blur":fm})
	
		if args.verbose or args.show or args.grid:
			# if the focus measure is less than the supplied threshold,
			# then the image should be considered "blurry"
			#print(fm, " < ", args["threshold"])
			if fm < args.threshold:
				text = "Blurry"
			else:
				text = "Sharp"
			print("{:.2f} {} {}".format(fm, filio, text), flush=True)
			print("{:.2f} {} {}".format(fm, filio, text), file=blurf, flush=True)
		
			# show the image
			txxy = (12, 60)
			fontscale = 1.5
			cv2.putText(image, "{}: {:.2f} {}".format(text, fm, filio),
				(txxy[0]-1, txxy[1]-1), cv2.FONT_HERSHEY_SIMPLEX,
				fontscale, (0, 0, 0), 3)
			cv2.putText(image, "{}: {:.2f} {}".format(text, fm, filio),
				(txxy[0]+2, txxy[1]+2), cv2.FONT_HERSHEY_SIMPLEX,
				fontscale, (0, 0, 0), 3)
			cv2.putText(image, "{}: {:.2f} {}".format(text, fm, filio),
				(txxy[0]  , txxy[1]  ), cv2.FONT_HERSHEY_SIMPLEX,
				fontscale, (205, 100, 100), 3)
			if args.grid:
				blurar[-1]['img'] = image
			if args.show:
				#cv2.imshow("Image", image)
				#plt.imshow(image, cmap='gray')
				if pyplot_frame is None:
					pyplot_frame = plt.imshow(image)
				else:
					pyplot_frame.set_data(image)
				plt.ion()
				plt.show()
				plt.pause(0.05)
				print("(h)prev, (l)next, (q)uit: ", flush=True)
				ch = getch.getch()
				if ch == 'h': print("Previous"); imgidx -= 2
				elif ch == 'l': print("Next"); pass # auto increment anyway
				elif ch == 'q': print("Quit"); quit=True
				else: print(f"Unknown ({ch}) - Going to Next")
				plt.pause(0.01)
		imgidx += 1
		# We're done if it's just a single image
		#  or if we finished all video frames
		if not is_vid or imgidx >= last_frame or quit:
			break
		# if imgidx > 4: break

def show_grid():
	tot = len(blurar)
	if args.max > -1: tot = min(tot, args.max)
	cols = int(math.sqrt(tot))
	rows = math.ceil(tot/cols)

	grid = imgutils.PicGrid(cols=cols, rows=rows)

	for i in range(tot):
		img = blurar[i]['img']
		wid = min(img.shape[1], 600)
		hi = int(img.shape[0] * wid/img.shape[1])
		grid.add(cv2.resize(img, (wid,hi), interpolation=cv2.INTER_AREA))
	
	plt.axis('off')
	plt.ioff()
	imggrid = grid.make_img_grid() # PIL image
	print("Saving image to /tmp/foo.jpg ...")
	imggrid.save("/tmp/foo.jpg")
	print("Image saved.")
	plt.imshow(imggrid)
	plt.show()

	# ar = np.array(blurar)
	# ar = np.hstack((range(ar.shape[0]), ar))
	
main()
if args.grid:
	show_grid()
