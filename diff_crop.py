# USAGE
# python detect_drowsiness.py --shape-predictor shape_predictor_68_face_landmarks.dat
# python detect_drowsiness.py --shape-predictor shape_predictor_68_face_landmarks.dat --alarm alarm.wav

# import the necessary packages
import os
import time
import argparse
from threading import Thread
import cv2
import dlib
import numpy as np
import imutils
import random
from moviepy.editor import VideoFileClip
from imutils import face_utils
from imutils.video import FileVideoStream
from scipy.spatial import distance as dist

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", help="video path input")
ap.add_argument("-t", type=float, default=0.57, help="threshold for mouth open")
ap.add_argument("-f", type=int, default=8, help="continous closed mouth frames to split")
ap.add_argument("-w", action="store_true", help="write frames")
ap.add_argument("-s", default='shape_predictor_68_face_landmarks.dat', help="path to facial landmark predictor")
args = vars(ap.parse_args())

# define one constants, for mouth aspect ratio to indicate open mouth
MOUTH_AR_THRESH = args["t"] #0.53
frames_closed_threshold = args["f"] # 5

words = open('words.txt').read().splitlines()

# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor
print("[INFO] loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args['s'])

# grab the indexes of the facial landmarks for the mouth
(mStart, mEnd) = (49, 68)

def mouth_aspect_ratio(mouth):
	# compute the euclidean distances between the two sets of
	# vertical mouth landmarks (x, y)-coordinates
	A = dist.euclidean(mouth[2], mouth[9]) # 51, 59
	B = dist.euclidean(mouth[4], mouth[7]) # 53, 57

	# compute the euclidean distance between the horizontal
	# mouth landmark (x, y)-coordinates
	C = dist.euclidean(mouth[0], mouth[6]) # 49, 55

	# compute the mouth aspect ratio
	mar = (A + B) / (2.0 * C)

	# return the mouth aspect ratio
	return mar

if __name__ == '__main__':
	# videos = ['meAllWords2.mp4', 'meAllWords.mp4', 'meAllWords3.mp4']
	videos = ['meAllWords3.mp4']

	for video in videos:
		dir = 'output_{}'.format(video)

		if os.path.exists(dir):
			for f in os.listdir(dir):
				os.remove(os.path.join(dir, f))
		else:
			os.mkdir(dir)

		# start the video stream thread
		print("[INFO] starting video stream thread...")
		cap = cv2.VideoCapture(video)

		width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))   # float `width`
		height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
		fps = int(cap.get(cv2.CAP_PROP_FPS))

		idx = 0
		ret, frame = cap.read()

		aspect_diff_count, previous_aspect_ratio = 0, -1
		frame_idx, last_diff_frame_idx = 0, 0
		
		delta = 0.02
		start_idx = 0
		buff_start = 0

		clip = VideoFileClip(video)

		def get_timestamp(frame_idx):
			return round((frame_idx/fps), 3) 

		def cut_and_write(ts):
			start_ts, end_ts = ts[0], ts[1]
			o = words[idx] if idx < len(words) else idx
			clip.subclip(start_ts, end_ts).to_videofile(os.path.join(dir, '{}.avi'.format(o)), codec="libx264", temp_audiofile='t.m4a', remove_temp=True, audio_codec='aac')

		# loop over frames from the video stream
		while ret:
			# grab the frame from the threaded and convert it to grayscale channels)
			gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

			# detect faces in the grayscale frame
			rects = detector(gray, 0)

			# loop over the face detections
			for rect in rects:
				# determine the facial landmarks for the face region, then
				# convert the facial landmark (x, y)-coordinates to a NumPy array
				shape = predictor(gray, rect)
				shape = face_utils.shape_to_np(shape)

				# extract the mouth coordinates, then use the
				# coordinates to compute the mouth aspect ratio
				mouth = shape[mStart:mEnd]
				aspect_ratio = mouth_aspect_ratio(mouth)

				diff = abs(previous_aspect_ratio - aspect_ratio) if previous_aspect_ratio > -1 else 0

				previous_aspect_ratio = aspect_ratio

				if diff > delta:
					aspect_diff_count += 1
					if aspect_diff_count == 2:
						buff_start = random.randint(0, 8)
						start_idx = max(0, last_diff_frame_idx-buff_start)
					last_diff_frame_idx = frame_idx

				# compute the convex hull for the mouth, then
				# visualize the mouth
				mouthHull = cv2.convexHull(mouth)

				if frame_idx - last_diff_frame_idx > 5:
					if aspect_diff_count > 7 and (frame_idx-start_idx+buff_start > 30):
						# write without sound to keep checking. 
						# Write for the last selected frame.
						end_idx = frame_idx+random.randint(0, 8)
						cut_and_write([ get_timestamp(start_idx), get_timestamp(end_idx) ])

						fname = words[idx] if idx < len(words) else idx
						print('\n', fname, [start_idx, end_idx], frame_idx-start_idx+buff_start, '\n')
						idx += 1

					aspect_diff_count = 0
					last_diff_frame_idx = frame_idx

			ret, frame = cap.read()
			frame_idx += 1
