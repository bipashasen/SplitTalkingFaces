import cv2
import dlib
from imutils import face_utils
import os

shape = '../../shape_predictor_68_face_landmarks.dat'
paths = os.listdir('.')
paths_mp4 = []
for path in paths:
	if '.mp4' in path:
		paths_mp4.append(path)

face_bound = dlib.get_frontal_face_detector()
landmarks = dlib.shape_predictor(shape)

def extract_faces_mouth_in_center(filename):
	n = 0
	landmarksSet = []
	cap = cv2.VideoCapture(filename)
	while(cap.isOpened()):
		ret, frame = cap.read()
		if ret:
			gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
			faces = face_bound(gray, 0)
			# if (len(faces) < 1):
			# 	return n, False
			for face in faces:
				(x, y, w, h) = face_utils.rect_to_bb(face)
				cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
				landmark = landmarks(gray, face)
				landmark = face_utils.shape_to_np(landmark) # mouth landmarks:- 48 to 68.
				faceHull = cv2.convexHull(landmark)
				landmark = landmark[48:68]
				landmarksSet.append(landmark)
				cv2.drawContours(frame, [faceHull], -1, (0, 255, 0), 1)
				for (x, y) in landmark:
					cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)
			n += 1
		else:
			break
	cap.release()
	cv2.destroyAllWindows()
	return n, len(landmarksSet), True

for path in paths_mp4:
	print(path, extract_faces_mouth_in_center(path))

# path = 'EVERYBODY-17089.npz'
# print(extract_faces_mouth_in_center(path))

# cv2.imshow('test', frame)
# cv2.waitKey()

