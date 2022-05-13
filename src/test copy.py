from calendar import c
import streamlit as st
import pandas as pd
import numpy as np
import imutils
import time
import cv2
import os
import datetime
import scipy.spatial.distance as dist
from tensorflow.keras.applications import imagenet_utils				# Preprocess for any models
from tensorflow.keras.applications.inception_v3 import preprocess_input	# Preprocess for Inception and Xception
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
import io
from PIL import Image

# load our serialized face detector model from disk
prototxtPath = r"src\face_detector\deploy.prototxt"
weightsPath = r"src\face_detector\res10_300x300_ssd_iter_140000.caffemodel"
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

# load the face mask detector model from disk
model_name = "InceptionV3.model"
model_path = os.path.join("src\models", model_name)
maskNet = load_model(model_path)

def clear_session_state():
	for key in st.session_state.keys():
		del st.session_state[key]

def calc_engagementrate(engagedcount, disengagedcount):
	total = engagedcount+disengagedcount
	engagedrate = (float(engagedcount)/float(total))*100
	engagedrate = round(engagedrate, 2)
	disengagedrate = 100 - engagedrate
	return [engagedrate, disengagedrate]

def detect_and_predict_mask(frame, faceNet, maskNet): 	#recv
	# grab the dimensions of the frame and then construct a blob
	# from it
	(h, w) = frame.shape[:2]
	blob = cv2.dnn.blobFromImage(frame, 1.0, (299, 299),
		(104.0, 177.0, 123.0))

	# pass the blob through the network and obtain the face detections
	faceNet.setInput(blob)
	detections = faceNet.forward()
	print(detections.shape)

	# initialize our list of faces, their corresponding locations,
	# and the list of predictions from our face mask network
	faces = []
	locs = []
	centroids = []
	preds = []

	# loop over the detections
	for i in range(0, detections.shape[2]):
		# extract the confidence (i.e., probability) associated with
		# the detection
		confidence = detections[0, 0, i, 2]

		# filter out weak detections by ensuring the confidence is
		# greater than the minimum confidence
		if confidence > 0.5:
			# compute the (x, y)-coordinates of the bounding box for
			# the object
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")

			# ensure the bounding boxes fall within the dimensions of
			# the frame
			(startX, startY) = (max(0, startX), max(0, startY))
			(endX, endY) = (min(w - 1, endX), min(h - 1, endY))

			# extract the face ROI, convert it from BGR to RGB channel
			# ordering, resize it to 224x224, and preprocess it
			face = frame[startY:endY, startX:endX]
			face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
			face = cv2.resize(face, (299, 299))
			face = img_to_array(face)
			face = preprocess_input(face)

			# add the face and bounding boxes to their respective
			# lists
			faces.append(face)
			locs.append((startX, startY, endX, endY))

	# only make a predictions if at least one face was detected
	if len(faces) > 0:
		# for faster inference we'll make batch predictions on *all*
		# faces at the same time rather than one-by-one predictions
		# in the above `for` loop
		faces = np.array(faces, dtype="float32")
		preds = maskNet.predict(faces, batch_size=32)

	# return a 2-tuple of the face locations and their corresponding
	# locations
	return (locs, preds)

def convert_df(df):
	# IMPORTANT: Cache the conversion to prevent computation on every rerun
	return df.to_csv().encode('utf-8')
	
def stopwebcam(engagedcount, disengagedcount):
	st.write("Webcam stopped")
	engagementrate = calc_engagementrate(engagedcount,disengagedcount)
	st.write("Engaged: ",engagementrate[0],"%, Disengaged:",engagementrate[1],"%")

def stopbuttoncallback():
	st.session_state.stopbutton_clicked = True

def startwebcam(): 	#annotate image
	# initialize the video stream
	frame_window = st.image([])
	vs = cv2.VideoCapture(0)
	stopbutton = st.button("Stop", st.session_state.stopbutton_clicked)
	engagedcount=0
	disengagedcount=0
	# stopbutton = st.button("Stop", on_click=stopwebcam(engagedcount,disengagedcount))

	# loop over the frames from the video stream
	while True:
		# grab the frame from the threaded video stream and resize it
		# to have a maximum width of 400 pixels
		_, frame = vs.read()
		frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
		frame = imutils.resize(frame, width=400)

		# detect faces in the frame and determine if they are wearing a
		# face mask or not
		(locs, preds) = detect_and_predict_mask(frame, faceNet, maskNet)

		# loop over the detected face locations and their corresponding
		# locations
		for (box, pred) in zip(locs, preds):
			# unpack the bounding box and predictions
			(startX, startY, endX, endY) = box
			(disengaged, engaged) = pred

			# determine the class label and color we'll use to draw
			# the bounding box and text
			if engaged > disengaged:
				label = "Engaged"
				engagedcount = engagedcount+1
				color = (0, 255, 0)

			else:
				label = "Disengaged"
				disengagedcount = disengagedcount+1
				color = (0, 0, 255)
			
			st.session_state.engagedcount = engagedcount
			st.session_state.disengagedcount = disengagedcount
			st.session_state.totalcount = engagedcount+disengagedcount
			# include the probability in the label
			label = "{}: {:.2f}%".format(label, max(disengaged, engaged) * 100)

			# display the label and bounding box rectangle on the output
			# frame
			cv2.putText(frame, label, (startX, startY - 10),
				cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
			cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

		# show the output frame
		frame_window.image(frame)
		

   #     vs.release()
	#frame_window.empty()
	#vs.release()
	#cv2.destroyAllWindows

def predictimage(uploadedimage):
	img1 = Image.open(uploadedimage)
	frame = np.array(img1)
	# frame = cv2.imdecode(np.fromstring(uploadedimage.read(), np.uint8), cv2.IMREAD_UNCHANGED)
	frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
	frame = imutils.resize(frame, width=400)
	# detect faces in the frame and determine if they are wearing a
	# face mask or not
	(locs, preds) = detect_and_predict_mask(frame, faceNet, maskNet)
	# loop over the detected face locations and their corresponding
	# locations
	for (box, pred) in zip(locs, preds):
		# unpack the bounding box and predictions
		(startX, startY, endX, endY) = box
		(disengaged, engaged) = pred
		# determine the class label and color we'll use to draw
		# the bounding box and text
		label = "Disengaged" if disengaged > engaged else "Engaged"
		color = (0, 255, 0) if label == "disengaged" else (0, 0, 255)
		# include the probability in the label
		label = "{}: {:.2f}%".format(label, max(disengaged, engaged) * 100)

		# display the label and bounding box rectangle on the output
		# frame
		
		cv2.putText(frame, label, (startX, startY - 10),
			cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
		cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
		# show the output frame

	frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
	predictedimage = st.image(frame)
	st.subheader("Result: ",label)
	cv2.destroyAllWindows()

def predictvideo(uploadedvideo):
	vid = uploadedvideo.name
	with open(vid, mode='wb') as f:
		f.write(uploadedvideo.read()) #save the video to disk
	
	engagedcount=0
	disengagedcount=0
	st_video = open(vid,'rb')
	video_bytes = st_video.read()
	st.video(video_bytes)
	st.write("Uploaded Video")
	cap = cv2.VideoCapture(vid)
	count = 0
	while True:
		_, frame = cap.read()
		if _ != False:
			frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
			frame = imutils.resize(frame, width=400)

			(locs, preds) = detect_and_predict_mask(frame, faceNet, maskNet)
			# loop over the detected face locations and their corresponding
			# locations
			for (box, pred) in zip(locs, preds):
				# unpack the bounding box and predictions
				(startX, startY, endX, endY) = box
				(disengaged, engaged) = pred

				# determine the class label and color we'll use to draw
				# the bounding box and text
				if engaged > disengaged:
					label = "Engaged"
					engagedcount = engagedcount+1
					color = (0, 255, 0)

				else:
					label = "Disengaged"
					disengagedcount = disengagedcount+1
					color = (0, 0, 255)
				
				# include the probability in the label
				label = "{}: {:.2f}%".format(label, max(disengaged, engaged) * 100)

				# display the label and bounding box rectangle on the output
				# frame
				cv2.putText(frame, label, (startX, startY - 10),
					cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
				cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

		else:
			break

	engagementrate = calc_engagementrate(engagedcount,disengagedcount)
	st.write("Engaged: ",engagementrate[0],"%, Disengaged:",engagementrate[1],"%")
	cap.release()
	cv2.destroyAllWindows()

def page_webcam():
	startbutton = st.button("Start webcam", on_click=startwebcam)
	

def page_image():
	uploadedimage = st.file_uploader("Upload an image", type=['jpeg','png','jpg','gif'], accept_multiple_files=False)
	if uploadedimage != None:
		predictimage(uploadedimage)

def page_video():
		uploadedvideo = st.file_uploader("Upload a video", type=['mp4','mpeg','mov'], accept_multiple_files=False)
		if uploadedvideo != None:
			predictvideo(uploadedvideo)

pages = {
	"Real time (Webcam)": page_webcam,
	"Image": page_image,
	"Video": page_video
}

if "stopbutton_clicked" not in st.session_state:
	st.session_state.stopbutton_clicked = False

if st.session_state.stopbutton_clicked == True:
	stopwebcam(st.session_state.engagedcount, st.session_state.disengagedcount)
	restartbutton = st.button(label="Restart",on_click=clear_session_state)

if st.session_state.stopbutton_clicked == False:
	st.title('Engagement Detection')
	st.subheader('Detecting engagement status through facial expression and deep learning.')

	selected_page = st.selectbox(
	"Choose page",
	pages.keys()
	)

	pages[selected_page]()




# Button to start the webcam
#  startbutton = st.button("Start webcam", on_click=startwebcam)
	# opencv frame

# Stop button    

# Time elapsed
# Average engagement: % Disengaged, % Engaged


# Show graph/results of the engagement status
# Run at least 1 minute to show statistics
# - 

# Download results button

# csv = convert_df(my_large_df)
# st.download_button(
#      label="Download data as CSV",
#      data=csv,
#      file_name='large_df.csv',
#      mime='text/csv',
#  )