# Inspired by https://github.com/opencv/opencv/blob/master/samples/dnn/action_recognition.py
# and https://github.com/kenshohara/video-classification-3d-cnn-pytorch
# and by https://www.pyimagesearch.com/2019/11/25/human-activity-recognition-with-opencv-and-deep-learning/

# Altenative TensorFlow example https://www.tensorflow.org/hub/tutorials/action_recognition_with_tf_hub
# import the necessary packages
import numpy as np
import os
import argparse
import imutils
import sys
import cv2
import json
from pathlib import Path
from opensearch import *

# Files - assumes these files are in the same file/directory as the Jupyter Notebook
names = "action_recognition_kinetics.txt"
model = "resnet-34_kinetics.onnx"
video = "example_activities.mp4"
with open(names) as l:
    CLASSES = l.read().strip().split("\n")

CLASSES
# load the contents of the class labels file, then define the sample
# duration (i.e., # of frames for classification) and sample size
# (i.e., the spatial dimensions of the frame)

SAMPLE_DURATION = 16 
SAMPLE_SIZE = 112 
FRAME_SPEED = 30 #frames per sec
# load the human activity recognition model
print("[INFO] loading human activity recognition model...")
net = cv2.dnn.readNet(model)
# grab a pointer to the input video stream
print("[INFO] accessing video stream...")

vs = cv2.VideoCapture(video)
# initialize the list / dictionaries to captured classified frames
classifiedFrames = []
framesDict = []
metaList = []
dictKey = {}
n = 0
# Run the model
# loop until we explicitly break from it
FRAME_NO = 1
flag = True
while True:
  # initialize the batch of frames that will be passed through the model
  
  frames = []

  n += 1
    
  # loop over the number of required sample frames
  for i in range(0, SAMPLE_DURATION):
    # read a frame from the video stream
    (grabbed, frame) = vs.read()
    
    #Identiy the frame number
    pos_frame = vs.get(cv2.CAP_PROP_POS_FRAMES)
    
    # if the frame was not grabbed then we've reached the end of
    # the video stream so exit the script
    if not grabbed:
      print("[INFO] no frame read from stream - exiting")
      #sys.exit(0)
      flag = False
      break
    # otherwise, the frame was read so resize it and add it to
    # our frames list
    frame = imutils.resize(frame, width=400)
    frames.append(frame)
  if not flag:
    break
  
  # now that our frames array is filled we can construct our blob
  blob = cv2.dnn.blobFromImages(frames, 1.0, (SAMPLE_SIZE, SAMPLE_SIZE), (114.7748, 107.7354, 99.4750), 
                                swapRB=True, crop=True)
  blob = np.transpose(blob, (1, 0, 2, 3))
  blob = np.expand_dims(blob, axis=0)
  
  # pass the blob through the network to obtain our human activity
  # recognition predictions
  net.setInput(blob)
  outputs = net.forward()
  label = CLASSES[np.argmax(outputs)]
  
  #capture the frames and labels for future database search engine 
  # fill blank frames
  for i in range(SAMPLE_DURATION):
    frame_no = int(pos_frame+(i+1)-SAMPLE_DURATION)
    metaList.append((label,frame_no,video))

  # loop over our frames
  frame_no = FRAME_NO
  for frame in frames:
    # # draw the predicted activity on the frame

    time_no = np.round( frame_no / FRAME_SPEED , 2)  
    cv2.rectangle(frame, (0, 0), (400, 40), (0, 0, 0), -1) #300, 40
    cv2.putText(frame, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(frame, "F:"+str(frame_no), (210, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(frame, "T:"+str(time_no), (300, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    # display the frame to our screen
    cv2.imshow("Activity Recognition", frame)
    frame_no += 1
    
    key = cv2.waitKey(1) & 0xFF
    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
      break
  
  FRAME_NO += SAMPLE_DURATION
  
print("Start seding the result")
outfile = create_document(metaList,is_face=False)
send_results(outfile)
print("Complete seding the result")  

# release VideoCapture()
vs.release()
# close all frames and video windows
cv2.destroyAllWindows()
activity = 'yoga'
activity_in_list = [activity in list for list in classifiedFrames]

for frame in classifiedFrames:
  if activity in frame:
    print(frame)
framesDict


