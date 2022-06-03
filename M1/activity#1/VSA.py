# Inspired by https://github.com/opencv/opencv/blob/master/samples/dnn/action_recognition.py
# and https://github.com/kenshohara/video-classification-3d-cnn-pytorch
# and by https://www.pyimagesearch.com/2019/11/25/human-activity-recognition-with-opencv-and-deep-learning/

# Altenative TensorFlow example https://www.tensorflow.org/hub/tutorials/action_recognition_with_tf_hub
# import the necessary packages
import numpy as np
import argparse
import imutils
import sys
import cv2
import json
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
# load the human activity recognition model
print("[INFO] loading human activity recognition model...")
net = cv2.dnn.readNet(model)
# grab a pointer to the input video stream
print("[INFO] accessing video stream...")

vs = cv2.VideoCapture(video)
# initialize the list / dictionaries to captured classified frames
classifiedFrames = []
framesDict = []
dictKey = {}
n = 0
# Run the model
# loop until we explicitly break from it
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
      sys.exit(0)
    # otherwise, the frame was read so resize it and add it to
    # our frames list
    frame = imutils.resize(frame, width=400)
    frames.append(frame)
  
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
  classifiedFrames.append([pos_frame, label])
  dictKey = {"Index": {"_index" : "actions", "_id" : n}}
  value = {"frame" : pos_frame , "activity" : label}
  framesDict.append(dictKey)
  framesDict.append(value)


# =============================================================================
#   # loop over our frames
#   for frame in frames:
#     # draw the predicted activity on the frame
#     cv2.rectangle(frame, (0, 0), (300, 40), (0, 0, 0), -1) #300, 40
#     cv2.putText(frame, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
#     # display the frame to our screen
#     cv2.imshow("Activity Recognition", frame)
#     
#     key = cv2.waitKey(1) & 0xFF
#     # if the `q` key was pressed, break from the loop
#     if key == ord("q"):
#       break
# # release VideoCapture()
# vs.release()
# # close all frames and video windows
# cv2.destroyAllWindows()
# activity = 'yoga'
# activity_in_list = [activity in list for list in classifiedFrames]
# 
# for frame in classifiedFrames:
#   if activity in frame:
#     print(frame)
# framesDict
# 
# file_path = "C:/Users/MinwooChoi/Desktop/CMU/Class/StudioPJ/VSAsample.json"
# with open(file_path, 'w') as outfile:
#     json.dump(framesDict, outfile) 
# 
# =============================================================================
