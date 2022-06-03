import face_recognition
import cv2
import numpy as np

# stuff for sending to opensearch
from opensearch import *

##############################
# CONSTANTS YO
max_frames = 300
frame_log_freq = 20
detect_log_freq = 1
UNKNOWN_NAME = 'Unknown'
##############################

video = "/home/vsa/face/data/ShortTC-TG.mp4"
video_capture = cv2.VideoCapture(video)

frame_width = int(video_capture.get(3))
frame_height = int(video_capture.get(4))

fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')

out = cv2.VideoWriter('/home/vsa/face/output/output_topgun.mp4', fourcc, 15.0, (frame_width, frame_height))

# Load a sample picture and learn how to recognize it.
tom_image = face_recognition.load_image_file("./pics/tom.jpg")
tom_face_encoding = face_recognition.face_encodings(tom_image)[0]

# Load a second sample picture and learn how to recognize it.
anthony_image = face_recognition.load_image_file("./pics/anthony.jpg")
anthony_face_encoding = face_recognition.face_encodings(anthony_image)[0]

# Load a third sample picture and learn how to recognize it.
val_image = face_recognition.load_image_file("./pics/val.jpg")
val_face_encoding = face_recognition.face_encodings(val_image)[0]

# Load a fourth sample picture and learn how to recognize it.
skerritt_image = face_recognition.load_image_file("./pics/skerritt.jpg")
skerritt_face_encoding = face_recognition.face_encodings(skerritt_image)[0]

# Create arrays of known face encodings and their names
known_face_encodings = [
    tom_face_encoding,
    anthony_face_encoding,
    val_face_encoding,
    skerritt_face_encoding
]
known_face_names = [
    "Tom Cruise",
    "Anthony",
    "Val Kilmer",
    "Tom Skerritt"
]

# Initialize some variables
face_locations = []
face_encodings = []
face_names = []
process_this_frame = True
frames = []
count = 0
results = []
while True:
    # Grab a single frame of video
    ret, frame = video_capture.read()
    # import pdb;pdb.set_trace()

    # Resize frame of video to 1/4 size for faster face recognition processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    count += 1
    if count == max_frames:
        break
    elif count % frame_log_freq == 0:
        print("[%d] frames processed" % count)
    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    rgb_small_frame = small_frame[:, :, ::-1]
    
    # Only process every other frame of video to save time
    if process_this_frame:
        # Find all the faces and face encodings in the current frame of video
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []
        for face_encoding in face_encodings:
            # See if the face is a match for the known face(s)
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = UNKNOWN_NAME

            # # If a match was found in known_face_encodings, just use the first one.
            # if True in matches:
            #     first_match_index = matches.index(True)
            #     name = known_face_names[first_match_index]

            # import pdb;pdb.set_trace()

            # Or instead, use the known face with the smallest distance to the new face
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]

            face_names.append(name)

    process_this_frame = not process_this_frame
    

    # Display the results
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # import pdb;pdb.set_trace()
        # Scale back up face locations since the frame we detected in was scaled to 1/4 size
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # Draw a label with a name below the face
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

        # add this for search
        if name != UNKNOWN_NAME:
            results.append((name,count,video.split("/")[-1]))

        # if count % detect_log_freq == 0:
        #     print('[Frame # %d] Found %s' % (count,name))

    out.write(frame)


# add results to search db
outfile = create_document(results,is_face=True)
send_results(outfile)

# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()