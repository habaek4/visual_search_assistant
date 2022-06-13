import face_recognition
import cv2
import numpy as np
import argparse
import os

def get_video_name(pth):
    return pth.split("/")[-1].split('.')[0]

def face_detection_batch(input_pth,output_pth=None):
    if output_pth is None:
        name = get_video_name(input_pth)
        new_name = name + '_detection'
        output_root = '/home/spock-the-wizard/cmu/project/outputs'
        output_pth = os.path.join(output_root,new_name+'.mp4')
    
    video_capture = cv2.VideoCapture(input_pth)

    frame_width = int(video_capture.get(3))
    frame_height = int(video_capture.get(4))

    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')

    out = cv2.VideoWriter(output_pth, fourcc, 30.0, (frame_width, frame_height))

    frame_count = 0
    frames = []
    while True:
        ret, frame = video_capture.read()

        if ret is False:
            break
        if frame_count % 50 == 49:
            print('Processed %d frames'% frame_count)
        face_locations = face_recognition.face_locations(frame[:,:,::-1])

        for (top,right,bottom,left) in face_locations:
            # Draw a box around the face
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
            # Draw a label with a name below the face
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)


        out.write(frame)
        frame_count += 1

    video_capture.release()
    print('Done Processing %s' % input_pth)


def face_detection(input_pth,output_pth=None,use_gpu=False):
    batch_len = 128

    if output_pth is None:
        name = get_video_name(input_pth)
        new_name = name + '_detection'
        output_root = '/home/spock-the-wizard/cmu/project/outputs'
        output_pth = os.path.join(output_root,new_name+'.mp4')
    
    video_capture = cv2.VideoCapture(input_pth)

    frame_width = int(video_capture.get(3))
    frame_height = int(video_capture.get(4))

    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')

    out = cv2.VideoWriter(output_pth, fourcc, 30.0, (frame_width, frame_height))

    frame_count = 0
    frames = []
    while video_capture.isOpened():
        ret, frame = video_capture.read()

        if not ret:
            break
        if frame_count % 50 == 49:
            print('Processed %d frames'% frame_count)
        
        frame_count += 1
        if use_gpu:
            frames.append(frame)
        if use_gpu and len(frames) == batch_len:
            
            import pdb;pdb.set_trace()
            batch_face_locations = face_recognition.batch_face_locations(frames)
            for frame_number_in_batch, face_locations in enumerate(batch_face_locations):
                number_of_faces_in_frame = len(face_locations)

                frame_number = frame_count - 128 + frame_number_in_batch

                frame = frames[frame_number]
                for (top,right,bottom,left) in face_locations:
                    # Draw a box around the face
                    cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
                    # Draw a label with a name below the face
                    cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
                out.write(frame)

        elif not use_gpu:
            face_locations = face_recognition.face_locations(frame[:,:,::-1])

            for (top,right,bottom,left) in face_locations:
                # Draw a box around the face
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
                # Draw a label with a name below the face
                cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)

            out.write(frame)
        

    video_capture.release()
    print('Done Processing %s' % input_pth)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_pth',type=str)
    parser.add_argument('--output_pth',type=str)
    parser.add_argument('--mode',choices=['detection'],default='detection')
    args = parser.parse_args()
    face_detection(args.input_pth,use_gpu=True)
