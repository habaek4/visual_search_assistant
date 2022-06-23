import os
import cv2
import face_recognition
# Basic constants

OUTPUT_DIR = '/home/ubuntu/visual_search_assistant/M3/results'
FACE_LIB_DIR = None
DATA_DIR = '/home/ubuntu/visual_search_assistant/data'
SAMPLE_FRAME_FREQ = 2
LOG_FREQ = 50

batch_size = 128

input_video = None
output_video = None


def process_video(input_pth,output_pth=None,use_gpu=True):
    if output_pth is None:
        output_pth = os.path.join(OUTPUT_DIR,input_pth.split('/')[-1])
    if not use_gpu:
        batch_size = 1
    else:
        batch_size = 16
    video_capture = cv2.VideoCapture(input_pth)

    frame_width = int(video_capture.get(3))
    frame_height = int(video_capture.get(4))
    
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    out = cv2.VideoWriter(output_pth, fourcc, 30.0, (frame_width, frame_height))
    
    frame_count = 0
    
    frames = []
    print('='*20,'Start Face Detection and Recognition','='*20)
    while video_capture.isOpened():
        ret, frame = video_capture.read()
        
        if not ret:
            break
        if frame_count % LOG_FREQ == LOG_FREQ -1:
            print('Processed %d frames'% frame_count)
            
        frame_count += 1
        
        # skip frames
        if frame_count % SAMPLE_FRAME_FREQ > 0:
            continue
            
        frames.append(frame)
        if len(frames) == batch_size:
            import pdb;pdb.set_trace()
            batch_face_locations = face_recognition.batch_face_locations(frames)
            for idx,face_locations in enumerate(batch_face_locations):
                number_of_faces_in_frame = len(face_locations)
                
                fno = frame_count - batch_size + idx
                frame = frames[frame_number]
                for (top,right,bottom,left) in face_locations:
                    # Draw a box around the face
                    cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
                    # Draw a label with a name below the face
                    cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
                out.write(frame)
            frames = []
                
                
    video_capture.release()
    out.release()
    print('='*15,'Done Processing %s to %s' % (input_pth,output_pth),'='*15)
        
if __name__ == "__main__":
# import face_recognition
    image = face_recognition.load_image_file("/home/ubuntu/visual_search_assistant/library/awkwafina.PNG")
    face_locations = face_recognition.batch_face_locations([image],number_of_times_to_upsample=0)

    # import dlib
    dlib.load_rgb_image("/home/ubuntu/visual_search_assistant/library/awkwafina.PNG")
    
