import os
import cv2
import face_recognition
import numpy as np
# Basic constants

OUTPUT_DIR = '/dli/task/visual_search_assistant/M3/results'
FACE_LIB_DIR = '/dli/task/visual_search_assistant/M3/library/'
DATA_DIR = '/dli/task/visual_search_assistant/M3/data'
SAMPLE_FRAME_FREQ = 1
SAMPLE_CLUSTER_FREQ = 20
LOG_FREQ = 50

# batch_size = 128

input_video = None
output_video = None

# import os
# import face_recognition
import pickle

meta_pth = '/dli/task/visual_search_assistant/M3/library/meta.npz' #'/home/ubuntu/visual_search_assistant/M3/library/meta.npz'
all_embeddings = []
all_names = []
with open(meta_pth,'rb') as f:
    metadata = pickle.load(f)
    
    for idx,(key,val) in enumerate(metadata.items()):
        encoding = val['encoding']
        # import pdb;pdb.set_trace()
        if not encoding.shape and np.isnan(encoding):
            continue
        all_embeddings.append(val['encoding'])
        all_names.append(key)

print("List of all_names: \n",all_names)
print("Shape of single face encoding: \n", all_embeddings[0].shape)

# import os
import cv2
# import pickle
import numpy as np


def process_video(input_pth,output_pth=None,use_gpu=True,recognition=False,detection_threshold=0.6,save_samples=True,
                  result_dir=None,max_frames=None,sample_cluster_freq=2,sample_freq=2,batch_size=1):
    if output_pth is None:
        output_pth = os.path.join(OUTPUT_DIR,input_pth.split('/')[-1])

    input_name = input_pth.split('/')[-1].split('.')[0]

    if result_dir is None:
        result_dir = os.path.join(OUTPUT_DIR,input_name)
    if not os.path.exists(result_dir):
        sampled_face_dir = os.path.join(result_dir,'sampled_faces')
        os.makedirs(sampled_face_dir)
    else:
        print('Error result dir %s already exists! aborting' % result_dir)
        sampled_face_dir = os.path.join(result_dir,'sampled_faces')
#         return
    
    # print configurations
    if use_gpu:
        batch_size = batch_size
        print("Using GPU, batch size is %d" % batch_size)
    if save_samples:
        print("Saving detected faces as separate image files")
    else:
        print("Not saving sampled faces, vanilla inference pipeline")

    cluster_dict = {}
    
    video_capture = cv2.VideoCapture(input_pth)

    frame_width = int(video_capture.get(3))
    frame_height = int(video_capture.get(4))
    
    fps = video_capture.get(cv2. CAP_PROP_FPS)
    nframes = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
    length = nframes/fps
    
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    print(output_pth)
    out = cv2.VideoWriter(output_pth, fourcc, 30.0, (frame_width, frame_height))
    
    # statistics
    frame_count = 0
    num_detected_faces = 0
    
    frames = []
    print('='*5,'Start Face Detection and Recognition','='*5)
    while video_capture.isOpened():
        # print(frame_count)
        ret, frame = video_capture.read()
        if not ret or (max_frames and frame_count > max_frames):
            break
        if frame_count % LOG_FREQ == LOG_FREQ -1:
            print('Processed %d frames'% frame_count)
            
        frame_count += 1
        
        # skip frames
        if frame_count % sample_freq > 0:
            continue
            
        frames.append(frame)
        
        # import pdb;pdb.set_trace()
        if len(frames) == batch_size:
            print('Processing in batch! Frame number %d'% frame_count)
            batch_face_locations = face_recognition.batch_face_locations(frames)
            for frame_idx,face_locations in enumerate(batch_face_locations):

                num_detected_faces += len(face_locations)
                number_of_faces_in_frame = len(face_locations)
                
                fno = frame_count - batch_size + frame_idx
                frame = frames[frame_idx]
                frame_ = frame
                
                # import pdb;pdb.set_trace()
                embeddings = face_recognition.face_encodings(frame,face_locations)
                for face_idx,(embd,(top,right,bottom,left)) in enumerate(zip(embeddings,face_locations)):
                    # Draw a box around the face
                    cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
                    # Draw a label with a name below the face
                    cv2.rectangle(frame, (left, bottom + 35), (right, bottom), (0, 0, 255), cv2.FILLED)
                    
                    
                    # write label
                    face_dist = face_recognition.face_distance(all_embeddings,embd)
                    label_idx = np.argmin(face_dist)
                    name = all_names[label_idx] if face_dist[label_idx] > detection_threshold else 'Unknown'
                    font = cv2.FONT_HERSHEY_DUPLEX
                    cv2.putText(frame, name, (left + 6, bottom + 6), font, 0.5, (255, 255, 255), 1)
                    
                    if save_samples:
                        # save face image for clustering
                        if frame_idx % sample_cluster_freq ==0:
                            fname = '%.3d_%.3d.png'%(fno,face_idx)
                            face_img = frame_[top:bottom,left:right,:]
                            # save face image
                            cv2.imwrite(os.path.join(sampled_face_dir,fname),face_img)
                            # save embd in dict
                            cluster_dict[fname.split('.')[0]] = embd
                                
                out.write(frame)
            frames = []
                
                
    video_capture.release()
    out.release()
    
    # save sampled embeddings as pickle file
#     import pdb;pdb.set_trace()
    if save_samples:
        cluster_meta_file = os.path.join(result_dir,'embeddings.pickle')
        with open(cluster_meta_file,'wb') as f:
            pickle.dump(cluster_dict,f,protocol=pickle.HIGHEST_PROTOCOL)
        print("Saving detected images and embeddings as %s" % cluster_meta_file)
    else:
        print("No embeddings and samples saved! ")

    print('='*5,'Done Processing %s' % (input_name),'='*5)
    print("[RESULT] Video \"%s\" No. Frames: %d, No. faces processed: %d" % (input_name,frame_count,num_detected_faces))
    return input_name,nframes,length

        
if __name__ == "__main__":
    test_input_video = '/dli/task/visual_search_assistant/M3/data/parks_and_rec_0_10.mp4' #'/home/ubuntu/visual_search_assistant/data/radio_star_10_20.mp4'
    import time
    print('[TIMER] Start timer')
    start_time = time.time()
    num_frames = 64
    vid_name,vid_nframes,vid_length = process_video(test_input_video,max_frames=num_frames,batch_size=2)
    elapsed_time = time.time()-start_time
    print('[TIMER] elapsed: %.2f seconds',elapsed_time)
    print('[RESULT] FPS: %.3f' % (num_frames/float(elapsed_time)))
    # image = face_recognition.load_image_file("/dli/task/visual_search_assistant/M3/library/gookjin_kim/gookjin_kim.jpeg")
    # import pdb;pdb.set_trace()
    
    # face_locations = face_recognition.batch_face_locations([image,]*50,number_of_times_to_upsample=0)

    # import dlib
    # dlib.load_rgb_image("/home/ubuntu/visual_search_assistant/library/awkwafina.PNG")

