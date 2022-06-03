import json
import requests
import os
import cv2

import awswrangler as wr

host = 'https://search-face-and-action-y6b5mzj5mupvipkof4pdaqlidy.us-east-2.es.amazonaws.com/'
auth = None # It's a secret

# curl -XPOST -u 'master-user:master-user-password' 'domain-endpoint/_bulk' --data-binary @bulk_movies.json -H 'Content-Type: application/json'
"""
curl command converted to python code
"""
def send_results(fname):
    headers = {
        'Content-Type': 'application/json',
    }
    with open(fname, 'rb') as f:
        data = f.read()
    response = requests.post(host+'_bulk', headers=headers, data=data, auth=auth)
    print(response.text)


# function to create tmp json for adding to db
def create_document(results,is_face=False):
    """
    name (str): action name or face name
    frame (int) : frame number
    video (str) : input video name (not path) ex. ShortTC-TG.mp4
    type (str, ['F', 'A']) : F for face, A for action
    """
    outfile = 'tmp.json'
    detection_type = 'F' if is_face else 'A'
    with open(outfile, 'w+') as f:
        for i,(name,fno,video_name) in enumerate(results):
            data = {
                'name': name,
                'frame': fno,
                'video':video_name,
                'type': detection_type
            }
            metadata = {
                'index': {
                    '_index':'faces' if is_face else 'actions',
                    '_id': name+video_name+str(fno)+detection_type
                }
            }
            meta = json.dumps(metadata)+'\n'
            f.write(meta)
            data = json.dumps(data) +'\n'
            f.write(data)
        
        return outfile

def frame2sec(frames,video_pth):
    video = cv2.VideoCapture(video_pth)
    fps = video.get(cv2.CAP_PROP_FPS)
    frame_count = video.get(cv2.CAP_PROP_FRAME_COUNT)
    
    converted = [round(f/frame_count*fps,2) for f in frames]

    return converted

def get_query_results(video_name,actors,is_face=True):
    search_type = 'faces' if is_face else 'actions'
    query_string = 'Frames with %s in %s'%(' and '.join(actors),video_name)

    params = {
        'q': 'name:'+','.join(actors)+'&video:'+video_name,
        'pretty': 'true',
    }
    # import pdb;pdb.set_trace()
    response = requests.get(host+search_type+'/_search',params=params, auth=auth)

    results = json.loads(response.text)['hits']['hits']
    print(response.text)
    frames = []
    for res in results:
        res = res['_source']
        frame = res['frame']
        frames.append(frame)

    frames = sorted(list(set(frames)))
    seconds = frame2sec(frames,video_pth=os.path.join('/home/vsa/face/data',video_name))
    
    print('='*20, query_string,'='*20)
    print('[Frame No.]:',frames)
    print('[Seconds  ]:',seconds)

    return frames

if __name__ == "__main__":
    get_query_results('ShortTC-TG.mp4',['Tom Cruise'])

    # # 리졀트 이렇게 tuple 의 리스트로 만들어주고
    # results = [('name',2,'test.mp4')]

    # # create_document 로 보내면 tmp.json 을 만들어서
    # outfile = create_document(results,is_face=True) 

    # # db로 보내줄거에요!
    # send_results(outfile)

