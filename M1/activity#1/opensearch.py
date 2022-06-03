import json
import requests
import os

host = 'https://search-face-and-action-y6b5mzj5mupvipkof4pdaqlidy.us-east-2.es.amazonaws.com/'
auth = ('master','@Master1234')

# curl -XPOST -u 'master-user:master-user-password' 'domain-endpoint/_bulk' --data-binary @bulk_movies.json -H 'Content-Type: application/json'
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
    name (str): detect 된 action 이나 face 의 id 혹은 이름
    frame (int) : frame number
    video (str) : input video 이름 (경로말고) ex. ShortTC-TG.mp4
    type (str, ['F', 'A']) : F 면 face A면 action
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


if __name__ == "__main__":

    # 리졀트 이렇게 tuple 의 리스트로 만들어주고
    results = [('name',2,'test.mp4')]

    # create_document 로 보내면 tmp.json 을 만들어서
    outfile = create_document(results,is_face=True) 

    # db로 보내줄거에요!
    send_results(outfile)