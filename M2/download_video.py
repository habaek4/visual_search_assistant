from pytube import YouTube
import argparse
from moviepy.editor import *

def download_video(link,output_path):
    yt = YouTube(link)
    video_stream = yt.streams.filter(file_extension="mp4",only_audio=False).first()
    video_stream.download(output_path=output_path)
    print('Download complete for %s' % video_stream.title)

def clip_video(video_pth,output_pth,time):
    start,end=time
    clip = VideoFileClip(video_pth).subclip(start,end)
    if output_pth is None:
        name = video_pth.split('/')[-1].split('.')[0]
        new_name = name + '_%d_%d'%(start,end)
        output_pth = video_pth.replace(name,new_name)

    clip.write_videofile(output_pth)

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--link',type=str,default=None)
    parser.add_argument('--output_path',type=str,default=None)
    parser.add_argument('--clip_time',nargs=2,default=[1,2],type=int)
    parser.add_argument('--mode',choices=['download','clip'])
    parser.add_argument('--clip_input_pth',type=str,default='/home/spock-the-wizard/cmu/project/data/[티전드] LA갈비구이와 소시지 버섯 구이! 대식가 가족의 가볍고() 맛있는 아침 먹방🍽  둥지탈출3 Diggle.mp4')
    parser.add_argument('--clip_output_pth',type=str,default=None)
    args = parser.parse_args()

    if args.mode == 'download':
        download_video(args.link,args.output_path)
    elif args.mode == 'clip':
        clip_video(video_pth=args.clip_input_pth,output_pth=args.clip_output_pth,time=args.clip_time)