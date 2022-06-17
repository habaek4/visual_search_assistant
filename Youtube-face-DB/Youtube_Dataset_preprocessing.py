# -*- coding: utf-8 -*-
"""
Created on Thu Jun 16 14:58:50 2022

@author: MinwooChoi
"""

import math
import numpy as np
import os
import scipy.io
from shutil import copyfile

source_root = '../aligned_images_DB/'
    
## 폴더0 - 폴더1(이름) - 폴더2(숫자) - 파일

## identity.txt 파일 만들기

identities = {}
identity = 0
folder1_list = os.listdir(source_root)
for folder1 in folder1_list:
    folder2_list = os.listdir(source_root+folder1+'/')
    for folder2 in folder2_list:
        file_list = os.listdir(source_root+folder1+'/'+folder2+'/')
        for file in file_list:
            if file.endswith('.jpg'):
                file_name = str(folder1+'/'+folder2+'/'+file)
                identities[file_name] = identity
    identity += 1
    
with open('./Youtube-face-DB-identity.txt','w',encoding='UTF-8') as f:
    for file,identity in identities.items():
        f.write(f'{file} {identity}\n')
        
print(f'There are {len(set(identities.values()))} identities.')
print(f'There are {len(identities.keys())} images.')

## 파일들을 target dir 에 복사하기

target_root = './Youtube_identity_dataset/'
file_list = os.listdir(source_root)


folder1_list = os.listdir(source_root)
for folder1 in folder1_list:
    folder2_list = os.listdir(source_root+folder1+'/')
    for folder2 in folder2_list:
        file_list = os.listdir(source_root+folder1+'/'+folder2+'/')
        for file in file_list:
            file_name = str(folder1+'/'+folder2+'/'+file)
            identity = identities[file_name]
            source = os.path.join(source_root, file_name)
            target = os.path.join(target_root, str(identity), file)
            if not os.path.exists(os.path.join(target_root, str(identity))):
                os.makedirs(os.path.join(target_root, str(identity)))
            copyfile(source, target)
                    
