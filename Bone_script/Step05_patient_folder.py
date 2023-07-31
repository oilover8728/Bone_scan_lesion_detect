import glob, cv2, os, sys
import argparse

import numpy as np
import json
from PIL import Image
import shutil

from torchvision import transforms
import copy

# === Code ===  

## Parameter

parser = argparse.ArgumentParser()
parser.add_argument('--in_folder', type=str, default='./Step4_patch_data/', help='input lesion images path')
parser.add_argument('--out_folder', type=str, default='./Step5_patient_data/', help='input lesion images path')
opt = parser.parse_args()

def creat_folder(path):
    if not os.path.isdir(path):
        os.mkdir(path)
        
def find_file_name(file):
    start_num = 0
    for i in range(len(file)-7):
        if(file[i:i+7]=='BS00001'):
            start_num=i
            break      
    return start_num

# '''fixed train-valid data'''
'''
    ==== 先複製一份input到output的資料夾 ====
'''

now_images = glob.glob(opt.in_folder+'images/*.png')
now_labels = glob.glob(opt.in_folder+'labels/*.json')
now_images.sort()
now_labels.sort()
print("Input Data (images/labels) : ", len(now_images), len(now_labels))

creat_folder(opt.out_folder)
creat_folder(opt.out_folder + 'images')
creat_folder(opt.out_folder + 'labels')

# print(now_images[0])
img_num=find_file_name(now_images[0])
lab_num=find_file_name(now_labels[0])     

for i in range(len(now_images)):
    img_name = now_images[i][img_num:]
    shutil.copyfile(now_images[i], opt.out_folder + 'images/' + img_name)

for i in range(len(now_labels)):
    lab_name = now_labels[i][lab_num:]
    shutil.copyfile(now_labels[i], opt.out_folder + 'labels/' + lab_name)

# '''
#     ==== 把所有Data按照病人編號放入其編號資料夾中 ====
# '''

images = glob.glob(opt.out_folder + 'images/*.png')
labels = glob.glob(opt.out_folder + 'labels/*.json')

images.sort()
labels.sort()
# print(images[0])

img_num=find_file_name(images[0])
lab_num=find_file_name(labels[0])    
        
for i in range(len(images)):
    patient_name = images[i][:img_num+7]
#     print(patient_name)
    if os.path.isdir(patient_name):
        shutil.move(images[i], patient_name)
    else:
        os.mkdir(patient_name)
        shutil.move(images[i], patient_name)
    

for i in range(len(labels)):
    patient_name = labels[i][:lab_num+7]
#     print(patient_name)
    if os.path.isdir(patient_name):
        shutil.move(labels[i], patient_name)
    else:
        os.mkdir(patient_name)
        shutil.move(labels[i], patient_name)
