import glob, cv2, os, sys
import numpy as np
import json
import argparse

import copy
import shutil
float_formatter = "{:.2f}".format

# === Code ===  

## Parameter

parser = argparse.ArgumentParser()
parser.add_argument('--in_images', type=str, default='./JPEGImages/', help='input images path')
parser.add_argument('--in_labels', type=str, default='./20211220 Cleaned labelme JSON/', help='input labels path')
parser.add_argument('--out_folder', type=str, default='./Step1_data/', help='output folder path')
opt = parser.parse_args()

def Check_jpg_and_json(images, labels):
    COUNT=0
    img_name = images[-26:-4] 
    lab_name = labels[-27:-5]
    if(img_name != lab_name):
        print(i)
        COUNT+=1
    
    return COUNT

def swap(a,b):
    temp = a
    a = b
    b = temp
    return a , b

def Check_jpg_and_json_all(images, labels):
    COUNT=0
    for i in range(len(images)):
        img_name = images[i][-26:-4] 
        lab_name = labels[i][-27:-5]
        if(img_name != lab_name):
            print(i)
            COUNT+=1
    
    return COUNT

def creat_folder(path):
    if not os.path.isdir(path):
        os.mkdir(path)
        
## input data



images_folder = glob.glob(opt.in_images+'*.jpg')
labels_folder = glob.glob(opt.in_labels+'*.json')
images_folder.sort()
labels_folder.sort()
print("Images number",len(images_folder))
print("Labels number",len(labels_folder))

dataset = []
for i in range(len(images_folder)):
    package=[]
    package.append(images_folder[i])
    package.append(labels_folder[i])
    dataset.append(package)

for i in range(len(images_folder)):
    Check_jpg_and_json(dataset[i][0],dataset[i][1])
    

creat_folder(opt.out_folder + 'lesion')
creat_folder(opt.out_folder + 'lesion/images')
creat_folder(opt.out_folder + 'lesion/labels')

creat_folder(opt.out_folder + 'normal')
creat_folder(opt.out_folder + 'normal/images')
creat_folder(opt.out_folder + 'normal/labels')

creat_folder(opt.out_folder + 'superscan')
creat_folder(opt.out_folder + 'superscan/images')
creat_folder(opt.out_folder + 'superscan/labels')

lesion_path = opt.out_folder + 'lesion'
normal_path = opt.out_folder + 'normal'
supercan_path = opt.out_folder + 'superscan'

for i in range(len(dataset)):
    label_iter = json.load(open(dataset[i][1]))
    bbox_list = []
    category_list = []
    flag = 0
    img_name = dataset[i][0][-26:]
    lab_name = dataset[i][1][-27:]
    for json_t in label_iter['shapes']:
#         print(json_t['label'])
        category =json_t['label'][0]
        category_list.append(category)
        
        if(category=='n'):
            shutil.copyfile(dataset[i][0], normal_path+'/images'+img_name)
            shutil.copyfile(dataset[i][1], normal_path+'/labels'+lab_name)
            flag=1
            
        elif(category=='S'):
            shutil.copyfile(dataset[i][0], supercan_path+'/images'+img_name)
            shutil.copyfile(dataset[i][1], supercan_path+'/labels'+lab_name)
            flag=1
            
    if(flag==0):
        shutil.copyfile(dataset[i][0], lesion_path+'/images'+img_name)
        shutil.copyfile(dataset[i][1], lesion_path+'/labels'+lab_name)

'''lesion data'''

images_folder = glob.glob( lesion_path + '/images/*.jpg')
labels_folder = glob.glob( lesion_path + '/labels/*.json')

images_folder.sort()
labels_folder.sort()
print("Lesion data : ",len(images_folder), len(labels_folder),', incorrect :', Check_jpg_and_json_all(images_folder, labels_folder))

'''normal data'''

images_folder = glob.glob(normal_path+'/images/*.jpg')
labels_folder = glob.glob(normal_path+'/labels/*.json')

images_folder.sort()
labels_folder.sort()
print("Normal data : ",len(images_folder), len(labels_folder),', incorrect :', Check_jpg_and_json_all(images_folder, labels_folder))

'''superscan data'''

images_folder = glob.glob(supercan_path+'/images/*.jpg')
labels_folder = glob.glob(supercan_path+'/labels/*.json')

images_folder.sort()
labels_folder.sort()
print("Superscan data : ",len(images_folder), len(labels_folder),', incorrect :', Check_jpg_and_json_all(images_folder, labels_folder))