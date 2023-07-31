import glob, cv2, os, sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import numpy as np
import json
import math
import shutil
import sklearn
import argparse
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import albumentations as albu
import copy
import random

from PIL import Image
from utils import utils
from Dataset_ import Pair_Whole_ImageDataset
from Dataset_ import Pair_ImageDataset
from Dataset_ import Pair_Registration_ImageDataset
from torch.utils.data import DataLoader
from utils.engine import train_one_epoch, evaluate
from skimage.metrics import structural_similarity
from model import Net

# === Code ===  

## Parameter

parser = argparse.ArgumentParser()
parser.add_argument('--in_fold', type=str, default='./02_Patch_data/', help='input images path')
parser.add_argument('--weight', type=str, default='./weight/0606_FCOS_real_STOD_reg_pseudo_Deform_SENet_seed115_recallbest', help='pre-trained weight for generate pseudo labels') 
parser.add_argument('--kfold', type=int, default=6, help='set k as validation fold')
parser.add_argument('--patch', type=str, default='True', help='patch data or not')
parser.add_argument('--name', type=str, default='save_pseudo', help='name of pseudo fold') 
parser.add_argument('--set_device', type=str, default='gpu', help='cuda or not')
parser.add_argument('--out_fold', type=str, default='./04_Pseudo_labels/', help='output pseudo labels fold')
opt = parser.parse_args()

def swap(a,b):
    temp = a
    a = b
    b = temp
    return a , b

def creat_folder(path):
    if not os.path.isdir(path):
        os.mkdir(path)
        
creat_folder(opt.out_fold)
creat_folder(opt.out_fold+opt.name+'/')
creat_folder(opt.out_fold+opt.name+'/images/')
creat_folder(opt.out_fold+opt.name+'/labels/')

images_dir = glob.glob(opt.in_fold+'data/images/*')
labels_dir = glob.glob(opt.in_fold+'data/labels/*') 
images_dir.sort()
labels_dir.sort()

'''
   # 切Kfold 預設第6個fold為validation
'''
      
all_dir=[]
for i in range(len(images_dir)):
    package = []
    package.append(images_dir[i])
    package.append(labels_dir[i])
    all_dir.append(package)

kfold = sklearn.model_selection.KFold(n_splits=10, shuffle=False, random_state=None)
kfold_train_data = [[],[],[],[],[],[],[],[],[],[]]
kfold_test_data = [[],[],[],[],[],[],[],[],[],[]]

i=0
for train_index, test_index in kfold.split(all_dir):
    for index in train_index:
        kfold_train_data[i].append(all_dir[index])
    for index in test_index:
        kfold_test_data[i].append(all_dir[index])
    i+=1

pseduo=[]
select = opt.kfold

for i in range(len(kfold_test_data[select])):
    img_paths = glob.glob(kfold_test_data[select][i][0]+'/*.png')
    lab_paths = glob.glob(kfold_test_data[select][i][1]+'/*.json')
    img_paths.sort()
    lab_paths.sort()
    
    for j in range(len(img_paths)):
        package=[]
        package.append(img_paths[j])
        package.append(lab_paths[j])
        pseduo.append(package)

pseduo_pair_images = []
pseduo_pair_labels = []

pseduo.sort()

patient_name_flag = 0
if(opt.patch == 'True'):
    patient_name_flag=-28
else:
    patient_name_flag=-26
    
# add first data
pseduo_pair_images.append([pseduo[0][0], pseduo[0][0]])
pseduo_pair_labels.append(pseduo[0][1])
for i in range(1,len(pseduo)):
    file_1 = pseduo[i-1][0][patient_name_flag:-18]
    file_2 = pseduo[i][0][patient_name_flag:-18] 
    image_t = []
    if(file_1 == file_2):
        image_t.append(pseduo[i][0])
        image_t.append(pseduo[i-1][0])
        pseduo_pair_images.append(image_t)
        pseduo_pair_labels.append(pseduo[i][1])
    else:
        image_t.append(pseduo[i][0])
        image_t.append(pseduo[i][0])
        pseduo_pair_images.append(image_t)
        pseduo_pair_labels.append(pseduo[i][1])
        
print('pair images number : ',len(pseduo_pair_images))

pair_pseduo=[]
for i in range(len(pseduo_pair_images)):
    package=[]
    package.append(pseduo_pair_images[i])
    package.append(pseduo_pair_labels[i])
    pair_pseduo.append(package)

if(opt.patch == 'True'):
    img_hegiht = 256
    img_width = 512
else:
    img_hegiht = 1024
    img_width = 1024
    
valid_transform = albu.Compose([
    albu.Resize(height=img_hegiht, width=img_width, p=1),
], bbox_params=albu.BboxParams(format='pascal_voc', label_fields=[]))

if(opt.patch=='True'):
    pseudo_dataset = Pair_ImageDataset(pair_pseduo, valid_transform, valid_transform)
else:
    pseudo_dataset = Pair_Whole_ImageDataset(pair_pseduo, valid_transform, valid_transform)

model = Net()
device = torch.device('cpu')
if(opt.set_device=='gpu' and torch.cuda.is_available()):
    print('==== GPU ====')
    device = torch.device('cuda')
else:
    print('==== CPU ====')

model.to(device)
model_path =  opt.weight
model.load_state_dict(torch.load(model_path))
model.eval()
    
blank_json_path = './BS00000_00000000_1234.json'
count = 0
patient_name_flag = 0
if(opt.patch=='True'):
    patient_name_flag=-28
else:
    patient_name_flag=-26

for i in range(len(pseudo_dataset)):
    img, img_2, time_interval, _, target = pseudo_dataset[i]    
    u_imgs = img.unsqueeze(0).cuda()
    u_imgs_2 = img_2.unsqueeze(0).cuda()
    time_interval = torch.tensor([time_interval]).cuda()


    loss_dict = model(u_imgs, u_imgs_2, time_interval)
    boxes = loss_dict[0]['boxes']
    labels = loss_dict[0]['labels']
    boxes_threshold = boxes[loss_dict[0]['scores']>0.5].cpu().detach().numpy().tolist()
    labels_threshold = labels[loss_dict[0]['scores']>0.5].cpu().numpy().tolist()

    json_path = pair_pseduo[i][1]
    img_path = pair_pseduo[i][0][0]
    img_2_path = pair_pseduo[i][0][1]
    save_img_path = opt.out_fold + opt.name + '/images/' + img_path[patient_name_flag:]
    save_json_path = opt.out_fold + opt.name + '/labels/' + json_path[patient_name_flag-1:]
    # 要創建一個新的img and json檔
    if(len(labels_threshold)>0):
        print('box : ', boxes_threshold)
        print('label : ', labels_threshold)

        # 先處理現在這張圖
        if(img_path[patient_name_flag:-4]!=json_path[patient_name_flag-1:-5] or os.path.isfile(save_json_path)):
            print("Invalid file :")
            print(img_path[patient_name_flag:-4])
            print(json_path[patient_name_flag-1:-5])
        else:

            shutil.copyfile(img_path, save_img_path)
            with open(blank_json_path,'rb') as f:
                    params = json.load(f)
                    dict_p = params
                    for row in range(len(labels_threshold)):
#                         print(labels_threshold[row])
                        dict_p['shapes'].append({'label': str(labels_threshold[row]), 'points':[boxes_threshold[row]]})

                    f.close()

            with open(save_json_path,'w') as r:

                json.dump(dict_p,r)

                r.close()

    else:
        count+=1
        print('No box : ',save_json_path)
print('No box number : ',count)
