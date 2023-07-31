import glob, cv2, os, sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse

import numpy as np
import json
from PIL import Image
import shutil

import matplotlib.pyplot as plt
import matplotlib.patches as patches

import albumentations as albu
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import copy
import xml.etree.ElementTree as ET

from tqdm import tqdm

from utils import utils
import torchvision
from utils.engine import train_one_epoch, evaluate
float_formatter = "{:.2f}".format

import torchvision
from torchvision.models.detection import *
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.fcos import FCOS

# === Code ===  

## Parameter

parser = argparse.ArgumentParser()
parser.add_argument('--in_folder', type=str, default='./Step1_data/lesion/', help='input lesion images path')
parser.add_argument('--train_path', type=str, default='./Crop_label_data/', help='trainging_data_path')
parser.add_argument('--out_folder', type=str, default='./Step3_crop_data/', help='output folder path')
opt = parser.parse_args()

def Check_jpg_and_json_all(images, labels):
    COUNT=0
    for i in range(len(images)):
        img_name = images[i][-26:-4] 
        lab_name = labels[i][-27:-5]
        if(img_name != lab_name):
            print(i)
            COUNT+=1
    
    return COUNT

def Check_jpg_and_xml_all(images, labels):
    COUNT=0
    for i in range(len(images)):
        img_name = images[i][-26:-4] 
        lab_name = labels[i][-26:-4]
        if(img_name != lab_name):
            print(i)
            COUNT+=1
    
    return COUNT

def creat_folder(path):
    if not os.path.isdir(path):
        os.mkdir(path)

def swap(a,b):
    temp = a
    a = b
    b = temp
    return a , b

def draw_bounding_box(pane, rect_coordinates,category_list):
    # Show bounding boxes

    # Create figure and axes
    fig,ax = plt.subplots(1,figsize=(12, 8))
    # Display the image
    ax.imshow(pane)

    # Create a Rectangle patch
    for e , c in zip(rect_coordinates,category_list):
        (x, y, xmax, ymax) = e 
        (x, y, w, h) = (x, y, xmax-x, ymax-y)
        rect = patches.Rectangle((x,y),w,h,linewidth=1,edgecolor='r',facecolor='none')
        ax.add_patch(rect)
        ax.text(x,y,category_id_to_name[c],color='blue',fontsize=9)
        
def FCOS_model():
    
    backbone = torchvision.models.detection.fcos_resnet50_fpn(weights=FCOS_ResNet50_FPN_Weights.DEFAULT,trainable_backbone_layers=2)
    num_classes = 2
    backbone.out_channels = num_classes
        
    return backbone
        
category_id_to_name = {1: 'Benign', 2: 'Probably benign', 3: 'Equivocal', 4: 'Probably malignant', 5: 'Malignant', 6: 'normal', 7: 'regress', 8:'Superscan'} 

''' 
    ============
    Labeled data for cropping people
    ============
''' 

train_images = glob.glob( opt.train_path+'crop_images/*.jpg')
train_labels = glob.glob( opt.train_path+'crop_labels/*.xml')
train_jsons = glob.glob( opt.train_path +'crop_jsons/*.json')

train_images.sort()
train_labels.sort()
train_jsons.sort()
print("Training data (images/xml/labels): ", len(train_images), len(train_labels), len(train_jsons))
print('Training File incorrect : ', Check_jpg_and_xml_all(train_images, train_labels))

''' 
    ============
    Copy 1 份沒resize的檔案作為Output的位置
    ============
''' 

now_images = glob.glob(opt.in_folder+'images/*.jpg')
now_labels = glob.glob(opt.in_folder+'labels/*.json')
now_images.sort()
now_labels.sort()
print("Lesion data (images/labels) : ",len(now_images), len(now_labels))

copy_path = opt.out_folder
creat_folder(copy_path)
creat_folder(copy_path + 'images')
creat_folder(copy_path + 'labels')
# creat_folder(copy_path + 'images_result')
# creat_folder(copy_path + 'labels_result')

for i in range(len(now_images)):
    label_iter = json.load(open(now_labels[i]))
    bbox_list = []
    category_list = []
    flag = 0
    img_name = now_images[i][-26:]
    lab_name = now_labels[i][-27:]
    
#     shutil.copyfile(now_images[i], copy_path+'images'+img_name)
    shutil.copyfile(now_labels[i], copy_path+'labels'+lab_name)
#     shutil.copyfile(now_labels[i], copy_path+'labels_result'+lab_name)

''' 
    ============
    Dataset
    ============
''' 

infer_images = glob.glob(opt.in_folder + 'images/*.jpg')
infer_labels = glob.glob(opt.in_folder + 'labels/*.json')
infer_jsons = glob.glob(opt.out_folder + 'labels/*.json')

infer_images.sort()
infer_labels.sort()
infer_jsons.sort()

train_transform = albu.Compose([
    albu.HorizontalFlip(p=0.5),
    albu.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.1, always_apply=False, p=0.2),
    albu.Resize(height=1024, width=1024, p=1),  # can try height=1050, width=790
   
], bbox_params=albu.BboxParams(format='pascal_voc', label_fields=[]))


valid_transform = albu.Compose([
#     albu.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    albu.Resize(height=1024, width=1024, p=1),
], bbox_params=albu.BboxParams(format='pascal_voc', label_fields=[]))

train=[]
for i in range(len(train_images)):
    package=[]
    package.append(train_images[i])
    package.append(train_labels[i])
    package.append(train_jsons[i])
    train.append(package)
    
print("Training data : ",len(train))

infer=[]
for i in range(len(infer_images)):
    package=[]
    package.append(infer_images[i])
    package.append(infer_labels[i])
    package.append(infer_jsons[i])
    infer.append(package)
    
print("Cropping data : ",len(infer))

class ImageDataset(Dataset):
    def __init__(self, data, augmentation):
        super().__init__()
        self.images = [x[0] for x in data]
        self.labels = [x[1] for x in data]
        self.augmentation = augmentation

    # 讓這個class可以用indexing的方式取東西
    def __getitem__(self, index):

        # get path
        img = self.images[index]
        label_iter = ET.parse(self.labels[index])
        bbox_list = []
        category_list = []
        
        # read image
        img = cv2.imread(img).astype('float32')
        img = img/255
#         img = cv2.addWeighted(img, 1.5, img, -0.5, 0)
        h,w,c = img.shape

        root = label_iter.getroot()
        bbox_list = []
        bbox = [int(point.text) for point in root[6][4]]
        
        category = 1

        if(bbox[0]>bbox[2]): 
            bbox[0],bbox[2] = swap(bbox[0],bbox[2])
        if(bbox[1]>bbox[3]): 
            bbox[1],bbox[3] = swap(bbox[1],bbox[3])
        bbox_list.append([bbox[0], bbox[1], bbox[2], bbox[3]])
        category_list.append(category)
        
        boxes = torch.as_tensor(bbox_list, dtype=torch.float32)

        # apply augmentations
        if self.augmentation:
            sample = self.augmentation(image=img, bboxes=bbox_list)
            img, bboxes = sample['image'], sample['bboxes']    
        
        # 增加channel這個維度
#         img = img[:,:,np.newaxis]
#         img = np.concatenate((img,np.concatenate((img,img),axis=2)),axis=2)
        img = torch.tensor(img).float().permute(2,0,1) #.unsqueeze(0)

        bbox_trans = []
        for bbox_temp in bboxes:
            one_bbox = []
            one_bbox+=(bbox_temp[0], bbox_temp[1], bbox_temp[2],bbox_temp[3])
            bbox_trans.append(one_bbox)
        
        bbox_trans_np = np.array(bbox_trans)
        area = (bbox_trans_np[:, 3] - bbox_trans_np[:, 1]) * (bbox_trans_np[:, 2] - bbox_trans_np[:, 0])
        image_id = torch.tensor([index])
        iscrowd = torch.zeros((len(category_list),), dtype=torch.int64)
        
        target = {}
        target["area"] = torch.as_tensor(area)
        target['boxes'] = torch.as_tensor(bbox_trans, dtype=torch.float32)
        target['labels'] = torch.as_tensor(category_list)
        target["image_id"] = image_id
        target["iscrowd"] = iscrowd


        return img, target
  
    def __len__(self):
        #有多少筆數的資料
        return len(self.images) 

class Infer_ImageDataset(Dataset):
    def __init__(self, data, augmentation):
        super().__init__()
        self.images = [x[0] for x in data]
        self.labels = [x[1] for x in data]
        self.json_path = [x[2] for x in data]
        self.augmentation = augmentation

    # 讓這個class可以用indexing的方式取東西
    def __getitem__(self, index):

        # get path
        img = self.images[index]
        label_iter = json.load(open(self.labels[index]))
        json_path = self.json_path[index]
        # read image
        img = cv2.imread(img).astype('float32')
        img = img/255
        h,w,c = img.shape
        bbox_list = []
        category_list = []

        for json_t in label_iter['shapes']:
            bbox = [float(point) for val in json_t['points'] for point in val ]
            category = int(json_t['label'][0])+1
            
            # Get the left-top and right-bottom points
            if(bbox[0]>bbox[2]): 
                bbox[0],bbox[2] = swap(bbox[0],bbox[2])
            if(bbox[1]>bbox[3]): 
                bbox[1],bbox[3] = swap(bbox[1],bbox[3])
            bbox_list.append([bbox[0], bbox[1], bbox[2], bbox[3]])
            category_list.append(category)
        
        boxes = torch.as_tensor(bbox_list, dtype=torch.float32)
            
        # apply augmentations
        if self.augmentation:
            sample = self.augmentation(image=img, bboxes=bbox_list)
            img, bboxes = sample['image'], sample['bboxes']    
        
        # 增加channel這個維度
        img = torch.tensor(img).float().permute(2,0,1) #.unsqueeze(0)

        bbox_trans = []
        for bbox_temp in bboxes:
            one_bbox = []
            one_bbox+=(bbox_temp[0], bbox_temp[1], bbox_temp[2],bbox_temp[3])
            bbox_trans.append(one_bbox)
        
        bbox_trans_np = np.array(bbox_trans)
        area = (bbox_trans_np[:, 3] - bbox_trans_np[:, 1]) * (bbox_trans_np[:, 2] - bbox_trans_np[:, 0])
        image_id = torch.tensor([index])
        iscrowd = torch.zeros((len(category_list),), dtype=torch.int64)

        target = {}
        target["area"] = torch.as_tensor(area)
        target['boxes'] = torch.as_tensor(bbox_trans, dtype=torch.float32)
        target['labels'] = torch.as_tensor(category_list)
        target["image_id"] = image_id
        target["iscrowd"] = iscrowd
        
        return img, target, json_path 
  
    def __len__(self):
        #有多少筆數的資料
        return len(self.images) 
    
train_dataset = ImageDataset(train, train_transform)  
infer_dataset = Infer_ImageDataset(infer,valid_transform)

''' 
    ============
    Model
    ============
''' 

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
# device = torch.device('cpu')

model = FCOS_model()
model.to(device)
model_path = './weight/crop'
model.load_state_dict(torch.load(model_path))
model.eval()

''' 
    ============
    Crop process
    ============
''' 

# print("Cropping data : ",len(infer_dataset))

save_path = opt.out_folder

for i in range(len(infer_dataset)):
#     print(i)
    data = infer_dataset[i]
    img, target, json_path = data
    imgs = img.to(device)
    u_imgs = imgs.unsqueeze(0)
    loss_dict = model(u_imgs)
    pre_bbox = loss_dict[0]['boxes']
    point = loss_dict[0]['boxes'][0].detach().cpu().numpy().astype(int)
    Crop_IMG = imgs.cpu()[0][point[1]:point[3], point[0]:point[2]].numpy()*255
    name = json_path[-26:-5] + '.png'
    
#     寫入圖檔
    cv2.imwrite(save_path + 'images/' + name , Crop_IMG)
#     draw_bounding_box(Crop_IMG, bbox_list,category_list)
    dict_p={}
    with open(json_path,'rb') as f:
    #定义为只读模型，并定义名称为f
    
        params = json.load(f)
        for i in range(len(params['shapes'])):
            # x0
            params['shapes'][i]['points'][0][0]=target['boxes'].numpy()[i][0].item()-loss_dict[0]['boxes'][0].detach().cpu().numpy()[0]
            # x1
            params['shapes'][i]['points'][1][0]=target['boxes'].numpy()[i][2].item()-loss_dict[0]['boxes'][0].detach().cpu().numpy()[0]
            # y0
            params['shapes'][i]['points'][0][1]=target['boxes'].numpy()[i][1].item()-loss_dict[0]['boxes'][0].detach().cpu().numpy()[1]
            # y1
            params['shapes'][i]['points'][1][1]=target['boxes'].numpy()[i][3].item()-loss_dict[0]['boxes'][0].detach().cpu().numpy()[1]

        dict_p = params
        f.close()

    with open(json_path,'w') as r:
        #定义为写模式，名称定义为r

        json.dump(dict_p,r)
        #将dict写入名称为r的文件中

        r.close()
        #加载json文件中的内容给params
print("Crop Done")
        
''' 
    ============
    將crop後的結果轉成1024x1024 方便切Patch
    ============
''' 
resize_images = glob.glob(opt.out_folder+'images/*.png')
resize_labels = glob.glob(opt.out_folder+'labels/*.json')

resize_images.sort()
resize_labels.sort()
print("Resize data (images/labels) : ", len(resize_images), len(resize_labels))

resize=[]
for i in range(len(resize_images)):
    package=[]
    package.append(resize_images[i])
    package.append(resize_labels[i])
    package.append(resize_labels[i])
    resize.append(package)
    
# from RandAugment import RandAugment
import torchvision.transforms as transforms
resize_transform = albu.Compose([
    albu.Resize(height=1024, width=1024, p=1),  # can try height=1050, width=790
], bbox_params=albu.BboxParams(format='pascal_voc', label_fields=[]))

class resize_ImageDataset(Dataset):
    def __init__(self, data, augmentation):
        super().__init__()
        self.images = [x[0] for x in data]
        self.labels = [x[1] for x in data]
        self.json_paths = [x[2] for x in data]
        self.augmentation = augmentation

    # 讓這個class可以用indexing的方式取東西
    def __getitem__(self, index):

        # get path
        img = self.images[index]
        label_iter = json.load(open(self.labels[index]))
        json_path = self.json_paths[index]
        bbox_list = []
        category_list = []
        
        # read image
        img = cv2.imread(img).astype('float32')
        img = img/255
        h,w,c = img.shape
        
        # Check if bbox be cut
        flag = 0
        
        # read lable
        for json_t in label_iter['shapes']:
            bbox = [float(point) for val in json_t['points'] for point in val ]
            c_temp = json_t['label']
            category = int(c_temp[0])
            if(bbox[0]>bbox[2]): 
                bbox[0],bbox[2] = swap(bbox[0],bbox[2])
            if(bbox[1]>bbox[3]): 
                bbox[1],bbox[3] = swap(bbox[1],bbox[3])
            bbox_list.append([bbox[0], bbox[1], bbox[2], bbox[3]])
            category_list.append(category)
        
        
        boxes = torch.as_tensor(bbox_list, dtype=torch.float32)

        # apply augmentations
        if self.augmentation:
            sample = self.augmentation(image=img, bboxes=bbox_list)
            img, bboxes = sample['image'], sample['bboxes']    
        
        img = torch.tensor(img).float().permute(2,0,1) #.unsqueeze(0)

        bbox_trans = []
        for bbox_temp in bboxes:
            one_bbox = []
            one_bbox+=(bbox_temp[0], bbox_temp[1], bbox_temp[2],bbox_temp[3])
            bbox_trans.append(one_bbox)
        
        bbox_trans_np = np.array(bbox_trans)
        area = (bbox_trans_np[:, 3] - bbox_trans_np[:, 1]) * (bbox_trans_np[:, 2] - bbox_trans_np[:, 0])
        image_id = torch.tensor([index])
        iscrowd = torch.zeros((len(category_list),), dtype=torch.int64)
        
        target = {}
        target["area"] = torch.as_tensor(area)
        target['boxes'] = torch.as_tensor(bbox_trans, dtype=torch.float32)
        target['labels'] = torch.as_tensor(category_list)
        target["image_id"] = image_id
        target["iscrowd"] = iscrowd


        return img, target, json_path
  
    def __len__(self):
        #有多少筆數的資料
        return len(self.images) 

resize_dataset = resize_ImageDataset(resize, resize_transform)

''' 
    ============
    Generate new json with 1024x1024 bbox
    ============
''' 

toPIL = transforms.ToPILImage()
for k in range(len(resize_dataset)):
    if(k%100==0):
        print("====    ", k , " done    ====")
    img, target, json_path = resize_dataset[k]
    bbox_list = target['boxes'].numpy().tolist()
    new_img_PIL = toPIL(img)
    name = resize_images[k][-26:]
    new_img_PIL.save(opt.out_folder+'images' + name)
    with open(json_path,'rb') as f:
    #定义为只读模型，并定义名称为f

        params = json.load(f)

        for i in reversed(range(len(params['shapes']))):
#             print(i)
#             print(params['shapes'][i]['points'])
            # x0
#             print(type(params['shapes'][i]['points'][0][0]))
            params['shapes'][i]['points'][0][0]=bbox_list[i][0]
            # x1
            params['shapes'][i]['points'][1][0]=bbox_list[i][2]
            # y0
            params['shapes'][i]['points'][0][1]=bbox_list[i][1]
            # y1
            params['shapes'][i]['points'][1][1]=bbox_list[i][3]

        dict_p = params
        f.close()
    
    with open(json_path,'w') as r:
        #定义为写模式，名称定义为r

        json.dump(dict_p,r)
        #将dict写入名称为r的文件中

        r.close()
            #加载json文件中的内容给params
print("Resize Done")
