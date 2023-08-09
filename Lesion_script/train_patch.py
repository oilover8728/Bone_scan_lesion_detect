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
from Dataset_ import Pair_ImageDataset
from Dataset_ import Pseudo_Pair_ImageDataset
from torch.utils.data import DataLoader
from utils.engine import train_one_epoch, evaluate
from skimage.metrics import structural_similarity
from model import Net
from torch.optim.lr_scheduler import CosineAnnealingLR
from warmup_scheduler import GradualWarmupScheduler

# === Code ===  

## Parameter

parser = argparse.ArgumentParser()
parser.add_argument('--in_fold', type=str, default='./02_Patch_data/', help='input images path') 
parser.add_argument('--pseudo', type=str, default='False', help='Train with pseudo images or not') 
parser.add_argument('--seed', type=int, default=115, help='set random seed')
parser.add_argument('--kfold', type=int, default=6, help='set k as validation fold')
parser.add_argument('--epoch', type=int, default=1, help='set epoch number') 
parser.add_argument('--batch_size', type=int, default=1, help='set batch size') 
parser.add_argument('--name', type=str, default='save_model', help='save model name') 
parser.add_argument('--train', type=str, default='True', help='Train/Valid')
parser.add_argument('--weight', type=str, default='./weight/0606_FCOS_real_STOD_reg_pseudo_Deform_SENet_seed115_recallbest', help='pre-trained weight')
parser.add_argument('--set_device', type=str, default='gpu', help='cuda or not')
parser.add_argument('--out_weight', type=str, default='./weight/', help='output training weight path')
opt = parser.parse_args()


random.seed(opt.seed)
np.random.seed(opt.seed)
torch.manual_seed(opt.seed)
torch.cuda.manual_seed(opt.seed) 
torch.cuda.manual_seed_all(opt.seed)
def seed_worker(worker_id):
    numpy.random.seed(worker_seed)
    random.seed(worker_seed)

g = torch.Generator()
g.manual_seed(opt.seed)

def swap(a,b):
    temp = a
    a = b
    b = temp
    return a , b

def get_ap30_50_95(coco_eval):
    # https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/cocoeval.py
    cat_30 = []
    cat_50 = []
    eval_dict = coco_eval.coco_eval['bbox'].eval
    p = coco_eval.coco_eval['bbox'].params
    
    areaRng='all'
    areaRnd_small ='small'
    maxDets=100
    aind = [i for i, aRng in enumerate(p.areaRngLbl) if aRng == areaRng]
    aind_small = [i for i, aRng in enumerate(p.areaRngLbl) if aRng == areaRnd_small]
    mind = [i for i, mDet in enumerate(p.maxDets) if mDet == maxDets]
        
    # s[0] => ap50
    # reference threshold [0.3, 0.35, 0.4, 0.45, 0.5 , 0.55, 0.6 , 0.65, 0.7 , 0.75, 0.8 , 0.85, 0.9 , 0.95]
    s = eval_dict['precision']
    # T,R,K,A,M
    r = eval_dict['recall']
    
    T,R,K,A,M = s.shape
    # T,K,A,M
    for i in range(K):
        cat_30.append(s[0][:,i,aind,mind].mean())
        cat_50.append(s[4][:,i,aind,mind].mean())
    
    s_30 = s[0][:,:,aind,mind]
    s_50 = s[4][:,:,aind,mind]
    s_75 = s[9][:,:,aind,mind]
    r_30 = s[0][:,:,aind,mind]
    r_50 = r[4][:, aind, mind]
    

    s_30_95 = s[:, :, :, aind, mind]
    s_50_95 = s[4:, :, :, aind, mind]
    r_30_95 = r[:, :, aind, mind]
    r_50_95 = r[4:, :, aind, mind]
    return s_30.mean(), s_50.mean(), s_75.mean(), s_30_95.mean(), s_50_95.mean(), r_50.mean(), r_30_95.mean(), r_50_95.mean(), cat_30, cat_50

def train_a_epoch(model, optimizer, data_loader, device, epoch, print_freq):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)

    lr_scheduler = None
    if epoch == 0:
        warmup_factor = 1. / 1000
        warmup_iters = min(1000, len(data_loader) - 1)

        lr_scheduler = utils.warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor)

    for images, images_2, times, loss_weights,  targets in metric_logger.log_every(data_loader, print_freq, header):
        images = list(image.to(device) for image in images)
        images_2 = list(image_2.to(device) for image_2 in images_2)
        times = torch.tensor(list(times)).to(device)
        loss_weights = torch.tensor(list(loss_weights)).to(device)
        
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, images_2, times, loss_weights, targets)

        losses = sum(loss for loss in loss_dict.values())

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())

        loss_value = losses_reduced.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        if lr_scheduler is not None:
            lr_scheduler.step()

        metric_logger.update(loss=losses_reduced, **loss_dict_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    return metric_logger

training_reports = {
    'tr_loss' : [],
    'tr_cls_loss' : [],
    'tr_bbox_loss' : [],
    'val_AP_30' : [],
    'val_AP_50' : [],
    'val_AP_75' : [],
    'val_AP_50_95' : [],
    'val_AR_50_95' : [],
    'val_AR_50' : [],
    'lr' : [],
    'best_epoch' : -1,
    'seed' : [],
}

'''
    # fixed train-valid data
'''

images_dir = glob.glob(opt.in_fold+'data/images/*')
labels_dir = glob.glob(opt.in_fold+'data/labels/*') 
images_dir.sort()
labels_dir.sort()

print('Data (patients) : ',len(images_dir))

if(opt.pseudo == 'True'):
    pseudo_images = glob.glob(opt.in_fold+'pseudo/images/*.png')
    pseudo_labels = glob.glob(opt.in_fold+'pseudo/labels/*.json')
    pseudo_images.sort()
    pseudo_labels.sort()
    print('Pseudo Label (images/labels) : ',len(pseudo_images), len(pseudo_labels))

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

# which fold is validation set
select = opt.kfold
# 整理Train data
if(opt.train=='True'):
    train=[]
    for i in range(len(kfold_train_data[select])):
        img_paths = glob.glob(kfold_train_data[select][i][0]+'/*.png')
        lab_paths = glob.glob(kfold_train_data[select][i][1]+'/*.json')
        img_paths.sort()
        lab_paths.sort()

        for j in range(len(img_paths)):
            package=[]
            package.append(img_paths[j])
            package.append(lab_paths[j])
            train.append(package)

    train_pair_images = []
    train_pair_labels = []
    train.sort()
    # add first data
    train_pair_images.append([train[0][0], train[0][0]])
    train_pair_labels.append(train[0][1])
    for i in range(1,len(train)):
        file_1 = train[i-1][0][-28:-18] 
        file_2 = train[i][0][-28:-18] 
        image_t = []
        if(file_1 == file_2):
            image_t.append(train[i][0])
            image_t.append(train[i-1][0])
            train_pair_images.append(image_t)
            train_pair_labels.append(train[i][1])
        else:
            image_t.append(train[i][0])
            image_t.append(train[i][0])
            train_pair_images.append(image_t)
            train_pair_labels.append(train[i][1])

    print(' === Train stage === ')
    print('pair number : ',len(train_pair_images))

    pair_train=[]
    for i in range(len(train_pair_images)):
        package=[]
        package.append(train_pair_images[i])
        package.append(train_pair_labels[i])
        pair_train.append(package)
    
valid=[]
for i in range(len(kfold_test_data[select])):
    img_paths = glob.glob(kfold_test_data[select][i][0]+'/*.png')
    lab_paths = glob.glob(kfold_test_data[select][i][1]+'/*.json')
    img_paths.sort()
    lab_paths.sort()
    
    for j in range(len(img_paths)):
        package=[]
        package.append(img_paths[j])
        package.append(lab_paths[j])
        valid.append(package)

valid_pair_images = []
valid_pair_labels = []

valid.sort()
# add first data
valid_pair_images.append([valid[0][0], valid[0][0]])
valid_pair_labels.append(valid[0][1])
for i in range(1,len(valid)):
    file_1 = valid[i-1][0][-28:-18]
    file_2 = valid[i][0][-28:-18] 
    image_t = []
    if(file_1 == file_2):
        image_t.append(valid[i][0])
        image_t.append(valid[i-1][0])
        valid_pair_images.append(image_t)
        valid_pair_labels.append(valid[i][1])
    else:
        image_t.append(valid[i][0])
        image_t.append(valid[i][0])
        valid_pair_images.append(image_t)
        valid_pair_labels.append(valid[i][1])
        
print(' === Valid stage ===')
print('pair number : ',len(valid_pair_images))

pair_valid=[]
for i in range(len(valid_pair_images)):
    package=[]
    package.append(valid_pair_images[i])
    package.append(valid_pair_labels[i])
    pair_valid.append(package)
    
'''
    # Pseudo label(Optinal)
'''
if(opt.pseudo == 'True'):
    pseudo_train=[]
    for i in range(len(pseudo_images)):
        package = []
        package.append(pseudo_images[i])
        package.append(pseudo_labels[i])
        pseudo_train.append(package)

    ## pair
    pseudo_train_pair_images = []
    pseudo_train_pair_labels = []

    pseudo_train.sort()
    # add first data
    pseudo_train_pair_images.append([pseudo_train[0][0], pseudo_train[0][0]])
    pseudo_train_pair_labels.append(pseudo_train[0][1])
    for i in range(1,len(pseudo_train)):
        file_1 = pseudo_train[i-1][0][-28:-18] 
        file_2 = pseudo_train[i][0][-28:-18] 
        image_t = []
        if(file_1 == file_2):
            image_t.append(pseudo_train[i][0])
            image_t.append(pseudo_train[i-1][0])
            pseudo_train_pair_images.append(image_t)
            pseudo_train_pair_labels.append(pseudo_train[i][1])
        else:
            image_t.append(pseudo_train[i][0])
            image_t.append(pseudo_train[i][0])
            pseudo_train_pair_images.append(image_t)
            pseudo_train_pair_labels.append(pseudo_train[i][1])

    print('Pseudo data : ',len(pseudo_train_pair_images))

    pseudo_pair_train=[]
    for i in range(len(pseudo_train_pair_images)):
        package=[]
        package.append(pseudo_train_pair_images[i])
        package.append(pseudo_train_pair_labels[i])
        pseudo_pair_train.append(package)

# Transform and Dataset 
img_hegiht = 256
img_width = 512
train_transform = albu.Compose([
     albu.GlassBlur(p=0.2),
     albu.GaussNoise(var_limit=(10,50),p=0.2), 
     albu.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, always_apply=False, p=0.2),
     albu.HorizontalFlip(p=0.2),
     albu.ShiftScaleRotate(scale_limit=0.1, rotate_limit=(-5,5),p=0.2),    
    
    albu.Resize(height=img_hegiht, width=img_width, p=1),
], bbox_params=albu.BboxParams(format='pascal_voc', label_fields=[]))

train2_transform = albu.Compose([
     albu.GlassBlur(p=0.2),
     albu.GaussNoise(var_limit=(10,50),p=0.2), 
     albu.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, always_apply=False, p=0.2),
     albu.HorizontalFlip(p=0.2),
     albu.ShiftScaleRotate(scale_limit=0.1),
    
    albu.Resize(height=img_hegiht, width=img_width, p=1),
], bbox_params=albu.BboxParams(format='pascal_voc', label_fields=[]))


valid_transform = albu.Compose([
    albu.Resize(height=img_hegiht, width=img_width, p=1),
], bbox_params=albu.BboxParams(format='pascal_voc', label_fields=[]))

if(opt.train == 'True'):
    train_dataset = Pair_ImageDataset(pair_train, train_transform, train2_transform)
    print("Dataset (Train) : ",len(train_dataset))

valid_dataset = Pair_ImageDataset(pair_valid, valid_transform, valid_transform)
if(opt.pseudo == 'True'):
    pseudo_train_dataset = Pseudo_Pair_ImageDataset(pseudo_pair_train, train_transform, train2_transform)
    print("Dataset (Pseudo) : ",len(pseudo_train_dataset))


print("Dataset (Valid) : ",len(valid_dataset))

model = Net()
device = torch.device('cpu')
if(opt.set_device=='gpu' and torch.cuda.is_available()):
    print('==== GPU ====')
    device = torch.device('cuda')
else:
    print('==== CPU ====')
    
torch.cuda.empty_cache()
category_id_to_name = {1: 'Benign', 2: 'Equivocal', 3: 'Malignant'} 
category_id_to_index = {1: '1_B', 2: '2_E', 3: '3_M'} 

num_epochs = opt.epoch
warm_up_epoches = 1
lr = 1e-4
batch = opt.batch_size
num_workers = 0
model.to(device)
# Optimizer
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.AdamW(model.parameters(), lr = lr, weight_decay=0.01)
# Scheduler
scheduler_steplr = CosineAnnealingLR(optimizer, num_epochs)
scheduler_warmup = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=warm_up_epoches, after_scheduler=scheduler_steplr)


if(opt.train == 'True'):
    train_Dataloader = DataLoader(dataset = train_dataset, batch_size = batch, shuffle = True, num_workers = num_workers, collate_fn=utils.collate_fn, worker_init_fn=seed_worker, generator=g)
if(opt.pseudo == 'True'):
    pseudo_train_Dataloader = DataLoader(dataset = pseudo_train_dataset, batch_size = batch, shuffle = True, num_workers = num_workers, collate_fn=utils.collate_fn, worker_init_fn=seed_worker, generator=g)
valid_Dataloader = DataLoader(dataset = valid_dataset, batch_size = batch, shuffle = False, num_workers = num_workers, collate_fn=utils.collate_fn, worker_init_fn=seed_worker, generator=g)

lr_rate = []
# load model
# model_path = './weight/'
# model.load_state_dict(torch.load(model_path))

Name = opt.name

max_map_score = 0
Best_cat_AP50 = [0,0,0,0,0]
save_best_path = opt.out_weight + Name + '_best'
save_path = opt.out_weight + Name

if(opt.train=='True'):
    for epoch in range(num_epochs):

        # train for one epoch, printing every 10 iterations
        if(opt.pseudo == 'True'):
            print("Pseudo stage :")
            metric_logger = train_a_epoch(model, optimizer, pseudo_train_Dataloader, device, epoch, print_freq=100)
        print("Ground truth stage :")
        metric_logger = train_a_epoch(model, optimizer, train_Dataloader, device, epoch, print_freq=100)

        # Scheduler
        scheduler_warmup.step()
        lr_rate.append(optimizer.param_groups[0]['lr'])
        # evaluate on the test dataset
        coco_eval = evaluate(model, valid_Dataloader, device=device)
        ap30, ap50, ap75, map30_95, map50_95, mar_50, mar30_95, mar50_95, cat30_score, cat50_score = get_ap30_50_95(coco_eval)
        # track training stats
        training_reports['tr_loss'].append(metric_logger.loss.avg)
        training_reports['lr'].append(optimizer.param_groups[0]['lr'])
        #     training_reports['tr_cls_loss'].append(metric_logger.loss_classifier.avg)
        #     training_reports['tr_bbox_loss'].append(metric_logger.loss_box_reg.avg)
        training_reports['tr_cls_loss'].append(metric_logger.classification.avg)
        training_reports['tr_bbox_loss'].append(metric_logger.bbox_regression.avg)
        training_reports['val_AP_30'].append(ap30)
        training_reports['val_AP_50'].append(ap50)
        training_reports['val_AP_75'].append(ap75)
        training_reports['val_AP_50_95'].append(map50_95)
        training_reports['val_AR_50_95'].append(mar50_95)
        training_reports['val_AR_50'].append(mar_50)

        # if max_map_score < map_score:
        Best_cat_AP50 = cat50_score
        print('==== mAP50 for each class ====')
        for i in range(1,len(cat50_score)+1):
            print(category_id_to_name[i], ' : ', cat50_score[i-1])
        if(ap50 > max_map_score):
            max_map_score = ap50
            training_reports['best_epoch'] = epoch
            torch.save(model.state_dict(), save_best_path)
        torch.save(model.state_dict(), save_path)
        print('Model Save!!')
        print("============================================================")
    print('==== BEST model result ====')
    model_path = save_best_path
    model.load_state_dict(torch.load(model_path))
    model = model.to(device)
    model.eval()

    valid_Dataloader = DataLoader(dataset = valid_dataset, batch_size = batch, shuffle = False, num_workers = 0, collate_fn=utils.collate_fn, worker_init_fn=seed_worker, generator=g)
    model.eval()
    coco_eval = evaluate(model, valid_Dataloader, device=device)
    ap30, ap50, ap75, map30_95, map50_95, mar_50, mar30_95, mar50_95, cat30_score, cat50_score = get_ap30_50_95(coco_eval)
    print("mAP75 : ",ap75)
    print("Recall : " ,mar_50)
    print('==== mAP50 for each class ====')
    for i in range(1,len(cat50_score)+1):
        print(category_id_to_name[i], ' : ', cat50_score[i-1])
else:
    print('==== Validation result ====')
    if(opt.weight != 'None'):
        model_path = opt.weight
        model.load_state_dict(torch.load(model_path))
    model = model.to(device)
    model.eval()

    valid_Dataloader = DataLoader(dataset = valid_dataset, batch_size = batch, shuffle = False, num_workers = 0, collate_fn=utils.collate_fn, worker_init_fn=seed_worker, generator=g)
    model.eval()
    coco_eval = evaluate(model, valid_Dataloader, device=device)
    ap30, ap50, ap75, map30_95, map50_95, mar_50, mar30_95, mar50_95, cat30_score, cat50_score = get_ap30_50_95(coco_eval)
    print("mAP75 : ",ap75)
    print("Recall : " ,mar_50)
    print('==== mAP50 for each class ====')
    for i in range(1,len(cat50_score)+1):
        print(category_id_to_name[i], ' : ', cat50_score[i-1])