import glob, cv2, os, sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import numpy as np
import json
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torchvision
import time

from PIL import Image
from utils import utils
from utils.engine import train_one_epoch, evaluate
from skimage.metrics import structural_similarity
from model import Net

# === Code ===  

## Parameter

parser = argparse.ArgumentParser()
parser.add_argument('--img', type=str, default='./example/BS00145_20150923_c564.png', help='input indiviual image') #img/fold
parser.add_argument('--img_2', type=str, default='./example/BS00145_20160219_fb7b.png', help='input indiviual image') #img/fold
parser.add_argument('--patch', type=str, default='not', help='input format is patch or not')
parser.add_argument('--patch_img', type=str, default='./example/BS00145_13_20150923_c564.png', help='input indiviual image') #img/fold
parser.add_argument('--patch_img_2', type=str, default='./example/BS00145_13_20160219_fb7b.png', help='input indiviual image') #img/fold
parser.add_argument('--weight', type=str, default='./weight/0606_FCOS_real_STOD_reg_pseudo_Deform_SENet_seed115_recallbest', help='pre-trained weight')
parser.add_argument('--threshold', type=int, default=0.56, help='pre-trained weight')
parser.add_argument('--set_device', type=str, default='gpu', help='cuda or not')
parser.add_argument('--out_folder', type=str, default='./result/', help='output training weight folder path')
opt = parser.parse_args()

category_id_to_name = {0:'Normal' , 1: 'Benign', 2: 'Equivocal', 3: 'Malignant'} 
category_id_to_index = {0: 'N', 1: '1_B', 2: '2_E', 3: '3_M'} 

def draw_bounding_box(pane, rect_coordinates,category_list, name, text=False):
    # Show bounding boxes

    # Create figure and axes
    fig,ax = plt.subplots(1,figsize=(12, 8))
    # Display the image
    ax.imshow(pane,cmap='gray')

    # Create a Rectangle patch
    for e , c in zip(rect_coordinates,category_list):
        (x, y, xmax, ymax) = e 
        (x, y, w, h) = (x, y, xmax-x, ymax-y)
        rect = patches.Rectangle((x,y),w,h,linewidth=1,edgecolor='r',facecolor='none')
        ax.add_patch(rect)
        ax.text(x,y,category_id_to_index[c],color='blue',fontsize=9)
    
    plt.savefig(opt.out_folder + name)
    plt.close()
    
def image_registration(img1_color,img2_color):
#     img1_color = img1
#     img2_color = img2

    # Convert to grayscale.
    img1 = cv2.cvtColor(img1_color, cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(img2_color, cv2.COLOR_BGR2GRAY)
    height, width = img2.shape

    # Create ORB detector with 5000 features.
    orb_detector = cv2.ORB_create(1000)

    # Find keypoints and descriptors.
    # The first arg is the image, second arg is the mask
    #  (which is not required in this case).
    kp1, d1 = orb_detector.detectAndCompute(img1, None)
    kp2, d2 = orb_detector.detectAndCompute(img2, None)

    # Match features between the two images.
    # We create a Brute Force matcher with
    # Hamming distance as measurement mode.
    if type(d1)!=type(None) and type(d2)!=type(None):

        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck = True)

        # Match the two sets of descriptors.
        matches = matcher.match(d1, d2)

        # Sort matches on the basis of their Hamming distance.
        matches.sort(key = lambda x: x.distance)

        # Take the top 90 % matches forward.
        matches = matches[:int(len(matches)*0.9)]
        no_of_matches = len(matches)

        # Define empty matrices of shape no_of_matches * 2.
        p1 = np.zeros((no_of_matches, 2))
        p2 = np.zeros((no_of_matches, 2))

        for i in range(len(matches)):
            p1[i, :] = kp1[matches[i].queryIdx].pt
            p2[i, :] = kp2[matches[i].trainIdx].pt

        n_p, _ = p1.shape
        if(n_p<10):
            return img2_color
        else:
            
            # Find the homography matrix.
            homography, mask = cv2.findHomography(p1, p2, cv2.RANSAC)
            
#             print("hemography:",homography) # In my case this was None, which is why I got an error
#             print("h:",height,"w:", width) # Check width/height seems correct
            # Use this matrix to transform the
            # colored image wrt the reference image.
            transformed_img = cv2.warpPerspective(img1_color,
                                homography, (width, height))

            return transformed_img
    else:
        return img2_color
def get_IoU(pred_bbox, gt_bbox):
    """
    return iou score between pred / gt bboxes
    :param pred_bbox: predict bbox coordinate
    :param gt_bbox: ground truth bbox coordinate
    :return: iou score
    """

    # bbox should be valid, actually we should add more judgements, just ignore here...
    # assert ((abs(pred_bbox[2] - pred_bbox[0]) > 0) and
    #         (abs(pred_bbox[3] - pred_bbox[1]) > 0))
    # assert ((abs(gt_bbox[2] - gt_bbox[0]) > 0) and
    #         (abs(gt_bbox[3] - gt_bbox[1]) > 0))

    # -----0---- get coordinates of inters
    ixmin = max(pred_bbox[0], gt_bbox[0])
    iymin = max(pred_bbox[1], gt_bbox[1])
    ixmax = min(pred_bbox[2], gt_bbox[2])
    iymax = min(pred_bbox[3], gt_bbox[3])
    iw = np.maximum(ixmax - ixmin + 1., 0.)
    ih = np.maximum(iymax - iymin + 1., 0.)

    # -----1----- intersection
    inters = iw * ih

    # -----2----- union, uni = S1 + S2 - inters
    uni = ((pred_bbox[2] - pred_bbox[0] + 1.) * (pred_bbox[3] - pred_bbox[1] + 1.) +
           (gt_bbox[2] - gt_bbox[0] + 1.) * (gt_bbox[3] - gt_bbox[1] + 1.) -
           inters)

    # -----3----- iou
    overlaps = inters / uni

    return overlaps

        
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

if(os.path.isfile(opt.img) and os.path.isfile(opt.img_2)):
    if(opt.patch == 'not'):
        print('Input whole images')
        imgs_all = []
        pre_bbox_all = []
        pre_label_all = []
    #     get = 2
        img_file = opt.img_2
        img_2_file = opt.img

        # read image
        img = cv2.imread(img_file)
        img_2 = cv2.imread(img_2_file)
        img = 255-img
        img_2 = 255-img_2

        date_1 = img_file[-17:-9]
        date_2 = img_2_file[-17:-9]
        time_diff =  (int(date_1[:4])-int(date_2[:4]))*365 + (int(date_1[4:6])-int(date_2[4:6]))*30 + (int(date_1[6:])-int(date_2[6:]))
        print('time interval : ', time_diff)
        img_2 = image_registration(img_2, img)
        img = img.astype('float32')/255
        img_2 = img_2.astype('float32')/255


        imgs_all.append(img)
        start_time_ = time.time()

        pre_bbox_the_list = []
        pre_label_the_list = []
        time_diff = torch.tensor([time_diff]).to(device)
        for col in range(2):
            for row in range(7):
                t = row*128
                b = t+256
                l = col*512
                r = (col+1)*512
    #             print('top : ',t, ' bot : ', b)
                u_imgs = img[t:b,l:r]
                u_imgs = torch.from_numpy(u_imgs)
                u_imgs = u_imgs.float().permute(2,0,1)
                u_imgs = u_imgs.unsqueeze(0).to(device)

                u_imgs_2 = img_2[t:b,l:r]
                u_imgs_2 = torch.from_numpy(u_imgs_2)
                u_imgs_2 = u_imgs_2.float().permute(2,0,1)
                u_imgs_2 = u_imgs_2.unsqueeze(0).to(device)
                loss_dict = model(u_imgs,u_imgs_2, time_diff)
    #             loss_dict = model(u_imgs)
                pre_bbox = loss_dict[0]['boxes']
                pre_bbox[:,0] += l
                pre_bbox[:,1] += t
                pre_bbox[:,2] += l
                pre_bbox[:,3] += t
                pre_label = loss_dict[0]['labels']
                pre_bbox_the = pre_bbox[loss_dict[0]['scores']>opt.threshold].cpu().detach().numpy().tolist()
        #         print(pre_bbox_the)
                pre_label_the = pre_label[loss_dict[0]['scores']>opt.threshold].cpu().numpy().tolist()
                pre_bbox_the_list += pre_bbox_the
                pre_label_the_list += pre_label_the
                show_img = u_imgs.cpu()[0].detach().float().permute(1,2,0)

        imgs_all.append(img)
        pre_bbox_all.append(pre_bbox_the_list)
        pre_label_all.append(pre_label_the_list)

        for index, box in enumerate(pre_bbox_the_list):
            index2=len(pre_bbox_the_list)-1
            while index2 > index:
                if(get_IoU(box,pre_bbox_the_list[index2]) >= 0.7):
                    pre_bbox_the_list.pop(index2)
                index2-=1


        draw_bounding_box(img, pre_bbox_the_list, pre_label_the_list, opt.img_2[-25:], False)
        print(' done')
        end_time_ = time.time()
        print("Time elapsed in this example code: ", end_time_ - start_time_)
    else:
        print('Input patch images')
        imgs_all = []
        pre_bbox_all = []
        pre_label_all = []
    #     get = 2
        img_file = opt.patch_img_2
        img_2_file = opt.patch_img

        # read image
        img = cv2.imread(img_file)
        img_2 = cv2.imread(img_2_file)
        img = 255-img
        img_2 = 255-img_2

        date_1 = img_file[-17:-9]
        date_2 = img_2_file[-17:-9]
        time_diff =  (int(date_1[:4])-int(date_2[:4]))*365 + (int(date_1[4:6])-int(date_2[4:6]))*30 + (int(date_1[6:])-int(date_2[6:]))
        print('time interval : ', time_diff)
        img_2 = image_registration(img_2, img)
        img = img.astype('float32')/255
        img_2 = img_2.astype('float32')/255


        imgs_all.append(img)
        start_time_ = time.time()

        pre_bbox_the_list = []
        pre_label_the_list = []
        time_diff = torch.tensor([time_diff]).to(device)

        u_imgs = img
        u_imgs = torch.from_numpy(u_imgs)
        u_imgs = u_imgs.float().permute(2,0,1)
        u_imgs = u_imgs.unsqueeze(0).to(device)

        u_imgs_2 = img_2
        u_imgs_2 = torch.from_numpy(u_imgs_2)
        u_imgs_2 = u_imgs_2.float().permute(2,0,1)
        u_imgs_2 = u_imgs_2.unsqueeze(0).to(device)
        loss_dict = model(u_imgs,u_imgs_2, time_diff)
#             loss_dict = model(u_imgs)
        pre_bbox = loss_dict[0]['boxes']
        pre_label = loss_dict[0]['labels']
        pre_bbox_the = pre_bbox[loss_dict[0]['scores']>opt.threshold].cpu().detach().numpy().tolist()
#         print(pre_bbox_the)
        pre_label_the = pre_label[loss_dict[0]['scores']>opt.threshold].cpu().numpy().tolist()
        pre_bbox_the_list += pre_bbox_the
        pre_label_the_list += pre_label_the
        show_img = u_imgs.cpu()[0].detach().float().permute(1,2,0)

        imgs_all.append(img)
        pre_bbox_all.append(pre_bbox_the_list)
        pre_label_all.append(pre_label_the_list)

        for index, box in enumerate(pre_bbox_the_list):
            index2=len(pre_bbox_the_list)-1
            while index2 > index:
                if(get_IoU(box,pre_bbox_the_list[index2]) >= 0.7):
                    pre_bbox_the_list.pop(index2)
                index2-=1

        draw_bounding_box(img, pre_bbox_the_list, pre_label_the_list, opt.patch_img_2[-28:], False)
        print(' done')
        end_time_ = time.time()
        print("Time elapsed in this example code: ", end_time_ - start_time_)
