import glob, cv2, os, sys
import torch
import numpy as np
import json
import math
import sklearn
import albumentations as albu
import random

from torch.utils.data import Dataset
from skimage.metrics import structural_similarity

# Open the patch image files.
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
            # Use this matrix to transform the
            # colored image wrt the reference image.
            transformed_img = cv2.warpPerspective(img1_color,
                                homography, (width, height))

            return transformed_img
    else:
        return img2_color
    
def image_similarity(img1_color,img2_color):
    img1 = cv2.cvtColor(img1_color, cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(img2_color, cv2.COLOR_BGR2GRAY)

    ssim = structural_similarity(img1, img2)
    return ssim

# for pair data 
class Pair_Whole_ImageDataset(Dataset):
    def __init__(self, data, augmentation, augmentation_2):
        super().__init__()
        self.images = [x[0][0] for x in data]
        self.images_2 = [x[0][1] for x in data]
        self.labels = [x[1] for x in data]
        self.augmentation = augmentation
        self.augmentation_2 = augmentation_2

    # 讓這個class可以用indexing的方式取東西
    def __getitem__(self, index):

        # get path
        img_file = self.images[index]
        img_2_file = self.images_2[index]
        label_iter = json.load(open(self.labels[index]))
        bbox_list = []
        category_list = []
        
        # read image
        img = cv2.imread(img_file)
        img_2 = cv2.imread(img_2_file)
        img = 255-img
        img_2 = 255-img_2
        h,w,c = img.shape
        h_2,w_2,c_2 = img_2.shape

        # ========== Registration ============
        img_2 = image_registration(img_2, img)
        # ========== Similarity ============
        similarity = image_similarity(img_2, img)
        weighted_loss = 1/similarity
        
        img = img.astype('float32')/255
        img_2 = img_2.astype('float32')/255
        
        # Check if bbox be cut
        flag = 0
        # read lable
        for json_t in label_iter['shapes']:
            bbox = [float(point) for val in json_t['points'] for point in val ]
            c_temp = json_t['label']

            if(c_temp[0]=='0'):
                category = 1
            elif(c_temp[0]=='1'):
                category = 2
            elif(c_temp[0]=='2'):
                category = 2
            elif(c_temp[0]=='3'):
                category = 2
            else:
                category = 3
            
            # Get the left-top and right-bottom points
            if(bbox[0]>bbox[2]): 
                bbox[0],bbox[2] = swap(bbox[0],bbox[2])
            if(bbox[1]>bbox[3]): 
                bbox[1],bbox[3] = swap(bbox[1],bbox[3])

            bbox[0] = 0 if bbox[0]-10 <= 0 else bbox[0]-10
            bbox[1] = 0 if bbox[1]-10 <= 0 else bbox[1]-10
            bbox[2] = w if bbox[2]+10 >= w else bbox[2]+10
            bbox[3] = h if bbox[3]+10 >= h else bbox[3]+10
                
            bbox_list.append([bbox[0], bbox[1], bbox[2], bbox[3]])
            category_list.append(category)
        
        boxes = torch.as_tensor(bbox_list, dtype=torch.float32)
        bbox_list_2=[[0,0,1,1]]
        aug_flag=0
        
        if self.augmentation:
            sample = self.augmentation(image=img, bboxes=bbox_list)
            img_aug, bboxes = sample['image'], sample['bboxes']
            sample_2 = self.augmentation(image=img_2, bboxes=bbox_list_2)
            img_aug_2, bboxes_2 = sample_2['image'], sample_2['bboxes'] 
        
        if(len(bboxes)==0):
            aug_flag=1

        if(aug_flag):
            sample = self.augmentation_2(image=img, bboxes=bbox_list)
            img_aug, bboxes = sample['image'], sample['bboxes'] 
            sample_2 = self.augmentation_2(image=img_2, bboxes=bbox_list_2)
            img_aug_2, bboxes_2 = sample_2['image'], sample_2['bboxes'] 

        # ========== Time ===========
        date_1 = self.images[index][-17:-9]
        date_2 = self.images_2[index][-17:-9]
        time_diff =  (int(date_1[:4])-int(date_2[:4]))*365 + (int(date_1[4:6])-int(date_2[4:6]))*30 + (int(date_1[6:])-int(date_2[6:]))
        img = torch.tensor(img_aug).float().permute(2,0,1)
        img_2 = torch.tensor(img_aug_2).float().permute(2,0,1)
            
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

        return img, img_2, time_diff, weighted_loss ,target
  
    def __len__(self):
        #有多少筆數的資料
        return len(self.images) 

# for Pseudo_Pair data 

from torch.utils.data import Dataset

class Pseudo_Pair_Whole_ImageDataset(Dataset):
    def __init__(self, data, augmentation, augmentation_2):
        super().__init__()
        self.images = [x[0][0] for x in data]
        self.images_2 = [x[0][1] for x in data]
        self.labels = [x[1] for x in data]
        self.augmentation = augmentation
        self.augmentation_2 = augmentation_2

    # 讓這個class可以用indexing的方式取東西
    def __getitem__(self, index):

        img_file = self.images[index]
        img_2_file = self.images_2[index]

        label_iter = json.load(open(self.labels[index]))
        bbox_list = []
        category_list = []
        
        # read image
        img = cv2.imread(img_file)
        img_2 = cv2.imread(img_2_file)
        img = 255-img
        img_2 = 255-img_2
        h,w,c = img.shape
        h_2,w_2,c_2 = img_2.shape

        # ========== Registration ============
        img_2 = image_registration(img_2, img)
        # ========== Similarity ============
        similarity = image_similarity(img_2, img)
        weighted_loss = 1/similarity
        
        # Check if bbox be cut
        flag = 0
        # read lable
        for json_t in label_iter['shapes']:
            bbox = [float(point) for val in json_t['points'] for point in val ]
            c_temp = json_t['label']

            if(c_temp[0]=='1'):
                category = 1
            elif(c_temp[0]=='2'):
                category = 2
            elif(c_temp[0]=='3'):
                category = 3
            
            # Get the left-top and right-bottom points
            if(bbox[0]>bbox[2]): 
                bbox[0],bbox[2] = swap(bbox[0],bbox[2])
            if(bbox[1]>bbox[3]): 
                bbox[1],bbox[3] = swap(bbox[1],bbox[3])
            
            bbox_list.append([bbox[0], bbox[1], bbox[2], bbox[3]])
            category_list.append(category)
        
        boxes = torch.as_tensor(bbox_list, dtype=torch.float32)
        
        bbox_list_2=[[0,0,1,1]]
        
        aug_flag=0
        
        aug_key = random.randint(0, 65535)
        random.seed(aug_key)
        if self.augmentation:
            sample = self.augmentation(image=img, bboxes=bbox_list)
            img_aug, bboxes = sample['image'], sample['bboxes']
            random.seed(aug_key)
            sample_2 = self.augmentation(image=img_2, bboxes=bbox_list_2)
            img_aug_2, bboxes_2 = sample_2['image'], sample_2['bboxes'] 
        
        if(len(bboxes)==0):
            aug_flag=1
        
        random.seed(aug_key)
        if(aug_flag):
            sample = self.augmentation_2(image=img, bboxes=bbox_list)
            img_aug, bboxes = sample['image'], sample['bboxes'] 
            random.seed(aug_key)
            sample_2 = self.augmentation_2(image=img_2, bboxes=bbox_list_2)
            img_aug_2, bboxes_2 = sample_2['image'], sample_2['bboxes'] 

        # ========== Time ===========
        date_1 = self.images[index][-17:-9]
        date_2 = self.images_2[index][-17:-9]
        img_aug = img_aug.astype('float32')/255
        img_aug_2 = img_aug_2.astype('float32')/255
        time_diff =  (int(date_1[:4])-int(date_2[:4]))*365 + (int(date_1[4:6])-int(date_2[4:6]))*30 + (int(date_1[6:])-int(date_2[6:]))

        img = torch.tensor(img_aug).float().permute(2,0,1)

        img_2 = torch.tensor(img_aug_2).float().permute(2,0,1)
    
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

        return img, img_2, time_diff, weighted_loss, target
  
    def __len__(self):
        #有多少筆數的資料
        return len(self.images) 
    
class Pair_ImageDataset(Dataset):
    def __init__(self, data, augmentation, augmentation_2):
        super().__init__()
        self.images = [x[0][0] for x in data]
        self.images_2 = [x[0][1] for x in data]
        self.labels = [x[1] for x in data]
        self.augmentation = augmentation
        self.augmentation_2 = augmentation_2

    # 讓這個class可以用indexing的方式取東西
    def __getitem__(self, index):

        # get path
        
        img_file = self.images[index]
        img_2_file = self.images_2[index]

#         print(time_diff)
        label_iter = json.load(open(self.labels[index]))
        bbox_list = []
        category_list = []
        
        # read image
        img = cv2.imread(img_file)
        img_2 = cv2.imread(img_2_file)
        img = 255-img
        img_2 = 255-img_2
        h,w,c = img.shape
        h_2,w_2,c_2 = img_2.shape

        # ========== Registration ============
#         img_2 = image_registration(img_2, img)
        # ========== Similarity ============
        similarity = image_similarity(img_2, img)
        weighted_loss = 1/similarity
        
#         img = img.astype('float32')/255
#         img_2 = img_2.astype('float32')/255
#         img_2 = image_concat_one_after.astype('float32')/255
        
#         img = 0.8*img + 0.2*img_2
        
        # Check if bbox be cut
        flag = 0
        # read lable
        for json_t in label_iter['shapes']:
            bbox = [float(point) for val in json_t['points'] for point in val ]
            c_temp = json_t['label']

            if(c_temp[0]=='0'):
                category = 1
            elif(c_temp[0]=='1'):
                category = 2
            elif(c_temp[0]=='2'):
                category = 2
            elif(c_temp[0]=='3'):
                category = 2
            else:
                category = 3
            
            # Get the left-top and right-bottom points
            if(bbox[0]>bbox[2]): 
                bbox[0],bbox[2] = swap(bbox[0],bbox[2])
            if(bbox[1]>bbox[3]): 
                bbox[1],bbox[3] = swap(bbox[1],bbox[3])

            bbox[0] = 0 if bbox[0]-10 <= 0 else bbox[0]-10
            bbox[1] = 0 if bbox[1]-10 <= 0 else bbox[1]-10
            bbox[2] = w if bbox[2]+10 >= w else bbox[2]+10
            bbox[3] = h if bbox[3]+10 >= h else bbox[3]+10
                
            bbox_list.append([bbox[0], bbox[1], bbox[2], bbox[3]])
            category_list.append(category)
        
        boxes = torch.as_tensor(bbox_list, dtype=torch.float32)
        
        bbox_list_2=[[0,0,1,1]]
        
        aug_flag=0

        aug_key = random.randint(0, 65535)
        random.seed(aug_key)
        
        if self.augmentation:
            sample = self.augmentation(image=img, bboxes=bbox_list)
            img_aug, bboxes = sample['image'], sample['bboxes']
            random.seed(aug_key)
            sample_2 = self.augmentation(image=img_2, bboxes=bbox_list_2)
            img_aug_2, bboxes_2 = sample_2['image'], sample_2['bboxes'] 
        
        if(len(bboxes)==0):
            aug_flag=1

        
        random.seed(aug_key)
        if(aug_flag):
            sample = self.augmentation_2(image=img, bboxes=bbox_list)
            img_aug, bboxes = sample['image'], sample['bboxes'] 
            random.seed(aug_key)
            sample_2 = self.augmentation_2(image=img_2, bboxes=bbox_list_2)
            img_aug_2, bboxes_2 = sample_2['image'], sample_2['bboxes'] 
        

        # ========== Time ===========
        date_1 = self.images[index][-17:-9]
        date_2 = self.images_2[index][-17:-9]
#         print(date_1)
#         print(date_2)
        img_aug = img_aug.astype('float32')/255
        img_aug_2 = img_aug_2.astype('float32')/255
        
        time_diff =  (int(date_1[:4])-int(date_2[:4]))*365 + (int(date_1[4:6])-int(date_2[4:6]))*30 + (int(date_1[6:])-int(date_2[6:]))
#         if(time_limit==1):
#         img_aug = img_aug * (1-time_decay) + img_aug_2 * time_decay

        img = torch.tensor(img_aug).float().permute(2,0,1)
        img_2 = torch.tensor(img_aug_2).float().permute(2,0,1)
            
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

        return img, img_2, time_diff, weighted_loss ,target
#         return img ,target
  
    def __len__(self):
        #有多少筆數的資料
        return len(self.images) 
    
    
# for Pseudo_Pair data 
class Pseudo_Pair_ImageDataset(Dataset):
    def __init__(self, data, augmentation, augmentation_2):
        super().__init__()
        self.images = [x[0][0] for x in data]
        self.images_2 = [x[0][1] for x in data]
        self.labels = [x[1] for x in data]
        self.augmentation = augmentation
        self.augmentation_2 = augmentation_2

    # 讓這個class可以用indexing的方式取東西
    def __getitem__(self, index):

        img_file = self.images[index]
        img_2_file = self.images_2[index]

        label_iter = json.load(open(self.labels[index]))
        bbox_list = []
        category_list = []
        
        # read image
        img = cv2.imread(img_file)
        img_2 = cv2.imread(img_2_file)
        img = 255-img
        img_2 = 255-img_2
        h,w,c = img.shape
        h_2,w_2,c_2 = img_2.shape

        # ========== Registration ============
        img_2 = image_registration(img_2, img)
        # ========== Similarity ============
        similarity = image_similarity(img_2, img)
        weighted_loss = 1/similarity
        
        flag = 0
        # read lable
        for json_t in label_iter['shapes']:
            bbox = [float(point) for val in json_t['points'] for point in val ]
            c_temp = json_t['label']

            if(c_temp[0]=='1'):
                category = 1
            elif(c_temp[0]=='2'):
                category = 2
            elif(c_temp[0]=='3'):
                category = 3
            
            # Get the left-top and right-bottom points
            if(bbox[0]>bbox[2]): 
                bbox[0],bbox[2] = swap(bbox[0],bbox[2])
            if(bbox[1]>bbox[3]): 
                bbox[1],bbox[3] = swap(bbox[1],bbox[3])
            
            bbox_list.append([bbox[0], bbox[1], bbox[2], bbox[3]])
            category_list.append(category)
        
        boxes = torch.as_tensor(bbox_list, dtype=torch.float32)
        
        bbox_list_2=[[0,0,1,1]]
        
        aug_flag=0
        
        aug_key = random.randint(0, 65535)
        random.seed(aug_key)
        if self.augmentation:
            sample = self.augmentation(image=img, bboxes=bbox_list)
            img_aug, bboxes = sample['image'], sample['bboxes']
            random.seed(aug_key)
            sample_2 = self.augmentation(image=img_2, bboxes=bbox_list_2)
            img_aug_2, bboxes_2 = sample_2['image'], sample_2['bboxes'] 
        
        if(len(bboxes)==0):
            aug_flag=1
        
        random.seed(aug_key)
        if(aug_flag):
            sample = self.augmentation_2(image=img, bboxes=bbox_list)
            img_aug, bboxes = sample['image'], sample['bboxes'] 
            random.seed(aug_key)
            sample_2 = self.augmentation_2(image=img_2, bboxes=bbox_list_2)
            img_aug_2, bboxes_2 = sample_2['image'], sample_2['bboxes'] 

        # ========== Time ===========
        date_1 = self.images[index][-17:-9]
        date_2 = self.images_2[index][-17:-9]
        img_aug = img_aug.astype('float32')/255
        img_aug_2 = img_aug_2.astype('float32')/255
        time_diff =  (int(date_1[:4])-int(date_2[:4]))*365 + (int(date_1[4:6])-int(date_2[4:6]))*30 + (int(date_1[6:])-int(date_2[6:]))

        img = torch.tensor(img_aug).float().permute(2,0,1)

        img_2 = torch.tensor(img_aug_2).float().permute(2,0,1)
    
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

        return img, img_2, time_diff, weighted_loss, target
#         return img ,target
  
    def __len__(self):
        #有多少筆數的資料
        return len(self.images) 
    
# for pair data 
class Pair_Registration_ImageDataset(Dataset):
    def __init__(self, data, augmentation, augmentation_2):
        super().__init__()
        self.images = [x[0][0] for x in data]
        self.images_2 = [x[0][1] for x in data]
        self.labels = [x[1] for x in data]
        self.augmentation = augmentation
        self.augmentation_2 = augmentation_2

    # 讓這個class可以用indexing的方式取東西
    def __getitem__(self, index):

        # get path
        
        img_file = self.images_2[index]
        img_2_file = self.images[index]

        label_iter = json.load(open(self.labels[index]))
        bbox_list = []
        category_list = []
        
        # read image
        img = cv2.imread(img_file)
        img_2 = cv2.imread(img_2_file)
        img = 255-img
        img_2 = 255-img_2
        h,w,c = img.shape
        h_2,w_2,c_2 = img_2.shape

        # ========== Registration ============
#         img_2 = image_registration(img_2, img)
        # ========== Similarity ============
        similarity = image_similarity(img_2, img)
        weighted_loss = 1/similarity
        
        # Check if bbox be cut
        flag = 0
        # read lable
        for json_t in label_iter['shapes']:
            bbox = [float(point) for val in json_t['points'] for point in val ]
            c_temp = json_t['label']

            if(c_temp[0]=='0'):
                category = 1
            elif(c_temp[0]=='1'):
                category = 2
            elif(c_temp[0]=='2'):
                category = 2
            elif(c_temp[0]=='3'):
                category = 2
            else:
                category = 3
            
            # Get the left-top and right-bottom points
            if(bbox[0]>bbox[2]): 
                bbox[0],bbox[2] = swap(bbox[0],bbox[2])
            if(bbox[1]>bbox[3]): 
                bbox[1],bbox[3] = swap(bbox[1],bbox[3])

            bbox[0] = 0 if bbox[0]-10 <= 0 else bbox[0]-10
            bbox[1] = 0 if bbox[1]-10 <= 0 else bbox[1]-10
            bbox[2] = w if bbox[2]+10 >= w else bbox[2]+10
            bbox[3] = h if bbox[3]+10 >= h else bbox[3]+10
                
            bbox_list.append([bbox[0], bbox[1], bbox[2], bbox[3]])
            category_list.append(category)
        
        boxes = torch.as_tensor(bbox_list, dtype=torch.float32)
        
        bbox_list_2=[[0,0,1,1]]
        
        aug_flag=0

        aug_key = random.randint(0, 65535)
        random.seed(aug_key)
        
        if self.augmentation:
            sample = self.augmentation(image=img, bboxes=bbox_list)
            img_aug, bboxes = sample['image'], sample['bboxes']
            random.seed(aug_key)
            sample_2 = self.augmentation(image=img_2, bboxes=bbox_list_2)
            img_aug_2, bboxes_2 = sample_2['image'], sample_2['bboxes'] 
        
        if(len(bboxes)==0):
            aug_flag=1

        
        random.seed(aug_key)
        if(aug_flag):
            sample = self.augmentation_2(image=img, bboxes=bbox_list)
            img_aug, bboxes = sample['image'], sample['bboxes'] 
            random.seed(aug_key)
            sample_2 = self.augmentation_2(image=img_2, bboxes=bbox_list_2)
            img_aug_2, bboxes_2 = sample_2['image'], sample_2['bboxes'] 
        

        img_aug = img_aug.astype('float32')/255
        img_aug_2 = img_aug_2.astype('float32')/255
        # ========== Time ===========
        date_1 = self.images[index][-20:-12]
        date_2 = self.images_2[index][-20:-12]
        time_diff =  (int(date_2[:4])-int(date_1[:4]))*365 + (int(date_2[4:6])-int(date_1[4:6]))*30 + (int(date_2[6:])-int(date_1[6:]))

        img = torch.tensor(img_aug).float().permute(2,0,1)
        img_2 = torch.tensor(img_aug_2).float().permute(2,0,1)
        # test
#         img_2 = img_2 - img
            
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

        return img, img_2, time_diff, weighted_loss ,target
  
    def __len__(self):
        #有多少筆數的資料
        return len(self.images) 
    
# for Pseudo_Pair data 

class Pseudo_Pair_Registration_ImageDataset(Dataset):
    def __init__(self, data, augmentation, augmentation_2):
        super().__init__()
        self.images = [x[0][0] for x in data]
        self.images_2 = [x[0][1] for x in data]
        self.labels = [x[1] for x in data]
        self.augmentation = augmentation
        self.augmentation_2 = augmentation_2

    # 讓這個class可以用indexing的方式取東西
    def __getitem__(self, index):
        img_file = self.images_2[index]
        img_2_file = self.images[index]

        label_iter = json.load(open(self.labels[index]))
        bbox_list = []
        category_list = []
        
        # read image
        img = cv2.imread(img_file)
        img_2 = cv2.imread(img_2_file)
        img = 255-img
        img_2 = 255-img_2
        h,w,c = img.shape
        h_2,w_2,c_2 = img_2.shape

        # ========== Registration ============
#         img_2 = image_registration(img_2, img)
        # ========== Similarity ============
        similarity = image_similarity(img_2, img)
        weighted_loss = 1/similarity
        
        # Check if bbox be cut
        flag = 0
        # read lable
        for json_t in label_iter['shapes']:
            bbox = [float(point) for val in json_t['points'] for point in val ]
            c_temp = json_t['label']

            if(c_temp[0]=='1'):
                category = 1
            elif(c_temp[0]=='2'):
                category = 2
            elif(c_temp[0]=='3'):
                category = 3
            
            # Get the left-top and right-bottom points
            if(bbox[0]>bbox[2]): 
                bbox[0],bbox[2] = swap(bbox[0],bbox[2])
            if(bbox[1]>bbox[3]): 
                bbox[1],bbox[3] = swap(bbox[1],bbox[3])
            
            bbox_list.append([bbox[0], bbox[1], bbox[2], bbox[3]])
            category_list.append(category)
        
        boxes = torch.as_tensor(bbox_list, dtype=torch.float32)
#         print(boxes)
        bbox_list_2=[[0,0,1,1]]
        
        aug_flag=0

        aug_key = random.randint(0, 65535)
        random.seed(aug_key)
        if self.augmentation:
            sample = self.augmentation(image=img, bboxes=bbox_list)
            img_aug, bboxes = sample['image'], sample['bboxes']
            random.seed(aug_key)
            sample_2 = self.augmentation(image=img_2, bboxes=bbox_list_2)
            img_aug_2, bboxes_2 = sample_2['image'], sample_2['bboxes'] 
        
        if(len(bboxes)==0):
            aug_flag=1
        
        random.seed(aug_key)
        if(aug_flag):
            sample = self.augmentation_2(image=img, bboxes=bbox_list)
            img_aug, bboxes = sample['image'], sample['bboxes'] 
            random.seed(aug_key)
            sample_2 = self.augmentation_2(image=img_2, bboxes=bbox_list_2)
            img_aug_2, bboxes_2 = sample_2['image'], sample_2['bboxes'] 

        img_aug = img_aug.astype('float32')/255
        img_aug_2 = img_aug_2.astype('float32')/255
        date_1 = self.images[index][-20:-12]
        date_2 = self.images_2[index][-20:-12]
        
        # ========== Time ===========
        time_diff =  (int(date_2[:4])-int(date_1[:4]))*365 + (int(date_2[4:6])-int(date_1[4:6]))*30 + (int(date_2[6:])-int(date_1[6:]))
        img = torch.tensor(img_aug).float().permute(2,0,1)

        img_2 = torch.tensor(img_aug_2).float().permute(2,0,1)
#         img_2 = img_2 - img
    
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

        return img, img_2, time_diff, weighted_loss, target
  
    def __len__(self):
        #有多少筆數的資料
        return len(self.images) 