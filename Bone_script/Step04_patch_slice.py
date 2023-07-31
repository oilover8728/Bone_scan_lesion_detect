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
parser.add_argument('--in_folder', type=str, default='./Step3_crop_data/', help='input lesion images path')
parser.add_argument('--patch_registration', type=str, default='False', help="decide patch slicing after registration")
parser.add_argument('--out_folder', type=str, default='./Step4_patch_data/', help='output folder path')
opt = parser.parse_args()

def creat_folder(path):
    if not os.path.isdir(path):
        os.mkdir(path)
        
def image_registration(img1_color,img2_color):
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
    
'''
    ==== 要切Patch的Data ====
'''

train_images = glob.glob(opt.in_folder+'images/*.png')
train_labels = glob.glob(opt.in_folder+'labels/*.json')

train_images.sort()
train_labels.sort()

print("Data (images/labels) : ",len(train_images), len(train_labels))

train=[]
        
for i in range(len(train_images)):
    img_paths = train_images[i]
    lab_paths = train_labels[i]
    
    package=[]
    package.append(img_paths)
#         print(img_paths[j])
    package.append(lab_paths)
    train.append(package)
    
len(train), train[0], train[-1]

## pair
# for i in range(1,len(train_images)):
train_pair_images = []
train_pair_labels = []

train.sort()
# add first data
train_pair_images.append([train[0][0], train[0][0]])
train_pair_labels.append(train[0][1])
for i in range(1,len(train)):
    file_1 = train[i-1][0][-25:-18] 
    file_2 = train[i][0][-25:-18] 
#     print(file_1)
#     print(file_2)
#     break
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
        
print("Pair data : ",len(train_pair_images))


'''
    ==== 創資料夾 ====
'''

# name = 'patch_'
creat_folder( opt.out_folder + 'images')
creat_folder( opt.out_folder + 'labels')

'''
    ==== Clean 重疊的bbox (Raw data 有一個重複的框，非Raw data則直接註解掉下面這段) ====
'''

json_path = opt.in_folder +'labels/BS00075_20170628_1a6c.json'
with open(json_path,'rb') as f:
    #定义为只读模型，并定义名称为f

    params = json.load(f)
#     for json_t in label_iter['shapes']:
#         if(json_t['label'][0]=='r'):
    for i in reversed(range(len(params['shapes']))):
#         print(i)
        if(i==0):
            params['shapes'].pop(i)

    dict_p = params
    f.close()


with open(json_path,'w') as r:
    #定义为写模式，名称定义为r

    json.dump(dict_p,r)
    #将dict写入名称为r的文件中

    r.close()
    #加载json文件中的内容给params
    
'''
    ==== 兩種切Patch方式 (直接切/先做完registration再切) ==== 
'''

# 用來寫進新的label值的空白檔案
blank_json_path = './BS00000_00000000_1234.json'

# 直接對原圖切patch
def Store_data(now_img_path, now_json_path, target_patch, now_bbox):
    """
    now_img_path       now_json_path   圖片和標註讀取路徑
    target_patch                       目標patch圖
    now_bbox                           目標label
    """
    if os.path.isfile(now_json_path):
        with open(now_json_path,'rb') as f:
                params = json.load(f)
                dict_p = params
                dict_p['shapes'].append({'label': str(clss), 'points':[[now_bbox[0],now_bbox[1]],[now_bbox[2],now_bbox[3]]]})

                f.close()

        with open(now_json_path,'w') as r:

            json.dump(dict_p,r)

            r.close()

    else:
        # 要創建一個新的img and json檔
        img_PIL = toPIL(target_patch)
        img_PIL.save(now_img_path)
        with open(blank_json_path,'rb') as f:
                params = json.load(f)
                dict_p = params
                dict_p['shapes'].append({'label': str(clss), 'points':[[now_bbox[0],now_bbox[1]],[now_bbox[2],now_bbox[3]]]})

                f.close()

        with open(now_json_path,'w') as r:

            json.dump(dict_p,r)

            r.close()
            
# 先registration再切patch
def Store_pair_data(now_img_path, now_img_2_path, now_json_path, target_patch, target_2_patch, now_bbox):
    """
    now_img_path       now_json_path   圖片和標註讀取路徑
    target_patch                       目標patch圖
    now_bbox                           目標label
    """
    if os.path.isfile(now_json_path):
        with open(now_json_path,'rb') as f:
                params = json.load(f)
                dict_p = params
                dict_p['shapes'].append({'label': str(clss), 'points':[[now_bbox[0],now_bbox[1]],[now_bbox[2],now_bbox[3]]]})

                f.close()

        with open(now_json_path,'w') as r:

            json.dump(dict_p,r)

            r.close()

    else:
        # 要創建一個新的img and json檔
        img_PIL = toPIL(target_patch)
        img_2_PIL = toPIL(target_2_patch)

        img_PIL.save(now_img_path)
        img_2_PIL.save(now_img_2_path)
        with open(blank_json_path,'rb') as f:
                params = json.load(f)
                dict_p = params
                dict_p['shapes'].append({'label': str(clss), 'points':[[now_bbox[0],now_bbox[1]],[now_bbox[2],now_bbox[3]]]})

                f.close()

        with open(now_json_path,'w') as r:

            json.dump(dict_p,r)

            r.close()
   

'''
    ==== 判斷是哪種切的方式並動作 ==== 
'''

toPIL = transforms.ToPILImage()

if(opt.patch_registration=='False'):
    print("Normal patch.")
    # Overlap Training data
    save_img_path = opt.out_folder + 'images'
    save_lab_path = opt.out_folder + 'labels'
    # 7 overlap region
    # [  0 :  256]
    # [128 :  384]
    # [256 :  512]
    # [384 :  640]
    # [512 :  768]
    # [640 :  896]
    # [768 :  1024]
    for i in range(len(train_images)):
        img_path = train_images[i]
        json_path = train_labels[i]

        img = cv2.imread(img_path)
        h,w,c = img.shape
        label_iter = json.load(open(json_path))

        patch_img_list = []
        left = 0
        right = 256
        # get patch
        for j in range(8):
            patch_row = []
            patch_row.append(img[left:right,0:512])
            patch_row.append(img[left:right,512:1024])
            patch_img_list.append(patch_row)
            left += 128
            right += 128

        # get bbox
        bbox_list = []
        for json_t in label_iter['shapes']:
            bbox = [int(point) for val in json_t['points'] for point in val ]
            clss = int(json_t['label'][0])

            col = 0 if bbox[0]<512 else 1

            row = 0 if bbox[1]<128 else 1 if bbox[1]<256 else 2 if bbox[1]<384 else 3 if bbox[1]<512 else 4 if bbox[1]<640 else 5 if bbox[1]<768 else 6 if bbox[1]<896 else 7

            target_patch = patch_img_list[row][col]
            bbox[0] -= col*512                  #x0
            bbox[2] -= col*512                  #x1

            gap = 0 if(row==0) else 128 if(row==1) else 256 if(row==2) else 384 if(row==3) else 512 if(row==4) else 640 if(row==5) else 768 if(row==6) else 896 
            bbox[1] -= gap
            bbox[3] -= gap

            # 要存的path
            now_img_path = save_img_path + img_path[-26:-18] + '_' + str(4*row+col).zfill(2) + img_path[-18:]
            now_json_path = save_lab_path + json_path[-27:-19] + '_' + str(4*row+col).zfill(2) + json_path[-19:]

            now_bbox = bbox.copy()
            area = (bbox[2]-bbox[0])*(bbox[3]-bbox[1])
            # 如果框沒有太接近邊緣就存
            now_bbox[0] = 0 if now_bbox[0] <= 0 else now_bbox[0]
            now_bbox[1] = 0 if now_bbox[1] <= 0 else now_bbox[1]
            now_bbox[2] = 512 if now_bbox[2] >= 512 else now_bbox[2]
            now_bbox[3] = 256 if now_bbox[3] >= 256 else now_bbox[3]
            now_area = (now_bbox[2]-now_bbox[0])*(now_bbox[3]-now_bbox[1])
            if(now_area/area > 0.45 or (now_bbox[3]-now_bbox[1])>50):
                Store_data(now_img_path, now_json_path, target_patch, now_bbox)

            # 往下檢查有沒有跨Patch
            while row < 7 :
                if(bbox[3] >= 128):
                    row += 1
                    bbox[1] -= 128
                    bbox[3] -= 128
                    target_patch = patch_img_list[row][col]
                    now_bbox = bbox.copy()

                    now_img_path = save_img_path + img_path[-26:-18] + '_' + str(4*row+col).zfill(2) + img_path[-18:]
                    now_json_path = save_lab_path + json_path[-27:-19] + '_' + str(4*row+col).zfill(2) + json_path[-19:]

                    now_bbox[0] = 0 if now_bbox[0] <= 0 else now_bbox[0]
                    now_bbox[1] = 0 if now_bbox[1] <= 0 else now_bbox[1]
                    now_bbox[2] = 512 if now_bbox[2] >= 512 else now_bbox[2]
                    now_bbox[3] = 256 if now_bbox[3] >= 256 else now_bbox[3]
                    now_area = (now_bbox[2]-now_bbox[0])*(now_bbox[3]-now_bbox[1])
                    if(now_area/area > 0.45 or (now_bbox[3]-now_bbox[1])>50):
                        Store_data(now_img_path, now_json_path, target_patch, now_bbox)
                else:
                    break

    print("Done")
elif(opt.patch_registration=='True'):
    print("Patch after Registration!")
    # Overlap Pair Patch Training data
    save_img_path = opt.out_folder + 'images'
    save_lab_path = opt.out_folder + 'labels'
    # 7 overlap region
    # [  0 :  256]
    # [128 :  384]
    # [256 :  512]
    # [384 :  640]
    # [512 :  768]
    # [640 :  896]
    # [768 :  1024]
    count = 0
    for i in range(len(train_pair_images)):
        img_path = train_pair_images[i][0]
        img_2_path = train_pair_images[i][1]
        same = 0
        if(img_path==img_2_path):
            same = 1
        json_path = train_pair_labels[i]

        img = cv2.imread(img_path)
        img_2 = cv2.imread(img_2_path)
        img_2 = image_registration(img_2, img)


        h,w,c = img.shape
        label_iter = json.load(open(json_path))

        patch_img_list = []
        patch_img_2_list = []
        left = 0
        right = 256
        # get patch
        for j in range(7):
            patch_row = []
            patch_2_row = []
            patch_row.append(img[left:right,0:512])
            patch_2_row.append(img_2[left:right,0:512])
            patch_row.append(img[left:right,512:1024])
            patch_2_row.append(img_2[left:right,512:1024])
            patch_img_list.append(patch_row)
            patch_img_2_list.append(patch_2_row)
            left += 128
            right += 128
        # get bbox
        bbox_list = []
        for json_t in label_iter['shapes']:
            bbox = [int(point) for val in json_t['points'] for point in val ]
            clss = int(json_t['label'][0])

            col = 0 if bbox[0]<512 else 1

            row = 0 if bbox[1]<128 else 1 if bbox[1]<256 else 2 if bbox[1]<384 else 3 if bbox[1]<512 else 4 if bbox[1]<640 else 5 if bbox[1]<768 else 6 

            target_patch = patch_img_list[row][col]
            target_2_patch = patch_img_2_list[row][col]
            bbox[0] -= col*512                  #x0
            bbox[2] -= col*512                  #x1

            gap = 0 if(row==0) else 128 if(row==1) else 256 if(row==2) else 384 if(row==3) else 512 if(row==4) else 640 if(row==5) else 768
            bbox[1] -= gap
            bbox[3] -= gap

            # 要存的path    
            now_img_0_path = save_img_path + img_path[-26:-18] + '_' + str(2*row+col).zfill(2) + img_path[-18:-9] + '_00' + img_path[-9:]
            now_img_1_path = save_img_path + img_path[-26:-18] + '_' + str(2*row+col).zfill(2) + img_path[-18:-9] + '_01' + img_path[-9:]
            now_img_2_path = save_img_path + img_2_path[-26:-18] + '_' + str(2*row+col).zfill(2) + img_2_path[-18:-9] + '_02' + img_2_path[-9:]

            now_json_0_path = save_lab_path + json_path[-27:-19] + '_' + str(2*row+col).zfill(2) + '_00' + json_path[-19:]
            now_json_1_path = save_lab_path + json_path[-27:-19] + '_' + str(2*row+col).zfill(2) + '_01' + json_path[-19:]

            now_bbox = bbox.copy()
            area = (bbox[2]-bbox[0])*(bbox[3]-bbox[1])
            # 如果框沒有太接近邊緣就存
            now_bbox[0] = 0 if now_bbox[0] <= 0 else now_bbox[0]
            now_bbox[1] = 0 if now_bbox[1] <= 0 else now_bbox[1]
            now_bbox[2] = 512 if now_bbox[2] >= 512 else now_bbox[2]
            now_bbox[3] = 256 if now_bbox[3] >= 256 else now_bbox[3]
            now_area = (now_bbox[2]-now_bbox[0])*(now_bbox[3]-now_bbox[1])
            if(now_area/area > 0.45 or (now_bbox[3]-now_bbox[1])>50):
                if(same==1):
                    Store_data(now_img_0_path, now_json_0_path, target_patch, now_bbox)
                else:
                    Store_pair_data(now_img_1_path, now_img_2_path, now_json_1_path, target_patch, target_2_patch, now_bbox)

            # 往下檢查有沒有跨Patch
            while row < 6 :
                if(bbox[3] >= 128):
                    row += 1
                    bbox[1] -= 128
                    bbox[3] -= 128
                    target_patch = patch_img_list[row][col]
                    target_2_patch = patch_img_2_list[row][col]
                    now_bbox = bbox.copy()

                    now_img_0_path = save_img_path + img_path[-26:-18] + '_' + str(2*row+col).zfill(2) + img_path[-18:-9] + '_00' + img_path[-9:]
                    now_img_1_path = save_img_path + img_path[-26:-18] + '_' + str(2*row+col).zfill(2) + img_path[-18:-9] + '_01' + img_path[-9:]
                    now_img_2_path = save_img_path + img_2_path[-26:-18] + '_' + str(2*row+col).zfill(2) + img_2_path[-18:-9] + '_02' + img_2_path[-9:]

                    now_json_0_path = save_lab_path + json_path[-27:-19] + '_' + str(2*row+col).zfill(2) + '_00' + json_path[-19:]
                    now_json_1_path = save_lab_path + json_path[-27:-19] + '_' + str(2*row+col).zfill(2) + '_01' + json_path[-19:]

                    now_bbox[0] = 0 if now_bbox[0] <= 0 else now_bbox[0]
                    now_bbox[1] = 0 if now_bbox[1] <= 0 else now_bbox[1]
                    now_bbox[2] = 512 if now_bbox[2] >= 512 else now_bbox[2]
                    now_bbox[3] = 256 if now_bbox[3] >= 256 else now_bbox[3]
                    now_area = (now_bbox[2]-now_bbox[0])*(now_bbox[3]-now_bbox[1])
                    if(now_area/area > 0.45 or (now_bbox[3]-now_bbox[1])>50):
                        count+=1
                        if(same==1):
                            Store_data(now_img_0_path, now_json_0_path, target_patch, now_bbox)
                        else:
                            Store_pair_data(now_img_1_path, now_img_2_path, now_json_1_path, target_patch, target_2_patch, now_bbox)
                else:
                    break
    print("Done")
else:
    print("Patch slice must be True or False string type")
