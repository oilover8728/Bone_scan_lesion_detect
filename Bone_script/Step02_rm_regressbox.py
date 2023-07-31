import glob, cv2, os, sys
import argparse
import numpy as np
import json
import shutil
import copy

# === Code ===  

## Parameter

parser = argparse.ArgumentParser()
parser.add_argument('--in_folder', type=str, default='./Step1_data/lesion/', help='input images path')
# parser.add_argument('--out_folder', type=str, default='./Step1_data/', help='output folder path')
opt = parser.parse_args()

float_formatter = "{:.2f}".format

def Check_jpg_and_json_all(images, labels):
    COUNT=0
    for i in range(len(images)):
        img_name = images[i][-26:-4] 
        lab_name = labels[i][-27:-5]
        if(img_name != lab_name):
            print(i)
            COUNT+=1
    
    return COUNT


'''lesion data'''

images_folder = glob.glob(opt.in_folder + 'images/*.jpg')
labels_folder = glob.glob(opt.in_folder + 'labels/*.json')

images_folder.sort()
labels_folder.sort()
len(images_folder), len(labels_folder)

print('File incorrect : ', Check_jpg_and_json_all(images_folder, labels_folder))

# Show empty/regress data number

# for i in range(len(labels_folder)):
#     label_iter = json.load(open(labels_folder[i])) 
#     if(len(label_iter['shapes'])==0):
#         print(i)
#     for json_t in label_iter['shapes']:
#         if(json_t['label'][0]=='r'):
#             print(i)
            
# Remove process

count = 0
for json_path in labels_folder:
    with open(json_path,'rb') as f:
        #定义为只读模型，并定义名称为f

        params = json.load(f)
    #     for json_t in label_iter['shapes']:
    #         if(json_t['label'][0]=='r'):
        for i in reversed(range(len(params['shapes']))):
    #         print(i)
            if(params['shapes'][i]['label'][0]=='r'):
                params['shapes'].pop(i)
            elif(params['shapes'][i]['label'][0]=='n' or params['shapes'][i]['label'][0]=='S'):
                print('Find normal/Superscan data, ' ,json_path)
            else:
                count+=1

        dict_p = params
        f.close()


    with open(json_path,'w') as r:
        #定义为写模式，名称定义为r

        json.dump(dict_p,r)
        #将dict写入名称为r的文件中

        r.close()
        #加载json文件中的内容给params
        
# Check

space = []
for i in range(len(labels_folder)):
    label_iter = json.load(open(labels_folder[i])) 
    if(len(label_iter['shapes'])==0):
        space.append(i)
        print('empty',i)
    for json_t in label_iter['shapes']:
        if(json_t['label'][0]=='r'):
            print(i)

# Remove all regress box images and labels

for i in range(len(space), 0, -1):
#     print(images_folder[i])
#     print(labels_folder[i])
#     print(i)
    print(space[i-1])
    print("Remove : ", images_folder[space[i-1]])
    print("Remove : ", labels_folder[space[i-1]])
    os.remove(images_folder[space[i-1]])
    os.remove(labels_folder[space[i-1]])
    space.remove(space[i-1])
    
# Check again
warning = 0

labels_folder = glob.glob(opt.in_folder + 'labels/*.json')

labels_folder.sort()
for i in range(len(labels_folder)):
    label_iter = json.load(open(labels_folder[i])) 
    if(len(label_iter['shapes'])==0):
        warning = 1
        print('empty',i)
    for json_t in label_iter['shapes']:
        if(json_t['label'][0]=='r'):
            print('r',i)
            warning = 1

if(warning!=0):
    print("Exist not empty/regress data")
else:
    print("Done")