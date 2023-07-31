
## 前處理部分

### 檔案說明 : 
20211220 Cleaned labelme JSON :  代表醫師標註原JSON檔，內容包含相對應檔名的病灶標註  
JPEGImages : 代表病人原始影像  

Crop_label_data : 代表我自行標註的人物全身與反面全身檔案，用來訓練裁切人物的100+張訓練資料  

Step1_data          : 第一階段執行Step01_rm_useless_data.py 後，會將原始資料分為lesion/normal/superscan儲存於此，Step02_rm_regressbox.py 會直接處理lesion資料夾內含有regress標註標的資料   
Step3_crop_data     : 第三階段執行Step03_crop_people_resize.py，將Step1_data/lesion 中的資料將人物全身正反面部分裁取，並resize至1024x1024儲存於此  
Step4_patch_data    : 第四階段執行Step04_patch_slice.py，將切過patch的影像與標註資料儲存於此  
Step5_patient_data  : 第五階段執行Step05_patient_fold.py，將所有的資料按照病人病歷號歸檔儲存於此(方便切kfold訓練)  
  
utils : 一些有用的import  
vision : 計算object detection model的相關函式  
  
BS00000_0000000_1234.json 用來寫入新資料儲存的空白json檔  

### 使用方式 : 
1. Step01_rm_useless_data.py :
會將原始檔案20211220 Cleaned labelme JSON/JPEGImages中有正確病灶框的樣本集中到lesion資料，並分離出normal/superscan的資料  
指令 : `python Step01_rm_useless_data.py --in_images "images folder path" --in_labels "labels folder path" --out_folder "output folder path" ` 
default : in_images = './JPEGImages/'     in_labels = './20211220 Cleaned labelme JSON/'     out_folder = './Step1_data/'  
3.   
