## 模型訓練與病灶預測

### 檔案說明 : 
* 01_Whole_data  
* 02_Patch_data  
* 03_Patch_registration  
* 04_Pseudo_labels  
   
* utils : 一些有用的import   
* vision : 計算object detection model的相關函式    
* Dataset.py : 所有用來訓練或預測的Dataset設定，(包含whole/patch/registration)   
* model.py : 最後使用的model的詳細程式碼   
   
* example : lesion_detect.py的範例輸入  
* result : lesion_detect.py的輸出  
  
* lesion_detect.py : 輸入兩張同一個病人的兩張bone scan影像，使用設定好的weight預測結果   
* pseudo_label.py : 用來產生pseudo_label的程式碼  
* train.patch.py : 訓練一個輸入為兩張正常patch影響的模型  
* train_registration_patch.py : 訓練一個輸入為兩張先registration再切patch的影像的模型  
* train_whole.py : 訓練一個輸入為兩張病人全身影像的模型  
     
* BS00000_0000000_1234.json 用來寫入新資料儲存的空白json檔  

### 使用方式 : 
## 1. lesion_detect.py :  
會將輸入的兩張影像，按照設定好的weight/threshold/device，產生出模型預測結果放入output的資料夾中
指令 : `python lesion_detect.py --img "images1"  --img_2 "images2" --patch "patch data or not " --patch_img "image1" --patch_img_2 "image2" --weight "pre-trained weight" --threshold "confience score" --set_device "GPU/CPU" --out_folder "output folder path" `  
default : 
* img = './example/BS00145_20150923_c564.png'
* img_2 = './example/BS00145_20160219_fb7b.png'
* patch = 'False'
* patch_img = 'False'
* patch_img = './example/BS00145_13_20150923_c564.png'
* patch_img_2 = './example/BS00145_13_20160219_fb7b.png'
* weight = './weight/0606_FCOS_real_STOD_reg_pseudo_Deform_SENet_seed115_recallbest'
* threshold = 0.56
* set_device = 'gpu'
* out_folder = './result/'
## 2. pseudo_label.py :  
## 3. train_patch.py :  
## 4. train_registration_patch.py :  
## 5. train_whole.py :  
