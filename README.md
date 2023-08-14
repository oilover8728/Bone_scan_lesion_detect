# Bone_scan_lesion_detect

這是我的碩論內容整理出來的程式碼  
分為兩個資料夾分別代表資料前處理，以及模型訓練和預測的部分  
  
* Bone script (資料前處理)  
* Lesion script (模型訓練和預測)  
  
資料以及pre-trained weight請使用iir的帳號到雲端上下載    
  
資料部分 :
<https://drive.google.com/drive/u/0/folders/1zeV3bAjdG1tpcYXDBBUvgLOG5hIFmibL>  

* Original_data 請下載後直接將裡面包含的資料夾放到Bone_script/ 底下  
* Detect_data 內包含Lesion_script/內處理過可用來直接訓練的資料，解壓縮完將3個資料夾放到Lesion_script/ 底下  
  
Pre-trained weights :   
* crop  
  放入Bone_script/weight 資料夾內代表用來crop影像中anterior/posterior的模型weight   
* 0606_FCOS_real_STOD_reg_pseudo_Deform_SENet_seed115_recallbest
* 0809_FCOS_whole_pair_STOD_seed115_recallbest  
放入Lesion_script/weight 資料夾內分別代表Patch/Whole image訓練出來的weight  
即可確保所有程式正常運作  

請先下
`pip install torch==1.13.0+cu116 torchvision==0.14.0+cu116 torchaudio==0.13.0 --extra-index-url https://download.pytorch.org/whl/cu116`  
來確保可以使用GPU  
requirment.txt為所使用套件  
pip: -r requirements.txt  
安裝相應版本來保證所有程式運作正常  

