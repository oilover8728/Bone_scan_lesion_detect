# Bone_scan_lesion_detect

這是我的碩論內容整理出來的程式碼  
分為兩個資料夾分別代表資料前處理，以及模型訓練和預測的部分  
  
* Bone script (資料前處理)  
* Lesion script (模型訓練和預測)  
  
因為模型的weight檔案太大push不上來，請從雲端下載  
<https://drive.google.com/drive/u/0/folders/1zeV3bAjdG1tpcYXDBBUvgLOG5hIFmibL>  
* crop  
  放入Bone_script/weight 資料夾內代表用來crop影像中anterior/posterior的模型weight   
* 0606_FCOS_real_STOD_reg_pseudo_Deform_SENet_seed115_recallbest
* 0809_FCOS_whole_pair_STOD_seed115_recallbest  
放入Lesion_script/weight 資料夾內分別代表Patch/Whole image訓練出來的weight  
即可確保所有程式正常運作  
  
requirment.txt為所使用套件  
pip: -r requirements.txt  
安裝相應版本來保證所有程式運作正常  
