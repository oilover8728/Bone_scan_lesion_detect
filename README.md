# Bone_scan_lesion_detect

分為兩個資料夾分別代表資料前處理，以及模型訓練和預測的部分  
  
-Bone script (資料前處理)  
-Lesion script (模型訓練和預測)  
  
有個問題是模型的weight 因為檔案太大所以push不上來，可以從雲端下載  
<https://drive.google.com/drive/u/0/folders/1zeV3bAjdG1tpcYXDBBUvgLOG5hIFmibL>  
crop 放入bone script的weight資料夾內  
0606_FCOS_real_STOD_reg_pseudo_Deform_SENet_seed115_recallbest放入lesion script的weight資料夾內  
即可確保所有程式正常運作  
  
requirment.txt為所使用套件  
pip: -r requirements.txt  
安裝相應版本來保證所有程式運作正常  
