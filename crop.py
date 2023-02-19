import pydicom
import cv2
import os
import numpy as np
from tqdm import tqdm
import torch


if __name__ == '__main__':

    orig_data_dir = './Data/Original'
    mask_data_dir = './Data/Mask'
    
    save_crop_path = './Data/Crop'
    os.makedirs(save_crop_path, exist_ok=True)
    
    orig_data_list = [file for file in os.listdir(orig_data_dir) if file.endswith(".dcm")]
    orig_data_list = sorted(orig_data_list)
   
    for orig_data in tqdm(orig_data_list):
        file_name, ext = os.path.splitext(orig_data)
        orig_dcm = pydicom.read_file(os.path.join(orig_data_dir,orig_data))
        orig_arr=orig_dcm.pixel_array # value 2000~3000
        
        # find contour with yellow
        mask=cv2.imread(os.path.join(mask_data_dir,file_name+'.bmp'))
        cnt=[]
        for i in range(mask.shape[0]):
            for j in range(mask.shape[1]):
                if (mask[i][j]==np.array([0,255,255])).all():
                    cnt.append([i,j])
        cnt=np.array(cnt)
        
        
        #crop dicom using contour
        max_x=max(cnt[:,0])
        min_x=min(cnt[:,0])
        max_y=max(cnt[:,1])
        min_y=min(cnt[:,1])
    
        crop_dcm=orig_arr[min_x:max_x,min_y:max_y]
        crop_dcm=crop_dcm.astype(np.float32)
        crop_dcm_norm = (crop_dcm - crop_dcm.min())/(crop_dcm.max()-crop_dcm.min()) * 255
        crop_dcm_norm = crop_dcm_norm.astype(np.uint8)
        cv2.imwrite(os.path.join(save_crop_path,orig_data).replace('.dcm', '.bmp'), crop_dcm_norm)
        
        
