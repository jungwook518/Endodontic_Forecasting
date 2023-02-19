import pydicom
import cv2
import os
import numpy as np
from tqdm import tqdm
import shutil
from sklearn.cluster import KMeans
import math

if __name__ == '__main__':
           
    orig_data_dir = './Data/Original'
    mask_data_dir = './Data/Mask'
    
    save_crop_path = './Data/Crop'
    os.makedirs(save_crop_path, exist_ok=True)
    
    save_preprocessed_path = './Data/Preprocessed'
    os.makedirs(save_preprocessed_path, exist_ok=True)
    
    orig_data_list = [file for file in os.listdir(orig_data_dir) if file.endswith(".dcm")]
    orig_data_list = sorted(orig_data_list)
   
   
    th_dic={}
   
   
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
        crop_bmp=crop_dcm.astype(np.float32)
        crop_bmp_norm = (crop_bmp - crop_bmp.min())/(crop_bmp.max()-crop_bmp.min()) * 255
        crop_bmp_norm = crop_bmp_norm.astype(np.uint8)
        cv2.imwrite(os.path.join(save_crop_path,orig_data).replace('.dcm', '.bmp'), crop_bmp_norm)
        
        kmeans=KMeans(n_clusters=3).fit(crop_dcm.flatten().reshape(-1,1))
        centers=kmeans.cluster_centers_.flatten()
        centers.sort()
        
        mask=np.zeros(crop_dcm.shape)
        tmp_arr=[]
        
        # for i in range(0,len(mask)):
        #     for j in range(0,len(mask[i])):
        #         if crop_dcm[i][j]<=np.mean(centers[0:2]):
        #             tmp_arr.append(crop_dcm[i][j])
        #         else:
        #             mask[i][j]=255
        idx_ = np.where(crop_dcm<=np.mean(centers[0:2]))
        for i in range(len(idx_)):
            tmp_arr.append(crop_dcm[idx_[i][0],idx_[i][1]])
        tmp_arr=np.array(tmp_arr)
        mask[crop_dcm > np.mean(centers[0:2])] = 255
        
        kmeans2=KMeans(n_clusters=2).fit(tmp_arr.flatten().reshape(-1,1))
        centers2=kmeans2.cluster_centers_.flatten()
        centers2.sort()
        
        mask2=np.zeros(crop_dcm.shape)
        
        #for i in range(0,len(mask2)):
        #    for j in range(0,len(mask2[i])):
        #        if crop_dcm[i][j]<=np.mean(centers2):
        #            continue
        #        else:
        #            mask2[i][j]=255
        mask2[crop_dcm > np.mean(centers[0:2])] = 255 
        th_dic[file_name]=np.mean(centers2)

        
        #Windowing
        flatten=crop_dcm.flatten()
        median=np.median(flatten)
        upper_threshold=median*1.3
        lower_threshold=math.trunc(th_dic[file_name])

        crop_dcm[crop_dcm<lower_threshold]=lower_threshold
        crop_dcm[crop_dcm>upper_threshold]=upper_threshold
                
        #Scaling
        crop_dcm=crop_dcm.astype(np.float32)
        crop_dcm_norm = (crop_dcm - crop_dcm.min())/(crop_dcm.max()-crop_dcm.min()) * 255
        
        #Histogram Equalization
        preprocessed_crop_dcm_norm = crop_dcm_norm.astype(np.uint8)
        clahe=cv2.createCLAHE(clipLimit=2.0,tileGridSize=(8,8))
        preprocessed_crop_dcm_norm=clahe.apply(preprocessed_crop_dcm_norm)
        
        cv2.imwrite(os.path.join(save_preprocessed_path,orig_data).replace('.dcm', '.bmp'), preprocessed_crop_dcm_norm)
        
        
        
        
        
