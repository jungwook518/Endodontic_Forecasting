import torch
from torch.utils.data import Dataset
from torchvision import transforms
import torchvision.transforms.functional as TF
import os
from PIL import Image
import random
import numpy as np

class DentalDataset(Dataset):

    def __init__(self,preprocessed_data_dir, crop_data_dir,data_list,normalize,augmentation=False):
        self.preprocessed_data_dir=preprocessed_data_dir
        self.crop_data_dir=crop_data_dir
        self.data_list=data_list
        self.normalize=normalize
        self.augmentation=augmentation

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        data_file = self.data_list[idx][0]
        data_label = self.data_list[idx][1]
        preprocessed_data_img=Image.open(os.path.join(self.preprocessed_data_dir,data_file))
        crop_data_img=Image.open(os.path.join(self.crop_data_dir,data_file))
        
        #data augmentation
        if self.augmentation:
            #random horizontal, vertical flip
            if random.random()>0.5:
                preprocessed_data_img=TF.hflip(preprocessed_data_img)
                crop_data_img=TF.hflip(crop_data_img)
                
            if random.random() > 0.5:
                preprocessed_data_img = TF.vflip(preprocessed_data_img)
                crop_data_img = TF.vflip(crop_data_img)
                
            # if random.random() > 0.5:
            #     factor=round(random.uniform(0.5,1.2),2)
            #     data_img=TF.adjust_brightness(data_img,factor)
            # if random.random() > 0.5:
            #     factor=round(random.uniform(1,1.7),2)
            #     data_img=TF.adjust_contrast(data_img,factor)
            # if random.random() > 0.5:
            #     angle=np.random.randint(-30,30)
            #     data_img=TF.rotate(data_img,angle,expand=False)

            
        #resize
        resize=transforms.Resize((600,600))
        preprocessed_data_img=resize(preprocessed_data_img)
        crop_data_img=resize(crop_data_img)
       
        #normalize
        preprocessed_data_img=self.normalize['preprocessed'](preprocessed_data_img)
        crop_data_img=self.normalize['original'](crop_data_img)
        
        #final sample to return 
        sample=(torch.cat((preprocessed_data_img,crop_data_img),0), int(data_label),data_file)
        
        return sample