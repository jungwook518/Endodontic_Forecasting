import os
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
from torch import optim
from torch.optim import lr_scheduler
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random
import shutil
import time
from Dataloader.Dental_dataset import DentalDataset
from Model.models import Net2, Net_CAM
from training_module import train_3model2, test_model
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
import cv2


def generate_data_list(label_csv, train_indices, valid_indices, test_indices):
    label_file = pd.read_csv(label_csv, sep=',', header=0, encoding="CP949")

    data = []
    for i in range(len(label_file)):
        if label_file['Status'][i] == 1:
            if label_file['Result'][i] == 1:
                data.append((label_file['PatientID_new'][i] + '.bmp', 1))
            elif label_file['Result'][i] == 0:
                data.append((label_file['PatientID_new'][i] + '.bmp', 0))

    train_data_list = [data[i] for i in train_indices if i < len(data)]
    valid_data_list = [data[i] for i in valid_indices if i < len(data)]
    test_data_list = [data[i] for i in test_indices if i < len(data)]

    data_list = {'train': train_data_list,
                'valid': valid_data_list,
                'test': test_data_list}

    return data_list

def make_seed(random_seed):
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)


def grad_CAM(model,classes,dataloader,preprocessed_data_dir,grad_cam_save_path):
    
    for img,label,file_name in dataloader:

        img=img.to(device)
        pred=model(img)
        pred_class=model(img).argmax(dim=1)

        #if classes[label[0]]!='Fail' and classes[pred_class[0]]!='Fail':
        #if not(classes[label[0]]=='Fail' and classes[pred_class[0]]=='Fail'):

        
        '''if not classes[label[0]]=='Fail':
            continue '''
        

        print('*'*40)
        print('Patient ID : ',file_name[0])
        print('Label : ',classes[label[0]])
        print('Predicted Class : ',classes[pred_class[0]])
        pred[:,pred_class[0]].backward()
        gradients=model.get_activations_gradient()
        pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])
        activations = model.get_activations(img).detach()

        for i in range(0,16):
            activations[:, i, :, :] *= pooled_gradients[i]  
        
        heatmap = torch.mean(activations, dim=1).squeeze()
        heatmap = np.maximum(heatmap.cpu(), 0)
        heatmap /= torch.max(heatmap)
        
        
        '''minm=heatmap.min()
        maxm=0.5
        for i in range(0,len(heatmap)):
            for j in range(0,len(heatmap[i])):
                if heatmap[i][j]>=0.6:
                    heatmap[i][j]=0.6
                heatmap[i][j]=(heatmap[i][j]-minm)/(maxm-minm)'''
        
           
        
        '''plt.hist(heatmap)
        plt.show()'''
        

        img_new = cv2.imread(os.path.join(preprocessed_data_dir,file_name[0]+'.bmp'))
        heatmap = cv2.resize(np.array(heatmap), (img_new.shape[1], img_new.shape[0]),interpolation=cv2.INTER_LINEAR)

        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

        superimposed_img = np.uint8(heatmap * 0.2 + img_new*0.8)

        f, axarr = plt.subplots(1,3)
        axarr[0].imshow(img_new)
        axarr[0].axis('off')
        axarr[1].imshow(heatmap)
        axarr[1].axis('off')
        axarr[2].imshow(superimposed_img)
        axarr[2].axis('off')


        # plt.show()
        plt.savefig(os.path.join(grad_cam_save_path,file_name[0]+'.png'))


if __name__ == '__main__':

    RANDOM_SEED=1
    make_seed(RANDOM_SEED)


    # ============================================================================== #
    #                        0. Define Hyper-parameters
    # ============================================================================== #

    # Device configuration
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # print('Device Name : ',torch.cuda.get_device_name(device))
    # print('The number of devices: ',torch.cuda.device_count())

    exp_name = '5foldexp_0011'
    print("Experiment location in {}. Please Check if exp_name is overlap or not".format(exp_name))
    batch_size = 5
    learning_rate = 0.0001
    num_epochs = 5



    # ============================================================================== #
    #                        1. Load Data
    # ============================================================================== #

    label_csv = './all_label.csv'
    preprocessed_data_dir='./Data/Preprocessed'
    crop_data_dir='./Data/Crop'
    grad_cam_save_path = './result/' + exp_name +'/grad_cam'
    os.makedirs(grad_cam_save_path, exist_ok=True)

    k = 5
    label_file = pd.read_csv(label_csv, sep=',', header=0, encoding="CP949")
    all_indices = list(range(len(label_file)))

    # Split data into train-validation (80%) and test (20%) sets
    train_val_indices, test_indices = train_test_split(all_indices, test_size=0.2, random_state=RANDOM_SEED)

    # Split train-validation data using K-Fold cross-validation
    cv = KFold(n_splits=k, shuffle=True, random_state=RANDOM_SEED)
    for fold, (train_indices, valid_indices) in enumerate(cv.split(train_val_indices)):
        data_list = generate_data_list(label_csv, train_indices, valid_indices, test_indices)

        preprocessed_mean=0.59
        preprocessed_std=0.25
        org_mean=0.69
        org_std=0.20

        data_normalization={'preprocessed': transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((preprocessed_mean,), (preprocessed_std,))
        ]),
            'original': transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((org_mean,), (org_std,))
        ])          
        }


        
        dataset={'train': DentalDataset(preprocessed_data_dir=preprocessed_data_dir,
                                        crop_data_dir=crop_data_dir,
                                        data_list=data_list['train'],
                                        normalize=data_normalization,
                                        augmentation=True),
                    'val': DentalDataset(preprocessed_data_dir=preprocessed_data_dir,
                                        crop_data_dir=crop_data_dir,
                                        data_list=data_list['valid'],
                                        normalize=data_normalization,
                                        augmentation=False),
                    'test': DentalDataset(preprocessed_data_dir=preprocessed_data_dir,
                                        crop_data_dir=crop_data_dir,
                                        data_list=data_list['test'],
                                        normalize=data_normalization,
                                        augmentation=False),
                                        }

        dataset_sizes = {'train': dataset['train'].__len__(),
                            'val':dataset['val'].__len__(),
                            'test':dataset['test'].__len__()}

        print('Dataset Sizes : ',dataset_sizes)

        # ============================================================================== #
        #                        2. Define Dataloader
        # ============================================================================== #

        dataloaders={'train': torch.utils.data.DataLoader(dataset['train'],batch_size=1,shuffle=True),
                        'val':torch.utils.data.DataLoader(dataset['val'],batch_size=1,shuffle=False),
                        'test':torch.utils.data.DataLoader(dataset['test'],batch_size=1,shuffle=False)}


        # ============================================================================== #
        #                        3. Define Model & Setting save path
        # ============================================================================== #

        
        model = Net2().to(device)
        # if you want to use pre-trained model for initialization, like below.
        # model.load_state_dict(torch.load("./result/5foldexp_0010/model/accf1Atten2_fold_05_05-25-16-32.pt",map_location=device))
        model.eval()
        grad_cam_model=Net_CAM(model)
        grad_cam_model=grad_cam_model.to(device)
        grad_cam_model.eval()

        
        grad_CAM(grad_cam_model,['Success','Fail'],dataloaders['test'],preprocessed_data_dir,grad_cam_save_path)
        
      