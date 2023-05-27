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
from Model.models import Net2
from training_module import train_3model2, test_model
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split


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

if __name__ == '__main__':

    RANDOM_SEED=1
    make_seed(RANDOM_SEED)


    # ============================================================================== #
    #                        0. Define Hyper-parameters
    # ============================================================================== #

    # Device configuration
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('Device Name : ',torch.cuda.get_device_name(device))
    print('The number of devices: ',torch.cuda.device_count())

    exp_name = '5foldexp_0011'
    print("Experiment location in {}. Please Check if exp_name is overlap or not".format(exp_name))
    batch_size = 5
    learning_rate = 0.0001
    num_epochs = 5



    # ============================================================================== #
    #                        1. Load Data
    # ============================================================================== #

    label_csv = './label2.csv'

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


        preprocessed_data_dir='./Data/Preprocessed'
        crop_data_dir='./Data/Crop'

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

        dataloaders={'train': torch.utils.data.DataLoader(dataset['train'],batch_size=batch_size,shuffle=True),
                        'val':torch.utils.data.DataLoader(dataset['val'],batch_size=batch_size,shuffle=False),
                        'test':torch.utils.data.DataLoader(dataset['test'],batch_size=batch_size,shuffle=False)}


        # ============================================================================== #
        #                        3. Define Model & Setting save path
        # ============================================================================== #

        model = Net2().to(device)
        # if you want to use pre-trained model for initialization, like below.
        model.load_state_dict(torch.load("./result/5foldexp_0010/model/accf1Atten2_fold_05_05-25-16-32.pt",map_location=device))

        
        model_save_path = './result/' + exp_name +'/model'
        score_save_path = './result/' + exp_name +'/score'
        figure_save_path = './result/' + exp_name +'/figure'

        os.makedirs(model_save_path, exist_ok=True)
        os.makedirs(score_save_path, exist_ok=True)
        os.makedirs(figure_save_path, exist_ok=True)

        base_name = f"fold_{fold+1:02d}_{datetime.today().strftime('%m-%d-%H-%M')}"


        # ============================================================================== #
        #                        4. Set Loss & Optimizer
        # ============================================================================== #

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.2)



        # ============================================================================== #
        #                        5. Train / Save / Test
        # ============================================================================== #
        accf1model, accf1model_later, accmodel, accmodel_later, f1model, f1model_later= \
                train_3model2(model, criterion, optimizer, exp_lr_scheduler, dataloaders, dataset_sizes, device, num_epochs=num_epochs, file_name=os.path.join(figure_save_path,'loss_acc_'+base_name+'.png'))


        torch.save(accf1model.state_dict(),os.path.join(model_save_path,'accf1Atten2_'+base_name+'.pt'))
        torch.save(accf1model_later.state_dict(),os.path.join(model_save_path,'accf1Atten2_later_'+base_name+'.pt'))
        
        torch.save(accmodel.state_dict(),os.path.join(model_save_path,'accAtten2_'+base_name+'.pt'))
        torch.save(accmodel_later.state_dict(),os.path.join(model_save_path,'accAtten2_later_'+base_name+'.pt'))
        
        torch.save(f1model.state_dict(),os.path.join(model_save_path,'f1Atten2_'+base_name+'.pt'))
        torch.save(f1model_later.state_dict(),os.path.join(model_save_path,'f1Atten2_later_'+base_name+'.pt'))
        
        device_cpu=torch.device('cpu')
        
        acc5,confusion_matrix5=test_model(accf1model,dataloaders,device_cpu,os.path.join(score_save_path,'accf1Atten2_'+base_name+'.txt'))
        acc6,confusion_matrix6=test_model(accf1model_later,dataloaders,device_cpu,os.path.join(score_save_path,'accf1Atten2_later_'+base_name+'.txt'))
        
        acc1,confusion_matrix1=test_model(accmodel,dataloaders,device_cpu,os.path.join(score_save_path,'accAtten2_'+base_name+'.txt'))
        acc2,confusion_matrix2=test_model(accmodel_later,dataloaders,device_cpu,os.path.join(score_save_path,'accAtten2_later_'+base_name+'.txt'))
        
        acc3,confusion_matrix3=test_model(f1model,dataloaders,device_cpu,os.path.join(score_save_path,'f1Atten2_'+base_name+'.txt'))
        acc4,confusion_matrix4=test_model(f1model_later,dataloaders,device_cpu,os.path.join(score_save_path,'f1Atten2_later_'+base_name+'.txt'))