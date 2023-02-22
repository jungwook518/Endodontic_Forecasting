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

def generate_data_list(label_csv,train_valid_test_ratio):
    label_file=pd.read_csv(label_csv,sep=',',header=0,encoding = "CP949")

    status_eq_0 = 0
    status_eq_1 = 0

    success_data=[]
    fail_data=[]
    total_data = len(label_file)
    for i in range(0,total_data):
        if label_file['Status'][i]==1:
            status_eq_1 +=1
            if label_file['Tooth'][i]==14 or label_file['Tooth'][i]==24:
                pass
            else:
                if label_file['Result'][i]==1:
                    fail_data.append(label_file['PatientID_new'][i]+'.pt')
                elif label_file['Result'][i]==0:
                    success_data.append(label_file['PatientID_new'][i]+'.pt')
        else:
            status_eq_0+=1
    
    num_success_data = len(success_data)
    num_fail_data = len(fail_data)
    
    # print('Number of total data : {}, ''Status=1 data'' : {}, ''Status=0 data'' : {}'.format(total_data, status_eq_1, status_eq_0)) #682 658 24
    # print('Number of include data in ''Status=1'' data) : {}'.format(num_success_data + num_fail_data)) #589
    # print('Number of exclude data in ''Status=1'' data) : {}'.format(status_eq_1 - num_success_data - num_fail_data)) #69
    # print('Number of success data : {}'.format(num_success_data)) #400
    # print('Number of fail data : {}'.format(num_fail_data)) #189
    

    suc_train_num=round(num_success_data*train_valid_test_ratio['train']*0.01)
    suc_valid_num = round(num_success_data*train_valid_test_ratio['valid']*0.01)
    random.shuffle(success_data)

    suc_train_path_list=[]
    suc_valid_path_list=[]
    suc_test_path_list=[]
    
    for i in range(0, num_success_data):
      if i < suc_train_num:
        suc_train_path_list.append((success_data[i],0))
      elif i<suc_train_num + suc_valid_num:
        suc_valid_path_list.append((success_data[i],0))
      else:
        suc_test_path_list.append((success_data[i],0))
    
    print('<Success Label Data>')
    print("The number of training data : ",len(suc_train_path_list))
    print("The number of val data : ",len(suc_valid_path_list))
    print("The number of test data : ",len(suc_test_path_list))
    
    
    
    fail_train_num=round(num_fail_data*train_valid_test_ratio['train']*0.01)
    fail_valid_num = round(num_fail_data*train_valid_test_ratio['valid']*0.01)
    random.shuffle(fail_data)
    
    fail_train_path_list=[]
    fail_valid_path_list=[]
    fail_test_path_list=[]
    
    for i in range(0, num_fail_data):
      if i < fail_train_num:
        fail_train_path_list.append((fail_data[i],1))
      elif i<fail_train_num + fail_valid_num:
        fail_valid_path_list.append((fail_data[i],1))
      else:
        fail_test_path_list.append((fail_data[i],1))

    print('<Fail Label Data>')
    print("The number of training data : ",len(fail_train_path_list))
    print("The number of val data : ",len(fail_valid_path_list))
    print("The number of test data : ",len(fail_test_path_list))

    train_data_list = suc_train_path_list + fail_train_path_list
    valid_data_list = suc_valid_path_list + fail_valid_path_list
    test_data_list = suc_test_path_list + fail_test_path_list
    
    data_list ={'train' :train_data_list,
                'valid' : valid_data_list,
                'test' : test_data_list}

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
    
    exp_name = 'exp_0001'
    print("Experiment location in {}. Please Check if exp_name is overlap or not".format(exp_name))
    batch_size = 4
    learning_rate = 0.001
    num_epochs = 5
    
    
    
    # ============================================================================== #
    #                        1. Load Data
    # ============================================================================== #
    label_csv = './sample_label.csv'
    
    train_valid_test_ratio = {'train':80,
                              'valid':10,
                              'test':10}
    data_list = generate_data_list(label_csv,train_valid_test_ratio)
    
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
    
    
    preprocessed_data_dir='./Data/Preprocessed_tensor'
    crop_data_dir='./Data/Crop_tensor'
    
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
    # model.load_state_dict(torch.load("result/model/bestmodel.pth",map_location=device))
    
    model_save_path = './result/' + exp_name +'/model'
    score_save_path = './result/' + exp_name +'/score'
    figure_save_path = './result/' + exp_name +'/figure'
    os.makedirs(model_save_path, exist_ok=True)
    os.makedirs(score_save_path, exist_ok=True)
    os.makedirs(figure_save_path, exist_ok=True)
    
    base_name = datetime.today().strftime('%m-%d-%H:%M')
    
    
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
    
    torch.save(accmodel.state_dict(),os.path.join(model_save_path,'accAtten2_'+datetime.today().strftime('%m-%d-%H:%M')+'.pt'))
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
    
    
