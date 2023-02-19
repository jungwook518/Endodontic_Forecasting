import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import time
import copy
from datetime import datetime
from sklearn.metrics import confusion_matrix
import os
import pandas as pd
from tqdm import tqdm

def train_3model2(model, criterion, optimizer, scheduler, dataloaders, dataset_sizes,device, num_epochs, file_name):
    since = time.time()

    accmodel=copy.deepcopy(model)
    accmodel_later=copy.deepcopy(model)
    f1model=copy.deepcopy(model)
    f1model_later=copy.deepcopy(model)
    accf1model=copy.deepcopy(model)
    accf1model_later=copy.deepcopy(model)

    best_ACCmodel_wts = copy.deepcopy(model.state_dict())
    best_ACCmodel_wts_later = copy.deepcopy(model.state_dict())
    best_F1model_wts = copy.deepcopy(model.state_dict())
    best_F1model_wts_later = copy.deepcopy(model.state_dict())
    best_ACCF1model_wts = copy.deepcopy(model.state_dict())
    best_ACCF1model_wts_later = copy.deepcopy(model.state_dict())

    best_acc = 0.0
    best_f1=0.0
    best_accf1=0.0
    
    acc_epoch=0
    f1_epoch=0
    accf1_epoch=0
    acc_epoch2=0
    f1_epoch2=0
    accf1_epoch2=0
    
    ACCmodel_dic={}
    ACC_dic={}
    F1model_dic={}
    F1_dic={}
    
    train_loss=[]
    val_loss=[]

    train_acc=[]
    val_acc=[]
    
    for epoch in tqdm(range(num_epochs)):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        
        # Each epoch has a training and validation phase
        for phase in ['train','val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0
            running_cm=0
            # Iterate over data.
            for inputs, labels, file_names in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs) #file_names
                    _, preds = torch.max(outputs, 1)
                    loss=criterion(outputs,labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += (loss.item()) * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                running_cm+=confusion_matrix(labels.cpu(), preds.cpu(),labels=[0,1])

            if phase == 'train':
                scheduler.step()
            
            epoch_loss = running_loss / dataset_sizes[phase]
            #epoch_acc = running_corrects.double() / dataset_sizes[phase]
            epoch_acc=(running_cm[1][1]+running_cm[0][0])/(running_cm[1][1]+running_cm[1][0]+running_cm[0][1]+running_cm[0][0])
            epoch_sen = running_cm[1][1]/(running_cm[1][1]+running_cm[1][0])
            epoch_pre = running_cm[1][1]/(running_cm[1][1]+running_cm[0][1])
            epoch_f1=(2*epoch_sen*epoch_pre)/(epoch_sen+epoch_pre)
            
            if phase=='train':
                train_loss.append(epoch_loss)
                train_acc.append(np.float(epoch_acc))
            else:
                val_loss.append(epoch_loss)
                val_acc.append(np.float(epoch_acc))

            #print(phase,train_loss,val_loss)
            print('{} Loss: {:.4f} Acc: {:.4f} F1: {:.4f}'.format(
                phase, epoch_loss, epoch_acc,epoch_f1))
            
            # deep copy the model
            #ACCF1 model
            if phase == 'val' and (epoch_acc+epoch_f1)/2 > best_accf1:
                best_accf1 = (epoch_acc+epoch_f1)/2
                accf1_epoch=epoch
                best_ACCF1model_wts = copy.deepcopy(model.state_dict())
            
            if phase == 'val' and (epoch_acc+epoch_f1)/2 >= best_accf1:
                accf1_epoch2=epoch
                best_ACCF1model_wts_later = copy.deepcopy(model.state_dict())
            
            #ACC model
            if phase == 'val' and epoch_acc > best_acc:
                if not(epoch == accf1_epoch):
                    best_acc = epoch_acc
                    acc_epoch=epoch
                    best_ACCmodel_wts = copy.deepcopy(model.state_dict())
                else:
                    ACCmodel_dic[epoch] = copy.deepcopy(model.state_dict())
                    ACC_dic[epoch] = epoch_acc
            
            if phase == 'val' and epoch_acc >= best_acc:
                if not(epoch==accf1_epoch):
                    acc_epoch2=epoch
                    best_ACCmodel_wts_later = copy.deepcopy(model.state_dict())
                else:
                    ACCmodel_dic[epoch] = copy.deepcopy(model.state_dict())
                    ACC_dic[epoch] = epoch_acc
             
            #F1 model
            if phase == 'val' and epoch_f1 > best_f1:
                if not(epoch==accf1_epoch) and not(epoch==acc_epoch):
                    best_f1 = epoch_f1
                    f1_epoch=epoch
                    best_F1model_wts = copy.deepcopy(model.state_dict())
                else:
                    F1model_dic[epoch] = copy.deepcopy(model.state_dict())
                    F1_dic[epoch] = epoch_f1
                    
            if phase == 'val' and epoch_f1 >= best_f1:
                if not(epoch==accf1_epoch) and not(epoch==acc_epoch):
                    f1_epoch2=epoch
                    best_F1model_wts_later = copy.deepcopy(model.state_dict())
                else:
                    F1model_dic[epoch] = copy.deepcopy(model.state_dict())
                    F1_dic[epoch] = epoch_f1

        print()

    if accf1_epoch in ACC_dic.keys():
        del ACCmodel_dic[accf1_epoch]
        del ACC_dic[accf1_epoch]
    ACC_dic=dict(sorted(ACC_dic.items()))
    ACCmodel_dic=dict(sorted(ACCmodel_dic.items()))
    best_acc=best_acc
    print('ACC_dic : ',ACC_dic)
    print('ACCmodel_dic.keys() : ',ACCmodel_dic.keys())
    print('best_acc : ',best_acc)
    
    if bool(ACC_dic) and best_acc<=max(ACC_dic.values()):
        if best_acc==max(ACC_dic.values()):
            acc_idx= np.where(np.array(list(ACC_dic.values()))==best_acc)[0]
            print('acc_idx : ',acc_idx)
            if acc_epoch>list(ACC_dic.keys())[acc_idx[0]]:
                acc_epoch=list(ACC_dic.keys())[acc_idx[0]]
                best_ACCmodel_wts = ACCmodel_dic[acc_epoch]
            if acc_epoch2<list(ACC_dic.keys())[acc_idx[len(acc_idx)-1]]:      
                acc_epoch2=list(ACC_dic.keys())[acc_idx[len(acc_idx)-1]]
                best_ACCmodel_wts_later = ACCmodel_dic[acc_epoch2]
        else:
            best_acc = max(ACC_dic.values())
            acc_idx= np.where(np.array(list(ACC_dic.values()))==best_acc)[0]
            print('acc_idx : ',acc_idx)
            acc_epoch=list(ACC_dic.keys())[acc_idx[0]]
            acc_epoch2=list(ACC_dic.keys())[acc_idx[len(acc_idx)-1]]
            best_ACCmodel_wts = ACCmodel_dic[acc_epoch]
            best_ACCmodel_wts_later = ACCmodel_dic[acc_epoch2]
    
    if accf1_epoch in F1_dic.keys():
        del F1model_dic[accf1_epoch]
        del F1_dic[accf1_epoch]
    if acc_epoch in F1_dic.keys():
        del F1model_dic[acc_epoch]
        del F1_dic[acc_epoch]
    F1_dic=dict(sorted(F1_dic.items()))
    F1model_dic=dict(sorted(F1model_dic.items()))
    print('F1_dic : ',F1_dic)
    print('F1model_dic.keys() : ',F1model_dic.keys())
    print('best_f1 : ',best_f1)
        
    if bool(F1_dic) and best_f1<=max(F1_dic.values()):
        if best_f1==max(F1_dic.values()):
            f1_idx=np.where(np.array(list(F1_dic.values()))==best_f1)[0]
            print('f1_idx : ',f1_idx)
            if f1_epoch>list(F1_dic.keys())[f1_idx[0]]:
                f1_epoch=list(F1_dic.keys())[f1_idx[0]]
                best_F1model_wts=F1model_dic[f1_epoch]
            if f1_epoch2<list(F1_dic.keys())[f1_idx[len(f1_idx)-1]]:
                f1_epoch2=list(F1_dic.keys())[f1_idx[len(f1_idx)-1]]
                best_F1model_wts_later=F1model_dic[f1_epoch2]
                
        else:
            best_f1=max(F1_dic.values())
            f1_idx=np.where(np.array(list(F1_dic.values()))==best_f1)[0]
            print('f1_idx : ',f1_idx)
            f1_epoch=list(F1_dic.keys())[f1_idx[0]]
            f1_epoch2=list(F1_dic.keys())[f1_idx[len(f1_idx)-1]]
            best_F1model_wts=F1model_dic[f1_epoch]
            best_F1model_wts_later=F1model_dic[f1_epoch2]
    
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc+F1: {:4f}'.format(best_accf1))
    print('Epoch: ',accf1_epoch,accf1_epoch2)
    print('Best val Acc: {:4f}'.format(best_acc))
    print('Epoch: ',acc_epoch,acc_epoch2)
    print('Best val F1: {:4f}'.format(best_f1))
    print('Epoch: ',f1_epoch,f1_epoch2)
    
    print()

    plt.rcParams["figure.figsize"] = (8,8)
    fig, axs=plt.subplots(2)
    axs[0].set_title('model loss')
    axs[1].set_title('model accuracy')
    for ax in axs.flat:
        ax.set_ylim([0.0,1.0])
    axs[0].plot(train_loss,'r',val_loss,'g',)
    axs[1].plot(train_acc,'r',val_acc,'g')
    fig.tight_layout()
    for ax in axs.flat:
        leg=ax.legend(['train','val'])
    
    plt.savefig(file_name)
   
    # load best model weights
    accmodel.load_state_dict(best_ACCmodel_wts)
    accmodel_later.load_state_dict(best_ACCmodel_wts_later)

    f1model.load_state_dict(best_F1model_wts)
    f1model_later.load_state_dict(best_F1model_wts_later)

    accf1model.load_state_dict(best_ACCF1model_wts)
    accf1model_later.load_state_dict(best_ACCF1model_wts_later)
    
    return accf1model, accf1model_later, accmodel, accmodel_later, f1model, f1model_later

def train_3model(model, criterion, optimizer, scheduler, dataloaders, dataset_sizes,device, num_epochs, file_name):
    since = time.time()

    accmodel=copy.deepcopy(model)
    accmodel_later=copy.deepcopy(model)
    f1model=copy.deepcopy(model)
    f1model_later=copy.deepcopy(model)
    accf1model=copy.deepcopy(model)
    accf1model_later=copy.deepcopy(model)

    best_ACCmodel_wts = copy.deepcopy(model.state_dict())
    best_ACCmodel_wts_later = copy.deepcopy(model.state_dict())
    best_F1model_wts = copy.deepcopy(model.state_dict())
    best_F1model_wts_later = copy.deepcopy(model.state_dict())
    best_ACCF1model_wts = copy.deepcopy(model.state_dict())
    best_ACCF1model_wts_later = copy.deepcopy(model.state_dict())

    best_acc = 0.0
    best_f1=0.0
    best_accf1=0.0
    
    train_loss=[]
    val_loss=[]

    train_acc=[]
    val_acc=[]
    
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        
        # Each epoch has a training and validation phase
        for phase in ['train','val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0
            running_cm=0

            # Iterate over data.
            for inputs, labels, file_names in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs) #file_names
                    _, preds = torch.max(outputs, 1)
                    loss=criterion(outputs,labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += (loss.item()) * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                running_cm+=confusion_matrix(labels.cpu(), preds.cpu(),labels=[0,1])

            if phase == 'train':
                scheduler.step()
            
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            epoch_sen = (running_cm[1][1]/(running_cm[1][1]+running_cm[1][0]))
            epoch_pre = (running_cm[1][1]/(running_cm[1][1]+running_cm[0][1]))
            epoch_f1=(2*epoch_sen*epoch_pre)/(epoch_sen+epoch_pre)
            
            if phase=='train':
                train_loss.append(epoch_loss)
                train_acc.append(np.float(epoch_acc))
            else:
                val_loss.append(epoch_loss)
                val_acc.append(np.float(epoch_acc))

            #print(phase,train_loss,val_loss)
            print('{} Loss: {:.4f} Acc: {:.4f} F1: {:.4f}'.format(
                phase, epoch_loss, epoch_acc,epoch_f1))
            
            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_ACCmodel_wts = copy.deepcopy(model.state_dict())
            
            if phase == 'val' and epoch_acc >= best_acc:
                best_ACCmodel_wts_later = copy.deepcopy(model.state_dict())

            if phase == 'val' and epoch_f1 > best_f1:
                best_f1 = epoch_f1
                best_F1model_wts = copy.deepcopy(model.state_dict())
            
            if phase == 'val' and epoch_f1 >= best_f1:
                best_F1model_wts_later = copy.deepcopy(model.state_dict())

            if phase == 'val' and (epoch_acc+epoch_f1)/2 > best_accf1:
                best_accf1 = (epoch_acc+epoch_f1)/2
                best_ACCF1model_wts = copy.deepcopy(model.state_dict())
            
            if phase == 'val' and (epoch_acc+epoch_f1)/2 >= best_accf1:
                best_ACCF1model_wts_later = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))
    print('Best val F1: {:4f}'.format(best_f1))
    print('Best val Acc+F1: {:4f}'.format(best_accf1))
    print()

    plt.rcParams["figure.figsize"] = (8,8)
    fig, axs=plt.subplots(2)
    axs[0].set_title('model loss')
    axs[1].set_title('model accuracy')
    for ax in axs.flat:
        ax.set_ylim([0.0,1.0])
    axs[0].plot(train_loss,'r',val_loss,'g',)
    axs[1].plot(train_acc,'r',val_acc,'g')
    fig.tight_layout()
    for ax in axs.flat:
        leg=ax.legend(['train','val'])
    
    plt.savefig(file_name)
   
    # load best model weights
    accmodel.load_state_dict(best_ACCmodel_wts)
    accmodel_later.load_state_dict(best_ACCmodel_wts_later)

    f1model.load_state_dict(best_F1model_wts)
    f1model_later.load_state_dict(best_F1model_wts_later)

    accf1model.load_state_dict(best_ACCF1model_wts)
    accf1model_later.load_state_dict(best_ACCF1model_wts_later)
    
    return accmodel,accmodel_later,f1model,f1model_later,accf1model,accf1model_later

def train_model(model, criterion, optimizer, scheduler, dataloaders, dataset_sizes,device, num_epochs, file_name):
    since = time.time()
    model_later=copy.deepcopy(model)
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    
    train_loss=[]
    val_loss=[]

    train_acc=[]
    val_acc=[]
    
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        
        # Each epoch has a training and validation phase
        for phase in ['train','val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0
            running_cm=0

            # Iterate over data.
            for inputs, labels, file_names in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs) #file_names
                    _, preds = torch.max(outputs, 1)
                    
                    #labels=labels.float()
                    #labels_loss=labels_loss.view(labels.size()[0],1)
                    #outputs=outputs.view(outputs.size()[0])

                    '''print('***********outputs************')
                    print(outputs)
                    print('***********preds************')
                    print(preds)
                    print('***********labels************')
                    print(labels)
                    print()'''

                    
                    #print(outputs.size(),labels_loss.size())
                    #print(outputs)
                    #print(F.sigmoid(outputs))

                    loss=criterion(outputs,labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += (loss.item()) * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                running_cm+=confusion_matrix(labels.cpu(), preds.cpu(),labels=[0,1])

            if phase == 'train':
                scheduler.step()
            
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            epoch_sen = (running_cm[1][1]/(running_cm[1][1]+running_cm[1][0]))
            epoch_pre = (running_cm[1][1]/(running_cm[1][1]+running_cm[0][1]))
            epoch_f1=(2*epoch_sen*epoch_pre)/(epoch_sen+epoch_pre)
            #epoch_ppv=(running_cm[1][1]/(running_cm[1][1]+running_cm[0][1]))#(tp/(tp+fp))
            
            if phase=='train':
                train_loss.append(epoch_loss)
                train_acc.append(np.float(epoch_acc))
            else:
                val_loss.append(epoch_loss)
                val_acc.append(np.float(epoch_acc))

            #print(phase,train_loss,val_loss)
            print('{} Loss: {:.4f} Acc: {:.4f} Acc+Sen: {:.4f}'.format(
                phase, epoch_loss, epoch_acc,(epoch_acc*0.5+epoch_sen*0.5)))
            
            # deep copy the model
            if phase == 'val' and (epoch_acc*0.5+epoch_sen*0.5) > best_acc:
                best_acc = (epoch_acc*0.5+epoch_sen*0.5)
                best_model_wts = copy.deepcopy(model.state_dict())
            
            if phase == 'val' and (epoch_acc*0.5+epoch_sen*0.5) >= best_acc:
                #best_acc = epoch_acc
                best_model_wts_later = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc+Sen: {:4f}'.format(best_acc))

    plt.rcParams["figure.figsize"] = (8,8)
    fig, axs=plt.subplots(2)
    axs[0].set_title('model loss')
    axs[1].set_title('model accuracy')
    for ax in axs.flat:
        ax.set_ylim([0.0,1.0])
    axs[0].plot(train_loss,'r',val_loss,'g',)
    axs[1].plot(train_acc,'r',val_acc,'g')
    fig.tight_layout()
    for ax in axs.flat:
        leg=ax.legend(['train','val'])
    
    plt.savefig(file_name)
    '''
    #loss graph
    plt.figure()
    plt.rcParams["figure.figsize"] = (10,5)
    plt.title('model loss')
    plt.plot(train_loss,'r',val_loss,'g')
    plt.ylim([0.0, 1.0])
    plt.legend(['train','val'])
    plt.show()
    if file_name != None:
        #print(file_name+datetime.today().strftime('_LOSS_%m-%d-%H:%M')+'.png',' Saved!!')
        plt.savefig('/home/yslee/CNN_Prediction_Endo/figs/'+file_name+datetime.today().strftime('_LOSS_%m-%d-%H:%M')+'.png')

    plt.figure()
    plt.rcParams["figure.figsize"] = (10,5)
    plt.title('model accuracy')
    plt.plot(train_acc,'r',val_acc,'g')
    plt.legend(['train','val'])
    plt.show()
    if file_name != None:
        #print(file_name+datetime.today().strftime('_ACC_%m-%d-%H:%M')+'.png',' Saved!!')
        plt.savefig('/home/yslee/CNN_Prediction_Endo/figs/'+file_name+datetime.today().strftime('_ACC_%m-%d-%H:%M')+'.png')
    '''
    # load best model weights
    model.load_state_dict(best_model_wts)
    model_later.load_state_dict(best_model_wts_later)
    
    return model,model_later

def test_model(model,dataloaders,device,score_save_path_txt):
    CM=0
    model.to(device)
    model.eval()
    with torch.no_grad():
        for data in dataloaders['test']:
            images, labels, file_name = data
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images) #file_name
            preds = torch.argmax(outputs.data, 1)

            CM+=confusion_matrix(labels.cpu(), preds.cpu(),labels=[0,1])
            
        tn=CM[0][0]
        tp=CM[1][1]
        fp=CM[0][1]
        fn=CM[1][0]
        acc=np.sum(np.diag(CM)/np.sum(CM))
        sensitivity=tp/(tp+fn)
        precision=tp/(tp+fp)
        
        f = open(score_save_path_txt, 'w')
        f.write('Testset Accuracy(mean): %f %%' % (100 * acc)+'\n')
        f.write('Confusion Matirx\n')
        f.write('TN CM[0][0] : '+str(tn)+'\n')
        f.write('TP CM[1][1] : '+str(tp)+'\n')
        f.write('FP CM[0][1] : '+str(fp)+'\n')
        f.write('FN CM[1][0] : '+str(fn)+'\n')
        f.write('- Sensitivity : '+str((tp/(tp+fn))*100)+'\n')
        f.write('- Specificity : '+str((tn/(tn+fp))*100)+'\n')
        f.write('- Precision: '+str((tp/(tp+fp))*100)+'\n')
        f.write('- NPV: '+str((tn/(tn+fn))*100)+'\n')
        f.write('- F1 : '+str(((2*sensitivity*precision)/(sensitivity+precision))*100)+'\n')
        f.close()
        
        print('\nTestset Accuracy(mean): %f %%' % (100 * acc))
        print()
        print('Confusion Matirx : ')
        print(CM)
        print('- Sensitivity : ',(tp/(tp+fn))*100)
        print('- Specificity : ',(tn/(tn+fp))*100)
        print('- Precision: ',(tp/(tp+fp))*100)
        print('- NPV: ',(tn/(tn+fn))*100)
        print('- F1 : ',((2*sensitivity*precision)/(sensitivity+precision))*100)
        print()
                
    return acc, CM

'''def test_model2(model,dataloaders,device):
    CM=0

    model.eval()
    with torch.no_grad():
        for data in dataloaders['test']:
            images, labels, file_name = data
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            preds = torch.argmax(outputs.data, 1)

            CM+=confusion_matrix(labels.cpu(), preds.cpu(),labels=[0,1,2])
            
        acc=np.sum(np.diag(CM)/np.sum(CM))
        
        print('\nTestset Accuracy(mean): %f %%' % (100 * acc))
        print()
        print('Confusion Matirx : ')
        print(CM)
                
    return acc, CM'''

'''def get_clinical_features(path='/home/NAS_mount/yslee/dataset/clinical_features_all.csv'):
    df=pd.read_csv(path,encoding='CP949')
    df=df[['P_No','CV','PF','ST','PA']]
    
    label_file=pd.read_csv('/home/NAS_mount/yslee/dataset/premolar_labels_0810.csv')
    label_file=label_file.set_index('PatientID_new')
    
    P_No_old_list=[]
    label_list=[]
    for i in range(0,len(df)):
        P_No_old_list.append(label_file['PatientID'][df['P_No'][i]])
        label_list.append(label_file['Result'][df['P_No'][i]])
    df=pd.concat([pd.DataFrame(P_No_old_list),df,pd.DataFrame(label_list)],axis=1)

    df.columns=['P_No_old','P_No','CV','PF','ST','PA','Result']
    df=df.set_index('P_No_old')
    
    features=df[['CV','PF','ST','PA']]
    label=df['Result']

    label=label.astype(int)

    return features'''

'''def get_clinical_features(path='/home/NAS_mount/yslee/dataset/clinical_features_0922.csv'):
    df=pd.read_csv(path,encoding='CP949')
    df=df[['P_No','CV','PF','ST','PA','FV','CD','RR','PS','TM','BRG']]
    
    label_file=pd.read_csv('/home/NAS_mount/yslee/dataset/premolar_labels_0810.csv')
    label_file=label_file.set_index('PatientID_new')
    
    P_No_old_list=[]
    label_list=[]
    for i in range(0,len(df)):
        P_No_old_list.append(label_file['PatientID'][df['P_No'][i]])
        label_list.append(label_file['Result'][df['P_No'][i]])
    df=pd.concat([pd.DataFrame(P_No_old_list),df,pd.DataFrame(label_list)],axis=1)

    df.columns=['P_No_old','P_No','CV','PF','ST','PA','FV','CD','RR','PS','TM','BRG','Result']
    df=df.set_index('P_No')
    features=df[['CV','PF','ST','PA','FV','CD','RR','PS','TM','BRG']]
    label=df['Result']

    label=label.astype(int)

    return features'''

''' def get_clinical_features(path='/home/NAS_mount/yslee/dataset/clinical_features_0922.csv'):
    df=pd.read_csv(path,encoding='CP949')
    df=df[['P_No','CV','PF','ST','PA','FV','CD','RR','PS','TM','BRG']]
    
    label_file=pd.read_csv('/home/NAS_mount/yslee/dataset/premolar_labels_0810.csv')
    label_file=label_file.set_index('PatientID_new')
    
    P_No_old_list=[]
    label_list=[]
    for i in range(0,len(df)):
        P_No_old_list.append(label_file['PatientID'][df['P_No'][i]])
        label_list.append(label_file['Result'][df['P_No'][i]])
    df=pd.concat([pd.DataFrame(P_No_old_list),df,pd.DataFrame(label_list)],axis=1)

    df.columns=['P_No_old','P_No','CV','PF','ST','PA','FV','CD','RR','PS','TM','BRG','Result']
    df=df.set_index('P_No')
    features=df[['CV','PF','ST','PA','FV','CD','RR','PS','TM','BRG']]
    label=df['Result']

    label=label.astype(int)
    features=features.astype(np.float32)
    
    return features'''

