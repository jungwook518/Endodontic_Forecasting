import torch
import torch.nn as nn
from torch.nn.modules.dropout import Dropout
from torch.nn.modules.pooling import AvgPool2d
import torchvision
import torch.nn.functional as F
import copy
import numpy as np

class Net2(nn.Module):
    def __init__(self):
        super(Net2,self).__init__()
        self.conv0=nn.Conv2d(in_channels=2,out_channels=16,kernel_size=(5,5),stride=(2,2),padding=(2,2),bias=False)
        self.bn=nn.BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.maxpool0=nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
        self.relu=nn.ReLU(inplace=True)

        self.layer1=Block(16,16,maxpool=True)
        self.layer2=Block(16,32,maxpool=False)
        self.layer_downsample1=DownsampleLayer(16,32) #,stride=(1,1)

        self.attention_matmul=SelfAttention(32)

        self.layer3=Block(32,32,maxpool=True)
        self.layer4=Block(32,64,maxpool=False)
        self.layer_downsample2=DownsampleLayer(32,64)

        self.layer5=Block(64,64,maxpool=True)
        self.layer6=Block(64,128,maxpool=False)
        self.layer_downsample3=DownsampleLayer(64,128)

        self.layer7=Block(128,128,maxpool=False)
        self.layer8=Block(128,256,maxpool=False)
        self.layer_downsample4=DownsampleLayer(128,256,stride=(1,1))

        self.avgpool=nn.AdaptiveAvgPool2d((1,1))
        self.fc=nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256*1*1,256*1*1//2,bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256*1*1//2,2,bias=True)
        )

    def forward(self,x):
        x=self.conv0(x)
        x=self.bn(x)
        x=self.maxpool0(x)
        x=self.relu(x)

        layer_identity1=x
        x=self.layer1(x)
        x=self.layer2(x)
        x=x.clone()+self.layer_downsample1(layer_identity1)

        x=self.attention_matmul(x)
        x=self.relu(x)

        layer_identity2=x
        x=self.layer3(x)
        x=self.layer4(x)
        x=x.clone()+self.layer_downsample2(layer_identity2)     

        layer_identity3=x
        x=self.layer5(x)
        x=self.layer6(x)
        x=x.clone()+self.layer_downsample3(layer_identity3)

        layer_identity4=x
        x=self.layer7(x)
        x=self.layer8(x)
        x=x.clone()+self.layer_downsample4(layer_identity4)

        x=self.avgpool(x)
        x=x.view(-1,self.num_flat_features(x))
        x=self.fc(x)

        return x

    def num_flat_features(self,x):
        size=x.size()[1:]
        num_features=1
        for s in size:
            num_features*=s
        
        return num_features
        
class ClinicalModel(nn.Module):
    def __init__(self,model, clinical_features,device,):
        super(ClinicalModel, self).__init__()
        self.model = model
        self.model.fc=nn.Identity()        
        self.fc=nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256+len(clinical_features.columns),256//2,bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256//2,2,bias=True)
        )
        self.clinical_features=clinical_features
        self.device=device

    def forward(self,img,patID):
        patID_list=[p.split('_')[0] for p in patID]


        clinical_tensor=[]
        for i in range(0,len(patID_list)):
            feature_list=self.clinical_features.loc[patID_list[i]].values
            feature_list=feature_list.astype(np.float32)
            clinical_tensor.append(feature_list)

        clinical_tensor=torch.tensor(clinical_tensor)
        clinical_tensor=clinical_tensor.to(self.device)

        x=self.model(img)
        '''print(x.size())
        print(clinical_tensor.size())'''

        x=torch.cat((x,clinical_tensor),1)
        #print(x.size())
        x=self.fc(x)
        #print(x)
        return x

class SelfAttention(nn.Module):
    def __init__(self,n_channels):
        super(SelfAttention,self).__init__()
        self.query,self.key,self.value=[self._conv(n_channels,c) for c in (n_channels,n_channels,n_channels)]
        self.gamma=nn.Parameter(torch.tensor([0.]))
        
    def _conv(self,n_in,n_out):
        return nn.Conv2d(n_in,n_out,kernel_size=(1,1))
   
    def forward(self,x):
        size=x.size()

        f,g,h=self.query(x),self.key(x),self.value(x)
        f_size=f.size()
        g_size=g.size()
        h_size=h.size()

        f=f.view(*f_size[:2],-1)
        g=g.view(*g_size[:2],-1)
        h=h.view(*h_size[:2],-1)
        transposed=f.transpose(1,2)
       
        softmax=nn.Softmax(dim=1)
        beta=softmax(torch.bmm(transposed,g))
        o=self.gamma*(torch.bmm(h,beta))
        o=o.view(*size)
        o=o+x
        
        return o.contiguous()

class MultiheadAttention(nn.Module):
    def __init__(self,n_channels):
        super(MultiheadAttention,self).__init__()
        self.query,self.key,self.value=[self._conv(n_channels,c) for c in (n_channels//8,n_channels//8,n_channels)]
        self.query2,self.key2=[self._conv(n_channels,c) for c in (n_channels//8,n_channels//8)]

        self.gamma=nn.Parameter(torch.tensor([0.]))
        
    def _conv(self,n_in,n_out):
        return nn.Conv2d(n_in,n_out,kernel_size=(1,1))
   
    def forward(self,x):
        size=x.size()

        f,g,h=self.query(x),self.key(x),self.value(x)
        f2,g2=self.query2(x),self.key2(x)
        f_size=f.size()
        g_size=g.size()
        h_size=h.size()

        f=f.view(*f_size[:2],-1)
        g=g.view(*g_size[:2],-1)
        h=h.view(*h_size[:2],-1)
        f2=f2.view(*f_size[:2],-1)
        g2=g2.view(*g_size[:2],-1)

        transposed=f.transpose(1,2)
        transposed2=f2.transpose(1,2)
       
        softmax=nn.Softmax(dim=1)
        beta=torch.bmm(transposed,g)
        beta+=torch.bmm(transposed2,g2)
        beta=softmax(beta)
        o=self.gamma*(torch.bmm(h,beta))
        o=o.view(*size)
        o=o+x
        
        return o.contiguous()

class SelfAttention_dot(nn.Module):
    def __init__(self,n_channels):
        super(SelfAttention_dot,self).__init__()
        self.query,self.key,self.value=[self._conv(n_channels,c) for c in (n_channels,n_channels,n_channels)]
        self.gamma=nn.Parameter(torch.tensor([0.]))
        
    def _conv(self,n_in,n_out):
        return nn.Conv2d(n_in,n_out,kernel_size=(1,1))
   
    def forward(self,x):
        size=x.size()

        f,g,h=self.query(x),self.key(x),self.value(x)
        transposed=f.transpose(2,3)
        #transposed=torch.rot90(f.transpose(2,3),3,[2,3])
        softmax=nn.Softmax(dim=1)
        beta=softmax(transposed*g)
        o=self.gamma*(h*beta)
        o=o.view(*size)
        o=o+x
        
        return o.contiguous()

class MultiheadAttention_dot(nn.Module):
    def __init__(self,n_channels):
        super(MultiheadAttention_dot,self).__init__()
        self.query,self.key,self.value=[self._conv(n_channels,c) for c in (n_channels,n_channels,n_channels)]
        self.query2,self.key2=[self._conv(n_channels,c) for c in (n_channels,n_channels)]
        #self.query3,self.key3=[self._conv(n_channels,c) for c in (n_channels,n_channels)]
        self.gamma=nn.Parameter(torch.tensor([0.]))
        
    def _conv(self,n_in,n_out):
        return nn.Conv2d(n_in,n_out,kernel_size=(1,1))
   
    def forward(self,x):
        size=x.size()

        f,g,h=self.query(x),self.key(x),self.value(x)
        f2,g2=self.query2(x),self.key2(x)
        #f3,g3=self.query3(x),self.key3(x)
        transposed=f.transpose(2,3)
        transposed2=torch.rot90(f2.transpose(2,3),3,[2,3])
        #transposed3=torch.rot90(f3,3,[2,3])

        softmax=nn.Softmax(dim=1)
        beta=transposed*g
        beta+=transposed2*g2
        #beta+=transposed3*g3
        beta=softmax(beta)
        o=self.gamma*(h*beta)
        o=o.view(*size)
        o=o+x
        
        return o.contiguous()

class AttentionResNet18(nn.Module):
    def __init__(self):
        super(AttentionResNet18,self).__init__()
        resnet= torchvision.models.resnet18(pretrained=False)
        self.conv1=nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.bn1=resnet.bn1
        self.relu=resnet.relu
        self.maxpool=resnet.maxpool

        self.layer1=resnet.layer1
        self.layer2=resnet.layer2

        #self.attention1=SelfAttention(128)
        self.attention1=MultiheadAttention(128)

        self.layer3=resnet.layer3
        self.layer4=resnet.layer4

        self.attention2=MultiheadAttention_dot(512)
        #self.attention2=SelfAttention(512)

        self.avgpool=resnet.avgpool
        self.fc=nn.Linear(in_features=512, out_features=2, bias=True)

    def forward(self,x):
        x=self.conv1(x)
        x=self.bn1(x)
        x=self.relu(x)
        x=self.maxpool(x)

        x=self.layer1(x)
        x=self.layer2(x)

        x=self.attention1(x)
        x=self.relu(x)

        x=self.layer3(x)
        x=self.layer4(x)

        x=self.attention2(x)
        x=self.relu(x)

        x=self.avgpool(x)
        x=x.view(-1,self.num_flat_features(x))
        x=self.fc(x)

        return x

    def num_flat_features(self,x):
        size=x.size()[1:]
        num_features=1
        for s in size:
            num_features*=s
        
        return num_features

class AttentionResNet18_2(nn.Module):
    def __init__(self):
        super(AttentionResNet18_2,self).__init__()
        resnet= torchvision.models.resnet18(pretrained=False)
        self.conv1=nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.bn1=resnet.bn1
        self.relu=resnet.relu
        self.maxpool=resnet.maxpool

        self.layer1=resnet.layer1
        self.layer2=resnet.layer2

        self.attention1=MultiheadAttention_dot(128)
        self.attention2=MultiheadAttention(128)

        self.layer3=resnet.layer3
        self.layer4=resnet.layer4

        self.attention3=SelfAttention_dot(512)

        self.avgpool=resnet.avgpool
        self.fc=nn.Linear(in_features=512, out_features=2, bias=True)

    def forward(self,x):
        x=self.conv1(x)
        x=self.bn1(x)
        x=self.relu(x)
        x=self.maxpool(x)

        x=self.layer1(x)
        x=self.layer2(x)

        x=self.attention1(x)
        x=self.relu(x)

        x=self.attention2(x)
        x=self.relu(x)

        x=self.layer3(x)
        x=self.layer4(x)

        x=self.attention3(x)
        x=self.relu(x)

        x=self.avgpool(x)
        x=x.view(-1,self.num_flat_features(x))
        x=self.fc(x)

        return x

    def num_flat_features(self,x):
        size=x.size()[1:]
        num_features=1
        for s in size:
            num_features*=s
        
        return num_features

class AttentionResNet18_3(nn.Module):
    def __init__(self):
        super(AttentionResNet18_3,self).__init__()
        resnet= torchvision.models.resnet18(pretrained=False)
        self.conv1=nn.Conv2d(2, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.bn1=resnet.bn1
        self.relu=nn.LeakyReLU(0.01,inplace=True)#resnet.relu
        self.maxpool=resnet.maxpool

        self.layer1=resnet.layer1
        self.layer2=resnet.layer2

        #self.attention1=SelfAttention_dot(128)
        self.attention2=SelfAttention(128) #MultiheadAttention

        self.layer3=resnet.layer3
        self.layer4=copy.deepcopy(resnet.layer3) #resnet.layer4
        self.layer4[0].conv1=nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        self.layer4[0].downsample[0]=nn.Conv2d(256, 256, kernel_size=(1, 1), stride=(2, 2), bias=False)

        for layer in [self.layer1,self.layer2,self.layer3,self.layer4 ]: #
            for i in range(0,2):
                layer[i].relu=nn.LeakyReLU(0.01,inplace=True)

        #self.attention3=SelfAttention_dot(256)
        
        self.avgpool=resnet.avgpool #nn.AdaptiveAvgPool2d((3,3))
        #self.dropout=nn.Dropout(p=0.5)
        self.fc=nn.Linear(in_features=1*1*256, out_features=1*1*2, bias=True)
        '''self.leakyrelu=nn.LeakyReLU(0.01)
        self.fc2=nn.Linear(in_features=1*1*64,out_features=2)'''
        

    def forward(self,x):
        x=self.conv1(x)
        x=self.bn1(x)
        x=self.relu(x)
        x=self.maxpool(x)

        x=self.layer1(x)
        x=self.layer2(x)

        '''x=self.attention1(x)
        x=self.relu(x)'''

        x=self.attention2(x)
        x=self.relu(x)

        x=self.layer3(x)
        x=self.layer4(x)

        '''x=self.attention3(x)
        x=self.relu(x)'''

        x=self.avgpool(x)
        x=x.view(-1,self.num_flat_features(x))
        #x=self.dropout(x)
        x=self.fc(x)
        '''x=self.leakyrelu(x)
        x=self.dropout(x)
        x=self.fc2(x)'''

        return x

    def num_flat_features(self,x):
        size=x.size()[1:]
        num_features=1
        for s in size:
            num_features*=s
        
        return num_features

class Block(nn.Module):
    def __init__(self,in_channels,out_channels,maxpool,stride=(1,1)):
        super(Block,self).__init__()
        self.maxpool_bool=maxpool

        self.conv1=nn.Conv2d(in_channels,out_channels,kernel_size=(3,3),stride=(1,1),padding=(1,1),bias=False)
        self.bn1=nn.BatchNorm2d(out_channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu1=nn.ReLU(inplace=True)

        self.conv2=nn.Conv2d(out_channels,out_channels,kernel_size=(3,3),stride=stride,padding=(1,1),bias=False)
        self.bn2=nn.BatchNorm2d(out_channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        if self.maxpool_bool==True:
            self.maxpool=nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
        self.relu2=nn.ReLU(inplace=True)

        self.downsample1=nn.Sequential(
            nn.Conv2d(in_channels,out_channels,kernel_size=(1,1),stride=(1,1),bias=False),
            nn.BatchNorm2d(out_channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
        self.downsample2=nn.Sequential(
            nn.Conv2d(out_channels,out_channels,kernel_size=(1,1),stride=(2,2),bias=False),
            nn.BatchNorm2d(out_channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
        self.downsample3=nn.Sequential(
            nn.Conv2d(out_channels,out_channels,kernel_size=(1,1),stride=(1,1),bias=False),
            nn.BatchNorm2d(out_channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
        self.downsample4=nn.Sequential(
            nn.Conv2d(in_channels,out_channels,kernel_size=(1,1),stride=(2,2),bias=False),
            nn.BatchNorm2d(out_channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )

    def forward(self,x):
        identity1=x

        x=self.conv1(x)
        x=self.bn1(x)

        x+=self.downsample1(identity1)
        x=self.relu1(x)

        identity2=x

        x=self.conv2(x)
        x=self.bn2(x)
        if self.maxpool_bool==True:
            x=self.maxpool(x)
            x+=self.downsample2(identity2)
            x+=self.downsample4(identity1)
        else:
            x+=self.downsample3(identity2)
            x+=self.downsample1(identity1)

        x=self.relu2(x)

        return x

class DownsampleLayer(nn.Module):
    def __init__(self,in_channels,out_channels,stride=(2,2)):
        super(DownsampleLayer,self).__init__()
        self.conv=nn.Conv2d(in_channels,out_channels,kernel_size=(1,1),stride=stride,bias=False)
        self.bn=nn.BatchNorm2d(out_channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu=nn.ReLU(inplace=True)

    def forward(self,x):
        x=self.conv(x)
        x=self.bn(x)
        x=self.relu(x)

        return x

class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.conv0=nn.Conv2d(in_channels=2,out_channels=16,kernel_size=(5,5),stride=(2,2),padding=(2,2),bias=False)
        self.bn=nn.BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.maxpool0=nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
        self.relu=nn.ReLU(inplace=True)

        self.layer1=Block(16,16,maxpool=True)
        self.layer2=Block(16,32,maxpool=False)
        self.layer_downsample1=DownsampleLayer(16,32) #,stride=(1,1)

        #self.attention_dot1=SelfAttention_dot(32)
        self.attention_matmul=SelfAttention(32)

        self.layer3=Block(32,32,maxpool=True)
        self.layer4=Block(32,64,maxpool=False)
        self.layer_downsample2=DownsampleLayer(32,64)

        self.layer5=Block(64,64,maxpool=True)
        self.layer6=Block(64,128,maxpool=False)
        self.layer_downsample3=DownsampleLayer(64,128)

        self.layer7=Block(128,128,maxpool=False)
        self.layer8=Block(128,256,maxpool=False)
        self.layer_downsample4=DownsampleLayer(128,256,stride=(1,1)) #

        #self.attention_dot2=SelfAttention_dot(256)

        self.avgpool=nn.AdaptiveAvgPool2d((1,1))
        self.fc=nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256*1*1,256*1*1//2,bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256*1*1//2,2,bias=True)
        )
        '''nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(64*3*3,2,bias=True)'''

    def forward(self,x):
        x=self.conv0(x)
        x=self.bn(x)
        x=self.maxpool0(x)
        x=self.relu(x)

        layer_identity1=x
        x=self.layer1(x)
        x=self.layer2(x)
        #layer_identity1=F.interpolate(layer_identity1,size=(76,76))
        x=x.clone()+self.layer_downsample1(layer_identity1)

        '''x=self.attention_dot1(x)
        x=self.relu(x)'''
        #attention_identity=x
        x=self.attention_matmul(x)
        #x+=attention_identity
        x=self.relu(x)

        layer_identity2=x
        x=self.layer3(x)
        x=self.layer4(x)
        #layer_identity2=F.interpolate(layer_identity2,scale_factor=0.5)
        x=x.clone()+self.layer_downsample2(layer_identity2)     

        layer_identity3=x
        x=self.layer5(x)
        x=self.layer6(x)
        #layer_identity3=F.interpolate(layer_identity3,scale_factor=0.5)
        x=x.clone()+self.layer_downsample3(layer_identity3)

        layer_identity4=x
        x=self.layer7(x)
        x=self.layer8(x)
        #layer_identity4=F.interpolate(layer_identity4,size=(10,10))
        x=x.clone()+self.layer_downsample4(layer_identity4)

        '''x=self.attention_dot2(x)
        x=self.relu(x)'''

        x=self.avgpool(x)
        x=x.view(-1,self.num_flat_features(x))
        x=self.fc(x)

        return x

    def num_flat_features(self,x):
        size=x.size()[1:]
        num_features=1
        for s in size:
            num_features*=s
        
        return num_features



'''class Net2(nn.Module):
    def __init__(self):
        super(Net2,self).__init__()
        self.conv0=nn.Conv2d(in_channels=2,out_channels=32,kernel_size=(5,5),stride=(2,2),padding=(2,2),bias=False)
        self.bn=nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.maxpool0=nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
        self.relu=nn.ReLU(inplace=True)

        self.layer1=Block(32,32,maxpool=True)
        self.layer2=Block(32,64,maxpool=False)
        self.layer_downsample1=DownsampleLayer(32,64) #,stride=(1,1)

        self.layer3=Block(64,64,maxpool=True)
        self.layer4=Block(64,128,maxpool=False)
        self.layer_downsample2=DownsampleLayer(64,128)

        self.layer5=Block(128,128,maxpool=True)
        self.layer6=Block(128,256,maxpool=False)
        self.layer_downsample3=DownsampleLayer(128,256)

        self.avgpool=nn.AdaptiveAvgPool2d((1,1))
        self.fc=nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256*1*1,256*1*1//2,bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256*1*1//2,2,bias=True)
        )

    def forward(self,x):
        x=self.conv0(x)
        x=self.bn(x)
        x=self.maxpool0(x)
        x=self.relu(x)

        layer_identity1=x
        x=self.layer1(x)
        x=self.layer2(x)
        #layer_identity1=F.interpolate(layer_identity1,size=(76,76))
        x=x.clone()+self.layer_downsample1(layer_identity1)

        layer_identity2=x
        x=self.layer3(x)
        x=self.layer4(x)
        #layer_identity2=F.interpolate(layer_identity2,scale_factor=0.5)
        x=x.clone()+self.layer_downsample2(layer_identity2)

        layer_identity3=x
        x=self.layer5(x)
        x=self.layer6(x)
        #layer_identity3=F.interpolate(layer_identity3,scale_factor=0.5)
        x=x.clone()+self.layer_downsample3(layer_identity3)

        x=self.avgpool(x)
        x=x.view(-1,self.num_flat_features(x))
        x=self.fc(x)

        return x

    def num_flat_features(self,x):
        size=x.size()[1:]
        num_features=1
        for s in size:
            num_features*=s
        
        return num_features'''
     
class Ensemble(nn.Module):
    def __init__(self):
        super(Ensemble,self).__init__()
        self.model1=torchvision.models.resnet18(pretrained=True)
        self.model1.conv1=nn.Conv2d(2, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.model2=Net()

        self.model1.fc=nn.Identity()
        self.model2.fc=nn.Identity()

        self.fc=nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(512+256,(512+256)//4,bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear((512+256)//4,2,bias=True)
        )
    def forward(self,x):
        x1=self.model1(x)
        x2=self.model2(x)

        x=torch.cat((x1,x2),1)

        x=self.fc(x)
        return x

class WrappedModel(nn.Module):
    def __init__(self):
        super(WrappedModel, self).__init__()
        self.module = Net() # that I actually define.
    def forward(self, x):
        return self.module(x)     

class DenseAttentionResNet18(nn.Module):
    def __init__(self):
        super(DenseAttentionResNet18,self).__init__()
        resnet= torchvision.models.resnet18(pretrained=False)
        self.conv1=nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.bn1=resnet.bn1
        self.relu=resnet.relu
        self.maxpool=resnet.maxpool

        self.layer1=resnet.layer1
        self.layer2=resnet.layer2
        self.downsample1=nn.Sequential(
          nn.Conv2d(64, 128, kernel_size=(1, 1), stride=(2, 2), bias=False),
          nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
          nn.ReLU(inplace=True)
        )

        self.attention1=SelfAttention_dot(128)
        self.attention2=MultiheadAttention(128)

        self.layer3=resnet.layer3
        self.layer4=resnet.layer4

        self.downsample2=nn.Sequential(
          nn.Conv2d(128, 512, kernel_size=(1, 1), stride=(2, 2), bias=False),
          nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
          nn.ReLU(inplace=True),
          nn.AvgPool2d((3,3),2,1)
        )

        self.attention3=SelfAttention_dot(512)
        
        self.avgpool=resnet.avgpool
        self.fc=nn.Linear(in_features=512, out_features=2, bias=True)

    def forward(self,x):
        x=self.conv1(x)
        x=self.bn1(x)
        x=self.relu(x)
        x=self.maxpool(x)

        identity=x

        x=self.layer1(x)
        x=self.layer2(x)
        
        identity=self.downsample1(identity)
        x+=identity.clone()

        x=self.attention1(x)
        x=self.relu(x)

        x=self.attention2(x)
        x=self.relu(x)

        identity=x

        x=self.layer3(x)
        x=self.layer4(x)

        identity=self.downsample2(identity)
        x+=identity.clone()

        x=self.attention3(x)
        x=self.relu(x)

        x=self.avgpool(x)
        x=x.view(-1,self.num_flat_features(x))
        x=self.fc(x)

        return x

    def num_flat_features(self,x):
        size=x.size()[1:]
        num_features=1
        for s in size:
            num_features*=s
        
        return num_features

class ResNet18(nn.Module):
    def __init__(self):
        super(ResNet18,self).__init__()
        self.model= torchvision.models.resnet18(pretrained=False)
        self.conv1=nn.Conv2d(2, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.bn1=self.model.bn1
        self.relu=self.model.relu
        self.maxpool=self.model.maxpool

        self.layer1=self.model.layer1
        self.layer2=self.model.layer2
        self.layer3=self.model.layer3
        #self.layer4=self.model.layer4

        self.avgpool=self.model.avgpool
        self.fc=nn.Linear(in_features=256, out_features=2, bias=True)
        
    def forward(self,x):
        x=self.conv1(x)
        x=self.bn1(x)
        x=self.relu(x)
        x=self.maxpool(x)

        x=self.layer1(x)
        x=self.layer2(x)
        x=self.layer3(x)
        #x=self.layer4(x)

        x=self.avgpool(x)
        x=x.view(-1,self.num_flat_features(x))
        x=self.fc(x)

        return x

    def num_flat_features(self,x):
        size=x.size()[1:]
        num_features=1
        for s in size:
            num_features*=s
        
        return num_features

'''
class AttentionResNet14(nn.Module):
    def __init__(self):
        super(AttentionResNet14,self).__init__()
        resnet= torchvision.models.resnet18(pretrained=False)
        self.conv1=nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.bn1=resnet.bn1
        self.relu=resnet.relu
        self.maxpool=resnet.maxpool

        self.layer1=resnet.layer1
        self.layer2=resnet.layer2
        self.layer3=resnet.layer3
        self.layer4=resnet.layer4

        self.attention1=SelfAttention(256)#512

        self.avgpool=nn.AdaptiveAvgPool2d((3,3))#resnet.avgpool
        self.fc=nn.Linear(in_features=3*3*256, out_features=2, bias=True)
      
        
    def forward(self,x):
        x=self.conv1(x)
        x=self.bn1(x)
        x=self.relu(x)
        x=self.maxpool(x)

        x=self.layer1(x)
        x=self.layer2(x)
        x=self.layer3(x)
        #x=self.layer4(x)

        x=self.attention1(x)
        x=self.relu(x)

        x=self.avgpool(x)
        x=x.view(-1,self.num_flat_features(x))
        x=self.fc(x)

        return x

    def num_flat_features(self,x):
        size=x.size()[1:]
        num_features=1
        for s in size:
            num_features*=s
        
        return num_features

class ResNet18(nn.Module):

    def __init__(self):
        super(AttentionResNet,self).__init__()
        self.model= torchvision.models.resnet18(pretrained=False)
        self.conv1=nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.bn1=self.model.bn1
        self.relu=self.model.relu
        self.maxpool=self.model.maxpool

        self.layer1=self.model.layer1
        self.layer2=self.model.layer2
        self.layer3=self.model.layer3
        self.layer4=self.model.layer4


        self.avgpool=self.model.avgpool
        self.model.fc=nn.Linear(in_features=256*3*3, out_features=2, bias=True)

        
        
    def forward(self,x):
        x=self.conv1(x)
        x=self.bn1(x)
        x=self.relu(x)
        x=self.maxpool(x)

        x=self.layer1(x)
        x=self.layer2(x)
        x=self.layer3(x)
        x=self.layer4(x)

        x=self.avgpool(x)
        x=x.view(-1,self.num_flat_features(x))
        x=self.fc(x)

        return x

    def num_flat_features(self,x):
        size=x.size()[1:]
        num_features=1
        for s in size:
            num_features*=s
        
        return num_features

class AttentionResNet18_dot(nn.Module):
    def __init__(self):
        super(AttentionResNet18_dot,self).__init__()
        resnet= torchvision.models.resnet18(pretrained=False)
        self.conv1=nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.bn1=resnet.bn1
        self.relu=resnet.relu
        self.maxpool=resnet.maxpool

        #self.attention1=SelfAttention_dot(64)

        self.layer1=resnet.layer1
        self.layer2=resnet.layer2
        self.layer3=resnet.layer3
        self.layer4=resnet.layer4

        self.attention2=SelfAttention_dot(512)

        self.avgpool=resnet.avgpool
        self.fc=nn.Linear(in_features=512, out_features=2, bias=True)
      
        
    def forward(self,x):
        x=self.conv1(x)
        x=self.bn1(x)
        x=self.relu(x)
        x=self.maxpool(x)

        x=self.layer1(x)

        #x=self.attention1(x)
        #x=self.relu(x)

        x=self.layer2(x)
        x=self.layer3(x)
        x=self.layer4(x)

        x=self.attention2(x)
        x=self.relu(x)

        x=self.avgpool(x)
        x=x.view(-1,self.num_flat_features(x))
        x=self.fc(x)

        return x

    def num_flat_features(self,x):
        size=x.size()[1:]
        num_features=1
        for s in size:
            num_features*=s
        
        return num_features

'''