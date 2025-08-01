# --------------------------------------------------------
# DenseFusion 6D Object Pose Estimation by Iterative Dense Fusion
# Licensed under The MIT License [see LICENSE for details]
# Written by Chen
# --------------------------------------------------------

import os
import sys
sys.path.insert(0, os.getcwd())
import argparse
import os
import random
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
# from datasets.PoseIndustrial6D.dataloader_6DPMultiResize3_JPz_trim import PoseDataset2 as PoseDataset_pallets
from datasets.PoseIndustrial6D.dataloader_20m import PoseDataset2 as PoseDataset
# from datasets.PoseIndustrial6D.dataloader_annotate import PoseDataset2 as PoseDataset

from lib.network_attnMOD_Manuel import PoseNetMultiCUSTOMPointsX, PoseNetMultiCUSTOM

from lib.lossMOD_Manuel import Lossv2
from lib.utils import setup_logger

import matplotlib as mpl
import matplotlib.pyplot as plt
from discordwebhook import Discord
import torch.nn.functional as F
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm

from torchvision import datasets, models, transforms


class InputResizer(nn.Module):
    def __init__(self, original_model, target_size=(224, 224)):
        super(InputResizer, self).__init__()
        # Add the resize and channel conversion layer
        self.target_size = target_size

        self.original_model = original_model

    def forward(self, x):
        # Resize and convert 1-channel input to 3-channel
        x = F.interpolate(x, size=self.target_size, mode='bilinear', align_corners=False)
        # Pass it to the pretrained model
        return self.original_model(x)


class InputResizerVIT(nn.Module):
    def __init__(self, original_model, target_size=(224, 224)):
        super(InputResizerVIT, self).__init__()
        # Add the resize and channel conversion layer
        self.target_size = target_size

        self.original_model = original_model

        #self.featuresC = original_model.conv_proj # Initial conv layer (patch embedding)
        #self.featuresE = original_model.encoder  # Transformer encoder blocks
        #self.cls_token=original_model.cls_token
    def forward(self, x):
        # Resize and convert 1-channel input to 3-channel
        x = F.interpolate(x, size=self.target_size, mode='bilinear', align_corners=False)
        #print(x.shape)
        # Pass it to the pretrained model
        #x=self.featuresC(x)
        #x=x.flatten(2).transpose(1, 2) 
         # Prepare CLS token (get the CLS token from the model)
        #batch_size = x.shape[0]
        #cls_token = self.cls_token.expand(batch_size, -1, -1)  # Shape: [batch_size, 1, embedding_dim]
    
        # Concatenate CLS token to the patch embeddings
        #x = torch.cat((cls_token, x), dim=1)
        #print(x.shape)
        #return self.original_model.forward_features(x)

        x = self.original_model.conv_proj(x)  # [batch_size, 768, 14, 14]
        
        # 2. Flatten the patches and add the [CLS] token
        x = x.flatten(2).transpose(1, 2)  # [batch_size, 196, 768]
        
        # Add the [CLS] token
        cls_token = self.original_model.class_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_token, x), dim=1)  # [batch_size, 197, 768]
        
        # 3. Add positional embeddings
        x = x + self.original_model.encoder.pos_embedding[:, : x.size(1), :]
        
        # 4. Pass through transformer encoder layers
        x = self.original_model.encoder(x)  # [batch_size, 197, 768]
        
        # The first token (x[:, 0, :]) is the [CLS] token that aggregates global information

        #print(x.shape)
        return x.transpose(1, 2) 

def load_pre_trained_model( network):
    emb=512
    if network == "ResNet18":
        pretrained_model = models.resnet18(weights="IMAGENET1K_V1") #"IMAGENET1K_V1"
        pretrained_model=nn.Sequential(*list(pretrained_model.children())[:-2])
    elif network == 'ResNet50':
        pretrained_model = models.resnet50(weights="IMAGENET1K_V1")
        pretrained_model=nn.Sequential(*list(pretrained_model.children())[:-2])
        #print (pretrained_model)
        emb=2048
    elif network == 'ResNet101':
        pretrained_model = models.resnet101(weights="IMAGENET1K_V1")
        pretrained_model=nn.Sequential(*list(pretrained_model.children())[:-2])
        emb=2048
    elif network == 'DenseNet':
        pretrained_model = models.densenet121(weights="IMAGENET1K_V1")
        pretrained_model= InputResizer(nn.Sequential(*list(pretrained_model.children())[:-1]))
        emb=1024
        #print(pretrained_model)
    elif network == 'VGG16':
        pretrained_model = models.vgg16_bn(weights="IMAGENET1K_V1")
        pretrained_model.classifier[0] = nn.Identity()
        #print(pretrained_model)
        pretrained_model = InputResizer(nn.Sequential(*list(pretrained_model.children())[:-1]))
    elif network == 'MobileNetV2':
        pretrained_model = models.mobilenet_v2(weights="IMAGENET1K_V1")
        pretrained_model.classifier[0] = nn.Identity()
        pretrained_model = nn.Sequential(*list(pretrained_model.children())[:-1])
        emb=1280
        # ----------------------------- NEW Backbones ----------------------------- #
    elif network == 'ConvNext_Small': # 50M parameters
        pretrained_model = models.convnext_small(weights="IMAGENET1K_V1")
        pretrained_model.classifier[2] = nn.Identity()
        pretrained_model = nn.Sequential(*list(pretrained_model.children())[:-1])
        emb=768
    elif network == 'ConvNext_base': # 88M parameters
        pretrained_model = models.convnext_base(weights=None)
        pretrained_model.classifier[2] = nn.Identity()
        pretrained_model = nn.Sequential(*list(pretrained_model.children())[:-1])
        emb=1024
    elif network == 'ConvNext_large': # 197M parameters
        pretrained_model = models.convnext_large(weights="IMAGENET1K_V1")
        pretrained_model.classifier[2] = nn.Identity()
        pretrained_model = nn.Sequential(*list(pretrained_model.children())[:-1])
    elif network == 'efficientnet_b1': # 7.8M parameters
        #pretrained_model = models.efficientnet_b1(weights="IMAGENET1K_V1")
        #pretrained_model.classifier[0] = nn.Identity()
        #pretrained_model = nn.Sequential(*list(pretrained_model.children())[:-1])
        pretrained_model = models.efficientnet_b1(weights="IMAGENET1K_V1")
        pretrained_model.classifier[0] = nn.Identity()
        pretrained_model = nn.Sequential(*list(pretrained_model.children())[:-1])
        emb=1280
    elif network == 'efficientnet_b4': # 19M parameters
        pretrained_model = models.efficientnet_b4(weights="IMAGENET1K_V1")
        pretrained_model.classifier[0] = nn.Identity()
        pretrained_model = nn.Sequential(*list(pretrained_model.children())[:-1])
    elif network == 'efficientnet_b7': # 66M parameters
        pretrained_model = models.efficientnet_b7(weights="IMAGENET1K_V1")
        pretrained_model.classifier[0] = nn.Identity()
        pretrained_model = nn.Sequential(*list(pretrained_model.children())[:-1])
    elif network == 'efficientnet_v2_s': # 21M parameters
        pretrained_model = models.efficientnet_v2_s(weights=None)
        pretrained_model.classifier[0] = nn.Identity()
        pretrained_model = nn.Sequential(*list(pretrained_model.children())[:-1])
        emb=1280
    elif network == 'efficientnet_v2_m': # 54M parameters
        pretrained_model = models.efficientnet_v2_m(weights="IMAGENET1K_V1")
        pretrained_model.classifier[0] = nn.Identity()
        pretrained_model = nn.Sequential(*list(pretrained_model.children())[:-1])
    elif network == 'efficientnet_v2_l': # 118M parameters
        pretrained_model = models.efficientnet_v2_l(weights="IMAGENET1K_V1")
        pretrained_model.classifier[0] = nn.Identity()
        pretrained_model = nn.Sequential(*list(pretrained_model.children())[:-1])
    elif network == 'resnext50_32x4d': # 25M parameters
        pretrained_model = models.resnext50_32x4d(weights=None)
        pretrained_model.fc = nn.Identity()
        pretrained_model = nn.Sequential(*list(pretrained_model.children())[:-1])
    elif network == 'resnext101_64x4d': # 83M parameters
        pretrained_model = models.resnext101_64x4d(weights="IMAGENET1K_V1")
        pretrained_model.fc = nn.Identity()
        pretrained_model = nn.Sequential(*list(pretrained_model.children())[:-1])
    elif network == 'swin_t': # 28M parameters
        pretrained_model = models.swin_t(weights="IMAGENET1K_V1")
        pretrained_model.flatten = nn.Identity()
        pretrained_model.head = nn.Identity()
        #print(pretrained_model)
        pretrained_model = nn.Sequential(*list(pretrained_model.children())[:-1])
        emb=768
    elif network == 'swin_b': # 87M parameters
        pretrained_model = models.swin_b(weights=None)
        pretrained_model.flatten = nn.Identity()
        pretrained_model.head = nn.Identity()
        emb=1024
        pretrained_model = nn.Sequential(*list(pretrained_model.children())[:-1])
    elif network == 'vit_b_16': #86M parameters
        pretrained_model = models.vit_b_16(weights="IMAGENET1K_V1")
        #pretrained_model.heads = nn.Identity()
        #print(pretrained_model)
        pretrained_model = InputResizerVIT(pretrained_model)
        emb=768
    elif network == 'vit_b_32': #88M parameters
        pretrained_model = models.vit_b_32(weights="IMAGENET1K_V1")
        pretrained_model.heads = nn.Identity()
        pretrained_model = nn.Sequential(*list(pretrained_model.children())[:-1])
    elif network == 'maxvit_t': #31M parameters
        pretrained_model = models.maxvit_t(weights="IMAGENET1K_V1")
        print(pretrained_model)

        pretrained_model.classifier = nn.Identity()
        #pretrained_model = InputResizer(nn.Sequential(*list(pretrained_model.children())[:-1]))
        pretrained_model = InputResizer(pretrained_model)
    elif network == 'mnasnet0_5': #2218512 parameters
        pretrained_model = models.mnasnet0_5(weights="IMAGENET1K_V1")
        pretrained_model.heads.head = nn.Identity()
        pretrained_model = nn.Sequential(*list(pretrained_model.children())[:-1])
    elif network == 'squeezenet1_0': #1248424 parameters
        pretrained_model = models.squeezenet1_0(weights="IMAGENET1K_V1")
        pretrained_model.heads.head = nn.Identity()
        pretrained_model = nn.Sequential(*list(pretrained_model.children())[:-1])
    elif network == 'mobilenet_v3_small': #2542856 parameters
        pretrained_model = models.mobilenet_v3_small(weights="IMAGENET1K_V1")
        #print(pretrained_model)
        pretrained_model.classifier = nn.Identity()
        pretrained_model = nn.Sequential(*list(pretrained_model.children())[:-1])
        emb=576

    return pretrained_model,emb


class ModifiedModel(nn.Module):
    def __init__(self, original_model):
        super(ModifiedModel, self).__init__()
        self.upchanneler= nn.Conv2d(1, 3, kernel_size=1)
        self.original_model = original_model

    def forward(self, x):
        # Convert 1-channel input to 3-channel input
        x = self.upchanneler(x)
        # Pass it to the pretrained model
        return self.original_model(x)


def load_pre_trained_model_1to3( network):
    if network == "ResNet18":
        pretrained_model = models.resnet18(weights="IMAGENET1K_V1") #"IMAGENET1K_V1"
        #
        pretrained_model=nn.Sequential(*list(pretrained_model.children())[:-2])
    elif network == 'ResNet50':
        pretrained_model = models.resnet50(weights="IMAGENET1K_V1")
        #pretrained_model.fc = nn.Identity()
        #pretrained_model.avgpool = nn.Identity()
        #print (pretrained_model)
        pretrained_model=nn.Sequential(*list(pretrained_model.children())[:-2])
    elif network == 'ResNet101':
        pretrained_model = models.resnet101(weights="IMAGENET1K_V1")
        pretrained_model= nn.Sequential(*list(pretrained_model.children())[:-2])
    elif network == 'DenseNet':
        pretrained_model = models.densenet121(weights="IMAGENET1K_V1")
        pretrained_model= InputResizer(nn.Sequential(*list(pretrained_model.children())[:-1]))
        #pretrained_model.classifier = nn.Identity()
        #pretrained_model.features.pool0 = nn.Identity()
    elif network == 'VGG16':
        pretrained_model = models.vgg16_bn(weights="IMAGENET1K_V1")
        pretrained_model.classifier[0] = nn.Identity()
        pretrained_model = InputResizer(nn.Sequential(*list(pretrained_model.children())[:-1]))
    elif network == 'MobileNetV2':
        pretrained_model = models.mobilenet_v2(weights="IMAGENET1K_V1")
        pretrained_model.classifier[0] = nn.Identity()
        pretrained_model = nn.Sequential(*list(pretrained_model.children())[:-1])
        emb=1280
        # ----------------------------- NEW Backbones ----------------------------- #
    elif network == 'ConvNext_Small': # 50M parameters
        pretrained_model = models.convnext_small(weights="IMAGENET1K_V1")
        pretrained_model.classifier[2] = nn.Identity()
        pretrained_model = nn.Sequential(*list(pretrained_model.children())[:-1])
    elif network == 'ConvNext_base': # 88M parameters
        pretrained_model = models.convnext_base(weights=None)
        pretrained_model.classifier[2] = nn.Identity()
        pretrained_model = nn.Sequential(*list(pretrained_model.children())[:-1])
    elif network == 'ConvNext_large': # 197M parameters
        pretrained_model = models.convnext_large(weights="IMAGENET1K_V1")
        pretrained_model.classifier[2] = nn.Identity()
        pretrained_model = nn.Sequential(*list(pretrained_model.children())[:-1])
    elif network == 'efficientnet_b1': # 7.8M parameters
        pretrained_model = models.efficientnet_b1(weights="IMAGENET1K_V1")
        pretrained_model.classifier[0] = nn.Identity()
        pretrained_model = nn.Sequential(*list(pretrained_model.children())[:-1])
    elif network == 'efficientnet_b4': # 19M parameters
        pretrained_model = models.efficientnet_b4(weights="IMAGENET1K_V1")
        pretrained_model.classifier[0] = nn.Identity()
        pretrained_model = nn.Sequential(*list(pretrained_model.children())[:-1])
    elif network == 'efficientnet_b7': # 66M parameters
        pretrained_model = models.efficientnet_b7(weights="IMAGENET1K_V1")
        pretrained_model.classifier[0] = nn.Identity()
        pretrained_model = nn.Sequential(*list(pretrained_model.children())[:-1])
    elif network == 'efficientnet_v2_s': # 21M parameters
        pretrained_model = models.efficientnet_v2_s(weights=None)
        pretrained_model.classifier[0] = nn.Identity()
        pretrained_model = nn.Sequential(*list(pretrained_model.children())[:-1])
    elif network == 'efficientnet_v2_m': # 54M parameters
        pretrained_model = models.efficientnet_v2_m(weights="IMAGENET1K_V1")
        pretrained_model.classifier[0] = nn.Identity()
        pretrained_model = nn.Sequential(*list(pretrained_model.children())[:-1])
    elif network == 'efficientnet_v2_l': # 118M parameters
        pretrained_model = models.efficientnet_v2_l(weights="IMAGENET1K_V1")
        pretrained_model.classifier[0] = nn.Identity()
        pretrained_model = nn.Sequential(*list(pretrained_model.children())[:-1])
    elif network == 'resnext50_32x4d': # 25M parameters
        pretrained_model = models.resnext50_32x4d(weights=None)
        pretrained_model.fc = nn.Identity()
        pretrained_model = nn.Sequential(*list(pretrained_model.children())[:-1])
    elif network == 'resnext101_64x4d': # 83M parameters
        pretrained_model = models.resnext101_64x4d(weights="IMAGENET1K_V1")
        pretrained_model.fc = nn.Identity()
        pretrained_model = nn.Sequential(*list(pretrained_model.children())[:-1])
    elif network == 'swin_t': # 28M parameters
        pretrained_model = models.swin_t(weights="IMAGENET1K_V1")
        pretrained_model.head = nn.Identity()
        pretrained_model.flatten = nn.Identity()
        pretrained_model = nn.Sequential(*list(pretrained_model.children())[:-1])
        emb=768
    elif network == 'swin_b': # 87M parameters
        pretrained_model = models.swin_b(weights=None)
        pretrained_model.flatten = nn.Identity()
        pretrained_model.head = nn.Identity()
        pretrained_model = nn.Sequential(*list(pretrained_model.children())[:-1])
        emb=1024
    elif network == 'vit_b_16': #86M parameters
        pretrained_model = models.vit_b_16(weights="IMAGENET1K_V1")
        pretrained_model.heads = nn.Identity()
        #pretrained_model = InputResizer(nn.Sequential(*list(pretrained_model.children())[:-1]))
        pretrained_model = InputResizerVIT(pretrained_model)
        # print(pretrained_model)
        emb=768
    elif network == 'vit_b_32': #88M parameters
        pretrained_model = models.vit_b_32(weights="IMAGENET1K_V1")
        pretrained_model.heads.head = nn.Identity()
        pretrained_model = nn.Sequential(*list(pretrained_model.children())[:-1])
    elif network == 'maxvit_t': #31M parameters
        pretrained_model = models.maxvit_t(weights="IMAGENET1K_V1")
        pretrained_model.classifier = nn.Identity()
        #print(pretrained_model)
        pretrained_model = InputResizer(pretrained_model)
    elif network == 'mnasnet0_5': #2218512 parameters
        pretrained_model = models.mnasnet0_5(weights="IMAGENET1K_V1")
        pretrained_model.heads.head = nn.Identity()
        pretrained_model = nn.Sequential(*list(pretrained_model.children())[:-1])
    elif network == 'squeezenet1_0': #1248424 parameters
        pretrained_model = models.squeezenet1_0(weights="IMAGENET1K_V1")
        pretrained_model.heads.head = nn.Identity()
        pretrained_model = nn.Sequential(*list(pretrained_model.children())[:-1])
    elif network == 'mobilenet_v3_small': #2542856 parameters
        pretrained_model = models.mobilenet_v3_small(weights="IMAGENET1K_V1")
        pretrained_model.classifier = nn.Identity()
        pretrained_model = nn.Sequential(*list(pretrained_model.children())[:-1])

    return ModifiedModel(pretrained_model)


# discord = Discord(url="https://discord.com/api/webhooks/1316486031360393226/mCisg8dJBB3MnVrDJBKolqTNlqkSajQQznj9PYaKqV-6hXDVGp3Gc_HQIojGgvHP7blp")
discord = Discord(url="https://discord.com/api/webhooks/1387805049471897811/DYo4R4oqJfGr9H2KT6jrqSTOJb_lNbgh9ZPn0nkPAeMDFTgRrykxrqe9cCa6Nqd2yCtK")

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default = 'ycb', help='ycb or linemod')
parser.add_argument('--dataset_root', type=str, default = '', help='dataset root dir (''YCB_Video_Dataset'' or ''Linemod_preprocessed'')')
parser.add_argument('--batch_size', type=int, default = 64, help='batch size')
parser.add_argument('--workers', type=int, default = 10, help='number of data loading workers')
parser.add_argument('--lr', default=0.0001, help='learning rate',type=float)
parser.add_argument('--lr_rate', default=0.5, help='learning rate decay rate')
parser.add_argument('--w', default=0.015, help='learning rate')
parser.add_argument('--w_rate', default=0.9, help='learning rate decay rate')
parser.add_argument('--decay_margin', default=0.016, help='margin to decay lr & w')
parser.add_argument('--refine_margin', default=0.013, help='margin to start the training of iterative refinement')
parser.add_argument('--noise_trans', default=0.15, help='range of the random noise of translation added to the training data')
parser.add_argument('--iteration', type=int, default = 25, help='number of refinement iterations')
parser.add_argument('--nepoch', type=int, default=500, help='max number of epochs to train')
parser.add_argument('--resume_posenet', type=str, default = '',  help='resume PoseNet model')
parser.add_argument('--start_epoch', type=int, default = 1, help='which epoch to start')
opt = parser.parse_args()

def view_results(estimator, dataloader, option, modalities):
    estimator.eval()

    for i, data in tqdm(enumerate(dataloader, 0), total=len(dataloader), desc=f'', unit='batch'):
        pc_depth_W, pc_depth, pc_velodyne_W, pc_velodyne, pc_model_W, pc_model, img, depth_vel, modelPoints, modelPoints_W, rt, idx = data

        # view_dataloader(pc_depth_W, pc_depth, pc_velodyne_W, pc_velodyne, pc_model_W, pc_model, img, depth_vel, modelPoints, modelPointsGT, rt, idx)

        if modalities == 0:
            RGBEnable = float(1)
            Depth1Enable = float(1)
            Depth2Enable = float(1)
            PC1Enable = float(1)
            PC2Enable = float(1)
        elif modalities == 1:
            RGBEnable = float(1)
            Depth1Enable = float(0)
            Depth2Enable = float(0)
            PC1Enable = float(0)
            PC2Enable = float(0)
        elif modalities == 2:
            RGBEnable = float(1)
            Depth1Enable = float(1)
            Depth2Enable = float(0)
            PC1Enable = float(0)
            PC2Enable = float(0)
        elif modalities == 3:
            RGBEnable = float(1)
            Depth1Enable = float(1)
            Depth2Enable = float(0)
            PC1Enable = float(1)
            PC2Enable = float(0)

        points = Variable(pc_depth).cuda()  # cam
        target = Variable(pc_depth_W).cuda()
        velodyne = Variable(pc_velodyne).cuda()
        velodyne_gt = Variable(pc_velodyne_W).cuda()
        model = Variable(pc_model).cuda()
        model_gt = Variable(pc_model_W).cuda()

        img = Variable(img).cuda()
        depth_vel = Variable(depth_vel).cuda()
        depth_vel = depth_vel.permute(0, 3, 1, 2).contiguous()

        choose = torch.LongTensor([0])
        choose = Variable(choose).cuda()        
        idx = Variable(idx).cuda()
     
        modelPoints_W = Variable(modelPoints_W).cuda()
        modelPoints = Variable(modelPoints).cuda()

        img[:,0:3,:,:] = img[:,0:3,:,:] * RGBEnable
        img[:,3,:,:] = img[:,3,:,:] * Depth1Enable

        with torch.no_grad():
            if option == 1:
                pred_r, pred_t, pred_c, _ = estimator(img, depth_vel*Depth2Enable, model_gt*PC1Enable, velodyne_gt*PC2Enable, choose, idx)
            elif option == 2:
                pred_r, pred_t, pred_c, _ = estimator(img, depth_vel*Depth2Enable, target*PC1Enable, velodyne_gt*PC2Enable, choose, idx)
            elif option == 3:
                pred_r, pred_t, pred_c, _ = estimator(img, depth_vel*Depth2Enable, model_gt*PC1Enable, target*PC2Enable, choose, idx)
            elif option == 4:
                pred_r, pred_t, pred_c, _ = estimator(img, depth_vel*Depth2Enable, model_gt*PC1Enable, target*PC2Enable, choose, idx)
            elif option == 5:
                pred_r, pred_t, pred_c, _ = estimator(img, depth_vel*Depth2Enable, model_gt*PC1Enable, velodyne_gt*PC2Enable, choose, idx)

        criterion = Lossv2(opt.num_points)
        loss, dis, new_points, new_target = criterion(pred_r, pred_t, pred_c, modelPoints_W, modelPoints, idx, points, opt.w, opt.refine_start)

        for b in range(dis.shape[0]):
            import math
            angle = math.pi / 2  # 90 graus
            q = torch.tensor([0.0, 0.0, math.sin(angle/2), math.cos(angle/2)])  # eixo Z, [x, y, z, w]
            t = torch.zeros(3)

            T = computeT(q, t)
            print(T)

            T = computeT(pred_r[b], pred_t[b])
            T = T.detach().cpu().numpy()

            rt_numpy = rt[b].squeeze(0).cpu().numpy()
            np.set_printoptions(precision=8, suppress=True)

            # print(filename)
            print("RT:")
            print(rt_numpy)
            print()
            print("Matriz T:")
            print(T)

            print(dis[b])

            target_vs_pred(pc_depth_W[b], pc_velodyne_W[b], pc_model_W[b], modelPoints_W[b], pc_depth[b], pc_velodyne[b], pc_model[b], modelPoints[b], T, rt[b].detach().cpu().numpy())

        print()

def computeT(pred_r, pred_t):
    bs = 1
    num_p = 1

    a = torch.norm(pred_r, dim=0)
    if a>0.001:
        pred_r = pred_r / a
    else:
        pred_r[3]=1
    
    base = torch.cat(((1.0 - 2.0*(pred_r[2]**2 + pred_r[3]**2)).view(bs, num_p, 1),\
                      (2.0*pred_r[1]*pred_r[2] - 2.0*pred_r[0]*pred_r[3]).view(bs, num_p, 1), \
                      (2.0*pred_r[0]*pred_r[2] + 2.0*pred_r[1]*pred_r[3]).view(bs, num_p, 1), \
                      (2.0*pred_r[1]*pred_r[2] + 2.0*pred_r[3]*pred_r[0]).view(bs, num_p, 1), \
                      (1.0 - 2.0*(pred_r[1]**2 + pred_r[3]**2)).view(bs, num_p, 1), \
                      (-2.0*pred_r[0]*pred_r[1] + 2.0*pred_r[2]*pred_r[3]).view(bs, num_p, 1), \
                      (-2.0*pred_r[0]*pred_r[2] + 2.0*pred_r[1]*pred_r[3]).view(bs, num_p, 1), \
                      (2.0*pred_r[0]*pred_r[1] + 2.0*pred_r[2]*pred_r[3]).view(bs, num_p, 1), \
                      (1.0 - 2.0*(pred_r[1]**2 + pred_r[2]**2)).view(bs, num_p, 1)), dim=2).contiguous().view(bs * num_p, 3, 3)

    ori_base = base
    # base = base.contiguous().transpose(2, 1).contiguous()

    rot = base[0]
    trans = pred_t.view(3, 1)
    upper = torch.cat([rot, trans], dim=1)
    bottom = torch.tensor([[0.0, 0.0, 0.0, 1.0]], device=rot.device)
    transform = torch.cat([upper, bottom], dim=0)

    return transform

def view_dataloader(pc_depth_W, pc_depth, pc_velodyne_W, pc_velodyne, pc_model_W, pc_model, img, depth_vel, modelPoints, modelPointsGT, rt, idx):
    pc_depth_W = pc_depth_W.squeeze(0)
    pc_depth = pc_depth.squeeze(0)
    pc_velodyne_W = pc_velodyne_W.squeeze(0)
    pc_velodyne = pc_velodyne.squeeze(0)
    pc_model_W = pc_model_W.squeeze(0)
    pc_model = pc_model.squeeze(0)
    img = img.squeeze(0)
    depth_vel = depth_vel.squeeze(0)
    rt = rt.squeeze(0)

    fig = plt.figure()
    ax1 = fig.add_subplot(2, 3, 1, projection='3d')
    ax2 = fig.add_subplot(2, 3, 2, projection='3d')
    ax3 = fig.add_subplot(2, 3, 3, projection='3d')
    ax4 = fig.add_subplot(2, 3, 4)
    ax5 = fig.add_subplot(2, 3, 5)
    ax6 = fig.add_subplot(2, 3, 6)

    ax1.scatter(pc_depth[:, 0], pc_depth[:, 1], pc_depth[:, 2])
    ax2.scatter(pc_velodyne[:, 0], pc_velodyne[:, 1], pc_velodyne[:, 2])
    ax3.scatter(pc_model[:, 0], pc_model[:, 1], pc_model[:, 2])

    origin = [0, 0, 0]
    for ax in [ax1, ax2, ax3]:
        ax.quiver(*origin, 1, 0, 0, color='r')
        ax.quiver(*origin, 0, 1, 0, color='g')
        ax.quiver(*origin, 0, 0, 1, color='b')

    rgb = img[:3, :, :]
    depth = img[3, :, :]

    rgb_np = rgb.permute(1, 2, 0).numpy()
    depth_np = depth.cpu().numpy()

    ax4.imshow(rgb_np)
    ax5.imshow(depth_np, cmap="viridis")
    ax6.imshow(depth_vel, cmap="viridis")

    ax1.set_title('pointcloud depth')
    ax2.set_title('pointcloud velodyne')
    ax3.set_title('pointlcoud modelo')
    ax4.set_title('Imagem RGB crop')
    ax5.set_title('Mascara')
    ax6.set_title('DEPTH')

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(
        pc_depth_W[:, 0],
        pc_depth_W[:, 1],
        pc_depth_W[:, 2],
        s=1,
        alpha=0.5,
        label='pointcloud depth'
    )

    ax.scatter(
        pc_velodyne_W[:, 0],
        pc_velodyne_W[:, 1],
        pc_velodyne_W[:, 2],
        s=1,
        alpha=0.5,
        color='magenta',
        label='pointcloud velodyne'
    )

    ax.scatter(
        pc_model_W[:, 0],
        pc_model_W[:, 1],
        pc_model_W[:, 2],
        s=1,
        alpha=0.5,
        color='red',
        label='pointcloud modelo'
    )

    R = rt[:3, :3]
    t = rt[:3, 3]

    scale = 0.5

    x_axis = R[:, 0] * scale
    y_axis = R[:, 1] * scale
    z_axis = R[:, 2] * scale

    ax.scatter(*t, color='k', s=30, label="origin")

    # Desenhar eixos
    ax.quiver(*t, *x_axis, color='r', label='X')
    ax.quiver(*t, *y_axis, color='g', label='Y')
    ax.quiver(*t, *z_axis, color='b', label='Z')

    ax.set_title('PointCloud + Eixos do Modelo no mundo')
    ax.legend(loc='upper left')

    plt.show()

def target_vs_pred(pc_depth_W, pc_velodyne_W, pc_model_W, modelPoints_W, pc_depth, pc_velodyne, pc_model, modelPoints, pred_RT, RT):
    pc_depth_W = pc_depth_W.squeeze(0)
    pc_depth = pc_depth.squeeze(0)
    pc_velodyne_W = pc_velodyne_W.squeeze(0)
    pc_velodyne = pc_velodyne.squeeze(0)
    pc_model_W = pc_model_W.squeeze(0)
    pc_model = pc_model.squeeze(0)
    modelPoints_W = modelPoints_W.squeeze(0).detach().cpu().numpy()
    modelPoints = modelPoints.squeeze(0).detach().cpu().numpy()

    pred_rt_inv = np.linalg.inv(pred_RT)
    pred_rotation_inv = pred_rt_inv[:3, :3]
    pred_translation_inv = pred_rt_inv[:3, 3]
    
    pred_rt = pred_RT
    pred_rotation = pred_rt[:3, :3]
    pred_translation = pred_rt[:3, 3]

    # pred_pointcloud_cam = (pred_rotation_inv @ pc_depth_W.cpu().numpy().T).T + pred_translation_inv
    pc_depth_est_W = (pred_rotation @ pc_depth.cpu().numpy().T).T + pred_translation

    pred_model_points = (pred_rotation @ modelPoints.T).T + pred_translation

    # Figura 1
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    for i in range(1, len(modelPoints)):
        x1 = [modelPoints_W[0, 0], modelPoints_W[i, 0]]
        y1 = [modelPoints_W[0, 1], modelPoints_W[i, 1]]
        z1 = [modelPoints_W[0, 2], modelPoints_W[i, 2]]

        x_pred = [pred_model_points[0, 0], pred_model_points[i, 0]]
        y_pred = [pred_model_points[0, 1], pred_model_points[i, 1]]
        z_pred = [pred_model_points[0, 2], pred_model_points[i, 2]]

        ax.plot(x1, y1, z1, color='blue', linewidth=1)
        ax.plot(x_pred, y_pred, z_pred, color='orange', linewidth=1)

    ax.scatter(
        pc_depth_W[:, 0],
        pc_depth_W[:, 1],
        pc_depth_W[:, 2],
        s=1,
        alpha=0.5,
        color="blue",
        label='pointcloud depth'
    )

    ax.scatter(
        pc_depth_est_W[:, 0],
        pc_depth_est_W[:, 1],
        pc_depth_est_W[:, 2],
        s=1,
        alpha=0.5,
        color="orange",
        label='pointcloud depth pred'
    )

    ax.set_title('PointCloud + Eixos do Modelo no mundo')
    ax.legend(loc='upper left')

    plt.show()

def main():
    stale = 0
    opt.manualSeed = random.randint(1, 10000)
    random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)
    torch.cuda.set_device(0)

    modelString = "MobileNetV2"

    option = 4
    if option == 1:
        opt.outf = f'trained_models/{modelString}/1-img_model-velodyne-pc_model-pc_velodyne'
        concat = "model"
        mask = "velodyne"
        print("option1- img_model-velodyne-pc_model-pc_velodyne")
    elif option == 2:
        opt.outf = f'trained_models/{modelString}/2-img_depth-velodyne-pc_depth-pc_velodyne'
        concat = "depth"
        mask = "velodyne"
        print("option2- img_depth-velodyne-pc_depth-pc_velodyne")
    elif option == 3:
        opt.outf = f'trained_models/{modelString}/3-img_model-depth-pc_model-pc_depth'
        concat = "model"
        mask = "depth"
        print("option3- img_model-depth-pc_model-pc_depth")
    elif option == 4:
        opt.outf = f'trained_models/{modelString}/4-img_depth-depth-pc_model-pc_velodyne'
        concat = "depth"
        mask = "depth"
        print("option4- img_depth-depth-pc_model-pc_velodyne")
    elif option == 5:
        opt.outf = f'trained_models/{modelString}/5-img_depth-velodyne-pc_model-pc_velodyne'
        concat = "depth"
        mask = "velodyne"
        print("option5- img_depth-velodyne-pc_model-pc_velodyne")

    # opt.outf += "/ssh batch1"

    modalities = 0
    if modalities == 0:
        opt.outf += '/All'
        print("Todas as modalidades ativas")        
    elif modalities == 1:
        opt.outf += '/RGB'
        print("RGB")
    elif modalities == 2:
        opt.outf += '/RGB-Depth1'
        print("RGB e Depth1")
    elif modalities == 3:
        opt.outf += '/RGB-Depth1-PC1'
        print("RGB, Depth1 e PC1")

    print()

    # opt.outf += " sem transportas"
    print("Metricas para pasta ", opt.outf)

    opt.num_objects = 7 # numero de classes
    opt.num_points = 1000
    opt.outfpre = 'trained_models/PoseIndustrial6DMultiRGBOnlyRandomized' + modelString

    opt.log_dir = 'experiments/PoseIndustrial6DMultiRGBOnlyRandomized' + modelString + str(opt.num_points)
    opt.repeat_epoch = 1

    modelRGB, embchannels = load_pre_trained_model(modelString)
    modelDepth = load_pre_trained_model_1to3(modelString)
    modelDepth2 = load_pre_trained_model_1to3(modelString)
    
    estimator = PoseNetMultiCUSTOMPointsX(modelRGB, modelDepth, modelDepth2, num_points=opt.num_points, num_obj=opt.num_objects,embchannels=embchannels)
    estimator.cuda()

    estimator.load_state_dict(torch.load(f"{opt.outf}/pose_model_best.pth"))

    dataset = PoseDataset('all', opt.num_points, concatmethod=concat, maskedmethod=mask)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batch_size, shuffle=False, num_workers=opt.workers)

    if not os.path.exists(opt.outf):
        os.makedirs(opt.outf)

    if not os.path.exists(opt.log_dir):
        os.makedirs(opt.log_dir)

    if opt.start_epoch == 1:
        for log in os.listdir(opt.log_dir):
            os.remove(os.path.join(opt.log_dir, log))
    st_time = time.time()
    # opt.refine_start = True
    opt.refine_start = False

    view_results(estimator, dataloader, option, modalities)


if __name__ == '__main__':
    main()
