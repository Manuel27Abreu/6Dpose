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
import math
from datetime import datetime
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

from lib.network_attnMOD_Manuel import PoseNetMultiCUSTOMPointsX,PoseNetMultiCUSTOM

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
        print(pretrained_model)
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

def train_epoch(estimator, criterion, optimizer, dataloader, epoch, train_dis_avg, train_count, option, modalities):
    # TREINO
    for i, data in tqdm(enumerate(dataloader, 0), total=len(dataloader), desc=f'Epoch {epoch}', unit='batch'):
        pc_depth_W, pc_depth, pc_velodyne_W, pc_velodyne, pc_model_W, pc_model, img, depth_vel, modelPoints, modelPointsGT, rt, idx = data

        """valid_indices = []
        for b in range(rt.shape[0]):
            x, y, z = rt[b][0][3], rt[b][1][3], rt[b][2][3]
            dist = math.sqrt(x**2 + y**2 + z**2)
            if 0.05 < dist < 20:
                valid_indices.append(b)

        if len(valid_indices) == 0:
            continue

        pc_depth = pc_depth[valid_indices]
        pc_depth_W = pc_depth_W[valid_indices]
        pc_velodyne = pc_velodyne[valid_indices]
        pc_velodyne_W = pc_velodyne_W[valid_indices]
        pc_model = pc_model[valid_indices]
        pc_model_W = pc_model_W[valid_indices]
        img = img[valid_indices]
        depth_vel = depth_vel[valid_indices]
        modelPoints = modelPoints[valid_indices]
        modelPointsGT = modelPointsGT[valid_indices]
        rt = rt[valid_indices]
        idx = idx[valid_indices]"""

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

        modelPoints = Variable(modelPoints).cuda()
        modelPointsGT = Variable(modelPointsGT).cuda()  # modelPoints_W

        img[:,0:3,:,:] = img[:,0:3,:,:] * RGBEnable
        img[:,3,:,:] = img[:,3,:,:] * Depth1Enable

        """print("---------------------")
        print("points.shape", points.shape)
        print("target.shape", target.shape)
        print("velodyne.shape", velodyne.shape)
        print("velodyne_gt.shape", velodyne_gt.shape)
        print("model.shape", model.shape)
        print("model_gt.shape", model_gt.shape)
        print("img.shape", img.shape)
        print("depth_vel.shape", depth_vel.shape)
        print("modelPoints.shape", modelPoints.shape)
        print("modelPointsGT.shape", modelPointsGT.shape)
        print("---------------------")"""

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

        # loss, dis, new_points, new_target = criterion(pred_r, pred_t, pred_c, target, points, idx, points, opt.w, opt.refine_start)
        loss, dis, new_points, new_target = criterion(pred_r, pred_t, pred_c, modelPointsGT, modelPoints, idx, points, opt.w, opt.refine_start)

        train_dis_avg += loss.item()
        train_count += 1

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return train_dis_avg, train_count


def eval_epoch(estimator, criterion, dataloader, epoch, option, modalities):
    test_dis = 0.0
    test_count = 0
    estimator.eval()
    
    #  EVAL
    for j, data in tqdm(enumerate(dataloader, 0), total=len(dataloader), desc=f'Epoch {epoch}(eval)', unit='batch'):
        pc_depth_W, pc_depth, pc_velodyne_W, pc_velodyne, pc_model_W, pc_model, img, depth_vel, modelPoints, modelPointsGT, rt, idx = data
        if math.sqrt(rt[0][0][3]**2 + rt[0][1][3]**2 + rt[0][2][3]**2) < 0.05:
            continue
        
        if math.sqrt(rt[0][0][3]**2 + rt[0][1][3]**2 + rt[0][2][3]**2) > 24:
            continue

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

        modelPoints = Variable(modelPoints).cuda()
        modelPointsGT = Variable(modelPointsGT).cuda()

        img[:,0:3,:,:] = img[:,0:3,:,:] * RGBEnable
        img[:,3,:,:] = img[:,3,:,:] * Depth1Enable

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

        loss, dis, new_points, new_target = criterion(pred_r, pred_t, pred_c, modelPointsGT, modelPoints, idx, points, opt.w, opt.refine_start)

        test_dis += loss.item()
        test_count += 1

    test_dis = test_dis / test_count

    return test_dis

def save_real_vs_reconstruction(estimator, dataloader, option, modalities):
    estimator.eval()

    data = next(iter(dataloader))

    pc_depth_W, pc_depth, pc_velodyne_W, pc_velodyne, pc_model_W, pc_model, img, depth_vel, modelPoints, modelPointsGT, rt, idx = data

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
    
    modelPoints = Variable(modelPoints).cuda()
    modelPointsGT = Variable(modelPointsGT).cuda()

    img[:,0:3,:,:] = img[:,0:3,:,:] * RGBEnable
    img[:,3,:,:] = img[:,3,:,:] * Depth1Enable

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

    T = computeT(pred_r[0], pred_t[0])
    
    target_vs_pred(pc_depth_W[0], pc_velodyne_W[0], pc_model_W[0], pc_depth[0], pc_velodyne[0], pc_model[0], T.detach().cpu().numpy(), rt[0].detach().cpu().numpy())

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

def target_vs_pred(pc_depth_W, pc_velodyne_W, pc_model_W, pc_depth, pc_velodyne, pc_model, pred_RT, RT):
    pc_depth_W = pc_depth_W.squeeze(0)
    pc_depth = pc_depth.squeeze(0)
    pc_velodyne_W = pc_velodyne_W.squeeze(0)
    pc_velodyne = pc_velodyne.squeeze(0)
    pc_model_W = pc_model_W.squeeze(0)
    pc_model = pc_model.squeeze(0)

    pred_rt_inv = np.linalg.inv(pred_RT)
    pred_rotation_inv = pred_rt_inv[:3, :3]
    pred_translation_inv = pred_rt_inv[:3, 3]

    pred_pointcloud_cam = (pred_rotation_inv @ pc_depth_W.cpu().numpy().T).T + pred_translation_inv
    pred_pointcloud_vel = (pred_rotation_inv @ pc_velodyne_W.cpu().numpy().T).T + pred_translation_inv
    pred_pointcloud_model = (pred_rotation_inv @ pc_model_W.cpu().numpy().T).T + pred_translation_inv

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(
        pc_depth[:, 0],
        pc_depth[:, 1],
        pc_depth[:, 2],
        s=1,
        alpha=0.5,
        label='pointcloud depth'
    )

    ax.scatter(
        pred_pointcloud_cam[:, 0],
        pred_pointcloud_cam[:, 1],
        pred_pointcloud_cam[:, 2],
        s=1,
        alpha=0.5,
        label='pointcloud pred depth'
    )

    ax.set_title('PointCloud + Eixos do Modelo no mundo')
    ax.legend(loc='upper left')

    plt.savefig('imgs/targetvspred.png')
    plt.close(fig)

def main():
    stale = 0
    opt.manualSeed = random.randint(1, 10000)
    random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)
    torch.cuda.set_device(0)

    modelString = "mobilenet_v3_small"

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
    
    opt.num_objects = 7 # numero de classes
    opt.num_points = 1000
    # opt.outf = 'trained_models/PoseIndustrial6DMultiRGBOnlyRandomized' + modelString + str(opt.num_points)
    opt.outfpre = 'trained_models/PoseIndustrial6DMultiRGBOnlyRandomized' + modelString

    opt.log_dir = 'experiments/PoseIndustrial6DMultiRGBOnlyRandomized' + modelString + str(opt.num_points)
    opt.repeat_epoch = 1

    modelRGB, embchannels = load_pre_trained_model(modelString)
    modelDepth = load_pre_trained_model_1to3(modelString)
    modelDepth2 = load_pre_trained_model_1to3(modelString)
    
    estimator = PoseNetMultiCUSTOMPointsX(modelRGB, modelDepth, modelDepth2, num_points=opt.num_points, num_obj=opt.num_objects, embchannels=embchannels)
    estimator.cuda()

    if opt.resume_posenet == 'pretrain':
        estimatorfake = PoseNetMultiCUSTOM(modelRGB,modelDepth,modelDepth2, num_points = opt.num_points, num_obj = opt.num_objects, embchannels=embchannels)

        estimatorfake.load_state_dict(torch.load('{0}/{1}'.format(opt.outfpre, "pose_model_best.pth")))
        estimator.cnn.encoder.rgb_backbone = estimatorfake.cnn.encoder.rgb_backbone 
        estimator.cnn.encoder.depth_backbone = estimatorfake.cnn.encoder.depth_backbone 
        estimator.cnn.encoder.depth_backbone2 = estimatorfake.cnn.encoder.depth_backbone2 
        estimator.cnn.attention_fusion = estimatorfake.cnn.attention_fusion
        estimator.cuda()

    elif opt.resume_posenet != '':
        print('Loading ', opt.resume_posenet)
        estimator.load_state_dict(torch.load('{0}/{1}'.format(opt.outf, opt.resume_posenet)))

    opt.refine_start = False
    opt.decay_start = False
    optimizer = optim.Adam(estimator.parameters(), lr=opt.lr, weight_decay=0.00001)

    dataset = PoseDataset('all', opt.num_points, concatmethod=concat, maskedmethod=mask)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batch_size, shuffle=True, num_workers=opt.workers)

    test_dataset = PoseDataset('all', opt.num_points, concatmethod=concat, maskedmethod=mask)
    testdataloader = torch.utils.data.DataLoader(test_dataset, batch_size=int(opt.batch_size/2), shuffle=True, num_workers=opt.workers)
    
    if not os.path.exists(opt.outf):
        os.makedirs(opt.outf)

    if not os.path.exists(opt.log_dir):
        os.makedirs(opt.log_dir)

    criterion = Lossv2(opt.num_points)
  
    best_test = 2000.2465

    modelCheck = False

    diz_loss = {'train_loss': [], 'eval_loss': []}

    if opt.start_epoch == 1:
        for log in os.listdir(opt.log_dir):
            os.remove(os.path.join(opt.log_dir, log))
    st_time = time.time()
    # opt.refine_start = True

    print(datetime.fromtimestamp(st_time).strftime('%d-%m-%y %H:%M:%S'))

    for epoch in range(opt.start_epoch, opt.nepoch):
        logger = setup_logger('epoch%d' % epoch, os.path.join(opt.log_dir, 'epoch_%d_log.txt' % epoch))
        # logger.info('Train time {0}'.format(time.strftime("%Hh %Mm %Ss", time.gmtime(time.time() - st_time)) + ', ' + 'Training started'))
        train_count = 0
        train_dis_avg = 0.0

        estimator.train()
        optimizer.zero_grad()

        for rep in range(opt.repeat_epoch):
            train_dis_avg, train_count = train_epoch(estimator, criterion, optimizer, dataloader, epoch, train_dis_avg, train_count, option, modalities)
        diz_loss['train_loss'].append(train_dis_avg / train_count)

        logger = setup_logger('epoch%d_test' % epoch, os.path.join(opt.log_dir, 'epoch_%d_test_log.txt' % epoch))
        # logger.info('Test time {0}'.format(time.strftime("%Hh %Mm %Ss", time.gmtime(time.time() - st_time)) + ', ' + 'Testing started'))
        
        test_dis = eval_epoch(estimator, criterion, testdataloader, epoch, option, modalities)
        # logger.info('Test time {0} Epoch {1} TEST FINISH Avg dis: {2}'.format(time.strftime("%Hh %Mm %Ss", time.gmtime(time.time() - st_time)), epoch, test_dis))
        diz_loss['eval_loss'].append(test_dis)

        tqdm.write(f"Epoch {epoch}/{opt.nepoch}\t Train_dis: {train_dis_avg / train_count:.6f} \t Eval_dis: {test_dis:.6f}")
        
        if test_dis <= best_test:
            best_test = test_dis

            torch.save(estimator.state_dict(), '{0}/pose_model_{1}_{2}.pth'.format(opt.outf, epoch, test_dis))
            torch.save(estimator.state_dict(), '{0}/pose_model_best.pth'.format(opt.outf))
            print(epoch, '>>>>>>>>----------BEST TEST MODEL SAVED---------<<<<<<<<')
            stale=0
        else:
            stale=stale+1

        # discord.post(content=" Windows Current 6D pose error Multi v3 (Randomized train) (No Resize) New Loss from GT pose Total points "+str(opt.num_points)+" ("+str(modelString)+"): "+str(test_dis)+" with best "+str(best_test)+"\n")

        if stale > 9:
            opt.lr *= opt.lr_rate
            estimator.load_state_dict(torch.load('{0}/pose_model_best.pth'.format(opt.outf)))
            optimizer = optim.Adam(estimator.parameters(), lr=opt.lr , weight_decay=0.00001)
            stale=0
        
    torch.save(estimator.state_dict(), '{0}/pose_model_final.pth'.format(opt.outf))

    estimator.load_state_dict(torch.load(f"{opt.outf}/pose_model_best.pth"))
    save_real_vs_reconstruction(estimator, dataloader, option, modalities)

    # Create a plot
    plt.figure(figsize=(8, 6))
    plt.plot(diz_loss['train_loss'], label='Train Loss', color='blue')
    plt.plot(diz_loss['eval_loss'], label='Evaluation Loss', color='red')
    plt.title('Train vs Evaluation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'imgs/train_vs_eval_loss_plot.png')
    plt.close()

    elapsed_time = time.time() - st_time
    days = int(elapsed_time // 86400)
    hours = int((elapsed_time % 86400) // 3600)
    minutes = int((elapsed_time % 3600) // 60)
    seconds = int(elapsed_time % 60)

    # Montar a string com pluralização correta
    formatted_time = (
        (f"{days}d " if days else "") +
        f"{hours}h {minutes}m {seconds}s"
    )
    discord.post(content=f"Treino finalizado {formatted_time}")

    discord.post(
        file={
                "file1": open(f"imgs/train_vs_eval_loss_plot.png", "rb"),
                "file2": open(f"imgs/targetvspred.png", "rb"),
        },
    )


if __name__ == '__main__':
    main()
