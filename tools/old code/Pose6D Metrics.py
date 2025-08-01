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
discord = Discord(url="https://discord.com/api/webhooks/1390414457381060698/butYB50GjmbOuIoQKcsp97wYuCtww8xN977XZl9EuSKdAzOhOpsDZ0dHn3hp9V-IiTyP")

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

def get_depth_bin_index(distance, thresholds):
    for i, th in enumerate(thresholds):
        if distance < th:
            return i
    return len(thresholds)

def compute_metrics(estimator, criterion, dataloader, option, modalities):
    estimator.eval()

    total_loss = 0.0
    total_batches = 0

    loss_cls0, loss_cls1, loss_cls2, loss_cls3, loss_cls4, loss_cls5, loss_cls6 = 0, 0, 0, 0, 0, 0, 0
    batch_cls0, batch_cls1, batch_cls2, batch_cls3, batch_cls4, batch_cls5, batch_cls6 = 0, 0, 0, 0, 0, 0, 0

    depththresholds = [4, 8, 16, 24, 34]
    num_bins = len(depththresholds)

    # Global loss acumulado até cada threshold
    loss_by_depth = [0.0 for _ in range(num_bins)]
    count_by_depth = [0 for _ in range(num_bins)]

    # Loss por classe até cada threshold
    loss_by_class_depth = [[0.0 for _ in range(num_bins)] for _ in range(7)]
    count_by_class_depth = [[0 for _ in range(num_bins)] for _ in range(7)]

    for i, data in tqdm(enumerate(dataloader, 0), total=len(dataloader), desc=f'', unit='batch'):
        pc_depth_W, pc_depth, pc_velodyne_W, pc_velodyne, pc_model_W, pc_model, img, depth_vel, modelPoints, modelPointsGT, rt, idx = data
        
        valid_indices = []
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
        idx = idx[valid_indices]

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

        loss, dis, new_points, new_target = criterion(pred_r, pred_t, pred_c, modelPointsGT, modelPoints, idx, points, opt.w, opt.refine_start)

        batch_size = pred_r.size(0)
        total_loss += dis.sum().item()
        total_batches += batch_size

        for b in range(batch_size):
            class_idx = idx[b].item()
            dis_b = dis[b].item()
            
            if class_idx == 0:
                loss_cls0 += dis_b
                batch_cls0 += 1
            elif class_idx == 1:
                loss_cls1 += dis_b
                batch_cls1 += 1
            elif class_idx == 2:
                loss_cls2 += dis_b
                batch_cls2 += 1
            elif class_idx == 3:
                loss_cls3 += dis_b
                batch_cls3 += 1
            elif class_idx == 4:
                loss_cls4 += dis_b
                batch_cls4 += 1
            elif class_idx == 5:
                loss_cls5 += dis_b
                batch_cls5 += 1
            elif class_idx == 6:
                loss_cls6 += dis_b
                batch_cls6 += 1

            # Processamento de rt
            t = rt[b, 0:3, 3].cpu().numpy()
            distancia = np.linalg.norm(t)

            for i, th in enumerate(depththresholds):
                if distancia < th:
                    loss_by_depth[i] += dis_b
                    count_by_depth[i] += 1
                    if class_idx < 7:
                        loss_by_class_depth[class_idx][i] += dis_b
                        count_by_class_depth[class_idx][i] += 1

    avg_loss = total_loss / total_batches
    
    if batch_cls0 == 0:
        loss_cls0 = 100.0
    else:
        loss_cls0 = loss_cls0 / batch_cls0
    if batch_cls1 == 0:
        loss_cls1 = 100.0
    else:
        loss_cls1 = loss_cls1 / batch_cls1
    if batch_cls2 == 0:
        loss_cls2 = 100.0
    else:
        loss_cls2 = loss_cls2 / batch_cls2
    if batch_cls3 == 0:
        loss_cls3 = 100.0
    else:
        loss_cls3 = loss_cls3 / batch_cls3
    if batch_cls4 == 0:
        loss_cls4 = 100.0
    else:
        loss_cls4 = loss_cls4 / batch_cls4
    if batch_cls5 == 0:
        loss_cls5 = 100.0
    else:
        loss_cls5 = loss_cls5 / batch_cls5
    if batch_cls6 == 0:
        loss_cls6 = 100.0
    else:
        loss_cls6 = loss_cls6 / batch_cls6

    avg_loss_by_depth = [
        loss_by_depth[i] / count_by_depth[i] if count_by_depth[i] > 0 else 0.0
        for i in range(num_bins)
    ]

    avg_loss_by_class_depth = [
        [
            loss_by_class_depth[cls][i] / count_by_class_depth[cls][i] if count_by_class_depth[cls][i] > 0 else 0.0
            for i in range(num_bins)
        ]
        for cls in range(7)
    ]

    return avg_loss, [loss_cls0, loss_cls1, loss_cls2, loss_cls3, loss_cls4, loss_cls5, loss_cls6], avg_loss_by_depth, avg_loss_by_class_depth

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
    base = base.contiguous().transpose(2, 1).contiguous()

    rot = base[0]
    trans = pred_t.view(3, 1)
    upper = torch.cat([rot, trans], dim=1)
    bottom = torch.tensor([[0.0, 0.0, 0.0, 1.0]], device=rot.device)
    transform = torch.cat([upper, bottom], dim=0)

    return transform


def main():
    stale = 0
    opt.manualSeed = random.randint(1, 10000)
    random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)
    torch.cuda.set_device(0)

    modelString = "MobileNetV2"

    opt.num_objects = 7
    opt.num_points = 1000

    msg = ""

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

    # opt.outf += '/20m'

    modalities = 0
    if modalities == 0:
        opt.outf += '/All'
        print("Todas as modalidades ativas")
        msg += f"modalitie0- All\n"
    elif modalities == 1:
        opt.outf += '/RGB'
        print("RGB")
        msg += f"modalitie1- RGB\n"
    elif modalities == 2:
        opt.outf += '/RGB-Depth1'
        print("RGB e Depth1")
        msg += f"modalitie2- RGB-Depth1\n"
    elif modalities == 3:
        opt.outf += '/RGB-Depth1-PC1'
        print("RGB, Depth1 e PC1")
        msg += f"modalitie3- RGB-Depth1-PC1\n"
    print()

    # opt.outf += f" ssh"

    msg += f"Metricas da pasta {opt.outf}\n"
    print(f"Metricas da pasta {opt.outf}")

    opt.outfpre = 'trained_models/PoseIndustrial6DMultiRGBOnlyRandomized' + modelString

    opt.log_dir = 'experiments/PoseIndustrial6DMultiRGBOnlyRandomized' + modelString + str(opt.num_points)
    opt.repeat_epoch = 1

    modelRGB, embchannels = load_pre_trained_model(modelString)
    modelDepth = load_pre_trained_model_1to3(modelString)
    modelDepth2 = load_pre_trained_model_1to3(modelString)
    
    estimator = PoseNetMultiCUSTOMPointsX(modelRGB, modelDepth, modelDepth2, num_points=opt.num_points, num_obj=opt.num_objects,embchannels=embchannels)
    estimator.cuda()

    estimator.load_state_dict(torch.load(f"{opt.outf}/pose_model_best.pth"))

    dataset = PoseDataset('all', 1000, concatmethod=concat, maskedmethod=mask)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batch_size, shuffle=True, num_workers=opt.workers)

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

    criterion = Lossv2(opt.num_points)

    # compute1_metrics(estimator, criterion, dataloader, option)

    depththresholds = [4, 8, 16, 24, 34]
    loss, loss_cls, loss_depth, loss_cls_depth = compute_metrics(estimator, criterion, dataloader, option, modalities)

    classes = ["Bidons", "Caixa", "Caixa encaxe", "Extintor", "Empilhadora", "Pessoas", "Toolboxes"]

    msg += f"Average loss over dataset: {loss:.4f}\n"
    msg += "Loss por classe:\n"
    msg += f"Bidons: {loss_cls[0]:.4f}\t Caixa: {loss_cls[1]:.4f}\t Caixa encaxe: {loss_cls[2]:.4f}\t Extintor: {loss_cls[3]:.4f}\t Empilhadora: {loss_cls[4]:.4f}\t Pessoas: {loss_cls[5]:.4f} \t Toolboxes: {loss_cls[6]:.4f}\n\n"

    msg += "Loss global por thresholds de profundidade:\n"
    for i, th in enumerate(depththresholds):
        msg += f"[0-{th}m]: {loss_depth[i]:.6f}\n"

    msg += "\nLoss por classe e thresholds de profundidade:\n"
    for i, th in enumerate(depththresholds):
        msg += f"[0-{th}m]:\t"
        for cls_idx, cls_name in enumerate(classes):
            msg += f"{cls_name}: {loss_cls_depth[cls_idx][i]:.6f}\t"
        msg += "\n"

    print(msg)

    discord.post(content=msg)


if __name__ == '__main__':
    main()
