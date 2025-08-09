# --------------------------------------------------------
# DenseFusion 6D Object Pose Estimation by Iterative Dense Fusion
# Licensed under The MIT License [see LICENSE for details]
# Written by Chen
# --------------------------------------------------------

import os
import sys
sys.path.insert(0, os.getcwd())
import argparse
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

from datasets.PoseIndustrial6D.dataloader_20m import PoseDataset2 as PoseDataset
from datasets.PoseIndustrial6D.dataloader_annotate import PoseDataset2 as AnotDataset

from tools.Train import Train
from tools.Run import Run
from tools.Metrics import Metrics
from tools.Annotate import Annotate

from lib.network_attnMOD_Manuel import PoseNetMultiCUSTOM
from lib.network_attnMOD_Manuel import PoseNetMultiCUSTOMPointsX_old as PoseNetMultiCUSTOMPointsX

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

discord_6dpose = Discord(url="https://discord.com/api/webhooks/1387805049471897811/DYo4R4oqJfGr9H2KT6jrqSTOJb_lNbgh9ZPn0nkPAeMDFTgRrykxrqe9cCa6Nqd2yCtK")
discord_metrics = Discord(url="https://discord.com/api/webhooks/1390414457381060698/butYB50GjmbOuIoQKcsp97wYuCtww8xN977XZl9EuSKdAzOhOpsDZ0dHn3hp9V-IiTyP")

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
parser.add_argument('--option', type=int,  required=True, help='opção a executar')
parser.add_argument('--modalities', type=int, default = 0, help='modalidades a executar')

parser.add_argument('--train', action='store_true', help='training mode')
parser.add_argument('--run', action='store_true', help='run mode')
parser.add_argument('--metrics', action='store_true', help='metrics mode')
parser.add_argument('--annotate', action='store_true', help='annotate mode')
parser.add_argument('--class_id', type=int, default = None, help='treinar apenas para a classe especifica')
opt = parser.parse_args()


if __name__ == '__main__':
    stale = 0
    opt.manualSeed = random.randint(1, 10000)
    random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)
    torch.cuda.set_device(0)

    modelString = "MobileNetV2"

    if opt.option == 1:
        opt.outf = f'trained_models/{modelString}/1-img_model-velodyne-pc_model-pc_velodyne'
        concat = "model"
        mask = "velodyne"
        print("option1- img_model-velodyne-pc_model-pc_velodyne")
    elif opt.option == 2:
        opt.outf = f'trained_models/{modelString}/2-img_depth-velodyne-pc_depth-pc_velodyne'
        concat = "depth"
        mask = "velodyne"
        print("option2- img_depth-velodyne-pc_depth-pc_velodyne")
    elif opt.option == 3:
        opt.outf = f'trained_models/{modelString}/3-img_model-depth-pc_model-pc_depth'
        concat = "model"
        mask = "depth"
        print("option3- img_model-depth-pc_model-pc_depth")
    elif opt.option == 4:
        opt.outf = f'trained_models/{modelString}/4-img_depth-depth-pc_model-pc_velodyne'
        concat = "depth"
        mask = "depth"
        print("option4- img_depth-depth-pc_model-pc_velodyne")
    elif opt.option == 5:
        opt.outf = f'trained_models/{modelString}/5-img_depth-velodyne-pc_model-pc_velodyne'
        concat = "depth"
        mask = "velodyne"
        print("option5- img_depth-velodyne-pc_model-pc_velodyne")
    elif opt.option == 6:
        opt.outf = f'trained_models/{modelString}/6-img_depth-depth-pc_velodyne-pc_model'
        concat = "depth"
        mask = "velodyne"
        print("option6- img_depth-velodyne-pc_velodyne-pc_model")
    elif opt.option == 7:
        opt.outf = f'trained_models/{modelString}/7-img_velodyne-model-pc_velodyne-pc_depth'
        concat = "velodyne"
        mask = "model"
        print("option7- img_velodyne-model-pc_velodyne-pc_depth")
    elif opt.option == 8:
        opt.outf = f'trained_models/{modelString}/8-img_velodyne-model-pc_velodyne-pc_model'
        concat = "velodyne"
        mask = "model"
        print("option8- img_velodyne-model-pc_velodyne-pc_model")

    opt.num_objects = 7
    opt.num_points = 1000
    opt.outfpre = 'trained_models/PoseIndustrial6DMultiRGBOnlyRandomized' + modelString

    opt.outf += f"/{opt.num_objects} objetos"

    opt.modalities = 0
    if opt.modalities == 0:
        opt.outf += '/All'
        print("Todas as modalidades ativas")
    elif opt.modalities == 1:
        opt.outf += '/RGB'
        print("RGB")
    elif opt.modalities == 2:
        opt.outf += '/RGB-Depth1'
        print("RGB e Depth1")
    elif opt.modalities == 3:
        opt.outf += '/RGB-Depth1-PC1'
        print("RGB, Depth1 e PC1")
    print()

    if opt.class_id != None:
        print("Vamos treinar um modelo para a classe ", opt.class_id)
        opt.outf += f' class{opt.class_id}'

    print("Out folder", opt.outf)
    print()
    
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

    anotdataloader = AnotDataset('all', opt.num_points, concatmethod=concat, maskedmethod=mask)
    anotdataloader = torch.utils.data.DataLoader(anotdataloader, batch_size=opt.batch_size, shuffle=False, num_workers=opt.workers)
    
    if not os.path.exists(opt.outf):
        os.makedirs(opt.outf)

    if not os.path.exists(opt.log_dir):
        os.makedirs(opt.log_dir)

    criterion = Lossv2(opt.num_points)

    train = Train(optimizer, dataloader, testdataloader, estimator, criterion, opt, discord_6dpose)
    run = Run(dataloader, estimator, criterion, opt)
    metrics = Metrics(testdataloader, estimator, criterion, opt, discord_metrics)
    anot = Annotate(anotdataloader, estimator, opt)

    if opt.train:
        train.main()
    
    if opt.run:
        estimator.load_state_dict(torch.load(f"{opt.outf}/pose_model_best.pth"))

        run.main()

    if opt.metrics:
        estimator.load_state_dict(torch.load(f"{opt.outf}/pose_model_best.pth"))

        metrics.main()

    if opt.annotate:
        estimator.load_state_dict(torch.load(f"{opt.outf}/pose_model_best.pth"))

        anot.main()
