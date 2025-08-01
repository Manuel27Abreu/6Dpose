# --------------------------------------------------------
# DenseFusion 6D Object Pose Estimation by Iterative Dense Fusion
# Licensed under The MIT License [see LICENSE for details]
# Written by Chen
# --------------------------------------------------------

import _init_paths
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
from datasets.PoseIndustrial6D.dataloader_6DPMultiResize3 import PoseDataset2 as PoseDataset_pallets

from lib.network_attnMOD import PoseNetMultiCUSTOMPointsX,PoseNetMultiCUSTOM

from lib.lossMOD import Lossv2
from lib.utils import setup_logger

import matplotlib as mpl
import matplotlib.pyplot as plt
from discordwebhook import Discord
import torch.nn.functional as F


 
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
        p#retrained_model = nn.Sequential(*list(pretrained_model.children())[:-1])
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




discord = Discord(url="https://discord.com/api/webhooks/1316486031360393226/mCisg8dJBB3MnVrDJBKolqTNlqkSajQQznj9PYaKqV-6hXDVGp3Gc_HQIojGgvHP7blp")



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



DepthEnable=1.0
RGBEnable=1.0
DepthVelodyneEnable=1.0
RGBDPointsEnable=1.0
VelodynePointsEnable=1.0


DepthEnableVAL=1.0
RGBEnableVAL=1.0
DepthVelodyneEnableVAL=1.0
RGBDPointsEnableVAL=1.0
VelodynePointsEnableVAL=1.0

def main():
    stale=0
    opt.manualSeed = random.randint(1, 10000)
    random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)
    torch.cuda.set_device(0)


    modelString="MobileNetV2"

    opt.num_objects = 1
    opt.num_points = 1000
    opt.outf = 'trained_models/PoseIndustrial6DMultiRGBOnlyRandomized'+modelString+str(opt.num_points)
    opt.outfpre = 'trained_models/PoseIndustrial6DMultiRGBOnlyRandomized'+modelString

    opt.log_dir = 'experiments/PoseIndustrial6DMultiRGBOnlyRandomized'+modelString+str(opt.num_points)
    opt.repeat_epoch = 1




    modelRGB,embchannels=load_pre_trained_model(modelString)
    modelDepth=load_pre_trained_model_1to3(modelString)
    modelDepth2=load_pre_trained_model_1to3(modelString)
    
    estimator = PoseNetMultiCUSTOMPointsX(modelRGB,modelDepth,modelDepth2,num_points = opt.num_points, num_obj = opt.num_objects,embchannels=embchannels)
    estimator.cuda()

 
    if opt.resume_posenet == 'pretrain':
        estimatorfake = PoseNetMultiCUSTOM(modelRGB,modelDepth,modelDepth2,num_points = opt.num_points, num_obj = opt.num_objects,embchannels=embchannels)

        estimatorfake.load_state_dict(torch.load('{0}/{1}'.format(opt.outfpre, "pose_model_best.pth")))
        estimator.cnn.encoder.rgb_backbone = estimatorfake.cnn.encoder.rgb_backbone 
        estimator.cnn.encoder.depth_backbone = estimatorfake.cnn.encoder.depth_backbone 
        estimator.cnn.encoder.depth_backbone2 = estimatorfake.cnn.encoder.depth_backbone2 
        estimator.cnn.attention_fusion=estimatorfake.cnn.attention_fusion
        estimator.cuda()

    elif opt.resume_posenet != '':
        print('Loading ',opt.resume_posenet)
        estimator.load_state_dict(torch.load('{0}/{1}'.format(opt.outf, opt.resume_posenet)))


    opt.refine_start = False
    opt.decay_start = False
    optimizer = optim.Adam(estimator.parameters(), lr=opt.lr, weight_decay=0.00001)

    dataset = PoseDataset_pallets('train', opt.num_points, True, opt.dataset_root, opt.noise_trans, opt.refine_start)

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=opt.workers)
    test_dataset = PoseDataset_pallets('test', opt.num_points, False, opt.dataset_root, 0.0, opt.refine_start)    

    testdataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=opt.workers)
    
 
    if not os.path.exists(opt.outf):
        os.makedirs(opt.outf)

    if not os.path.exists(opt.log_dir):
        os.makedirs(opt.log_dir)

    criterion = Lossv2(opt.num_points)
  
    best_test = 2000.2465

    modelCheck=False;

    if opt.start_epoch == 1:
        for log in os.listdir(opt.log_dir):
            os.remove(os.path.join(opt.log_dir, log))
    st_time = time.time()
    #opt.refine_start = True
    for epoch in range(opt.start_epoch, opt.nepoch):
        logger = setup_logger('epoch%d' % epoch, os.path.join(opt.log_dir, 'epoch_%d_log.txt' % epoch))
        logger.info('Train time {0}'.format(time.strftime("%Hh %Mm %Ss", time.gmtime(time.time() - st_time)) + ', ' + 'Training started'))
        train_count = 0
        train_dis_avg = 0.0

        estimator.train()
        optimizer.zero_grad()

 


        for rep in range(opt.repeat_epoch):
            currentloss=None
            counter=0

            for i, data in enumerate(dataloader, 0):
                target, points, rt, idx, index, img, choose, velodyne,velodyne_gt,depth_vel,gt,modelPoints,modelPointsGT = data
                if (gt[0][0][3]**2+gt[0][1][3]**2+gt[0][2][3]**2)<0.05:
                    continue
                
                # target,   points, rt, idx, index, img, choose, velodyne, velodyne_gt, depth_vel           ,gt,modelPoints,modelPointsGT
                # pc_cam_W, pc_cam, rt, idx, index, img, choose, pc_vel_W,     pc_vel,  depth_expanded

                """
                points = Variable(pc_depth).cuda()
                choose = Variable(choose).cuda()
                img = Variable(img).cuda()
                target = Variable(pc_depth_W).cuda()
                idx = Variable(idx).cuda()
                velodyne = Variable(pc_velodyne_W).cuda()

                velodyne_gt = Variable(pc_velodyne).cuda()
                depth_vel = Variable(depth_vel).cuda()

                modelPoints = Variable(modelPoints).cuda()
                modelPointsGT = Variable(modelPointsGT).cuda()
                """

                DepthEnable = float(random.randint(0, 1))
                RGBEnable = float(random.randint(0, 1))
                DepthVelodyneEnable = float(random.randint(0, 1))
                RGBDPointsEnable = float(random.randint(0, 1))
                VelodynePointsEnable = float(random.randint(0, 1)) 

                obj_centered = points
                model_points = obj_centered#torch.mm(obj_centered.float(), rt[0].float().T)
                #print(rt[0].float().T)

                
                points= Variable(points).cuda();
                choose=Variable(choose).cuda()
                img=  Variable(img).cuda()                                           
                target=Variable(target).cuda()                                              
                model_points=  Variable(model_points).cuda()                                           
                idx=    Variable(idx).cuda()
                velodyne = Variable(velodyne).cuda()

                velodyne_gt=  Variable(velodyne_gt).cuda()
                depth_vel=  Variable(depth_vel).cuda()

                modelPoints=  Variable(modelPoints).cuda()
                modelPointsGT=  Variable(modelPointsGT).cuda()



                img[:,3,:,:]=img[:,3,:,:]*DepthEnable
                img[:,0:3,:,:]=img[:,0:3,:,:]*RGBEnable
                pred_r, pred_t, pred_c, emb = estimator(img,depth_vel*DepthVelodyneEnable, points*RGBDPointsEnable,velodyne*VelodynePointsEnable, choose, idx)

                #pred_r, pred_t, pred_c, emb = estimator(img, points, choose, idx)
                #loss, dis, pred, points = criterion(pred_r, pred_t, pred_c, target, model_points, idx, points, opt.w, opt.refine_start)
                loss, dis, new_points, new_target = criterion(pred_r, pred_t, pred_c,  modelPointsGT, modelPoints, idx, points, opt.w, opt.refine_start)

                loss.backward()
                print(opt.lr,'% .4f' %loss.item(),'% .4f' %dis.item())#," R: ", '% .3f' %pred_r[0].item(), '% .3f' %pred_r[1].item(), '% .3f' %pred_r[2].item(), '% .3f' %pred_r[3].item()," T: ",'% .3f' %pred_t[0].item(), '% .3f' %pred_t[1].item(), '% .3f' %pred_t[2].item())

                train_dis_avg += dis.item()
                train_count += 1

                if train_count % opt.batch_size == 0:
                    logger.info('Train time {0} Epoch {1} Batch {2} Frame {3} Avg_dis:{4}'.format(time.strftime("%Hh %Mm %Ss", time.gmtime(time.time() - st_time)), epoch, int(train_count / opt.batch_size), train_count, train_dis_avg / opt.batch_size))
                    optimizer.step()
                    optimizer.zero_grad()
                    train_dis_avg = 0

                 

        print('>>>>>>>>----------epoch {0} train finish---------<<<<<<<<'.format(epoch))


        logger = setup_logger('epoch%d_test' % epoch, os.path.join(opt.log_dir, 'epoch_%d_test_log.txt' % epoch))
        logger.info('Test time {0}'.format(time.strftime("%Hh %Mm %Ss", time.gmtime(time.time() - st_time)) + ', ' + 'Testing started'))
        test_dis = 0.0
        test_count = 0
        estimator.eval()

        for j, data in enumerate(testdataloader, 0):
            target, points, rt, idx, index, img, choose, velodyne,velodyne_gt,depth_vel,gt,modelPoints,modelPointsGT = data
            if (gt[0][0][3]**2+gt[0][1][3]**2+gt[0][2][3]**2)<0.05:
                continue



            obj_centered = points
            model_points = obj_centered#torch.mm(obj_centered.float(), rt[0].float().T)
            #print(rt[0].float().T)

                
            points= Variable(points).cuda();
            choose=Variable(choose).cuda()
            img=  Variable(img).cuda()                                           
            target=Variable(target).cuda()                                              
            model_points=  Variable(model_points).cuda()                                           
            idx=    Variable(idx).cuda()
            velodyne = Variable(velodyne).cuda()
            velodyne_gt=  Variable(velodyne_gt).cuda()
            depth_vel=  Variable(depth_vel).cuda()

            modelPoints=  Variable(modelPoints).cuda()
            modelPointsGT=  Variable(modelPointsGT).cuda()

            img[:,3,:,:]=img[:,3,:,:]*DepthEnableVAL
            img[:,0:3,:,:]=img[:,0:3,:,:]*RGBEnableVAL

            pred_r, pred_t, pred_c, emb = estimator(img,depth_vel*DepthVelodyneEnableVAL, points*RGBDPointsEnableVAL,velodyne*VelodynePointsEnableVAL, choose, idx)
            _, dis, new_points, new_target = criterion(pred_r, pred_t, pred_c, modelPointsGT, modelPoints, idx, points, opt.w, opt.refine_start)
            minx=dis.item()
           
            test_dis += minx
            print(test_count,'% .4f' %minx)
            #logger.info('Test time {} Test Frame No.{} dis:{: .6f} -> R: {: .3f} {: .3f} {: .3f} {: .3f} T: {: .3f} {: .3f} {: .3f}'.format(time.strftime("%Hh %Mm %Ss", time.gmtime(time.time() - st_time)), test_count, minx,pred_r[0].item(),pred_r[1].item(),pred_r[2].item(),pred_r[3].item(),pred_t[0].item(),pred_t[1].item(),pred_t[2].item()))
            test_count += 1

        test_dis = test_dis / test_count
        logger.info('Test time {0} Epoch {1} TEST FINISH Avg dis: {2}'.format(time.strftime("%Hh %Mm %Ss", time.gmtime(time.time() - st_time)), epoch, test_dis))
        
        #discord.post(content="re0MOD on Windows Current 6D pose error Multi(on RGBD): "+str(test_dis)+"\n")
        if test_dis <= best_test:
            best_test = test_dis
            
            torch.save(estimator.state_dict(), '{0}/pose_model_{1}_{2}.pth'.format(opt.outf, epoch, test_dis))
            torch.save(estimator.state_dict(), '{0}/pose_model_best.pth'.format(opt.outf))
            print(epoch, '>>>>>>>>----------BEST TEST MODEL SAVED---------<<<<<<<<')
            stale=0
        else:
            stale=stale+1


        discord.post(content=" Windows Current 6D pose error Multi v3 (Randomized train) (No Resize) New Loss from GT pose Total points "+str(opt.num_points)+" ("+str(modelString)+"): "+str(test_dis)+" with best "+str(best_test)+"\n")

        print('debug 1')

        if stale>9:
            opt.lr *= opt.lr_rate
            estimator.load_state_dict(torch.load( '{0}/pose_model_best.pth'.format(opt.outf)))
            optimizer = optim.Adam(estimator.parameters(), lr=opt.lr , weight_decay=0.00001)
            stale=0

 

if __name__ == '__main__':
    main()
