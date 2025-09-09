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
import math
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
from scipy.spatial import KDTree

import matplotlib.pyplot as plt
from tqdm import tqdm


class Annotate:
    def __init__(self, dataloader, estimator, opt):
        self.dataloader = dataloader
        self.estimator = estimator
        self.option = opt.option
        self.modalities = opt.modalities

    def annotate(self):
        self.estimator.eval()
        bad_anot = "../Anot model/results modelo/mas anotações.txt"
        open(bad_anot, "w").close()

        for i, data in tqdm(enumerate(self.dataloader, 0), total=len(self.dataloader), desc=f'', unit='batch'):
            pc_depth_W, pc_depth, pc_velodyne_W, pc_velodyne, pc_model_W, pc_model, img, depth_vel, modelPoints, modelPointsGT, rt, idx, file_name = data

            if self.modalities == 0:
                RGBEnable = float(1)
                Depth1Enable = float(1)
                Depth2Enable = float(1)
                PC1Enable = float(1)
                PC2Enable = float(1)
            elif self.modalities == 1:
                RGBEnable = float(1)
                Depth1Enable = float(0)
                Depth2Enable = float(0)
                PC1Enable = float(0)
                PC2Enable = float(0)
            elif self.modalities == 2:
                RGBEnable = float(1)
                Depth1Enable = float(1)
                Depth2Enable = float(0)
                PC1Enable = float(0)
                PC2Enable = float(0)
            elif self.modalities == 3:
                RGBEnable = float(1)
                Depth1Enable = float(1)
                Depth2Enable = float(0)
                PC1Enable = float(1)
                PC2Enable = float(0)
            elif self.modalities == 4:
                RGBEnable = float(1)
                Depth1Enable = float(0)
                Depth2Enable = float(1)
                PC1Enable = float(0)
                PC2Enable = float(1)
            elif self.modalities == 5:
                RGBEnable = float(1)
                Depth1Enable = float(1)
                Depth2Enable = float(1)
                PC1Enable = float(0)
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
                if self.option == 1:
                    pred_r, pred_t, pred_c, _ = self.estimator(img, depth_vel*Depth2Enable, model_gt*PC1Enable, velodyne_gt*PC2Enable, choose, idx)
                elif self.option == 2:
                    pred_r, pred_t, pred_c, _ = self.estimator(img, depth_vel*Depth2Enable, target*PC1Enable, velodyne_gt*PC2Enable, choose, idx)
                elif self.option == 3:
                    pred_r, pred_t, pred_c, _ = self.estimator(img, depth_vel*Depth2Enable, model_gt*PC1Enable, target*PC2Enable, choose, idx)
                elif self.option == 4:
                    pred_r, pred_t, pred_c, _ = self.estimator(img, depth_vel*Depth2Enable, model_gt*PC1Enable, target*PC2Enable, choose, idx)
                elif self.option == 5:
                    pred_r, pred_t, pred_c, _ = self.estimator(img, depth_vel*Depth2Enable, model_gt*PC1Enable, velodyne_gt*PC2Enable, choose, idx)
                elif self.option == 6:
                    pred_r, pred_t, pred_c, _ = self.estimator(img, depth_vel*Depth2Enable, target*PC1Enable, model_gt*PC2Enable, choose, idx)
                elif self.option == 7:
                    pred_r, pred_t, pred_c, _ = self.estimator(img, depth_vel*Depth2Enable, velodyne_gt*PC1Enable, target*PC2Enable, choose, idx)
                elif self.option == 8:
                    pred_r, pred_t, pred_c, _ = self.estimator(img, depth_vel*Depth2Enable, velodyne_gt*PC1Enable, model_gt*PC2Enable, choose, idx)

            predictions = []
            for i in range(rt.shape[0]):
                centroide = velodyne_gt[i].mean(dim=0)
                dist_centroide = torch.norm(centroide).item()

                T = self.computeT(pred_r[i], pred_t[i])
                T = T.detach().cpu().numpy()

                x, y, z = T[0][3], T[1][3], T[2][3]
                dist = math.sqrt(x**2 + y**2 + z**2)
                if dist_centroide < 20:
                    # guardar no ficheiro
                    with open(file_name[i], 'w') as f:
                        for row in T:
                            formatted_row = ' '.join(f"{val:.5f}" for val in row)
                            f.write(formatted_row + " ; ") 

                    diff = abs(dist_centroide - dist)

                    pc = velodyne_gt[i].cpu().numpy()
                    p = np.array([x, y, z])
                    tree = KDTree(pc)
                    dist_nn, _ = tree.query(p)

                    dentro = dist_nn < 0.5

                    if not dentro:
                        with open(bad_anot, "a") as f:
                            f.write(f"{file_name[i]}\n")

                else:
                    # tqdm.write(f"Matriz identidade")
                    # escrever matriz identidade
                    identity = [
                        [1.0, 0.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0, 0.0],
                        [0.0, 0.0, 1.0, 0.0],
                        [0.0, 0.0, 0.0, 1.0]
                    ]
                    with open(file_name[i], 'w') as f:
                        for row in identity:
                            formatted_row = ' '.join(f"{val:.5f}" for val in row)
                            f.write(formatted_row + " ; ")

                

    def computeT(self, pred_r, pred_t):
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

        rot = base[0]
        trans = pred_t.view(3, 1)
        upper = torch.cat([rot, trans], dim=1)
        bottom = torch.tensor([[0.0, 0.0, 0.0, 1.0]], device=rot.device)
        transform = torch.cat([upper, bottom], dim=0)

        return transform

    def main(self):
        self.annotate()

