# --------------------------------------------------------
# DenseFusion 6D Object Pose Estimation by Iterative Dense Fusion
# Licensed under The MIT License [see LICENSE for details]
# Written by Chen
# --------------------------------------------------------

import os
import sys
sys.path.insert(0, os.getcwd())
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.autograd import Variable
from tqdm import tqdm

class Run:
    def __init__(self, dataloader, estimator, criterion, opt):
        self.dataloader = dataloader
        self.option = opt.option
        self.modalities = opt.modalities
        self.opt = opt
        self.estimator = estimator
        self.criterion = criterion

    def view_results(self):
        self.estimator.eval()

        for i, data in tqdm(enumerate(self.dataloader, 0), total=len(self.dataloader), desc=f'', unit='batch'):
            pc_depth_W, pc_depth, pc_velodyne_W, pc_velodyne, pc_model_W, pc_model, img, depth_vel, modelPoints, modelPoints_W, rt, idx = data

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

            loss, dis, new_points, new_target = self.criterion(pred_r, pred_t, pred_c, modelPoints_W, modelPoints, idx, points, self.opt.w, self.opt.refine_start)

            for b in range(dis.shape[0]):
                T = self.computeT(pred_r[b], pred_t[b])
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

                self.target_vs_pred(pc_depth_W[b], pc_velodyne_W[b], pc_model_W[b], modelPoints_W[b], pc_depth[b], pc_velodyne[b], pc_model[b], modelPoints[b], T, rt[b].detach().cpu().numpy())

            print()

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

    def target_vs_pred(self, pc_depth_W, pc_velodyne_W, pc_model_W, modelPoints_W, pc_depth, pc_velodyne, pc_model, modelPoints, pred_RT, RT):
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

    def main(self):
       self.view_results()
