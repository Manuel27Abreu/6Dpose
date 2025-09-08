# --------------------------------------------------------
# DenseFusion 6D Object Pose Estimation by Iterative Dense Fusion
# Licensed under The MIT License [see LICENSE for details]
# Written by Chen
# --------------------------------------------------------

import os
import sys
sys.path.insert(0, os.getcwd())
import time
import math
from datetime import datetime
import numpy as np
import torch
import torch.optim as optim
from torch.autograd import Variable
import matplotlib.pyplot as plt

from tqdm import tqdm

class Train:
    def __init__(self, optimizer, dataloader, testdataloader, estimator, criterion, opt, discord):
        self.dataloader = dataloader
        self.testdataloader = testdataloader
        self.option = opt.option
        self.modalities = opt.modalities
        self.opt = opt
        self.estimator = estimator
        self.criterion = criterion
        self.discord = discord
        self.optimizer = optimizer

    def train_epoch(self, epoch, train_dis_avg, train_count):
        # TREINO
        for i, data in tqdm(enumerate(self.dataloader, 0), total=len(self.dataloader), desc=f'Epoch {epoch}', unit='batch'):
            pc_depth_W, pc_depth, pc_velodyne_W, pc_velodyne, pc_model_W, pc_model, img, depth_vel, modelPoints, modelPointsGT, rt, idx = data

            if self.opt.class_id != None:
                mask = idx == self.opt.class_id

                if mask.sum() == 0:
                    continue

                pc_depth_W = pc_depth_W[mask]
                pc_depth = pc_depth[mask]
                pc_velodyne_W = pc_velodyne_W[mask]
                pc_velodyne = pc_velodyne[mask]
                pc_model_W = pc_model_W[mask]
                pc_model = pc_model[mask]
                img = img[mask]
                depth_vel = depth_vel[mask]
                modelPoints = modelPoints[mask]
                modelPointsGT = modelPointsGT[mask]
                rt = rt[mask]
                idx = idx[mask]


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

            loss, dis, new_points, new_target = self.criterion(pred_r, pred_t, pred_c, modelPointsGT, modelPoints, idx, points, self.opt.w, self.opt.refine_start)

            train_dis_avg += loss.item()
            train_count += 1

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        return train_dis_avg, train_count


    def eval_epoch(self, epoch):
        test_dis = 0.0
        test_count = 0
        self.estimator.eval()
        
        #  EVAL
        for j, data in tqdm(enumerate(self.testdataloader, 0), total=len(self.testdataloader), desc=f'Epoch {epoch}(eval)', unit='batch'):
            pc_depth_W, pc_depth, pc_velodyne_W, pc_velodyne, pc_model_W, pc_model, img, depth_vel, modelPoints, modelPointsGT, rt, idx = data

            if self.opt.class_id != None:
                mask = idx == self.opt.class_id

                if mask.sum() == 0:
                    continue

                pc_depth_W = pc_depth_W[mask]
                pc_depth = pc_depth[mask]
                pc_velodyne_W = pc_velodyne_W[mask]
                pc_velodyne = pc_velodyne[mask]
                pc_model_W = pc_model_W[mask]
                pc_model = pc_model[mask]
                img = img[mask]
                depth_vel = depth_vel[mask]
                modelPoints = modelPoints[mask]
                modelPointsGT = modelPointsGT[mask]
                rt = rt[mask]
                idx = idx[mask]
            
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

            loss, dis, new_points, new_target = self.criterion(pred_r, pred_t, pred_c, modelPointsGT, modelPoints, idx, points, self.opt.w, self.opt.refine_start)

            test_dis += loss.item()
            test_count += 1

        test_dis = test_dis / test_count

        return test_dis

    def save_real_vs_reconstruction(self):
        self.estimator.eval()

        data = next(iter(self.dataloader))

        pc_depth_W, pc_depth, pc_velodyne_W, pc_velodyne, pc_model_W, pc_model, img, depth_vel, modelPoints, modelPointsGT, rt, idx = data

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
        
        modelPoints = Variable(modelPoints).cuda()
        modelPointsGT = Variable(modelPointsGT).cuda()

        img[:,0:3,:,:] = img[:,0:3,:,:] * RGBEnable
        img[:,3,:,:] = img[:,3,:,:] * Depth1Enable

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

        T = self.computeT(pred_r[0], pred_t[0])
        
        self.target_vs_pred(pc_depth_W[0], pc_velodyne_W[0], pc_model_W[0], pc_depth[0], pc_velodyne[0], pc_model[0], T.detach().cpu().numpy(), rt[0].detach().cpu().numpy())

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

        ori_base = base
        # base = base.contiguous().transpose(2, 1).contiguous()

        rot = base[0]
        trans = pred_t.view(3, 1)
        upper = torch.cat([rot, trans], dim=1)
        bottom = torch.tensor([[0.0, 0.0, 0.0, 1.0]], device=rot.device)
        transform = torch.cat([upper, bottom], dim=0)

        return transform

    def target_vs_pred(self, pc_depth_W, pc_velodyne_W, pc_model_W, pc_depth, pc_velodyne, pc_model, pred_RT, RT):
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

    def main(self):  
        best_test = 2000.2465

        diz_loss = {'train_loss': [], 'eval_loss': []}

        st_time = time.time()
        print(datetime.fromtimestamp(st_time).strftime('%d-%m-%y %H:%M:%S'))

        for epoch in range(self.opt.start_epoch, self.opt.nepoch):
            train_count = 0
            train_dis_avg = 0.0

            self.estimator.train()
            self.optimizer.zero_grad()

            for rep in range(self.opt.repeat_epoch):
                train_dis_avg, train_count = self.train_epoch(epoch, train_dis_avg, train_count)
            diz_loss['train_loss'].append(train_dis_avg / train_count)
            
            test_dis = self.eval_epoch(epoch)
            diz_loss['eval_loss'].append(test_dis)

            tqdm.write(f"Epoch {epoch}/{self.opt.nepoch}\t Train_dis: {train_dis_avg / train_count:.6f} \t Eval_dis: {test_dis:.6f}")
            
            if test_dis <= best_test:
                best_test = test_dis

                # torch.save(self.estimator.state_dict(), '{0}/pose_model_{1}_{2}.pth'.format(self.opt.outf, epoch, test_dis))
                torch.save(self.estimator.state_dict(), '{0}/pose_model_best.pth'.format(self.opt.outf))
                print(epoch, '>>>>>>>>----------BEST TEST MODEL SAVED---------<<<<<<<<')
                stale = 0
            else:
                stale = stale + 1

            # discord.post(content=" Windows Current 6D pose error Multi v3 (Randomized train) (No Resize) New Loss from GT pose Total points "+str(opt.num_points)+" ("+str(modelString)+"): "+str(test_dis)+" with best "+str(best_test)+"\n")

            if stale > 9:
                self.opt.lr *= self.opt.lr_rate
                self.estimator.load_state_dict(torch.load('{0}/pose_model_best.pth'.format(self.opt.outf)))
                self.optimizer = optim.Adam(self.estimator.parameters(), lr=self.opt.lr , weight_decay=0.00001)
                stale = 0
            
        torch.save(self.estimator.state_dict(), '{0}/pose_model_final.pth'.format(self.opt.outf))

        self.estimator.load_state_dict(torch.load(f"{self.opt.outf}/pose_model_best.pth"))
        self.save_real_vs_reconstruction()

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
        self.discord.post(content=f"Treino finalizado {formatted_time}")

        self.discord.post(
            file={
                    "file1": open(f"imgs/train_vs_eval_loss_plot.png", "rb"),
                    "file2": open(f"imgs/targetvspred.png", "rb"),
            },
        )
