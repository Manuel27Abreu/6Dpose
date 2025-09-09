import os
import sys
sys.path.insert(0, os.getcwd())
import math
import numpy as np
import torch
from torch.autograd import Variable
import matplotlib.pyplot as plt
from tqdm import tqdm


class Metrics:
    def __init__(self, testdataloader, estimator, criterion, opt, discord):
        self.testdataloader = testdataloader
        self.option = opt.option
        self.modalities = opt.modalities
        self.opt = opt
        self.estimator = estimator
        self.criterion = criterion
        self.discord = discord

    def compute_metrics(self):
        self.estimator.eval()

        total_loss = 0.0
        total_batches = 0

        loss_cls0, loss_cls1, loss_cls2, loss_cls3, loss_cls4, loss_cls5, loss_cls6 = 0, 0, 0, 0, 0, 0, 0
        batch_cls0, batch_cls1, batch_cls2, batch_cls3, batch_cls4, batch_cls5, batch_cls6 = 0, 0, 0, 0, 0, 0, 0

        depththresholds = [5, 10, 15, 20]
        num_bins = len(depththresholds)

        # Global loss acumulado até cada threshold
        loss_by_depth = [0.0 for _ in range(num_bins)]
        count_by_depth = [0 for _ in range(num_bins)]

        # Loss por classe até cada threshold
        loss_by_class_depth = [[0.0 for _ in range(num_bins)] for _ in range(7)]
        count_by_class_depth = [[0 for _ in range(num_bins)] for _ in range(7)]

        # <<< novo >>> thresholds de erro para curva accuracy
        num_classes = 7
        error_thresholds = np.arange(0.05, 0.85, 0.05).tolist()
        
        correct_by_threshold = [0 for _ in error_thresholds]
        total_predictions = 0
        
        correct_by_threshold_class = [[0 for _ in error_thresholds] for _ in range(num_classes)]
        total_predictions_class = [0 for _ in range(num_classes)]

        for i, data in tqdm(enumerate(self.testdataloader, 0), total=len(self.testdataloader), desc=f'', unit='batch'):
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

            loss, dis, new_points, new_target = self.criterion(pred_r, pred_t, pred_c, modelPointsGT, modelPoints, idx, points, self.opt.w, self.opt.refine_start)

            batch_size = pred_r.size(0)
            total_loss += dis.sum().item()
            total_batches += batch_size

            for b in range(batch_size):
                class_idx = idx[b].item()
                dis_b = dis[b].item()
                
                # <<< novo >>> contabilizar para accuracy vs threshold
                total_predictions += 1
                for j, th in enumerate(error_thresholds):
                    if dis_b <= th:
                        correct_by_threshold[j] += 1

                # Por classe <<< novo >>>
                total_predictions_class[class_idx] += 1
                for j, th in enumerate(error_thresholds):
                    if dis_b <= th:
                        correct_by_threshold_class[class_idx][j] += 1

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

        avg_loss_by_class_depth = [
            [
                loss_by_class_depth[cls][i] / count_by_class_depth[cls][i] if count_by_class_depth[cls][i] > 0 else 0.0
                for i in range(num_bins)
            ]
            for cls in range(7)
        ]

        # <<< novo >>> calcular accuracy vs threshold
        accuracy_by_threshold = [c / total_predictions for c in correct_by_threshold]

        accuracy_by_threshold_class = [
            [correct_by_threshold_class[cls][j] / total_predictions_class[cls] if total_predictions_class[cls] > 0 else 0.0
            for j in range(len(error_thresholds))]
            for cls in range(num_classes)
        ]

        return avg_loss, [loss_cls0, loss_cls1, loss_cls2, loss_cls3, loss_cls4, loss_cls5, loss_cls6], avg_loss_by_class_depth, accuracy_by_threshold, accuracy_by_threshold_class


    def compute_metrics_old(self):
        self.estimator.eval()

        total_loss = 0.0
        total_batches = 0

        loss_cls0, loss_cls1, loss_cls2, loss_cls3, loss_cls4, loss_cls5, loss_cls6 = 0, 0, 0, 0, 0, 0, 0
        batch_cls0, batch_cls1, batch_cls2, batch_cls3, batch_cls4, batch_cls5, batch_cls6 = 0, 0, 0, 0, 0, 0, 0

        depththresholds = [5, 10, 15, 20]
        num_bins = len(depththresholds)

        # Global loss acumulado até cada threshold
        loss_by_depth = [0.0 for _ in range(num_bins)]
        count_by_depth = [0 for _ in range(num_bins)]

        # Loss por classe até cada threshold
        loss_by_class_depth = [[0.0 for _ in range(num_bins)] for _ in range(7)]
        count_by_class_depth = [[0 for _ in range(num_bins)] for _ in range(7)]

        for i, data in tqdm(enumerate(self.testdataloader, 0), total=len(self.testdataloader), desc=f'', unit='batch'):
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

            loss, dis, new_points, new_target = self.criterion(pred_r, pred_t, pred_c, modelPointsGT, modelPoints, idx, points, self.opt.w, self.opt.refine_start)

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

        avg_loss_by_class_depth = [
            [
                loss_by_class_depth[cls][i] / count_by_class_depth[cls][i] if count_by_class_depth[cls][i] > 0 else 0.0
                for i in range(num_bins)
            ]
            for cls in range(7)
        ]

        return avg_loss, [loss_cls0, loss_cls1, loss_cls2, loss_cls3, loss_cls4, loss_cls5, loss_cls6], avg_loss_by_class_depth
    
    def compute_metrics_class(self):
        self.estimator.eval()

        total_loss = 0.0
        total_batches = 0

        loss_cls0, loss_cls1, loss_cls2, loss_cls3, loss_cls4, loss_cls5, loss_cls6 = 0, 0, 0, 0, 0, 0, 0
        batch_cls0, batch_cls1, batch_cls2, batch_cls3, batch_cls4, batch_cls5, batch_cls6 = 0, 0, 0, 0, 0, 0, 0

        depththresholds = [5, 10, 15, 20]
        num_bins = len(depththresholds)

        # Global loss acumulado até cada threshold
        loss_by_depth = [0.0 for _ in range(num_bins)]
        count_by_depth = [0 for _ in range(num_bins)]

        # Loss por classe até cada threshold
        loss_by_class_depth = [[0.0 for _ in range(num_bins)] for _ in range(7)]
        count_by_class_depth = [[0 for _ in range(num_bins)] for _ in range(7)]

        for i, data in tqdm(enumerate(self.testdataloader, 0), total=len(self.testdataloader), desc=f'', unit='batch'):
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

            loss, dis, new_points, new_target = self.criterion(pred_r, pred_t, pred_c, modelPointsGT, modelPoints, idx, points, self.opt.w, self.opt.refine_start)

            batch_size = pred_r.size(0)
            total_loss += dis.sum().item()
            total_batches += batch_size

            for b in range(batch_size):
                dis_b = dis[b].item()
                
                # Processamento de rt
                t = rt[b, 0:3, 3].cpu().numpy()
                distancia = np.linalg.norm(t)

                for i, th in enumerate(depththresholds):
                    if distancia < th:
                        loss_by_depth[i] += dis_b
                        count_by_depth[i] += 1

        avg_loss = total_loss / total_batches

        avg_loss_by_depth = []
        for i in range(len(depththresholds)):
            if count_by_depth[i] > 0:
                avg = loss_by_depth[i] / count_by_depth[i]
            else:
                avg = float('nan')
            avg_loss_by_depth.append(avg)

        return avg_loss, avg_loss_by_depth
        
    def main(self):
        msg = f"Metricas da pasta {self.opt.outf}\n"

        classes = ["Bidons", "Caixa", "Caixa encaxe", "Extintor", "Empilhadora", "Pessoas", "Toolboxes"]

        depththresholds = [5, 10, 15, 20]

        loss, loss_cls, loss_cls_depth, accuracy_by_threshold, accuracy_by_threshold_class = self.compute_metrics()

        msg += f"Average loss over dataset: {loss:.4f}\n"
        msg += "Loss por classe:\n"
        msg += f"Bidons: {loss_cls[0]:.4f}\t Caixa: {loss_cls[1]:.4f}\t Caixa encaxe: {loss_cls[2]:.4f}\t Extintor: {loss_cls[3]:.4f}\t Empilhadora: {loss_cls[4]:.4f}\t Pessoas: {loss_cls[5]:.4f} \t Toolboxes: {loss_cls[6]:.4f}\n\n"

        msg += "\nLoss por classe e thresholds de profundidade:\n"
        for i, th in enumerate(depththresholds):
            msg += f"[0-{th}m]:\t"
            for cls_idx, cls_name in enumerate(classes):
                msg += f"{cls_name}: {loss_cls_depth[cls_idx][i]:.4f}\t"
            msg += "\n"

        print(msg)

        self.discord.post(content=msg)

        error_thresholds = np.arange(0.05, 0.85, 0.05).tolist()

        """plt.figure(figsize=(8,5))
        plt.plot(error_thresholds, accuracy_by_threshold, linestyle='-', color='b', label="Accuracy")
        plt.xlabel("Threshold (m)")
        plt.ylabel("Accuracy")
        plt.xlim(0, 0.8)
        plt.ylim(0, 1.0) 
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.tight_layout()
        plt.show()"""

        plt.figure(figsize=(8,5))
        plt.plot(error_thresholds, accuracy_by_threshold, marker='o', label="Global", color="black")

        # Por classe
        colors = ["Orange", "Purple", "Red", "Yellow", "Cyan", "Green", "Blue"]
        class_labels = ["Industrial Drums", "Box", "Box slot", "Fire extinguisher", "Forklift", "People", "Toolbox"]

        for cls, acc_curve in enumerate(accuracy_by_threshold_class):        
            plt.plot(
                error_thresholds, 
                acc_curve, 
                linestyle='--', 
                color=colors[cls], 
                label=class_labels[cls]
            )

        plt.xlabel("Error threshold (m)")
        plt.ylabel("Accuracy")
        plt.xlim(0, 0.8)
        plt.ylim(0, 1)
        plt.legend()
        plt.xticks(error_thresholds)

        plt.tight_layout()
        plt.savefig("imgs/accuracy_vs_threshold.png", dpi=300)
        # plt.show()

        self.discord.post(
            file={
                    "file1": open(f"imgs/accuracy_vs_threshold.png", "rb"),
            },
        )
