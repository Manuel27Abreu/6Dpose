#include <pybind11/stl.h>
import torch.utils.data as data
from PIL import Image
import os
import os.path
from os import listdir, scandir
from sys import exit
import torch
import re

import numpy as np
import torchvision.transforms as transforms
import argparse
import time
import random
#from lib.transformations import quaternion_from_euler, euler_matrix, random_quaternion, quaternion_matrix
import numpy.ma as ma
import copy
import scipy.misc
import scipy.io as scio
import matplotlib.pyplot as plt
import numpy.ma as ma
import open3d as o3d
import open3d.visualization as vis

 
class PoseDataset2(data.Dataset):
    def __init__(self, mode="all", num_pt=15000, concatmethod="depth", maskedmethod="depth"):
        self.concatmethod = concatmethod
        self.maskedmethod = maskedmethod

        self.path_depth = "../Anot model/results"
        self.path_rgb = "../Anot model/results"

        all_folders = [d for d in os.listdir(self.path_depth) if os.path.isdir(os.path.join(self.path_depth, d))]
        all_folders.sort()

        total_images = len(all_folders)
        indices = list(range(total_images))

        # split points
        train_split = int(0.8 * total_images)

        random.Random(666).shuffle(indices)

        if mode == 'train':
            selected_folders = [all_folders[i] for i in indices[:train_split]]
        elif mode == 'test':
            selected_folders = [all_folders[i] for i in indices[train_split:]]
        elif mode == 'all':
            selected_folders = all_folders

        self.num_pt_mesh_large = num_pt
        self.num_pt_mesh_small = num_pt
        self.num_points = num_pt
        self.norm = transforms.Normalize(mean=[0.485, 0.456, 0.406,0.5], std=[0.229, 0.224, 0.225,0.5])

        self.list_pc = []
        self.list_pc_depth = []
        self.list_pc_velod = []
        self.list_pc_model = []

        self.list_vel = []
        self.list_label = []
        # self.list_label2d = []
        self.list_gt = []
        self.list_rgb = []
        self.list_mask = []
        self.list_depth = []
        self.list_depthV = []
        self.list_depthM = []
        self.list_T = []

        for folder in selected_folders:
            f_path = os.path.join(self.path_depth, folder)
            for file in os.scandir(f_path):
                if 'RGB' in file.name:
                    fn = file.name
                    fpls = fn.split('_')
                    d1 = fpls[-1][3:]
                    d2 = d1.split('.')
                    str_det = fpls[1] + '_' + fpls[2] + '_det' + d2[0]
                    str_seg = fpls[1] + '_' + fpls[2] + '_seg' + d2[0]
                    detX = 'det' + d2[0]

                    self.list_pc_depth.append(f"{f_path}/PC_DEPTH_{str_det}.ply")
                    self.list_pc_velod.append(f"{f_path}/PC_VELODYNE_{str_det}.ply")
                    self.list_pc_model.append(f"{f_path}/PC_MODEL_{str_det}.ply")

                    self.list_rgb.append(f"{f_path}/RGB_{str_det}.png")
                    self.list_mask.append(f"{f_path}/mask_{str_seg}.png")
                    self.list_depth.append(f"{f_path}/DEPTH_{str_det}.png")
                    self.list_depthV.append(f"{f_path}/VELODYNE_{str_det}.png")
                    self.list_depthM.append(f"{f_path}/MODEL_{str_det}.png")

                    self.list_label.append(f"{f_path}/{detX}.txt")

        self.length = len(self.list_rgb)

    def __getitem__(self, j):
        idx = int(re.search(r'class(\d+)', self.list_depth[j]).group(1))

        det_file_name = self.list_label[j]

        # Load pointclouds
        try:
            pointcloud_cam_W = np.asarray(o3d.io.read_point_cloud(self.list_pc_depth[j]).points)
            pointcloud_vel_W = np.asarray(o3d.io.read_point_cloud(self.list_pc_velod[j], format = "ply").points)
            pointcloud_model_W = np.asarray(o3d.io.read_point_cloud(self.list_pc_model[j], format = "ply").points)
        except FileNotFoundError:
            exit("ERROR: Necessary PC files not found. Exiting program")

        # Load rotation matrix
        try:
            with open(self.list_label[j], 'r') as f:
                rt = f.read().replace(';', ' ')
            rt = np.fromstring(rt, sep=' ').reshape((4, 4))
        except:
            rt = np.identity(4)
            # print("Matriz T nao encontrada = identidade")

        # Compute the inverse matrix
        rt_inv = np.linalg.inv(rt)

        rotation_inv = rt_inv[:3, :3]
        translation_inv = rt_inv[:3, 3]

        pointcloud_cam = (rotation_inv @ pointcloud_cam_W.T).T + translation_inv
        pointcloud_vel = (rotation_inv @ pointcloud_vel_W.T).T + translation_inv
        pointcloud_model = (rotation_inv @ pointcloud_model_W.T).T + translation_inv

        with Image.open(self.list_rgb[j]) as img_open, \
            Image.open(self.list_depth[j]) as depth_open, \
            Image.open(self.list_depthV[j]) as depth_openV, \
            Image.open(self.list_depthM[j]) as depth_openM, \
            Image.open(self.list_mask[j]) as mask_open:

            width, height = img_open.size
            resize_size = (224, 224)
            img_open = img_open.resize(resize_size)
            depth_open = depth_open.resize(resize_size)
            depth_openV = depth_openV.resize(resize_size)
            depth_openM = depth_openM.resize(resize_size)
            mask_open = mask_open.resize(resize_size)

            img = np.array(img_open)
            depth = np.array(depth_open)
            depthV = np.array(depth_openV)
            depthM = np.array(depth_openM)
            mask = np.array(mask_open)

        # Normalize depth to [0, 1] range
        depth_normalized = depth / 65535.0
        depth_expanded = np.expand_dims(depth_normalized, axis=-1)
        
        depth_normalizedV = depthV / 65535.0
        depth_expandedV = np.expand_dims(depth_normalizedV, axis=-1)

        depth_normalizedM = depthM / 65535.0
        depth_expandedM = np.expand_dims(depth_normalizedM, axis=-1)

        mask = np.expand_dims(mask, axis=-1)

        img4c = np.zeros((4, img.shape[1], img.shape[2]), dtype=np.float32)

        if self.concatmethod == "depth":
            concat = np.concatenate((img/256, depth_expanded), axis=-1)
        elif self.concatmethod == "velodyne":
            concat = np.concatenate((img/256, depth_expandedV), axis=-1)
        elif self.concatmethod == "model":
            concat = np.concatenate((img/256, depth_expandedM), axis=-1)
        img4c = np.transpose(concat, (2, 0, 1))
        # print(img4c.shape)

        img_masked = img4c
        img_masked = torch.from_numpy(img_masked.astype(np.float32))

        if self.maskedmethod == "depth":
            depth_expanded = torch.from_numpy(depth_expanded.astype(np.float32))
        elif self.maskedmethod == "velodyne":
            depth_expanded = torch.from_numpy(depth_expandedV.astype(np.float32))
        elif self.maskedmethod == "model":
            depth_expanded = torch.from_numpy(depth_expandedM.astype(np.float32))

        # Ensure we don't sample more points than are available
        array = random.choices(range(0, pointcloud_cam.shape[0]), k=self.num_points)
        pointcloud_cam = pointcloud_cam[array,:]
        pointcloud_cam_W = pointcloud_cam_W[array,:]

        array2 = random.choices(range(0, pointcloud_vel.shape[0]), k=self.num_points)
        pointcloud_vel = pointcloud_vel[array2,:]
        pointcloud_vel_W = pointcloud_vel_W[array2,:]

        array3 = random.choices(range(0, pointcloud_model.shape[0]), k=self.num_points)
        pointcloud_model = pointcloud_model[array3,:]
        pointcloud_model_W = pointcloud_model_W[array3,:]

        modelPoints = np.array([[0., 0., 0.],
                                [1., 0., 0.],
                                [0., 1., 0.],
                                [0., 0., 1.]],dtype=np.float32)

        rotation = rt[:3, :3]
        translation = rt[:3, 3]

        modelPoints_W = (rotation @ modelPoints.T).T + translation

        return  torch.from_numpy(pointcloud_cam_W.astype(np.float32)),\
                torch.from_numpy(pointcloud_cam.astype(np.float32)),\
                torch.from_numpy(pointcloud_vel_W.astype(np.float32)),\
                torch.from_numpy(pointcloud_vel.astype(np.float32)),\
                torch.from_numpy(pointcloud_model_W.astype(np.float32)),\
                torch.from_numpy(pointcloud_model.astype(np.float32)),\
                img_masked, depth_expanded, modelPoints, modelPoints_W, rt, idx, det_file_name

    def __len__(self):
        return self.length

    def get_num_points_mesh(self):
        if self.refine:
            return self.num_pt_mesh_large
        else:
            return self.num_pt_mesh_small