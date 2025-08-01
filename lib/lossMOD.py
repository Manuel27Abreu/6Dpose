from torch.nn.modules.loss import _Loss
from torch.autograd import Variable
import torch
import time
import numpy as np
import torch.nn as nn
import random
import torch.backends.cudnn as cudnn


def knn(x, y, k=1):
    _, dim, x_size = x.shape
    _, _, y_size = y.shape

    x = x.detach().squeeze().transpose(0, 1)
    y = y.detach().squeeze().transpose(0, 1)

    xx = (x**2).sum(dim=1, keepdim=True).expand(x_size, y_size)
    yy = (y**2).sum(dim=1, keepdim=True).expand(y_size, x_size).transpose(0, 1)

    dist_mat = xx + yy - 2 * x.matmul(y.transpose(0, 1))
    if k == 1:
        return dist_mat.argmin(dim=0)
    mink_idxs = dist_mat.argsort(dim=0)
    return mink_idxs[: k]

def loss_calculation(pred_r, pred_t, pred_c, target, model_points, idx, points, w, refine, num_point_mesh):
    #print(pred_r.size())
    bs=1;

    num_p=1;

    a=torch.norm(pred_r, dim=0)
    if a>0.001:
        pred_r = pred_r / a
    else:
        pred_r[3]=1

    
    base = torch.cat(((1.0 - 2.0*(pred_r[ 2]**2 + pred_r[ 3]**2)).view(bs, num_p, 1),\
                      (2.0*pred_r[ 1]*pred_r[ 2] - 2.0*pred_r[ 0]*pred_r[ 3]).view(bs, num_p, 1), \
                      (2.0*pred_r[ 0]*pred_r[ 2] + 2.0*pred_r[ 1]*pred_r[ 3]).view(bs, num_p, 1), \
                      (2.0*pred_r[ 1]*pred_r[ 2] + 2.0*pred_r[ 3]*pred_r[ 0]).view(bs, num_p, 1), \
                      (1.0 - 2.0*(pred_r[ 1]**2 + pred_r[ 3]**2)).view(bs, num_p, 1), \
                      (-2.0*pred_r[ 0]*pred_r[ 1] + 2.0*pred_r[ 2]*pred_r[ 3]).view(bs, num_p, 1), \
                      (-2.0*pred_r[ 0]*pred_r[ 2] + 2.0*pred_r[ 1]*pred_r[ 3]).view(bs, num_p, 1), \
                      (2.0*pred_r[ 0]*pred_r[ 1] + 2.0*pred_r[ 2]*pred_r[ 3]).view(bs, num_p, 1), \
                      (1.0 - 2.0*(pred_r[ 1]**2 + pred_r[ 2]**2)).view(bs, num_p, 1)), dim=2).contiguous().view(bs * num_p, 3, 3)

    ori_base = base
    base = base.contiguous().transpose(2, 1).contiguous()
    #print(model_points.shape)

    #del_points = model_points.view(bs, 1, num_point_mesh, 3).repeat(1, num_p, 1, 1).view(bs * num_p, num_point_mesh, 3)
    #ori_target = target
    #ori_t = pred_t

    averageposition = torch.mean(points, dim=1)
    #print(points.shape,averageposition.shape)
    pred = torch.add(torch.bmm(model_points, base), averageposition+pred_t)#torch.bmm(model_points, base)+ ( points+pred_t)

    dis = torch.norm((pred - target), dim=2)
    #print(dis.shape)
    loss = torch.mean(dis, dim=-1)
    #print(loss.shape)

    R_inv = np.linalg.inv(base.cpu().detach().numpy())
    points = torch.add(torch.bmm(points, torch.tensor(R_inv).cuda()), points - pred_t)

    if torch.isnan(loss.cpu()):
        print(pred_r,pred_t)
        quit()

    #-0.01*torch.log(pred_c)
    return loss, loss, pred, points

def quaternion_geodesic_loss(q1, q2):
    """
    Computes the geodesic distance between two quaternions as the loss.
    
    :param q1: Predicted quaternion tensor of shape (batch_size, 4)
    :param q2: Ground truth quaternion tensor of shape (batch_size, 4)
    :return: Geodesic distance loss.
    """
    # Normalize both quaternions to ensure they are unit quaternions
    q1 = F.normalize(q1, p=2, dim=-1)
    q2 = F.normalize(q2, p=2, dim=-1)

    # Compute dot product between quaternions
    dot_product = torch.abs(torch.sum(q1 * q2, dim=-1))
    
    # Geodesic distance (in radians)
    loss = 2 * torch.acos(torch.clamp(dot_product, -1.0, 1.0))  # clamp for numerical stability
    
    return torch.mean(loss)  # Return mean loss for the batch

def loss_calculationv2(pred_r, pred_t, pred_c, target, model_points, idx, points, w, refine, num_point_mesh):
    bs=1;

    num_p=1;

    a = torch.norm(pred_r, dim=0)
    if a>0.001:
        pred_r = pred_r / a
    else:
        pred_r[3]=1

    
    base = torch.cat(((1.0 - 2.0*(pred_r[ 2]**2 + pred_r[ 3]**2)).view(bs, num_p, 1),\
                      (2.0*pred_r[ 1]*pred_r[ 2] - 2.0*pred_r[ 0]*pred_r[ 3]).view(bs, num_p, 1), \
                      (2.0*pred_r[ 0]*pred_r[ 2] + 2.0*pred_r[ 1]*pred_r[ 3]).view(bs, num_p, 1), \
                      (2.0*pred_r[ 1]*pred_r[ 2] + 2.0*pred_r[ 3]*pred_r[ 0]).view(bs, num_p, 1), \
                      (1.0 - 2.0*(pred_r[ 1]**2 + pred_r[ 3]**2)).view(bs, num_p, 1), \
                      (-2.0*pred_r[ 0]*pred_r[ 1] + 2.0*pred_r[ 2]*pred_r[ 3]).view(bs, num_p, 1), \
                      (-2.0*pred_r[ 0]*pred_r[ 2] + 2.0*pred_r[ 1]*pred_r[ 3]).view(bs, num_p, 1), \
                      (2.0*pred_r[ 0]*pred_r[ 1] + 2.0*pred_r[ 2]*pred_r[ 3]).view(bs, num_p, 1), \
                      (1.0 - 2.0*(pred_r[ 1]**2 + pred_r[ 2]**2)).view(bs, num_p, 1)), dim=2).contiguous().view(bs * num_p, 3, 3)

    ori_base = base
    base = base.contiguous().transpose(2, 1).contiguous()

    #print(model_points.shape)

    #del_points = model_points.view(bs, 1, num_point_mesh, 3).repeat(1, num_p, 1, 1).view(bs * num_p, num_point_mesh, 3)
    #ori_target = target
    #ori_t = pred_t


    #averageposition = torch.mean(points, dim=1)
    #print(points.shape,averageposition.shape)
    pred = torch.add(torch.bmm(model_points, base), pred_t)#torch.bmm(model_points, base)+ ( points+pred_t)

    # print("pred_points.shape", pred.shape)
    # print("target_points.shape", target.shape)

    dis = torch.norm((pred - target), dim=2)
    #print(dis.shape)
    loss = torch.mean(dis, dim=-1)
    #print(loss.shape)

    #loss2=quaternion_geodesic_loss(pred_r,qr)+


    R_inv = np.linalg.inv(base.cpu().detach().numpy())
    points = torch.add(torch.bmm(points, torch.tensor(R_inv).cuda()), points - pred_t)


    if torch.isnan(loss.cpu()):
        print(pred_r,pred_t)
        quit()

#-0.01*torch.log(pred_c)
    return loss, loss, pred, points


class Loss(_Loss):
    def __init__(self, num_points_mesh):
        super(Loss, self).__init__(True)
        self.num_pt_mesh = num_points_mesh

    def forward(self, pred_r, pred_t, pred_c, target, model_points, idx, points, w, refine):

        return loss_calculation(pred_r, pred_t, pred_c, target, model_points, idx, points, w, refine, self.num_pt_mesh)

class Lossv2(_Loss):
    def __init__(self, num_points_mesh):
        super(Lossv2, self).__init__(True)
        self.num_pt_mesh = num_points_mesh

    def forward(self, pred_r, pred_t, pred_c, target, model_points, idx, points, w, refine):

        return loss_calculationv2(pred_r, pred_t, pred_c, target, model_points, idx, points, w, refine, self.num_pt_mesh)