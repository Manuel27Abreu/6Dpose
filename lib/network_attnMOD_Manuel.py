import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
from PIL import Image
import numpy as np
import pdb
import torch.nn.functional as F
from lib.pspnet_attn import PSPNet, PSPNetH, PSPNet4C, PSPNetH2b, PSPNetH2v2, PSPNetH2v2a, PSPNetH2v2b
from einops import rearrange

import inspect
__LINE__ = inspect.currentframe()

psp_models = {
    'resnet18': lambda: PSPNet(sizes=(1, 2, 3, 6), psp_size=512, deep_features_size=256, backend='resnet18'),
    'resnet34': lambda: PSPNet(sizes=(1, 2, 3, 6), psp_size=512, deep_features_size=256, backend='resnet34'),
    'resnet50': lambda: PSPNet(sizes=(1, 2, 3, 6), psp_size=2048, deep_features_size=1024, backend='resnet50'),
    'resnet101': lambda: PSPNet(sizes=(1, 2, 3, 6), psp_size=2048, deep_features_size=1024, backend='resnet101'),
    'resnet152': lambda: PSPNet(sizes=(1, 2, 3, 6), psp_size=2048, deep_features_size=1024, backend='resnet152')
}


# psp_modelsH = {
#     'resnet18': lambda: PSPNetH(sizes=(1, 2, 3, 6), psp_size=512, deep_features_size=256, backend='resnet18'),
#     'resnet34': lambda: PSPNetH(sizes=(1, 2, 3, 6), psp_size=512, deep_features_size=256, backend='resnet34'),
#     'resnet50': lambda: PSPNetH(sizes=(1, 2, 3, 6), psp_size=2048, deep_features_size=1024, backend='resnet50'),
#     'resnet101': lambda: PSPNetH(sizes=(1, 2, 3, 6), psp_size=2048, deep_features_size=1024, backend='resnet101'),
#     'resnet152': lambda: PSPNetH(sizes=(1, 2, 3, 6), psp_size=2048, deep_features_size=1024, backend='resnet152')
# }


#considering 1 epoch -> PSPNetH2a(0.027) better than PSPNetH2 (0.04)
#  PSPNetH2b ->0.02438354686590631                    p1 = rearrange(p1, 'd0 d1 d2 d3 -> d0 (d2 d3)  d1')

#  PSPNetH2b -> 0.05          p1 = rearrange(p1, 'd0 d1 d2 d3 ->  (d2 d3) d0 d1')
# PSPNetH2v2 -> 0.0279
# PSPNetH2v2a -> 0.029
# PSPNetH2v2b -> 0.03356962547355895


psp_modelsH = {
    'resnet18': lambda: PSPNetH2v2(sizes=(1, 2, 3, 6), psp_size=512, deep_features_size=256, backend='resnet18'),
    'resnet34': lambda: PSPNetH2v2(sizes=(1, 2, 3, 6), psp_size=512, deep_features_size=256, backend='resnet34'),
    'resnet50': lambda: PSPNetH2v2(sizes=(1, 2, 3, 6), psp_size=2048, deep_features_size=1024, backend='resnet50'),
    'resnet101': lambda: PSPNetH2v2(sizes=(1, 2, 3, 6), psp_size=2048, deep_features_size=1024, backend='resnet101'),
    'resnet152': lambda: PSPNetH2v2(sizes=(1, 2, 3, 6), psp_size=2048, deep_features_size=1024, backend='resnet152')
}

psp_models4C = {
    'resnet18': lambda: PSPNet4C(sizes=(1, 2, 3, 6), psp_size=512, deep_features_size=256, backend='resnet18'),
    'resnet34': lambda: PSPNet4C(sizes=(1, 2, 3, 6), psp_size=512, deep_features_size=256, backend='resnet34'),
    'resnet50': lambda: PSPNet4C(sizes=(1, 2, 3, 6), psp_size=2048, deep_features_size=1024, backend='resnet50'),
    'resnet101': lambda: PSPNet4C(sizes=(1, 2, 3, 6), psp_size=2048, deep_features_size=1024, backend='resnet101'),
    'resnet152': lambda: PSPNet4C(sizes=(1, 2, 3, 6), psp_size=2048, deep_features_size=1024, backend='resnet152')
}


class ModifiedResnet(nn.Module):

    def __init__(self, usegpu=True):
        super(ModifiedResnet, self).__init__()

        self.model = psp_models['resnet18'.lower()]()
        self.model = nn.DataParallel(self.model)

    def forward(self, x):
        x = self.model(x)
        return x


class ModifiedResnet4C(nn.Module):

    def __init__(self, usegpu=True):
        #super(ModifiedResnet, self).__init__()
        super(ModifiedResnet4C, self).__init__()

        self.model = psp_models4C['resnet18'.lower()]()
        self.model = nn.DataParallel(self.model)

    def forward(self, x):
        x = self.model(x)
        return x


class ModifiedResnetWDepth(nn.Module):

    def __init__(self, usegpu=True):
        super(ModifiedResnetWDepth, self).__init__()

        self.model = psp_modelsH['resnet18'.lower()]()
        self.model = nn.DataParallel(self.model)

    def forward(self, x,d):
        x = self.model(x,d)
        return x



class PoseNetFeat(nn.Module):
    def __init__(self, num_points):
        super(PoseNetFeat, self).__init__()
        self.conv1 = torch.nn.Conv1d(39, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)

        self.e_conv1 = torch.nn.Conv1d(5, 64, 1,padding=122)
        self.e_conv2 = torch.nn.Conv1d(64, 128, 1)

        self.conv5 = torch.nn.Conv1d(256, 512, 1)
        self.conv6 = torch.nn.Conv1d(512, 1024, 1)

        self.ap1 = torch.nn.AvgPool1d(num_points)
        self.num_points = num_points

    def forward(self, x, emb): 

        # x -> points
        # y -> image emb
        
        x = F.relu(self.conv1(x))
        emb = F.relu(self.e_conv1(emb))
        #print(x.shape,emb.shape)
        pointfeat_1 = torch.cat((x, emb), dim=1)

        x = F.relu(self.conv2(x))
        emb = F.relu(self.e_conv2(emb))
        pointfeat_2 = torch.cat((x, emb), dim=1)

        x = F.relu(self.conv5(pointfeat_2))
        x = F.relu(self.conv6(x))


        #print(pointfeat_1.shape,pointfeat_2.shape,x.shape)


        return torch.cat([pointfeat_1, pointfeat_2, x], 1) #128 + 256 + 1024


class TransformerEncoder(nn.Module):
    def __init__(self, n_features, n_heads):
        super(TransformerEncoder, self).__init__()

        self.norm       = nn.LayerNorm(n_features)
        self.norm2       = nn.LayerNorm(n_features)
        self.attention  = nn.MultiheadAttention(embed_dim= n_features, num_heads = n_heads)
        self.attention2  = nn.MultiheadAttention(embed_dim= n_features, num_heads = n_heads)
        self.fc         = nn.Linear(in_features = n_features, out_features = n_features, bias = True)
        self.fc2         = nn.Linear(in_features = n_features, out_features = n_features, bias = True)


    def forward(self, x,y):
        x1   = torch.clone(x)
        x    = self.norm(x)
        x, _ = self.attention(x,x,x) 
        x    = x + x1 
        x1   = torch.clone(x)
        x    = self.norm(x)
        x    = F.relu(self.fc(x))



        y1   = torch.clone(y)
        y    = self.norm2(y)
        y, _ = self.attention2(y,y,y) 
        y    = y + y1 
        y1   = torch.clone(y)
        y    = self.norm2(y)
        y    = F.relu(self.fc2(y))



        return x+x1,y+y1



class TransformerEncoderW3(nn.Module):
    def __init__(self, n_features, n_heads):
        super(TransformerEncoderW3, self).__init__()

        self.norm       = nn.LayerNorm(n_features)
        self.norm2       = nn.LayerNorm(n_features)
        self.norm3       = nn.LayerNorm(n_features)
        self.attention  = nn.MultiheadAttention(embed_dim= n_features, num_heads = n_heads)
        self.attention2  = nn.MultiheadAttention(embed_dim= n_features, num_heads = n_heads)
        self.attention3  = nn.MultiheadAttention(embed_dim= n_features, num_heads = n_heads)
        self.fc         = nn.Linear(in_features = n_features, out_features = n_features, bias = True)
        self.fc2         = nn.Linear(in_features = n_features, out_features = n_features, bias = True)
        self.fc3         = nn.Linear(in_features = n_features, out_features = n_features, bias = True)


    def forward(self, x,y,z):
        x1   = torch.clone(x)
        x    = self.norm(x)
        x, _ = self.attention(x,x,x) 
        x    = x + x1 
        x1   = torch.clone(x)
        x    = self.norm(x)
        x    = F.relu(self.fc(x))



        y1   = torch.clone(y)
        y    = self.norm2(y)
        y, _ = self.attention2(y,y,y) 
        y    = y + y1 
        y1   = torch.clone(y)
        y    = self.norm2(y)
        y    = F.relu(self.fc2(y))

        z1   = torch.clone(z)
        z    = self.norm3(z)
        z, _ = self.attention3(z,z,z) 
        z    = z + z1 
        z1   = torch.clone(z)
        z    = self.norm3(z)
        z    = F.relu(self.fc3(z))

        return x+x1,y+y1,z+z1


class StdDevPool3d(nn.Module):
    def __init__(self, kernel_size, stride=None, padding=(0,0,0)):
        super(StdDevPool3d, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride or kernel_size
        self.padding = padding

    def forward(self, x):
        b, d, h, w = x.size()
        kd, kh, kw = self.kernel_size
        sd, sh, sw = self.stride
        pd, ph, pw = self.padding

        # Pad input if needed
        if pd > 0 or ph > 0 or pw > 0:
            x = F.pad(x, (pw, pw, ph, ph, pd, pd), mode='constant', value=0)

        # Calculate local mean
        mean = F.avg_pool3d(x, kernel_size=self.kernel_size, stride=self.stride, padding=0)

        # Calculate local variance
        variance = F.avg_pool3d(x**2, kernel_size=self.kernel_size, stride=self.stride, padding=0) - mean**2

        # Standard deviation
        std = torch.sqrt(variance)

        return std


class KurtosisPool3d(nn.Module):
    def __init__(self, kernel_size, stride=None, padding=(0,0,0)):
        super(KurtosisPool3d, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride or kernel_size
        self.padding = padding

    def forward(self, x):
        b, d, h, w = x.size()
        kd, kh, kw = self.kernel_size
        sd, sh, sw = self.stride
        pd, ph, pw = self.padding

        # Pad input if needed
        if pd > 0 or ph > 0 or pw > 0:
            x = F.pad(x, (pw, pw, ph, ph, pd, pd), mode='constant', value=0)

        # Calculate local mean
        mean = F.avg_pool3d(x, kernel_size=self.kernel_size, stride=self.stride, padding=0)

        # Calculate local fourth central moment (kurtosis)
        moment_4 = F.avg_pool3d(x ** 4, kernel_size=self.kernel_size, stride=self.stride, padding=0)
        variance = F.avg_pool3d(x**2, kernel_size=self.kernel_size, stride=self.stride, padding=0) - mean**2
        kurtosis = moment_4 / (0.0001+variance ** 2) - 3

        return kurtosis

class SkewnessPool3d(nn.Module):
    def __init__(self, kernel_size, stride=None, padding=(0,0,0)):
        super(SkewnessPool3d, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride or kernel_size
        self.padding = padding

    def forward(self, x):
        b, d, h, w = x.size()
        kd, kh, kw = self.kernel_size
        sd, sh, sw = self.stride
        pd, ph, pw = self.padding

        # Pad input if needed
        if pd > 0 or ph > 0 or pw > 0:
            x = F.pad(x, (pw, pw, ph, ph, pd, pd), mode='constant', value=0)

        # Calculate local mean
        mean = F.avg_pool3d(x, kernel_size=self.kernel_size, stride=self.stride, padding=0)

        # Calculate local third central moment (skewness)
        moment_3 = F.avg_pool3d(x ** 3, kernel_size=self.kernel_size, stride=self.stride, padding=0)
        variance = F.avg_pool3d(x**2, kernel_size=self.kernel_size, stride=self.stride, padding=0) - mean**2
        skewness = moment_3 / (variance ** (3/2))

        return skewness



class FeatureSelector(nn.Module):
    def __init__(self, num_channels, kernel_size=1, stride=1):
        super(FeatureSelector, self).__init__()
        # Learnable weights for each channel
        self.weights = nn.Parameter(torch.randn(num_channels))
        self.kernel_size = kernel_size
        self.stride = stride
        self.pool = nn.MaxPool3d(kernel_size=kernel_size, stride=stride)

    def forward(self, x):
        # Normalize weights
        normalized_weights = F.softmax(self.weights, dim=0)
        
        # Scale feature maps
        scaled_feature_maps = x * normalized_weights.view(1, -1, 1, 1)
        
        # Optional: Apply pooling to scaled feature maps
        pooled_feature_maps = self.pool(scaled_feature_maps)
        
        return pooled_feature_maps


class FakeFeatureSelector(nn.Module):
    def __init__(self, num_channels, kernel_size=1, stride=1):
        super(FeatureSelector, self).__init__()
        # Learnable weights for each channel
        self.weights = nn.Parameter(torch.randn(num_channels))
        self.kernel_size = kernel_size
        self.stride = stride

    def forward(self, x):
        # Normalize weights
        normalized_weights = F.softmax(self.weights, dim=0)
        
        # Scale feature maps
        scaled_feature_maps = x * normalized_weights.view(1, -1, 1, 1)
                
        return scaled_feature_maps

class FakeFeatureSelectorFC(nn.Module):
    def __init__(self, num_channels, kernel_size=1, stride=1):
        super(FakeFeatureSelectorFC, self).__init__()
        # Learnable weights for each channel
        self.weights = nn.Parameter(torch.randn(num_channels))
        self.kernel_size = kernel_size
        self.stride = stride

    def forward(self, x):
        # Normalize weights
        normalized_weights = F.softmax(self.weights, dim=0)
        
        # Scale feature maps
        scaled_feature_maps = x * normalized_weights.view(1, -1)
                
        return scaled_feature_maps


class LearnableAvgPool3d(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size=1, stride=1):
        super(LearnableAvgPool3d, self).__init__()
        self.conv = nn.Conv3d(1, 1, kernel_size=(1,1,1))
        self.kernel_size=kernel_size
        self.stride=stride

    def forward(self, x):
        # Apply 1x1x1 convolution to input tensor
        weights = self.conv(x.unsqueeze(1))
        ##print(weights.shape)
        # Apply softmax to normalize the weights across channels
        weights = F.softmax(weights, dim=1)
        ##print(weights.shape)
        weights=weights.squeeze(1)
        # Element-wise multiplication of input tensor with normalized weights
        weighted_input = x * weights
        # Sum along the channel dimension to obtain the weighted average
        output = F.avg_pool3d(weighted_input, kernel_size=self.kernel_size, stride=self.stride, padding=0)
        ##print(output.shape)

        return output

 
class MedianPool3d(nn.Module):
    def __init__(self, kernel_size, stride=None, padding=(0,0,0)):
        super(MedianPool3d, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride or kernel_size
        self.padding = padding

    def forward(self, x):
        b, d, h, w = x.size()
        kd, kh, kw = self.kernel_size
        sd, sh, sw = self.stride
        pd, ph, pw = self.padding

        # Pad input if needed
        if pd > 0 or ph > 0 or pw > 0:
            x = F.pad(x, (pw, pw, ph, ph, pd, pd), mode='constant', value=0)

        # Initialize output tensor
        out_d = (d + 2 * pd - kd) // sd + 1
        out_h = (h + 2 * ph - kh) // sh + 1
        out_w = (w + 2 * pw - kw) // sw + 1
        output = torch.zeros(b, out_d, out_h, out_w, device=x.device)

        # Perform median pooling
        for i in range(out_d):
            for j in range(out_h):
                for k in range(out_w):
                    # Select the current pooling region
                    region = x[:, i * sd:i * sd + kd, j * sh:j * sh + kh, k * sw:k * sw + kw]

                    # Reshape and calculate median
                    median_value = torch.median(region.view(b, -1), dim=-1)[0]

                    # Set median value in output tensor
                    output[:, i, j, k] = median_value

        return output



class AvgStdDevSamplePool3d(nn.Module):
    def __init__(self, kernel_size, stride=None, padding=(0,0,0), num_samples=1):
        super(AvgStdDevSamplePool3d, self).__init__()
        self.avg_pool = nn.AvgPool3d(kernel_size, stride=stride, padding=padding)
        self.std_dev_pool = StdDevPool3d(kernel_size, stride=stride, padding=padding)
        self.num_samples = num_samples

    def forward(self, x):
        avg_output = self.avg_pool(x)
        std_dev_output = self.std_dev_pool(x)
        
        b, d, h, w = avg_output.size()


        ##print(avg_output.shape,std_dev_output.shape)
        
        # Reshape std_dev_output to match the shape of avg_output
        std_dev_output = std_dev_output.view(b, -1)
        avg_output = avg_output.view(b, -1)
        ##print(avg_output.shape,std_dev_output.shape)



        # Sample from Gaussian distribution parameterized by mean and standard deviation
        samples = torch.normal(avg_output, std_dev_output)

        ##print(samples.shape)
        # Take the average of the samples
        #combined_output = torch.mean(samples, dim=2)

        return samples

from functools import partial
from einops.layers.torch import Rearrange, Reduce


class FeatureMixingModule(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(FeatureMixingModule, self).__init__()
        self.depthwise_conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels)
        self.pointwise_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        # Depthwise convolution
        out = self.depthwise_conv(x)
        # Pointwise convolution
        out = self.pointwise_conv(out)
        return out



from torchvision.models import resnet34,resnet18
from torch.nn import MultiheadAttention





class RGBDEncoder(nn.Module):
    def __init__(self):
        super(RGBDEncoder, self).__init__()
        self.rgb_backbone = resnet34(pretrained=True)
        self.depth_backbone = resnet34(pretrained=True)
        
        # Remove the fully connected layers
        self.rgb_backbone = nn.Sequential(*list(self.rgb_backbone.children())[:-2])
        self.depth_backbone = nn.Sequential(*list(self.depth_backbone.children())[:-2])
        self.depth_backbone[0] = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    def forward(self, rgb, depth):
        #print(rgb.shape,depth.shape)
        rgb_features = self.rgb_backbone(rgb)
        depth_features = self.depth_backbone(depth)
        
        #print(rgb_features.shape,depth_features.shape)
                
        return rgb_features,depth_features

class RGBDEncoder2(nn.Module):
    def __init__(self):
        super(RGBDEncoder2, self).__init__()
        self.rgb_backbone = resnet34(pretrained=True)
        self.depth_backbone = resnet34(pretrained=True)
        self.depth_backbone2 = resnet34(pretrained=True)

        # Remove the fully connected layers
        self.rgb_backbone = nn.Sequential(*list(self.rgb_backbone.children())[:-2])
        self.depth_backbone = nn.Sequential(*list(self.depth_backbone.children())[:-2])
        self.depth_backbone[0] = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

        self.depth_backbone2 = nn.Sequential(*list(self.depth_backbone2.children())[:-2])
        self.depth_backbone2[0] = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    def forward(self, rgb, depth,depth2):
        rgb_features = self.rgb_backbone(rgb)
        depth_features = self.depth_backbone(depth)
        depth_features2 = self.depth_backbone2(depth2)
                
        return rgb_features,depth_features,depth_features2






class RGBDEncoder2CUSTOM(nn.Module):
    def __init__(self, modelRGB,modelDepth, modelDepth2):
        super(RGBDEncoder2CUSTOM, self).__init__()
        self.rgb_backbone = modelRGB
        self.depth_backbone = modelDepth
        self.depth_backbone2 = modelDepth2

    def forward(self, rgb, depth,depth2):
        rgb_features = self.rgb_backbone(rgb)
        depth_features = self.depth_backbone(depth)
        depth_features2 = self.depth_backbone2(depth2)
        # print("RGBDEncoder2", rgb_features.shape,depth_features.shape,depth_features2.shape)
                
        return rgb_features,depth_features,depth_features2





class RGBDEncoder2RESNET18(nn.Module):
    def __init__(self):
        super(RGBDEncoder2RESNET18, self).__init__()
        self.rgb_backbone = resnet18(pretrained=True)
        self.depth_backbone = resnet18(pretrained=True)
        self.depth_backbone2 = resnet18(pretrained=True)

        # Remove the fully connected layers
        self.rgb_backbone = nn.Sequential(*list(self.rgb_backbone.children())[:-2])
        self.depth_backbone = nn.Sequential(*list(self.depth_backbone.children())[:-2])
        self.depth_backbone[0] = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

        self.depth_backbone2 = nn.Sequential(*list(self.depth_backbone2.children())[:-2])
        self.depth_backbone2[0] = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    def forward(self, rgb, depth,depth2):
        rgb_features = self.rgb_backbone(rgb)
        depth_features = self.depth_backbone(depth)
        depth_features2 = self.depth_backbone2(depth2)
                
        return rgb_features,depth_features,depth_features2


class RGBEncoder(nn.Module):
    def __init__(self):
        super(RGBEncoder, self).__init__()
        self.rgb_backbone = resnet34(pretrained=True)

        # Remove the fully connected layers
        self.rgb_backbone = nn.Sequential(*list(self.rgb_backbone.children())[:-2])

    def forward(self, rgb):
        rgb_features = self.rgb_backbone(rgb)
      
                
        return rgb_features


class MultiHeadAttentionFusion(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(MultiHeadAttentionFusion, self).__init__()
        self.attentionRGB = MultiheadAttention(embed_dim, num_heads)
        self.attentionDepth = MultiheadAttention(embed_dim, num_heads)

    def forward(self, rgb_features, depth_features):
        # Flatten the spatial dimensions for multi-head attention
        rgb_flat = rgb_features.flatten(2).permute(2, 0, 1)
        depth_flat = depth_features.flatten(2).permute(2, 0, 1)
        fused_features, _ = self.attentionRGB(rgb_flat, rgb_flat, rgb_flat)
        fused_features2, _ = self.attentionDepth(depth_flat, depth_flat, depth_flat)

        #print(fused_features.shape,fused_features2.shape)
        return  torch.cat([fused_features, fused_features2], 0)


class MultiHeadAttentionFusionPlaceOlder(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(MultiHeadAttentionFusionPlaceOlder, self).__init__()
 

    def forward(self, rgb_features):
        # Flatten the spatial dimensions for multi-head attention
        rgb_flat = rgb_features.flatten(2).permute(2, 0, 1)


        return  rgb_flat


class MultiHeadAttentionFusion2(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(MultiHeadAttentionFusion2, self).__init__()
        self.attention = MultiheadAttention(embed_dim, num_heads)
        self.attention1 = MultiheadAttention(embed_dim, num_heads)
        self.attention2 = MultiheadAttention(embed_dim, num_heads)

    def forward(self, rgb_features, depth_features,depth2_features):
        # Flatten the spatial dimensions for multi-head attention
        #print(rgb_features.shape)
        rgb_flat = rgb_features.flatten(2).permute(2, 0, 1)
        depth_flat = depth_features.flatten(2).permute(2, 0, 1)
        depth2_flat = depth2_features.flatten(2).permute(2, 0, 1)
        fused_features1, _ = self.attention(rgb_flat, rgb_flat, rgb_flat)
        fused_features2, _ = self.attention1(depth_flat, depth_flat, depth_flat)
        fused_features3, _ = self.attention2(depth2_flat, depth2_flat, depth2_flat)

        return torch.cat([fused_features1,fused_features2, fused_features3], 0)


class RGBDFeatureExtractor(nn.Module):
    def __init__(self, num_heads=4):
        super(RGBDFeatureExtractor, self).__init__()
        self.encoder = RGBDEncoder()
        self.attention_fusion = MultiHeadAttentionFusion(embed_dim=512, num_heads=num_heads)

    def forward(self, rgb, depth):
        # Encode RGB and Depth images
        rgb_features,depth_features = self.encoder(rgb, depth)
        fused_features = self.attention_fusion(rgb_features, depth_features)

        return fused_features

class RGBFeatureExtractor(nn.Module):
    def __init__(self, num_heads=4):
        super(RGBFeatureExtractor, self).__init__()
        self.encoder = RGBEncoder()
        self.attention_fusion = MultiHeadAttentionFusionPlaceOlder(embed_dim=512, num_heads=num_heads)

    def forward(self, rgb):
        # Encode RGB and Depth images
        rgb_features= self.encoder(rgb)



       # ap_x=torch.cat([rgb_features, depth_features, depth2_features], 2)


        fused_features = self.attention_fusion(rgb_features)

        return fused_features

class RGBDFeatureExtractor2(nn.Module):
    def __init__(self, num_heads=4):
        super(RGBDFeatureExtractor2, self).__init__()
        self.encoder = RGBDEncoder2()
        self.attention_fusion = MultiHeadAttentionFusion2(embed_dim=512, num_heads=num_heads)

    def forward(self, rgb, depth, depth2):
        # Encode RGB and Depth images
        rgb_features,depth_features,depth2_features = self.encoder(rgb, depth,depth2)



       # ap_x=torch.cat([rgb_features, depth_features, depth2_features], 2)


        fused_features = self.attention_fusion(rgb_features, depth_features, depth2_features)

        return fused_features


class RGBDFeatureExtractor2CUSTOM(nn.Module):
    def __init__(self, modelRGB,modelDepth, modelDepth2, num_heads=4,emb=512):
        super(RGBDFeatureExtractor2CUSTOM, self).__init__()
        self.encoder = RGBDEncoder2CUSTOM(modelRGB,modelDepth, modelDepth2)
        self.attention_fusion = MultiHeadAttentionFusion2(embed_dim=emb, num_heads=num_heads)

    def forward(self, rgb, depth, depth2):
        # Encode RGB and Depth images
        rgb_features,depth_features,depth2_features = self.encoder(rgb, depth,depth2)



       # ap_x=torch.cat([rgb_features, depth_features, depth2_features], 2)


        fused_features = self.attention_fusion(rgb_features, depth_features, depth2_features)

        return fused_features


class RGBDFeatureExtractor2RESNET18(nn.Module):
    def __init__(self, num_heads=4):
        super(RGBDFeatureExtractor2RESNET18, self).__init__()
        self.encoder = RGBDEncoder2RESNET18()
        self.attention_fusion = MultiHeadAttentionFusion2(embed_dim=512, num_heads=num_heads)

    def forward(self, rgb, depth, depth2):
        # Encode RGB and Depth images
        rgb_features,depth_features,depth2_features = self.encoder(rgb, depth,depth2)



       # ap_x=torch.cat([rgb_features, depth_features, depth2_features], 2)


        fused_features = self.attention_fusion(rgb_features, depth_features, depth2_features)

        return fused_features


class StdDevPooling1D(nn.Module):
    def __init__(self, kernel_size, stride=None, padding=0):
        super(StdDevPooling1D, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size
        self.padding = padding

    def forward(self, x):
        x_unfold = F.unfold(x.unsqueeze(-1), (self.kernel_size, 1), stride=(self.stride, 1), padding=(self.padding, 0))
        x_unfold = x_unfold.view(x.size(0), x.size(1), self.kernel_size, -1)
        std_dev = torch.std(x_unfold, dim=2)
        return std_dev

class KurtosisPooling1D(nn.Module):
    def __init__(self, kernel_size, stride=None, padding=0):
        super(KurtosisPooling1D, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size
        self.padding = padding

    def forward(self, x):
        x_unfold = F.unfold(x.unsqueeze(-1), (self.kernel_size, 1), stride=(self.stride, 1), padding=(self.padding, 0))
        x_unfold = x_unfold.view(x.size(0), x.size(1), self.kernel_size, -1)
        mean = x_unfold.mean(dim=2, keepdim=True)
        variance = ((x_unfold - mean) ** 2).mean(dim=2, keepdim=True)
        fourth_moment = ((x_unfold - mean) ** 4).mean(dim=2)
        kurtosis = fourth_moment / (variance ** 2) - 3
        return kurtosis

class SkewnessPooling1D(nn.Module):
    def __init__(self, kernel_size, stride=None, padding=0):
        super(SkewnessPooling1D, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size
        self.padding = padding

    def forward(self, x):
        x_unfold = F.unfold(x.unsqueeze(-1), (self.kernel_size, 1), stride=(self.stride, 1), padding=(self.padding, 0))
        x_unfold = x_unfold.view(x.size(0), x.size(1), self.kernel_size, -1)
        mean = x_unfold.mean(dim=2, keepdim=True)
        variance = ((x_unfold - mean) ** 2).mean(dim=2, keepdim=True)
        skewness = (((x_unfold - mean) ** 3).mean(dim=2)) / (variance ** 1.5)
        return skewness

class MedianPooling1D(nn.Module):
    def __init__(self, kernel_size, stride=None, padding=0):
        super(MedianPooling1D, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size
        self.padding = padding

    def forward(self, x):
        x_unfold = F.unfold(x.unsqueeze(-1), (self.kernel_size, 1), stride=(self.stride, 1), padding=(self.padding, 0))
        x_unfold = x_unfold.view(x.size(0), x.size(1), self.kernel_size, -1)
        median = torch.median(x_unfold, dim=2)[0]
        return median



def weighted_average_quaternions(quaternions, weights):
    """
    Average multiple quaternions with specific weights

    :params quaternions: is a Nx4 numpy matrix and contains the quaternions
        to average in the rows.
        The quaternions are arranged as (w,x,y,z), with w being the scalar

    :params weights: The weight vector w must be of the same length as
        the number of rows in the

    :returns: the average quaternion of the input. Note that the signs
        of the output quaternion can be reversed, since q and -q
        describe the same orientation
    :raises: ValueError if all weights are zero
    """
    # Number of quaternions to average
    samples = quaternions.shape[0]
    mat_a = np.zeros(shape=(4, 4), dtype=np.float64)
    weight_sum = 0

    for i in range(0, samples):
        quat = quaternions[i, :]
        mat_a = weights[i] * np.outer(quat, quat) + mat_a
        weight_sum += weights[i]

    if weight_sum <= 0.0:
        raise ValueError("At least one weight must be greater than zero")

    # scale
    mat_a = (1.0/weight_sum) * mat_a

    # compute eigenvalues and -vectors
    eigen_values, eigen_vectors = np.linalg.eig(mat_a)

    # Sort by largest eigenvalue
    eigen_vectors = eigen_vectors[:, eigen_values.argsort()[::-1]]

    # return the real part of the largest eigenvector (has only real part)
    return np.real(np.ravel(eigen_vectors[:, 0]))





class Conv1DNetwork(nn.Module):
    def __init__(self, in_channels_):
        super(Conv1DNetwork, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=in_channels_, out_channels=16, kernel_size=3,  padding=1)
        self.conv1b = nn.Conv1d(in_channels=in_channels_, out_channels=16, kernel_size=3, padding=1)
        self.conv1a = nn.Conv1d(in_channels=16, out_channels=16, kernel_size=3, padding=1)
        self.bn1 = nn.InstanceNorm1d(16)
        self.bn1a = nn.InstanceNorm1d(16)

        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3,  padding=1)
        self.conv2b = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3,  padding=1)
        self.conv2a = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=3, padding=1)
        self.bn2 = nn.InstanceNorm1d(32)
        self.bn2a = nn.InstanceNorm1d(32)

        self.conv3 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3,  padding=1)
        self.conv3b = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.conv3a = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.bn3 = nn.InstanceNorm1d(64)
        self.bn3a = nn.InstanceNorm1d(64)


        self.conv4 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.conv4b = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.conv4a = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=3, padding=1)
        self.bn4 = nn.InstanceNorm1d(128)
        self.bn4a = nn.InstanceNorm1d(128)


        self.conv5 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3,  padding=1)
        self.conv5b = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.conv5a = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3, padding=1)
        self.bn5 = nn.InstanceNorm1d(256)
        self.bn5a = nn.InstanceNorm1d(256)


        self.conv6 = nn.Conv1d(in_channels=256, out_channels=512, kernel_size=3,  padding=1)
        self.conv6b = nn.Conv1d(in_channels=256, out_channels=512, kernel_size=3,  padding=1)
        self.conv6a = nn.Conv1d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.bn6 = nn.InstanceNorm1d(512)
        self.bn6a = nn.InstanceNorm1d(512)
        
        self.pool = nn.AvgPool1d(kernel_size=2, stride=2)
        
#https://d2l.ai/chapter_convolutional-modern/resnet.html
    def forward(self, x):
        x1 = self.conv1b(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.conv1a(x)
        x = self.bn1a(x)
        x=x+x1;
        x = F.relu(x)
        x = self.pool(x)


        x1 = self.conv2b(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.conv2a(x)
        x = self.bn2a(x)
        x=x+x1;
        x = F.relu(x)
        x = self.pool(x)

        x1 = self.conv3b(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.conv3a(x)
        x = self.bn3a(x)
        x=x+x1;
        x = F.relu(x)
        x = self.pool(x)

        x1 = self.conv4b(x)
        x = self.conv4(x)
        x = self.bn4(x)
        x = F.relu(x)
        x = self.conv4a(x)
        x = self.bn4a(x)
        x=x+x1;
        x = F.relu(x)
        x = self.pool(x)

        x1 = self.conv5b(x)
        x = self.conv5(x)
        x = self.bn5(x)
        x = F.relu(x)
        x = self.conv5a(x)
        x = self.bn5a(x)
        x=x+x1;
        x = F.relu(x)
        x = self.pool(x)

        x1 = self.conv6b(x)
        x = self.conv6(x)
        x = self.bn6(x)
        x = F.relu(x)
        x = self.conv6a(x)
        x = self.bn6a(x)
        x=x+x1;
        x = F.relu(x)
        x = self.pool(x)
        

        x = x.permute(2, 0, 1)  # Shape: [sequence_length, batch_size, embedding_dim]
        
        return x




class Conv1DNetwork_old(nn.Module):
    def __init__(self, in_channels_):
        super(Conv1DNetwork_old, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=in_channels_, out_channels=16, kernel_size=3,  padding=1)
        self.conv1b = nn.Conv1d(in_channels=in_channels_, out_channels=16, kernel_size=3, padding=1)
        self.conv1a = nn.Conv1d(in_channels=16, out_channels=16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(16)
        self.bn1a = nn.BatchNorm1d(16)

        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3,  padding=1)
        self.conv2b = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3,  padding=1)
        self.conv2a = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(32)
        self.bn2a = nn.BatchNorm1d(32)

        self.conv3 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3,  padding=1)
        self.conv3b = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.conv3a = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(64)
        self.bn3a = nn.BatchNorm1d(64)


        self.conv4 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.conv4b = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.conv4a = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm1d(128)
        self.bn4a = nn.BatchNorm1d(128)


        self.conv5 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3,  padding=1)
        self.conv5b = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.conv5a = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm1d(256)
        self.bn5a = nn.BatchNorm1d(256)


        self.conv6 = nn.Conv1d(in_channels=256, out_channels=512, kernel_size=3,  padding=1)
        self.conv6b = nn.Conv1d(in_channels=256, out_channels=512, kernel_size=3,  padding=1)
        self.conv6a = nn.Conv1d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.bn6 = nn.BatchNorm1d(512)
        self.bn6a = nn.BatchNorm1d(512)
        
        self.pool = nn.AvgPool1d(kernel_size=2, stride=2)
        
#https://d2l.ai/chapter_convolutional-modern/resnet.html
    def forward(self, x):
        x = self.conv1(x)

        x1=x.clone()
        x = self.bn1(x)
        x = F.relu(x)
        x = self.conv1a(x)
        x = self.bn1a(x)
        x=x+x1;
        x = F.relu(x)
        x = self.pool(x)


#        x1 = self.conv2b(x)
        x = self.conv2(x)
        x1=x.clone()
        x = self.bn2(x)
        x = F.relu(x)
        x = self.conv2a(x)
        x = self.bn2a(x)
        x=x+x1;
        x = F.relu(x)
        x = self.pool(x)

#        x1 = self.conv3b(x)
        x = self.conv3(x)
        x1=x.clone()
        x = self.bn3(x)
        x = F.relu(x)
        x = self.conv3a(x)
        x = self.bn3a(x)
        x=x+x1;
        x = F.relu(x)
        x = self.pool(x)

#        x1 = self.conv4b(x)
        x = self.conv4(x)
        x1=x.clone()
        x = self.bn4(x)
        x = F.relu(x)
        x = self.conv4a(x)
        x = self.bn4a(x)
        x=x+x1;
        x = F.relu(x)
        x = self.pool(x)

#        x1 = self.conv5b(x)
        x = self.conv5(x)
        x1=x.clone()
        x = self.bn5(x)
        x = F.relu(x)
        x = self.conv5a(x)
        x = self.bn5a(x)
        x=x+x1;
        x = F.relu(x)
        x = self.pool(x)

#        x1 = self.conv6b(x)
        x = self.conv6(x)
        x1=x.clone()
        x = self.bn6(x)
        x = F.relu(x)
        x = self.conv6a(x)
        x = self.bn6a(x)
        x=x+x1;
        x = F.relu(x)
        x = self.pool(x)
        

        x = x.permute(2, 0, 1)  # Shape: [sequence_length, batch_size, embedding_dim]
        
        return x

 



class PoseNet(nn.Module):
    def __init__(self, num_points, num_obj):
        super(PoseNet, self).__init__()
        self.num_points = num_points
        self.cnn = RGBDFeatureExtractor(num_heads=4)
        
        self.compressr2=torch.nn.Linear(8192, num_obj*4)
        self.compresst2=torch.nn.Linear(8192, num_obj*3)
        self.num_obj = num_obj
        self.attn = TransformerEncoder(512, 4); 
        self.net2=Conv1DNetwork(4);
        self.net1=Conv1DNetwork(3);
        self.netFinal=Conv1DNetwork(7);


    def embed_fn(self, x, L_embed=6):
        rets = []
        rets.append(x[0])
        rets.append(x[1])
        rets.append(x[2])
        for i in range(L_embed):
            for fn in [np.sin, np.cos]:
                a=fn(2.**i * x)
                rets.append(a[0])
                rets.append(a[1])
                rets.append(a[2])                
        return rets 

    def forward(self, img, x, choose, obj):
        #print(img)
        depth = img[:,3,:,:].unsqueeze(0)
        out_img = self.cnn(img[:,0:3,:,:],depth)

        bs, di, _ = out_img.size()
        emb = out_img.view(di, bs, -1)
        emb=F.adaptive_avg_pool2d(emb, (4,500))
        emb = emb.view(1, 4, 500)

        x = x.transpose(2, 1).contiguous()

        ap_x = self.net1(x)
        ap_y = self.net2(emb)
        ap_x,ap_y= self.attn(F.dropout(ap_x, p=0.25),F.dropout(ap_y, p=0.25))

        ap_x = ap_x.permute(1, 0, 2) 
        ap_y = ap_y.permute(1, 0, 2) 
        ap_x=torch.cat([ap_x, ap_y], 2)
        ap_x=self.netFinal(F.dropout(ap_x, p=0.25))
        ap_x=ap_x.flatten()
        rx= F.tanh(self.compressr2(F.dropout(ap_x, p=0.5)))
        tx= (self.compresst2(F.dropout(ap_x, p=0.5)))

        rx = rx.view(1, self.num_obj, 4)
        tx = tx.view(1, self.num_obj, 3)
      
        b = 0
        rx = torch.index_select(rx[b], 0, obj[b])
        tx = torch.index_select(tx[b], 0, obj[b])

        return rx.flatten(), tx.flatten(), None, emb.detach()















def replace_batchnorm_with_instancenorm(model):
    for name, module in model.named_children():
        if isinstance(module, nn.BatchNorm2d):
            # Replace with InstanceNorm2d
            instance_norm = nn.InstanceNorm2d(module.num_features, affine=True)
            setattr(model, name, instance_norm)
        elif isinstance(module, nn.Module):
            replace_batchnorm_with_instancenorm(module)






class PoseNetMulti(nn.Module):
    def __init__(self, num_points, num_obj):
        super(PoseNetMulti, self).__init__()
        self.num_points = num_points
        self.cnn = RGBDFeatureExtractor2(num_heads=4)
        
        self.compressr2=torch.nn.Linear(4096, num_obj*4)
        self.compresst2=torch.nn.Linear(4096, num_obj*3)
        self.num_obj = num_obj
        self.attn = TransformerEncoder(512, 4); 
        self.net2=Conv1DNetwork(50);
        self.net1=Conv1DNetwork(3);
        self.net3=Conv1DNetwork(3);
        self.netFinal=Conv1DNetwork(22);
        replace_batchnorm_with_instancenorm(self.cnn)
        replace_batchnorm_with_instancenorm(self.attn)

    def embed_fn(self, x, L_embed=6):
        rets = []
        rets.append(x[0])
        rets.append(x[1])
        rets.append(x[2])
        for i in range(L_embed):
            for fn in [np.sin, np.cos]:
                a=fn(2.**i * x)
                rets.append(a[0])
                rets.append(a[1])
                rets.append(a[2])                
        return rets 

    def forward(self, img, depth_vel, x,velodyne, choose, obj):
        #print(img)
        depth = img[:,3,:,:].unsqueeze(0)
        depth_vel = depth_vel.squeeze(3)
        depth_vel = depth_vel.unsqueeze(0)

        #print(depth.shape)
        #print(depth_vel.shape)

        out_img = self.cnn(img[:,0:3,:,:],depth,depth_vel)
        #print(out_img.shape)
        bs, di, _ = out_img.size()

        emb = out_img.view(di, bs, -1)
        emb=F.adaptive_avg_pool2d(emb, (50,500))
        emb = emb.view(1, 50, 500)

        x = x.transpose(2, 1).contiguous()
        velodyne = velodyne.transpose(2, 1).contiguous()

        ap_x = self.net1(x)
        ap_x = ap_x+self.net3(velodyne)
        ap_y = self.net2(emb)
        ap_x,ap_y= self.attn(F.dropout(ap_x, p=0.0025),F.dropout(ap_y, p=0.0025))

        ap_x = ap_x.permute(1, 0, 2) 
        ap_y = ap_y.permute(1, 0, 2) 
        ap_x=torch.cat([ap_x, ap_y], 1)
        ap_x=self.netFinal(F.dropout(ap_x, p=0.0025))
        ap_x=ap_x.flatten()
        rx= F.tanh(self.compressr2(F.dropout(ap_x, p=0.005)))
        tx= (self.compresst2(F.dropout(ap_x, p=0.005)))

        rx = rx.view(1, self.num_obj, 4)
        tx = tx.view(1, self.num_obj, 3)
      
        b = 0
        rx = torch.index_select(rx[b], 0, obj[b])
        tx = torch.index_select(tx[b], 0, obj[b])

        return rx.flatten(), tx.flatten(), None, emb.detach()





class PoseNetMultiRESNET18(nn.Module):
    def __init__(self, num_points, num_obj):
        super(PoseNetMultiRESNET18, self).__init__()
        self.num_points = num_points
        self.cnn = RGBDFeatureExtractor2RESNET18(num_heads=4)
        
        self.compressr2=torch.nn.Linear(4096, num_obj*4)
        self.compresst2=torch.nn.Linear(4096, num_obj*3)
        self.num_obj = num_obj
        self.attn = TransformerEncoder(512, 4); 
        self.net2=Conv1DNetwork(50);
        self.net1=Conv1DNetwork(3);
        self.net3=Conv1DNetwork(3);
        self.netFinal=Conv1DNetwork(22);
        replace_batchnorm_with_instancenorm(self.cnn)
        replace_batchnorm_with_instancenorm(self.attn)

    def embed_fn(self, x, L_embed=6):
        rets = []
        rets.append(x[0])
        rets.append(x[1])
        rets.append(x[2])
        for i in range(L_embed):
            for fn in [np.sin, np.cos]:
                a=fn(2.**i * x)
                rets.append(a[0])
                rets.append(a[1])
                rets.append(a[2])                
        return rets 

    def forward(self, img, depth_vel, x,velodyne, choose, obj):
        #print(img)
        depth = img[:,3,:,:].unsqueeze(0)
        depth_vel = depth_vel.squeeze(3)
        depth_vel = depth_vel.unsqueeze(0)

        #print(depth.shape)
        #print(depth_vel.shape)

        out_img = self.cnn(img[:,0:3,:,:],depth,depth_vel)
        #print(out_img.shape)
        bs, di, _ = out_img.size()

        emb = out_img.view(di, bs, -1)
        emb=F.adaptive_avg_pool2d(emb, (50,500))
        emb = emb.view(1, 50, 500)

        x = x.transpose(2, 1).contiguous()
        velodyne = velodyne.transpose(2, 1).contiguous()

        ap_x = self.net1(x)
        ap_x = ap_x+self.net3(velodyne)
        ap_y = self.net2(emb)
        ap_x,ap_y= self.attn(F.dropout(ap_x, p=0.0025),F.dropout(ap_y, p=0.0025))

        ap_x = ap_x.permute(1, 0, 2) 
        ap_y = ap_y.permute(1, 0, 2) 
        ap_x=torch.cat([ap_x, ap_y], 1)
        ap_x=self.netFinal(F.dropout(ap_x, p=0.0025))
        ap_x=ap_x.flatten()
        rx= F.tanh(self.compressr2(F.dropout(ap_x, p=0.005)))
        tx= (self.compresst2(F.dropout(ap_x, p=0.005)))

        rx = rx.view(1, self.num_obj, 4)
        tx = tx.view(1, self.num_obj, 3)
      
        b = 0
        rx = torch.index_select(rx[b], 0, obj[b])
        tx = torch.index_select(tx[b], 0, obj[b])

        return rx.flatten(), tx.flatten(), None, emb.detach()



class PoseNetFeat(nn.Module):
    def __init__(self, num_points):
        super(PoseNetFeat, self).__init__()
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv5 = torch.nn.Conv1d(128, 512, 1)
        self.conv6 = torch.nn.Conv1d(512, 1024, 1)

        self.ap1 = torch.nn.AvgPool1d(num_points)
        self.num_points = num_points

    def forward(self, x):
        x = F.relu(self.conv1(x))
        #emb = F.relu(self.e_conv1(emb))
        #pointfeat_1 = torch.cat((x, emb), dim=1)

        x = F.relu(self.conv2(x))
        #emb = F.relu(self.e_conv2(emb))
        #pointfeat_2 = torch.cat((x, emb), dim=1)

        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))

        ap_x = self.ap1(x)

        ap_x = ap_x.view(-1, 1024, 1).repeat(1, 1, self.num_points)
        return ap_x #1024



class PoseNetMultiCUSTOM(nn.Module):
    def __init__(self, modelRGB, modelDepth, modelDepth2, num_points, num_obj,embchannels):
        super(PoseNetMultiCUSTOM, self).__init__()
        self.num_points = num_points


        self.cnn = RGBDFeatureExtractor2CUSTOM(modelRGB,modelDepth, modelDepth2,num_heads=4,emb=embchannels)


        
        self.compressr2=torch.nn.Linear(4096, num_obj*4)
        self.compresst2=torch.nn.Linear(4096, num_obj*3)
        self.num_obj = num_obj
        self.attn = TransformerEncoder(512, 4); 
        self.net2=Conv1DNetwork(50);
        self.net1=Conv1DNetwork(3);
        self.net3=Conv1DNetwork(3);
        self.netFinal=Conv1DNetwork(22);
        replace_batchnorm_with_instancenorm(self.cnn)
        replace_batchnorm_with_instancenorm(self.attn)

    def embed_fn(self, x, L_embed=6):
        rets = []
        rets.append(x[0])
        rets.append(x[1])
        rets.append(x[2])
        for i in range(L_embed):
            for fn in [np.sin, np.cos]:
                a=fn(2.**i * x)
                rets.append(a[0])
                rets.append(a[1])
                rets.append(a[2])                
        return rets 

    def forward(self, img, depth_vel, x,velodyne, choose, obj):
        #print(img)
        depth = img[:,3,:,:].unsqueeze(0)
        depth_vel = depth_vel.squeeze(3)
        depth_vel = depth_vel.unsqueeze(0)

        #print(depth.shape)
        #print(depth_vel.shape)

        out_img = self.cnn(img[:,0:3,:,:],depth,depth_vel)
        #print(out_img.shape)
        bs, di, _ = out_img.size()

        emb = out_img.view(di, bs, -1)
        emb=F.adaptive_avg_pool2d(emb, (50,500))
        emb = emb.view(1, 50, 500)

        x = x.transpose(2, 1).contiguous()
        velodyne = velodyne.transpose(2, 1).contiguous()

        ap_x = self.net1(x)
        #print("PC vector: ",ap_x.shape)
        ap_x2=self.net3(velodyne);
        #print("PC2 vector: ",ap_x.shape)
        ap_x = ap_x+ap_x2
        ap_y = self.net2(emb)
        ap_x,ap_y= self.attn(F.dropout(ap_x, p=0.0025),F.dropout(ap_y, p=0.0025))

        ap_x = ap_x.permute(1, 0, 2) 
        ap_y = ap_y.permute(1, 0, 2) 
        ap_x=torch.cat([ap_x, ap_y], 1)
        ap_x=self.netFinal(F.dropout(ap_x, p=0.0025))
        ap_x=ap_x.flatten()
        rx= F.tanh(self.compressr2(F.dropout(ap_x, p=0.005)))
        tx= (self.compresst2(F.dropout(ap_x, p=0.005)))

        rx = rx.view(1, self.num_obj, 4)
        tx = tx.view(1, self.num_obj, 3)
      
        b = 0
        rx = torch.index_select(rx[b], 0, obj[b])
        tx = torch.index_select(tx[b], 0, obj[b])

        return rx.flatten(), tx.flatten(), None, emb.detach()
    
class PoseNetMultiCUSTOMPointsX(nn.Module):
    def __init__(self, modelRGB, modelDepth, modelDepth2, num_points, num_obj, embchannels):
        super(PoseNetMultiCUSTOMPointsX, self).__init__()
        self.num_points = num_points

        self.cnn = RGBDFeatureExtractor2CUSTOM(modelRGB,modelDepth, modelDepth2,num_heads=4,emb=embchannels)
        
        self.compressr2 = torch.nn.Linear(4096, num_obj*4)
        self.compresst2 = torch.nn.Linear(4096, num_obj*3)
        self.num_obj = num_obj
        self.attn = TransformerEncoder(512, 4)
        self.net2 = Conv1DNetwork(50)
        self.net1 = PoseNetFeat(num_points)
        self.net3 = PoseNetFeat(num_points)
        self.netFinal = Conv1DNetwork(22)

    def embed_fn(self, x, L_embed=6):
        rets = []
        rets.append(x[0])
        rets.append(x[1])
        rets.append(x[2])
        for i in range(L_embed):
            for fn in [np.sin, np.cos]:
                a=fn(2.**i * x)
                rets.append(a[0])
                rets.append(a[1])
                rets.append(a[2])                
        return rets 

    def forward(self, img, depth_vel, x,velodyne, choose, obj):
        # print("Dentro rede ------------------")


        #net1_input:  torch.Size([64, 3, 1000])
        #net1:  torch.Size([15, 64, 512])
        #net3_input:  torch.Size([64, 3, 1000])
        #net3:  torch.Size([15, 64, 512])
        #net2_input:  torch.Size([64, 50, 500])
        #net2 torch.Size([7, 64, 512])
        #netnetFinal_input:  torch.Size([64, 22, 512])
        #netnetFinal torch.Size([8, 64, 512])
        batch = img.shape[0]
 
        rgb = img[:, 0:3, :, :]
        depth = img[:, 3, :, :].unsqueeze(0).permute(1, 0, 2, 3)
        
        # print("img.shape", rgb.shape)
        # print("depth.shape", depth.shape)
        # print("depth_vel", depth_vel.shape)

        out_img = self.cnn(rgb, depth, depth_vel)
        # print("out img", out_img.shape)

        bs, di, _ = out_img.size()

        emb = out_img.view(di, bs, -1)
        emb = F.adaptive_avg_pool2d(emb, (50,500))
        emb = emb.view(di, 50, 500)
        # print("emb.shape", emb.shape)

        x = x.transpose(2, 1).contiguous()
        velodyne = velodyne.transpose(2, 1).contiguous()
        #print("net1_input: ", x.shape)
        ap_x = self.net1(x).contiguous()
        #print("net1: ", ap_x.shape)

        bs, di, _ = ap_x.size()

        ap_x = ap_x.view(bs, di, 1000)
        ap_x = F.adaptive_avg_pool2d(ap_x, (15, 512))
        ap_x = ap_x.view(15,bs, 512)
        #print("PC vector Adapt: ", ap_x.shape)
        #print("net3_input: ", velodyne.shape)

        ap_x2 = self.net3(velodyne).contiguous()
        #print("net3: ", ap_x2.shape)

        ap_x2 = ap_x2.view(bs, di, 1000)
        ap_x2 = F.adaptive_avg_pool2d(ap_x2, (15,512))
        ap_x2 = ap_x2.view( 15,bs, 512)
        #print("PC2 vector Adapt: ",ap_x2.shape)

        ap_x = ap_x + ap_x2
        #print("net2_input: ", emb.shape)

        ap_y = self.net2(emb).contiguous()
        #print("net2", ap_y.shape)

        ap_x, ap_y = self.attn(F.dropout(ap_x, p=0.0025),F.dropout(ap_y, p=0.0025))

        ap_x = ap_x.permute(1, 0, 2) 
        ap_y = ap_y.permute(1, 0, 2) 
    
        #print(ap_x.shape,ap_y.shape)
        ap_x = torch.cat([ap_x, ap_y], 1)
        #print("netnetFinal_input: ", ap_x.shape)

        ap_x = self.netFinal(F.dropout(ap_x, p=0.0025))
        #print("netnetFinal", ap_x.shape)

        ap_x = ap_x.permute(1, 0, 2)
        ap_x = ap_x.flatten(start_dim=1)
        rx = F.tanh(self.compressr2(F.dropout(ap_x, p=0.005)))
        tx = (self.compresst2(F.dropout(ap_x, p=0.005)))

        rx = rx.view(batch, self.num_obj, 4)
        tx = tx.view(batch, self.num_obj, 3)

        # print("rx.shape", rx.shape)
        # print("obj", obj)
        # print(rx)
        # print("tx.shape", tx.shape)

        batch_indices = torch.arange(rx.size(0)).cuda()

        rx = rx[batch_indices, obj]
        tx = tx[batch_indices, obj]

        # print("rx.shape", rx.shape)
        # print(rx)

        # print("rx.shape", rx.shape)
        # print("tx.shape", tx.shape)

        # print("Dentro rede ------------------")

        return rx, tx, None, emb.detach()
    
class PoseNetMultiCUSTOMPointsX_old(nn.Module):
    def __init__(self, modelRGB, modelDepth, modelDepth2, num_points, num_obj, embchannels):
        super(PoseNetMultiCUSTOMPointsX_old, self).__init__()
        self.num_points = num_points

        self.cnn = RGBDFeatureExtractor2CUSTOM(modelRGB,modelDepth, modelDepth2,num_heads=4,emb=embchannels)
        
        self.compressr2 = torch.nn.Linear(4096, num_obj*4)
        self.compresst2 = torch.nn.Linear(4096, num_obj*3)
        self.num_obj = num_obj
        self.attn = TransformerEncoder(512, 4)
        self.net2 = Conv1DNetwork(50)
        self.net1 = Conv1DNetwork(3)
        self.net3 = Conv1DNetwork(3)
        self.netFinal = Conv1DNetwork(22)
        replace_batchnorm_with_instancenorm(self.cnn)
        replace_batchnorm_with_instancenorm(self.attn)

    def embed_fn(self, x, L_embed=6):
        rets = []
        rets.append(x[0])
        rets.append(x[1])
        rets.append(x[2])
        for i in range(L_embed):
            for fn in [np.sin, np.cos]:
                a=fn(2.**i * x)
                rets.append(a[0])
                rets.append(a[1])
                rets.append(a[2])                
        return rets 

    def forward(self, img, depth_vel, x,velodyne, choose, obj):
        # print("Dentro rede ------------------")
        batch = img.shape[0]
 
        rgb = img[:, 0:3, :, :]
        depth = img[:, 3, :, :].unsqueeze(0).permute(1, 0, 2, 3)
        
        # print("img.shape", rgb.shape)
        # print("depth.shape", depth.shape)
        # print("depth_vel", depth_vel.shape)

        out_img = self.cnn(rgb, depth, depth_vel)
        # print("out img", out_img.shape)

        bs, di, _ = out_img.size()

        emb = out_img.view(di, bs, -1)
        emb = F.adaptive_avg_pool2d(emb, (50,500))
        emb = emb.view(di, 50, 500)
        # print("emb.shape", emb.shape)

        x = x.transpose(2, 1).contiguous()
        velodyne = velodyne.transpose(2, 1).contiguous()

        ap_x = self.net1(x).contiguous()
        print("net1: ", ap_x.shape)

        bs, di, _ = ap_x.size()

        ap_x = ap_x.view(di, bs, 512)
        ap_x = F.adaptive_avg_pool2d(ap_x, (15, 512))
        ap_x = ap_x.view(15, di, 512)
        # print("PC vector Adapt: ", ap_x.shape)

        ap_x2 = self.net3(velodyne).contiguous()
        print("net3: ", ap_x2.shape)

        ap_x2 = ap_x2.view(di, bs, 512)
        ap_x2 = F.adaptive_avg_pool2d(ap_x2, (15,512))
        ap_x2 = ap_x2.view(15, di, 512)
        # print("PC2 vector Adapt: ",ap_x2.shape)

        ap_x = ap_x + ap_x2

        ap_y = self.net2(emb).contiguous()
        print("net2", ap_y.shape)
        
        ap_x, ap_y = self.attn(F.dropout(ap_x, p=0.0025),F.dropout(ap_y, p=0.0025))

        ap_x = ap_x.permute(1, 0, 2) 
        ap_y = ap_y.permute(1, 0, 2) 

        ap_x = torch.cat([ap_x, ap_y], 1)
        ap_x = self.netFinal(F.dropout(ap_x, p=0.0025))
        ap_x = ap_x.permute(1, 0, 2)
        ap_x = ap_x.flatten(start_dim=1)
        rx = F.tanh(self.compressr2(F.dropout(ap_x, p=0.005)))
        tx = (self.compresst2(F.dropout(ap_x, p=0.005)))

        rx = rx.view(batch, self.num_obj, 4)
        tx = tx.view(batch, self.num_obj, 3)

        # print("rx.shape", rx.shape)
        # print("obj", obj)
        # print(rx)
        # print("tx.shape", tx.shape)

        batch_indices = torch.arange(rx.size(0)).cuda()

        rx = rx[batch_indices, obj]
        tx = tx[batch_indices, obj]

        # print("rx.shape", rx.shape)
        # print(rx)

        # print("rx.shape", rx.shape)
        # print("tx.shape", tx.shape)

        # print("Dentro rede ------------------")

        return rx, tx, None, emb.detach()

class PoseNetMultiCUSTOMPointsX_Manuel(nn.Module):
    def __init__(self, modelRGB, modelDepth, modelDepth2, num_points, num_obj, embchannels):
        super(PoseNetMultiCUSTOMPointsX_Manuel, self).__init__()
        self.num_points = num_points

        self.cnn = RGBDFeatureExtractor2CUSTOM(modelRGB,modelDepth, modelDepth2,num_heads=4,emb=embchannels)
        
        self.num_obj = num_obj
        self.attn = TransformerEncoder(512, 4)
        self.net2 = Conv1DNetwork(50)
        self.net1 = Conv1DNetwork(3)
        self.net3 = Conv1DNetwork(3)
        self.netFinal = Conv1DNetwork(22)
        replace_batchnorm_with_instancenorm(self.cnn)
        replace_batchnorm_with_instancenorm(self.attn)

    def embed_fn(self, x, L_embed=6):
        rets = []
        rets.append(x[0])
        rets.append(x[1])
        rets.append(x[2])
        for i in range(L_embed):
            for fn in [np.sin, np.cos]:
                a=fn(2.**i * x)
                rets.append(a[0])
                rets.append(a[1])
                rets.append(a[2])                
        return rets 

    def forward(self, img, depth_vel, x, velodyne, choose, obj):
        batch = img.shape[0]
        self.compressr2 = torch.nn.Linear(4096*batch, self.num_obj*4*batch).cuda()
        self.compresst2 = torch.nn.Linear(4096*batch, self.num_obj*3*batch).cuda()

        rgb = img[:,0:3,:,:]
        depth = img[:,3,:,:].unsqueeze(1)
                
        out_img = self.cnn(rgb, depth, depth_vel)
        bs, di, _ = out_img.size()

        emb = out_img.view(di, bs, -1)
        emb = F.adaptive_avg_pool2d(emb, (50,500))
        emb = emb.view(batch, 50, 500)

        x = x.transpose(2, 1).contiguous()
        velodyne = velodyne.transpose(2, 1).contiguous()

        ap_x = self.net1(x).contiguous()

        bs, di, _ = ap_x.size()

        ap_x = ap_x.view(di, bs, 512)
        ap_x = F.adaptive_avg_pool2d(ap_x, (15,512))
        ap_x = ap_x.view(15, di, 512)
        # print("PC vector Adapt: ",ap_x.shape)

        ap_x2 = self.net3(velodyne).contiguous()

        ap_x2 = ap_x2.view(di, bs, 512)
        ap_x2 = F.adaptive_avg_pool2d(ap_x2, (15,512))
        ap_x2 = ap_x2.view(15, di, 512)
        # print("PC2 vector Adapt: ",ap_x2.shape)

        ap_x = ap_x + ap_x2

        ap_y = self.net2(emb)
        ap_x, ap_y = self.attn(F.dropout(ap_x, p=0.0025),F.dropout(ap_y, p=0.0025))

        ap_x = ap_x.permute(1, 0, 2) 
        ap_y = ap_y.permute(1, 0, 2) 
        ap_x = torch.cat([ap_x, ap_y], 1)
        ap_x = self.netFinal(F.dropout(ap_x, p=0.0025))
        ap_x = ap_x.flatten()
        rx = F.tanh(self.compressr2(F.dropout(ap_x, p=0.005)))  # 4096*batch_size
        tx = (self.compresst2(F.dropout(ap_x, p=0.005)))

        rx = rx.view(self.num_obj*batch, 4)
        tx = tx.view(self.num_obj*batch, 3)

        rx = torch.index_select(rx, 0, obj)
        tx = torch.index_select(tx, 0, obj)

        return rx, tx, None, emb.detach()


class AdaptiveStdPool2d(torch.nn.Module):
    def __init__(self, output_size):
        """
        Initialize the adaptive standard deviation pooling layer.
        :param output_size: The target output size (single int or tuple).
        """
        super(AdaptiveStdPool2d, self).__init__()
        self.output_size = output_size

    def forward(self, x,mean):
        # Step 2: Upsample the mean to the original size of x
        mean_upsampled = F.interpolate(mean, size=x.shape, mode='nearest')
        
        # Step 3: Compute the variance
        variance = F.adaptive_avg_pool2d((x - mean_upsampled) ** 2, self.output_size)
        
        # Step 4: Compute the standard deviation
        std = torch.sqrt(variance + 1e-8)  # Adding epsilon for numerical stability
        
        return std

class PoseNetMultiFeatures(nn.Module):
    def __init__(self, num_points, num_obj):
        super(PoseNetMultiFeatures, self).__init__()
        self.num_points = num_points
        self.cnn = RGBDFeatureExtractor2(num_heads=4)
        
        self.compressr2=torch.nn.Linear(4096, num_obj*4)
        self.compresst2=torch.nn.Linear(4096, num_obj*3)
        self.num_obj = num_obj
        self.attn = TransformerEncoder(512, 4); 
        self.net2=Conv1DNetwork(50);
        self.net1=Conv1DNetwork(3);
        self.net3=Conv1DNetwork(3);
        self.netFinal=Conv1DNetwork(22);
        replace_batchnorm_with_instancenorm(self.cnn)
        replace_batchnorm_with_instancenorm(self.attn)
        #self.stdPool=AdaptiveStdPool2d((50,500))

    def embed_fn(self, x, L_embed=6):
        rets = []
        rets.append(x[0])
        rets.append(x[1])
        rets.append(x[2])
        for i in range(L_embed):
            for fn in [np.sin, np.cos]:
                a=fn(2.**i * x)
                rets.append(a[0])
                rets.append(a[1])
                rets.append(a[2])                
        return rets 

    def forward(self, img, depth_vel, x,velodyne, choose, obj):
        #print(img)
        depth = img[:,3,:,:].unsqueeze(0)
        depth_vel = depth_vel.squeeze(3)
        depth_vel = depth_vel.unsqueeze(0)

        #print(depth.shape)
        #print(depth_vel.shape)


        out_img = self.cnn(img[:,0:3,:,:],depth,depth_vel)
        #print(out_img.shape)
        bs, di, _ = out_img.size()

        emb = out_img.view(di, bs, -1)
        emb1=F.adaptive_avg_pool2d(emb, (50,500))
        embMAX=F.adaptive_max_pool2d(emb, (50,500))



        variance = F.adaptive_avg_pool2d((embMAX - emb1) ** 2, (50,500))
        
        # Step 4: Compute the standard deviation
        emb2 = torch.sqrt(variance + 1e-8)  # Adding epsilon for numerical stability


        emb2 = emb2.view(1, 50, 500)
        emb1 = emb1.view(1, 50, 500)

        






        emb = (embMAX-emb1)/(emb2+0.000001)
        mask = torch.abs(emb) < 3
        emb = torch.where(mask, emb, torch.zeros_like(emb))

        x = x.transpose(2, 1).contiguous()
        velodyne = velodyne.transpose(2, 1).contiguous()

        ap_x = self.net1(x)
        ap_x = ap_x+self.net3(velodyne)


        ap_y = self.net2(emb1)
        ap_x,ap_y= self.attn(F.dropout(ap_x, p=0.1),F.dropout(ap_y, p=0.1))

        ap_x = ap_x.permute(1, 0, 2) 
        ap_y = ap_y.permute(1, 0, 2) 
        ap_x=torch.cat([ap_x, ap_y], 1)
        ap_x=self.netFinal(F.dropout(ap_x, p=0.1))
        ap_x=ap_x.flatten()
        rx= F.tanh(self.compressr2(F.dropout(ap_x, p=0.1)))
        tx= (self.compresst2(F.dropout(ap_x, p=0.1)))

        rx = rx.view(1, self.num_obj, 4)
        tx = tx.view(1, self.num_obj, 3)
      
        b = 0
        rx = torch.index_select(rx[b], 0, obj[b])
        tx = torch.index_select(tx[b], 0, obj[b])

        return rx.flatten(), tx.flatten(), None, emb.detach()



# concat features
class PoseNetMultiv2(nn.Module):
    def __init__(self, num_points, num_obj):
        super(PoseNetMultiv2, self).__init__()
        self.num_points = num_points
        self.cnn = RGBDFeatureExtractor2(num_heads=4)
        
        self.compressr2=torch.nn.Linear(4096, num_obj*4)
        self.compresst2=torch.nn.Linear(4096, num_obj*3)
        self.num_obj = num_obj
        self.attn = TransformerEncoderW3(512, 4); 
        self.net2=Conv1DNetwork(50);
        self.net1=Conv1DNetwork(3);
        self.net3=Conv1DNetwork(3);
        self.netFinal=Conv1DNetwork(37);
        replace_batchnorm_with_instancenorm(self.cnn)
        replace_batchnorm_with_instancenorm(self.attn)

    def embed_fn(self, x, L_embed=6):
        rets = []
        rets.append(x[0])
        rets.append(x[1])
        rets.append(x[2])
        for i in range(L_embed):
            for fn in [np.sin, np.cos]:
                a=fn(2.**i * x)
                rets.append(a[0])
                rets.append(a[1])
                rets.append(a[2])                
        return rets 

    def forward(self, img, depth_vel, x,velodyne, choose, obj):
        #print(img)
        depth = img[:,3,:,:].unsqueeze(0)
        depth_vel = depth_vel.squeeze(3)
        depth_vel = depth_vel.unsqueeze(0)

        #print(depth.shape)
        #print(depth_vel.shape)

        out_img = self.cnn(img[:,0:3,:,:],depth,depth_vel)
        #print(out_img.shape)
        bs, di, _ = out_img.size()

        emb = out_img.view(di, bs, -1)
        emb=F.adaptive_avg_pool2d(emb, (50,500))
        emb = emb.view(1, 50, 500)

        x = x.transpose(2, 1).contiguous()
        velodyne = velodyne.transpose(2, 1).contiguous()

        ap_x = self.net1(x)
        ap_z =self.net3(velodyne)
        ap_y = self.net2(emb)
        ap_x,ap_y,ap_z= self.attn(F.dropout(ap_x, p=0.0025),F.dropout(ap_y, p=0.0025),F.dropout(ap_z, p=0.0025))

        ap_x = ap_x.permute(1, 0, 2) 
        ap_y = ap_y.permute(1, 0, 2) 
        ap_z = ap_z.permute(1, 0, 2) 
        ap_x=torch.cat([ap_x, ap_y, ap_z], 1)
        ap_x=self.netFinal(F.dropout(ap_x, p=0.0025))
        ap_x=ap_x.flatten()
        rx= F.tanh(self.compressr2(F.dropout(ap_x, p=0.005)))
        tx= (self.compresst2(F.dropout(ap_x, p=0.005)))

        rx = rx.view(1, self.num_obj, 4)
        tx = tx.view(1, self.num_obj, 3)
      
        b = 0
        rx = torch.index_select(rx[b], 0, obj[b])
        tx = torch.index_select(tx[b], 0, obj[b])

        return rx.flatten(), tx.flatten(), None, emb.detach()



class PoseNetMultiv3(nn.Module):
    def __init__(self, num_points, num_obj):
        super(PoseNetMultiv3, self).__init__()
        self.num_points = num_points
        self.cnn = RGBFeatureExtractor(num_heads=4)
        
        self.compressr2=torch.nn.Linear(4096, num_obj*4)
        self.compresst2=torch.nn.Linear(4096, num_obj*3)
        self.num_obj = num_obj
        self.attn = TransformerEncoderW3(512, 4); 
        self.net2=Conv1DNetwork(50);
        self.net1=Conv1DNetwork(3);
        self.net3=Conv1DNetwork(3);
        self.netFinal=Conv1DNetwork(37);
        replace_batchnorm_with_instancenorm(self.cnn)
        replace_batchnorm_with_instancenorm(self.attn)

    def embed_fn(self, x, L_embed=6):
        rets = []
        rets.append(x[0])
        rets.append(x[1])
        rets.append(x[2])
        for i in range(L_embed):
            for fn in [np.sin, np.cos]:
                a=fn(2.**i * x)
                rets.append(a[0])
                rets.append(a[1])
                rets.append(a[2])                
        return rets 

    def forward(self, img, depth_vel, x,velodyne, choose, obj):
        #print(img)
        depth = img[:,3,:,:].unsqueeze(0)
        depth_vel = depth_vel.squeeze(3)
        depth_vel = depth_vel.unsqueeze(0)

        #print(depth.shape)
        #print(depth_vel.shape)

        out_img = self.cnn(img[:,0:3,:,:])
        #print(out_img.shape)
        bs, di, _ = out_img.size()

        emb = out_img.view(di, bs, -1)
        emb=F.adaptive_avg_pool2d(emb, (50,500))
        emb = emb.view(1, 50, 500)

        x = x.transpose(2, 1).contiguous()
        velodyne = velodyne.transpose(2, 1).contiguous()

        ap_x = self.net1(x)
        ap_z =self.net3(velodyne)
        ap_y = self.net2(emb)
        ap_x,ap_y,ap_z= self.attn(F.dropout(ap_x, p=0.0025),F.dropout(ap_y, p=0.0025),F.dropout(ap_z, p=0.0025))

        ap_x = ap_x.permute(1, 0, 2) 
        ap_y = ap_y.permute(1, 0, 2) 
        ap_z = ap_z.permute(1, 0, 2) 
        ap_x=torch.cat([ap_x, ap_y, ap_z], 1)
        ap_x=self.netFinal(F.dropout(ap_x, p=0.0025))
        ap_x=ap_x.flatten()
        rx= F.tanh(self.compressr2(F.dropout(ap_x, p=0.005)))
        tx= (self.compresst2(F.dropout(ap_x, p=0.005)))

        rx = rx.view(1, self.num_obj, 4)
        tx = tx.view(1, self.num_obj, 3)
      
        b = 0
        rx = torch.index_select(rx[b], 0, obj[b])
        tx = torch.index_select(tx[b], 0, obj[b])

        return rx.flatten(), tx.flatten(), None, emb.detach()






class PoseNetMultiFake(nn.Module):
    def __init__(self, num_points, num_obj):
        super(PoseNetMultiFake, self).__init__()
        self.num_points = num_points
        self.cnn = RGBDFeatureExtractor(num_heads=4)
        
        self.compressr2=torch.nn.Linear(4096, num_obj*4)
        self.compresst2=torch.nn.Linear(4096, num_obj*3)
        self.num_obj = num_obj
        self.attn = TransformerEncoder(512, 4); 
        self.net2=Conv1DNetwork(50);
        self.net1=Conv1DNetwork(3);
        self.netFinal=Conv1DNetwork(22);


        replace_batchnorm_with_instancenorm(self.cnn)
        replace_batchnorm_with_instancenorm(self.attn)
    def embed_fn(self, x, L_embed=6):
        rets = []
        rets.append(x[0])
        rets.append(x[1])
        rets.append(x[2])
        for i in range(L_embed):
            for fn in [np.sin, np.cos]:
                a=fn(2.**i * x)
                rets.append(a[0])
                rets.append(a[1])
                rets.append(a[2])                
        return rets 

    def forward(self, img, depth_vel, x,velodyne, choose, obj):
        #print(img.shape,depth_vel.shape,x.shape,velodyne.shape)
        depth = img[:,3,:,:].unsqueeze(0)
        
        out_img = self.cnn(img[:,0:3,:,:],depth)

        bs, di, _ = out_img.size()
        #print(out_img.shape)
        emb = out_img.view(di, bs, -1)
        emb=F.adaptive_avg_pool2d(emb, (50,500))

        

        emb = emb.view(1, 50, 500)

        x = x.transpose(2, 1).contiguous()

        ap_x = self.net1(x)
        ap_y = self.net2(emb)
        ap_x,ap_y= self.attn(F.dropout(ap_x, p=0.0025),F.dropout(ap_y, p=0.0025))

        ap_x = ap_x.permute(1, 0, 2) 
        ap_y = ap_y.permute(1, 0, 2) 
        #print(ap_x.shape,ap_y.shape)
        ap_x=torch.cat([ap_x, ap_y], 1)
        ap_x=self.netFinal(F.dropout(ap_x, p=0.0025))
        ap_x=ap_x.flatten()
        rx= F.tanh(self.compressr2(F.dropout(ap_x, p=0.005)))
        tx= (self.compresst2(F.dropout(ap_x, p=0.005)))

        rx = rx.view(1, self.num_obj, 4)
        tx = tx.view(1, self.num_obj, 3)
      
        b = 0
        rx = torch.index_select(rx[b], 0, obj[b])
        tx = torch.index_select(tx[b], 0, obj[b])

        return rx.flatten(), tx.flatten(), None, emb.detach()







class PoseNetEduardo(nn.Module):
    def __init__(self, num_points, num_obj):
        super(PoseNetEduardo, self).__init__()
        self.num_points = num_points
        self.cnn = RGBDFeatureExtractor2(num_heads=4)
        
        self.compressr2=torch.nn.Linear(8192, num_obj*1) 

        self.compresss2=torch.nn.Linear(8192, num_obj*3) 
        self.compresst2=torch.nn.Linear(8192, num_obj*3)
        self.num_obj = num_obj
        self.attn = TransformerEncoder(512, 4); 
        self.net2=Conv1DNetwork(4);
        self.netFinal=Conv1DNetwork(7);
 
    def forward(self, rgb,depth,depthlidar):
        #print(img)
      
        out_img = self.cnn(rgb,depth,depthlidar)

        bs, di, _ = out_img.size()
        emb = out_img.view(di, bs, -1)
        emb=F.adaptive_avg_pool2d(emb, (4,500))
        emb = emb.view(1, 4, 500)


        ap_y = self.net2(emb)
        ap_x,ap_y= self.attn(F.dropout(ap_y, p=0.125),F.dropout(ap_y, p=0.125))

        ap_x = ap_x.permute(1, 0, 2) 
        ap_y = ap_y.permute(1, 0, 2) 
        ap_x=torch.cat([ap_x, ap_y], 2)
        ap_x=self.netFinal(F.dropout(ap_x, p=0.125))
        ap_x=ap_x.flatten()
        rx= (self.compressr2(F.dropout(ap_x, p=0.15)))
        tx= (self.compresst2(F.dropout(ap_x, p=0.15)))
        sx= (self.compresss2(F.dropout(ap_x, p=0.15)))

        rx = rx.view(1, self.num_obj, 1)
        tx = tx.view(1, self.num_obj, 3)
        sx = sx.view(1, self.num_obj, 3)
      
        #b = 0
        #rx = torch.index_select(rx[b], 0, 0)
        #tx = torch.index_select(tx[b], 0, 0)
        #sx = torch.index_select(sx[b], 0, 0)

        #print(rx.shape,tx.shape,sx.shape)

        return rx.flatten(), tx.flatten(),sx.flatten() 


class PoseNetEduardo2(nn.Module):
    def __init__(self, num_points, num_obj):
        super(PoseNetEduardo2, self).__init__()
        self.num_points = num_points
        self.cnn = RGBDFeatureExtractor2(num_heads=4)
        
        self.compressr2=torch.nn.Linear(3588, num_obj*1) 

        self.compresss2=torch.nn.Linear(3588, num_obj*3) 
        self.compresst2=torch.nn.Linear(3588, num_obj*3)
        self.num_obj = num_obj
        self.net2=Conv1DNetwork(4);
 
    def forward(self, rgb,depth,depthlidar,bb):
        #print(img)
      
        out_img = self.cnn(rgb,depth,depthlidar)

        bs, di, _ = out_img.size()
        emb = out_img.view(di, bs, -1)
        emb=F.adaptive_avg_pool2d(emb, (4,500))
        emb = emb.view(1, 4, 500)


        ap_x = self.net2(emb).flatten()
        #print(ap_x.shape,bb.shape)
        ap_x=torch.cat([ap_x, bb.flatten()], 0)

        rx= F.tanh(self.compressr2(ap_x))*3.1415
        tx=  self.compresst2(ap_x)

        sx= F.relu(self.compresss2(ap_x))


        #b = 0
        #rx = torch.index_select(rx[b], 0, 0)
        #tx = torch.index_select(tx[b], 0, 0)
        #sx = torch.index_select(sx[b], 0, 0)

        #print(rx.shape,tx.shape,sx.shape)

        return rx.flatten(), tx.flatten(),sx.flatten() 



class PoseNetEduardo3(nn.Module):
    def __init__(self, num_points, num_obj):
        super(PoseNetEduardo3, self).__init__()
        self.num_points = num_points
        self.cnn = RGBDFeatureExtractor2(num_heads=4)
        


        self.overcompressor1=torch.nn.Linear(8192, 256) 
        self.overcompressor2=torch.nn.Linear(8192, 256) 
        self.overcompressor3=torch.nn.Linear(8192, 256) 
        self.overcompressorbb1=torch.nn.Linear(4, 128) 
        self.overcompressorbb2=torch.nn.Linear(128, 256) 

        self.compressr2=torch.nn.Linear(512, num_obj*1) 
        self.compresss2=torch.nn.Linear(512, num_obj*3) 
        self.compresst2=torch.nn.Linear(512, num_obj*3)
        self.num_obj = num_obj
        self.attn = TransformerEncoder(512, 4); 
        self.net2=Conv1DNetwork(12);
        self.netFinal=Conv1DNetwork(7);
 
    def forward(self, rgb,depth,depthlidar,bb):
        #print(img)
      
        out_img = self.cnn(rgb,depth,depthlidar)

        bs, di, _ = out_img.size()
        emb = out_img.view(di, bs, -1)
        emb=F.adaptive_avg_pool2d(emb, (12,500))
        emb = emb.view(1, 12, 500)


        ap_y = self.net2(emb)
        ap_x,ap_y= self.attn(F.dropout(ap_y, p=0.01),F.dropout(ap_y, p=0.01))

        ap_x = ap_x.permute(1, 0, 2) 
        ap_y = ap_y.permute(1, 0, 2) 
        ap_x=torch.cat([ap_x, ap_y], 2)
        ap_x=self.netFinal(F.dropout(ap_x, p=0.01))
        ap_x=ap_x.flatten()


        ap_x1= F.relu(self.overcompressor1(F.dropout(ap_x, p=0.01)))
        ap_x2= F.relu(self.overcompressor2(F.dropout(ap_x, p=0.01)))
        ap_x3= F.relu(self.overcompressor3(F.dropout(ap_x, p=0.01)))


        bb=F.relu(self.overcompressorbb1(bb))
        bb=F.relu(self.overcompressorbb2(bb))


        ap_x1=torch.cat([ap_x1, bb.flatten()], 0)
        ap_x2=torch.cat([ap_x2, bb.flatten()], 0)
        ap_x3=torch.cat([ap_x3, bb.flatten()], 0)




        rx= (self.compressr2(F.dropout(ap_x1, p=0.01)))
        tx= (self.compresst2(F.dropout(ap_x2, p=0.01)))
        sx= (self.compresss2(F.dropout(ap_x3, p=0.01)))

        rx = rx.view(1, self.num_obj, 1)
        tx = tx.view(1, self.num_obj, 3)
        sx = sx.view(1, self.num_obj, 3)
      
        #b = 0
        #rx = torch.index_select(rx[b], 0, 0)
        #tx = torch.index_select(tx[b], 0, 0)
        #sx = torch.index_select(sx[b], 0, 0)

        #print(rx.shape,tx.shape,sx.shape)

        return rx.flatten(), tx.flatten(),sx.flatten() 






class PoseNetEduardo3_4D_old(nn.Module):
    def __init__(self, num_points, num_obj):
        super(PoseNetEduardo3_4D_old, self).__init__()
        self.num_points = num_points
        self.cnn = RGBDFeatureExtractor2(num_heads=4)
        


        self.overcompressor1=torch.nn.Linear(8192, 256) 
        self.overcompressor2=torch.nn.Linear(8192, 256) 
        #self.overcompressor3=torch.nn.Linear(8192, 256) 
        self.overcompressorbb1=torch.nn.Linear(4, 128) 
        self.overcompressorbb2=torch.nn.Linear(128, 256) 

        self.compressr2=torch.nn.Linear(512, num_obj*1) 
    #self.compresss2=torch.nn.Linear(512, num_obj*3) 
        self.compresst2=torch.nn.Linear(512, num_obj*3)
        self.num_obj = num_obj
        self.attn = TransformerEncoder(512, 4); 
        self.net2=Conv1DNetwork(12);
        self.netFinal=Conv1DNetwork(7);
 
    def forward(self, rgb,depth,depthlidar,bb):
        #print(img)
      
        out_img = self.cnn(rgb,depth,depthlidar)

        bs, di, _ = out_img.size()
        emb = out_img.view(di, bs, -1)
        emb=F.adaptive_avg_pool2d(emb, (12,500))
        emb = emb.view(1, 12, 500)


        ap_y = self.net2(emb)
        ap_x,ap_y= self.attn(F.dropout(ap_y, p=0.01),F.dropout(ap_y, p=0.01))

        ap_x = ap_x.permute(1, 0, 2) 
        ap_y = ap_y.permute(1, 0, 2) 
        ap_x=torch.cat([ap_x, ap_y], 2)
        ap_x=self.netFinal(F.dropout(ap_x, p=0.01))
        ap_x=ap_x.flatten()


        ap_x1= F.relu(self.overcompressor1(F.dropout(ap_x, p=0.01)))
        ap_x2= F.relu(self.overcompressor2(F.dropout(ap_x, p=0.01)))


        bb=F.relu(self.overcompressorbb1(bb))
        bb=F.relu(self.overcompressorbb2(bb))


        ap_x1=torch.cat([ap_x1, bb.flatten()], 0)
        ap_x2=torch.cat([ap_x2, bb.flatten()], 0)




        rx= F.tanh(self.compressr2(F.dropout(ap_x1, p=0.01)))*3.1415
        tx= (self.compresst2(F.dropout(ap_x2, p=0.01)))
        #sx= F.relu(self.compresss2(F.dropout(ap_x3, p=0.01)))

        rx = rx.view(1, self.num_obj, 1)
        tx = tx.view(1, self.num_obj, 3)
        #sx = sx.view(1, self.num_obj, 3)
      
        #b = 0
        #rx = torch.index_select(rx[b], 0, 0)
        #tx = torch.index_select(tx[b], 0, 0)
        #sx = torch.index_select(sx[b], 0, 0)

        #print(rx.shape,tx.shape,sx.shape)

        return rx.flatten(), tx.flatten() 

class PoseNetEduardo3_4D(nn.Module):
    def __init__(self, num_points, num_obj):
        super(PoseNetEduardo3_4D, self).__init__()
        self.num_points = num_points
        self.cnn = RGBDFeatureExtractor2(num_heads=4)
        
        self.overcompressor1=torch.nn.Linear(7168, 256) 
        self.overcompressor2=torch.nn.Linear(7168, 256) 
        self.overcompressorbb1=torch.nn.Linear(4, 32) 
        self.overcompressorbb2=torch.nn.Linear(32, 64) 

        self.compressr2=torch.nn.Linear(256+64, 128) 
        self.compresst2=torch.nn.Linear(256+64, 128)

        self.compressr2a=torch.nn.Linear(128, num_obj*1) 
        self.compresst2a=torch.nn.Linear(128, num_obj*3) 

 
        self.num_obj = num_obj
        self.attn = TransformerEncoder(512, 4); 
        self.net2=Conv1DNetwork(12);

         
 
    def forward(self, rgb,depth,depthlidar,bb):
        #print(img)
      
        out_img = self.cnn(rgb,depth,depthlidar)

        bs, di, _ = out_img.size()
        emb = out_img.view(di, bs, -1)
        emb=F.adaptive_avg_pool2d(emb, (12,500))
        emb = emb.view(1, 12, 500)


        ap_y = self.net2(emb)
        ap_x,ap_y= self.attn(F.dropout(ap_y, p=0.0001),F.dropout(ap_y, p=0.0001))

        ap_x = ap_x.permute(1, 0, 2) 
        ap_y = ap_y.permute(1, 0, 2) 
        ap_x=torch.cat([ap_x, ap_y], 2)
        #ap_x=self.netFinal(F.dropout(ap_x, p=0.0001))

        ap_x=ap_x.flatten()

        #print(ap_x.shape)
        ap_x1= F.relu((self.overcompressor1(F.dropout(ap_x, p=0.0001))))
        ap_x2= F.relu((self.overcompressor2(F.dropout(ap_x, p=0.0001))))


        bb=F.relu(self.overcompressorbb1(bb))
        bb=(F.relu(self.overcompressorbb2(bb)))


        ap_x1=torch.cat([ap_x1, bb.flatten()], 0)
        ap_x2=torch.cat([ap_x2, bb.flatten()], 0)




        rx= F.tanh(self.compressr2a(F.sigmoid(self.compressr2(F.dropout(ap_x1, p=0.0001)))))*3.1415
        tx= self.compresst2a(F.sigmoid((self.compresst2(F.dropout(ap_x2, p=0.0001)))))
        #sx= F.relu(self.compresss2(F.dropout(ap_x3, p=0.01)))

        rx = rx.view(1, self.num_obj, 1)
        tx = tx.view(1, self.num_obj, 3)
        #sx = sx.view(1, self.num_obj, 3)
      
        #b = 0
        #rx = torch.index_select(rx[b], 0, 0)
        #tx = torch.index_select(tx[b], 0, 0)
        #sx = torch.index_select(sx[b], 0, 0)

        #print(rx.shape,tx.shape,sx.shape)

        return rx.flatten(), tx.flatten() 
