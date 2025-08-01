import torch
from torch import nn
from torch.nn import functional as F
import lib.extractors_attn as extractors
from einops import rearrange

# ------------------------------------------------------------------
# This file defines several classes for a semantic segmentation
# model based on the Pyramid Scene Network (PSPNet) architecture.
# ------------------------------------------------------------------

class PSPModule(nn.Module):
    def __init__(self, features, out_features=1024, sizes=(1, 2, 3, 6)):
        super(PSPModule, self).__init__()
        self.stages = []
        self.stages = nn.ModuleList([self._make_stage(features, size) for size in sizes])
        self.bottleneck = nn.Conv2d(features * (len(sizes) + 1), out_features, kernel_size=1)
        self.relu = nn.ReLU()

    def _make_stage(self, features, size):
        prior = nn.AdaptiveAvgPool2d(output_size=(size, size))
        conv = nn.Conv2d(features, features, kernel_size=1, bias=False)
        return nn.Sequential(prior, conv)

    def forward(self, feats):
        h, w = feats.size(2), feats.size(3)
        priors = [F.upsample(input=stage(feats), size=(h, w), mode='bilinear') for stage in self.stages] + [feats]
        bottle = self.bottleneck(torch.cat(priors, 1))
        return self.relu(bottle)

# ------------------------------------------------------------------
# 'PSPUpsample': 
# 
# Defines the upsampling block, which consists
# of bilinear upsampling followed by a 3x3 convolution and a PReLU
# activation.
# ------------------------------------------------------------------

class PSPUpsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(PSPUpsample, self).__init__()
        self.conv = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.PReLU()
        )

    def forward(self, x):
        return self.conv(x)

# ------------------------------------------------------------------
# 'PSPNet':
# 
# Defines the main PSPNet model. It includes a feature 
# exctrator (backend), the PSP module, and several upsampling blocks.
# The final layer consists of a 1x1 convolution followed by a log
# softmax activation.
# ------------------------------------------------------------------

class PSPNet(nn.Module):
    def __init__(self, n_classes=21, sizes=(1, 2, 3, 6), psp_size=2048, deep_features_size=1024, backend='resnet18',
                 pretrained=False):
        super(PSPNet, self).__init__()
        self.feats = getattr(extractors, backend)(pretrained)
        self.psp = PSPModule(psp_size, 1024, sizes)
        self.drop_1 = nn.Dropout2d(p=0.3)

        self.up_1 = PSPUpsample(1024, 256)
        self.up_2 = PSPUpsample(256, 64)
        self.up_3 = PSPUpsample(64, 64)

        self.drop_2 = nn.Dropout2d(p=0.15)
        self.final = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=1),
            nn.LogSoftmax()
        )

        self.classifier = nn.Sequential(
            nn.Linear(deep_features_size, 256),
            nn.ReLU(),
            nn.Linear(256, n_classes)
        )

    def forward(self, x):
        f, class_f = self.feats(x) 
        p = self.psp(f)
        p = self.drop_1(p)

        p = self.up_1(p)
        p = self.drop_2(p)

        p = self.up_2(p)
        p = self.drop_2(p)

        p = self.up_3(p)
        
        return self.final(p)

# ------------------------------------------------------------------
# 'PSPNetHv2, PSPNetH2v2b, PSPNetH2v2a, PSPNetH2v2, PSPNetH':

# These classes are variations of the PSPNet with different modifications,
# including the addition of attention mechanisms (Multihead Attention)
# and changes in the number of parameters and layers.
# ------------------------------------------------------------------

class PSPNetHv3(nn.Module):
    def __init__(self, n_classes=21, sizes=(1, 2, 3, 6), psp_size=2048, deep_features_size=1024, backend='resnet18',
                 pretrained=False):
        super(PSPNetHv3, self).__init__()
        self.feats = getattr(extractors, backend)(pretrained)
        self.featsD = getattr(extractors, backend+"Depth")(pretrained)
        self.featsD2 = getattr(extractors, backend+"Depth2")(pretrained)
        self.psp = PSPModule(psp_size, deep_features_size, sizes)

        self.drop_1 = nn.Dropout2d(p=0.3)

        self.up_1 = PSPUpsample(deep_features_size, 256)
        self.up_2 = PSPUpsample(256, 64)
        self.up_3 = PSPUpsample(64, 64)

        self.drop_2 = nn.Dropout2d(p=0.15)
        self.final = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=1),
            nn.LogSoftmax()
        )

        self.attn = nn.MultiheadAttention(1024, 1)


        self.classifier = nn.Sequential(
            nn.Linear(deep_features_size, 256),
            nn.ReLU(),
            nn.Linear(256, n_classes)
        )

    def forward(self, x,d):
        f, class_f = self.feats(x)
        fd, class_fd = self.featsD(d)
        fd2, class_fd = self.featsD2(d)

#        print('f',f.shape)
#        print('fd',fd.shape)

        # print('f',f.shape)
        # print('fd',fd.shape)
        f = f.permute((0,2, 3, 1))
        fd = fd.permute((0,2, 3, 1))
        fd2 = fd2.permute((0,2, 3, 1))

        p3=torch.cat((f,fd,fd2)) # confirmar dims aqui

        print(p3.shape)

        f, _= self.attn(p3.squeeze(0), p3.squeeze(0), p3.squeeze(0))
        print(f.shape)
        f=f.unsqueeze(0)
        f = f.permute((0,3, 1, 2))

        p = self.psp(f)


        #print('f',p1.shape)
        #print('fd',p2.shape)


        p = self.drop_1(p)

        p = self.up_1(p)
        p = self.drop_2(p)

        p = self.up_2(p)
        p = self.drop_2(p)

        p = self.up_3(p)

        return self.final(p)

class PSPNetHv2(nn.Module):
    def __init__(self, n_classes=21, sizes=(1, 2, 3, 6), psp_size=2048, deep_features_size=1024, backend='resnet18',
                 pretrained=False):
        super(PSPNetHv2, self).__init__()
        self.feats = getattr(extractors, backend)(pretrained)
        self.featsD = getattr(extractors, backend+"Depth")(pretrained)
        self.psp = PSPModule(psp_size, deep_features_size, sizes)

        self.drop_1 = nn.Dropout2d(p=0.3)

        self.up_1 = PSPUpsample(deep_features_size, 256)
        self.up_2 = PSPUpsample(256, 64)
        self.up_3 = PSPUpsample(64, 64)

        self.drop_2 = nn.Dropout2d(p=0.15)
        self.final = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=1),
            nn.LogSoftmax()
        )

        self.attn = nn.MultiheadAttention(1024, 1)


        self.classifier = nn.Sequential(
            nn.Linear(deep_features_size, 256),
            nn.ReLU(),
            nn.Linear(256, n_classes)
        )

    def forward(self, x,d):
        f, class_f = self.feats(x) 
        fd, class_fd = self.featsD(d) 

#        print('f',f.shape)
#        print('fd',fd.shape)

        # print('f',f.shape)
        # print('fd',fd.shape)
        f = f.permute((0,2, 3, 1))
        fd = fd.permute((0,2, 3, 1))

        p3=torch.cat((f,fd)) # confirmar dims aqui


        f, _= self.attn(p3.squeeze(0), p3.squeeze(0), p3.squeeze(0))
        f=f.unsqueeze(0)
        f = f.permute((0,3, 1, 2))

        p = self.psp(f)


        #print('f',p1.shape)
        #print('fd',p2.shape)
       

        p = self.drop_1(p)

        p = self.up_1(p)
        p = self.drop_2(p)

        p = self.up_2(p)
        p = self.drop_2(p)

        p = self.up_3(p)
        
        return self.final(p)


class PSPNetH2v2b(nn.Module):
    def __init__(self, n_classes=21, sizes=(1, 2, 3, 6), psp_size=2048, deep_features_size=1024, backend='resnet18',
                 pretrained=False):
        super(PSPNetH2v2b, self).__init__()
        self.feats = getattr(extractors, backend)(pretrained)
        self.featsD = getattr(extractors, backend+"Depth")(pretrained)
        self.psp = PSPModule(psp_size, deep_features_size, sizes)
        self.psp1 = PSPModule(psp_size, deep_features_size, sizes)

        self.drop_1 = nn.Dropout2d(p=0.3)

        self.up_1 = PSPUpsample(deep_features_size, 256)
        self.up_2 = PSPUpsample(256, 64)
        self.up_3 = PSPUpsample(64, 64)

        self.drop_2 = nn.Dropout2d(p=0.15)
        self.final = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=1),
            nn.LogSoftmax()
        )

        self.attn = nn.MultiheadAttention(256, 4) #confirmar os valores
        self.max3d=nn.MaxPool3d((2, 1, 1))

        self.classifier = nn.Sequential(
            nn.Linear(deep_features_size, 256),
            nn.ReLU(),
            nn.Linear(256, n_classes)
        )

    def forward(self, x,d):
        f, class_f = self.feats(x) 
        fd, class_fd = self.featsD(d) 

#        print('f',f.shape)
#        print('fd',fd.shape)


        # print('f',f.shape)
        # print('fd',fd.shape)
        

        p1 = self.psp(f)
        p2 = self.psp1(fd)


        #print('f',p1.shape)
        #print('fd',p2.shape)
        #p1 = p1.permute((0,2, 3, 1))
        #p2 = p2.permute((0,2, 3, 1))
        #print(p1.shape,p2.shape)
        p3=torch.cat((p1,p2),1)

        p3[:,::2, :,:] = p1   # Index every second row, starting from 0
        p3[:,1::2, :,:] = p2
#        p3=p1+p2
        #print(p3.shape)
        p3=self.max3d(p3)

        shape=p3.shape



        p3 = rearrange(p3, 'd0 d1 d2 d3 -> d0 (d2 d3)  d1')

        #print(p1.shape,p2.shape)

        p, _= self.attn(p3, p3, p3) 
        p = rearrange(p, 'd0 (d2 d3)  d1 -> d0 d1 d2 d3',d2=shape[2],d3=shape[3])
        #print(p.shape)


        p = self.drop_1(p)

        p = self.up_1(p)
        p = self.drop_2(p)
        #print(p.shape)

        p = self.up_2(p)
        p = self.drop_2(p)
        #print(p.shape)

        p = self.up_3(p)
        
        return self.final(p)


class PSPNetH2v2a(nn.Module):
    def __init__(self, n_classes=21, sizes=(1, 2, 3, 6), psp_size=2048, deep_features_size=1024, backend='resnet18',
                 pretrained=False):
        super(PSPNetH2v2a, self).__init__()
        self.feats = getattr(extractors, backend)(pretrained)
        self.featsD = getattr(extractors, backend+"Depth")(pretrained)
        self.psp = PSPModule(psp_size, deep_features_size, sizes)
        self.psp1 = PSPModule(psp_size, deep_features_size, sizes)

        self.drop_1 = nn.Dropout2d(p=0.3)

        self.up_1 = PSPUpsample(deep_features_size, 256)
        self.up_2 = PSPUpsample(256, 64)
        self.up_3 = PSPUpsample(64, 64)

        self.drop_2 = nn.Dropout2d(p=0.15)
        self.final = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=1),
            nn.LogSoftmax()
        )

        self.attn = nn.MultiheadAttention(512, 4) #confirmar os valores 512
        self.max3d=nn.MaxPool3d((2, 1, 1))

        self.classifier = nn.Sequential(
            nn.Linear(deep_features_size, 256),
            nn.ReLU(),
            nn.Linear(256, n_classes)
        )

    def forward(self, x,d):
        f, class_f = self.feats(x) 
        fd, class_fd = self.featsD(d) 

#        print('f',f.shape)
#        print('fd',fd.shape)


        # print('f',f.shape)
        # print('fd',fd.shape)
        

        p1 = self.psp(f)
        p2 = self.psp1(fd)


        #print('f',p1.shape)
        #print('fd',p2.shape)
        #p1 = p1.permute((0,2, 3, 1))
        #p2 = p2.permute((0,2, 3, 1))
        #print(p1.shape,p2.shape)
        p3=torch.cat((p1,p2),1)

        p3[:,::2, :,:] = p1   # Index every second row, starting from 0
        p3[:,1::2, :,:] = p2
#        p3=p1+p2
        #print(p3.shape)

        shape=p3.shape



        p3 = rearrange(p3, 'd0 d1 d2 d3 -> d0 (d2 d3)  d1')

        #print(p1.shape,p2.shape)

        p, _= self.attn(p3, p3, p3)
        p = rearrange(p, 'd0 (d2 d3)  d1 -> d0 d1 d2 d3',d2=shape[2],d3=shape[3])
        #print(p.shape)

        p=self.max3d(p)

        p = self.drop_1(p)

        p = self.up_1(p)
        p = self.drop_2(p)
        #print(p.shape)

        p = self.up_2(p)
        p = self.drop_2(p)
        #print(p.shape)

        p = self.up_3(p)
        
        return self.final(p)




class PSPNetH2v2(nn.Module):
    def __init__(self, n_classes=21, sizes=(1, 2, 3, 6), psp_size=2048, deep_features_size=1024, backend='resnet18',
                 pretrained=False):
        super(PSPNetH2v2, self).__init__()
        self.feats = getattr(extractors, backend)(pretrained)
        self.featsD = getattr(extractors, backend+"Depth")(pretrained)
        self.psp = PSPModule(psp_size, deep_features_size, sizes)
        self.psp1 = PSPModule(psp_size, deep_features_size, sizes)

        self.drop_1 = nn.Dropout2d(p=0.3)

        self.up_1 = PSPUpsample(deep_features_size, 256)
        self.up_2 = PSPUpsample(256, 64)
        self.up_3 = PSPUpsample(64, 64)

        self.drop_2 = nn.Dropout2d(p=0.15)
        self.final = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=1),
            nn.LogSoftmax()
        )

        self.attn = nn.MultiheadAttention(256, 4) #confirmar os valores


        self.classifier = nn.Sequential(
            nn.Linear(deep_features_size, 256),
            nn.ReLU(),
            nn.Linear(256, n_classes)
        )

    def forward(self, x,d):
        f, class_f = self.feats(x) 
        fd, class_fd = self.featsD(d) 

#        print('f',f.shape)
#        print('fd',fd.shape)


        # print('f',f.shape)
        # print('fd',fd.shape)
        

        p1 = self.psp(f)
        p2 = self.psp1(fd)


        #print('f',p1.shape)
        #print('fd',p2.shape)
        #p1 = p1.permute((0,2, 3, 1))
        #p2 = p2.permute((0,2, 3, 1))
#        p3=torch.cat((p1,p2),1)
        p3=p1+p2

        shape=p3.shape



        p3 = rearrange(p3, 'd0 d1 d2 d3 -> d0 (d2 d3)  d1')

        #print(p1.shape,p2.shape)

        p, _= self.attn(p3, p3, p3)
        p = rearrange(p, 'd0 (d2 d3)  d1 -> d0 d1 d2 d3',d2=shape[2],d3=shape[3])
 

        p = self.drop_1(p)

        p = self.up_1(p)
        p = self.drop_2(p)

        p = self.up_2(p)
        p = self.drop_2(p)

        p = self.up_3(p)
        
        return self.final(p)




class PSPNetH2v3(nn.Module):
    def __init__(self, n_classes=21, sizes=(1, 2, 3, 6), psp_size=2048, deep_features_size=1024, backend='resnet18',
                 pretrained=False):
        super(PSPNetH2v3, self).__init__()
        self.feats = getattr(extractors, backend)(pretrained)
        self.featsD = getattr(extractors, backend+"Depth")(pretrained)
        self.featsV = getattr(extractors, backend+"Depth")(pretrained)
        self.psp = PSPModule(psp_size, deep_features_size, sizes)
        self.psp1 = PSPModule(psp_size, deep_features_size, sizes)
        self.psp2= PSPModule(psp_size, deep_features_size, sizes)

        self.drop_1 = nn.Dropout2d(p=0.3)

        self.up_1 = PSPUpsample(deep_features_size, 256)
        self.up_2 = PSPUpsample(256, 64)
        self.up_3 = PSPUpsample(64, 64)

        self.drop_2 = nn.Dropout2d(p=0.15)
        self.final = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=1),
            nn.LogSoftmax()
        )

        self.attn = nn.MultiheadAttention(256, 4) #confirmar os valores


        self.classifier = nn.Sequential(
            nn.Linear(deep_features_size, 256),
            nn.ReLU(),
            nn.Linear(256, n_classes)
        )

    def forward(self, x,d,v):
        f, class_f = self.feats(x)
        fd, class_fd = self.featsD(d)
        fdv, class_fdv = self.featsV(d)



        p1 = self.psp(f)
        p2 = self.psp1(fd)
        p3 = self.psp2(fdv)


        #print('f',p1.shape)
        #print('fd',p2.shape)
        #p1 = p1.permute((0,2, 3, 1))
        #p2 = p2.permute((0,2, 3, 1))
#        p3=torch.cat((p1,p2),1)
        p3=p1+p2+p3

        shape=p3.shape



        p3 = rearrange(p3, 'd0 d1 d2 d3 -> d0 (d2 d3)  d1')

        #print(p1.shape,p2.shape)

        p, _= self.attn(p3, p3, p3)
        p = rearrange(p, 'd0 (d2 d3)  d1 -> d0 d1 d2 d3',d2=shape[2],d3=shape[3])


        p = self.drop_1(p)

        p = self.up_1(p)
        p = self.drop_2(p)

        p = self.up_2(p)
        p = self.drop_2(p)

        p = self.up_3(p)

        return self.final(p)



class PSPNetH(nn.Module):
    def __init__(self, n_classes=21, sizes=(1, 2, 3, 6), psp_size=2048, deep_features_size=1024, backend='resnet18',
                 pretrained=False):
        super(PSPNetH, self).__init__()
        self.feats = getattr(extractors, backend)(pretrained)
        self.featsD = getattr(extractors, backend+"Depth")(pretrained)
        self.psp = PSPModule(psp_size, deep_features_size, sizes)

        self.drop_1 = nn.Dropout2d(p=0.3)

        self.up_1 = PSPUpsample(deep_features_size, 256)
        self.up_2 = PSPUpsample(256, 64)
        self.up_3 = PSPUpsample(64, 64)

        self.drop_2 = nn.Dropout2d(p=0.15)
        self.final = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=1),
            nn.LogSoftmax()
        )

        self.attn = nn.MultiheadAttention(512, 2)


        self.classifier = nn.Sequential(
            nn.Linear(deep_features_size, 256),
            nn.ReLU(),
            nn.Linear(256, n_classes)
        )

    def forward(self, x,d):
        f, class_f = self.feats(x) 
        fd, class_fd = self.featsD(d) 

#        print('f',f.shape)
#        print('fd',fd.shape)

        # print('f',f.shape)
        # print('fd',fd.shape)
        f = f.permute((0,2, 3, 1))
        fd = fd.permute((0,2, 3, 1))

        f, _= self.attn(fd.squeeze(0), f.squeeze(0), f.squeeze(0))
        f=f.unsqueeze(0)
        f = f.permute((0,3, 1, 2))

        p = self.psp(f)


        #print('f',p1.shape)
        #print('fd',p2.shape)
       

        p = self.drop_1(p)

        p = self.up_1(p)
        p = self.drop_2(p)

        p = self.up_2(p)
        p = self.drop_2(p)

        p = self.up_3(p)
        
        return self.final(p)

class PSPNetH2(nn.Module):
    def __init__(self, n_classes=21, sizes=(1, 2, 3, 6), psp_size=2048, deep_features_size=1024, backend='resnet18',
                 pretrained=False):
        super(PSPNetH2, self).__init__()
        self.feats = getattr(extractors, backend)(pretrained)
        self.featsD = getattr(extractors, backend+"Depth")(pretrained)
        self.psp = PSPModule(psp_size, deep_features_size, sizes)
        self.psp1 = PSPModule(psp_size, deep_features_size, sizes)

        self.drop_1 = nn.Dropout2d(p=0.3)

        self.up_1 = PSPUpsample(deep_features_size, 256)
        self.up_2 = PSPUpsample(256, 64)
        self.up_3 = PSPUpsample(64, 64)

        self.drop_2 = nn.Dropout2d(p=0.15)
        self.final = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=1),
            nn.LogSoftmax()
        )

        self.attn = nn.MultiheadAttention(256, 4)


        self.classifier = nn.Sequential(
            nn.Linear(deep_features_size, 256),
            nn.ReLU(),
            nn.Linear(256, n_classes)
        )

    def forward(self, x,d):
        f, class_f = self.feats(x) 
        fd, class_fd = self.featsD(d) 

#        print('f',f.shape)
#        print('fd',fd.shape)


        # print('f',f.shape)
        # print('fd',fd.shape)
        

        p1 = self.psp(f)
        p2 = self.psp1(fd)


        #print('f',p1.shape)
        #print('fd',p2.shape)
        p1 = p1.permute((0,2, 3, 1))
        p2 = p2.permute((0,2, 3, 1))

        p, _= self.attn(p2.squeeze(0), p1.squeeze(0), p1.squeeze(0))
        p=p.unsqueeze(0)
        p = p.permute((0,3, 1, 2))

        p = self.drop_1(p)

        p = self.up_1(p)
        p = self.drop_2(p)

        p = self.up_2(p)
        p = self.drop_2(p)

        p = self.up_3(p)
        
        return self.final(p)

# ------------------------------------------------------------------
# 'PSPNetH2b, PSPNetH2a':
#
# These classes are further variations with additional modifications,
# including different attention mechanisms and dropout layers.
# ------------------------------------------------------------------

class PSPNetH2b(nn.Module):
    def __init__(self, n_classes=21, sizes=(1, 2, 3, 6), psp_size=2048, deep_features_size=1024, backend='resnet18',
                 pretrained=False):
        super(PSPNetH2b, self).__init__()
        self.feats = getattr(extractors, backend)(pretrained)
        self.featsD = getattr(extractors, backend+"Depth")(pretrained)
        self.psp = PSPModule(psp_size, deep_features_size, sizes)
        self.psp1 = PSPModule(psp_size, deep_features_size, sizes)

        self.drop_1 = nn.Dropout2d(p=0.3)

        self.up_1 = PSPUpsample(deep_features_size, 256)
        self.up_2 = PSPUpsample(256, 64)
        self.up_3 = PSPUpsample(64, 64)

        self.drop_2 = nn.Dropout2d(p=0.15)
        self.final = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=1),
            nn.LogSoftmax()
        )

        self.attn = nn.MultiheadAttention(256, 4)


        self.classifier = nn.Sequential(
            nn.Linear(deep_features_size, 256),
            nn.ReLU(),
            nn.Linear(256, n_classes)
        )

    def forward(self, x,d):
        f, class_f = self.feats(x) 
        fd, class_fd = self.featsD(d) 

#        print('f',f.shape)
#        print('fd',fd.shape)


        # print('f',f.shape)
        # print('fd',fd.shape)
        

        p1 = self.psp(f)
        p2 = self.psp1(fd)


        #print('f',p1.shape)
        #print('fd',p2.shape)
       

        #print(p1.shape,p2.shape)
        #print(torch.__version__)
        #p1 = p1.permute((0,2, 3, 1))
        #p2 = p2.permute((0,2, 3, 1))
        shape=p1.shape



        p1 = rearrange(p1, 'd0 d1 d2 d3 -> d0 (d2 d3)  d1')
        p2 = rearrange(p2, 'd0 d1 d2 d3 -> d0 (d2 d3)  d1')

        #print(p1.shape,p2.shape)

        p, _= self.attn(p2, p1, p1)
        p = rearrange(p, 'd0 (d2 d3)  d1 -> d0 d1 d2 d3',d2=shape[2],d3=shape[3])


        p = self.drop_1(p)

        p = self.up_1(p)
        p = self.drop_2(p)

        p = self.up_2(p)
        p = self.drop_2(p)

        p = self.up_3(p)
        
        return self.final(p)

class PSPNetH2a(nn.Module):
    def __init__(self, n_classes=21, sizes=(1, 2, 3, 6), psp_size=2048, deep_features_size=1024, backend='resnet18',
                 pretrained=False):
        super(PSPNetH2a, self).__init__()
        self.feats = getattr(extractors, backend)(pretrained)
        self.featsD = getattr(extractors, backend+"Depth")(pretrained)
        self.psp = PSPModule(psp_size, deep_features_size, sizes)
        self.psp1 = PSPModule(psp_size, deep_features_size, sizes)

        self.drop_1 = nn.Dropout2d(p=0.3)

        self.up_1 = PSPUpsample(deep_features_size, 256)
        self.up_2 = PSPUpsample(256, 64)
        self.up_3 = PSPUpsample(64, 64)

        self.drop_2 = nn.Dropout2d(p=0.15)
        self.final = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=1),
            #nn.LogSoftmax()
        )


        self.drop_1a = nn.Dropout2d(p=0.3)

        self.up_1a = PSPUpsample(deep_features_size, 256)
        self.up_2a = PSPUpsample(256, 64)
        self.up_3a = PSPUpsample(64, 64)

        self.drop_2a = nn.Dropout2d(p=0.15)
        self.finala = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=1),
            #nn.LogSoftmax()
        )


        self.attn = nn.MultiheadAttention(32, 4)


        #self.classifier = nn.Sequential(
        #    nn.Linear(deep_features_size, 256),
        #    nn.ReLU(),
        #    nn.Linear(256, n_classes)
        #)

    def forward(self, x,d):
        f, class_f = self.feats(x) 
        fd, class_fd = self.featsD(d) 

#        print('f',f.shape)
#        print('fd',fd.shape)


        # print('f',f.shape)
        # print('fd',fd.shape)
        

        p1 = self.psp(f)
        p2 = self.psp1(fd)


        #print('f',p1.shape)
        #print('fd',p2.shape)
        #p1 = p1.permute((0,2, 3, 1))
        #p2 = p2.permute((0,2, 3, 1))

        #p, _= self.attn(p2.squeeze(0), p1.squeeze(0), p1.squeeze(0))
        #p=p.unsqueeze(0)
        #p = p.permute((0,3, 1, 2))

        p = self.drop_1(p1)

        p = self.up_1(p)
        p = self.drop_2(p)

        p = self.up_2(p)
        p = self.drop_2(p)

        p = self.up_3(p)

        p1=self.final(p)



        p = self.drop_1a(p2)

        p = self.up_1a(p)
        p = self.drop_2a(p)

        p = self.up_2a(p)
        p = self.drop_2a(p)

        p = self.up_3a(p)

        p2=self.finala(p)
  
        #print(p1.shape,p2.shape)
        #print(torch.__version__)
        #p1 = p1.permute((0,2, 3, 1))
        #p2 = p2.permute((0,2, 3, 1))
        shape=p1.shape



        p1 = rearrange(p1, 'd0 d1 d2 d3 -> d0 (d2 d3)  d1')
        p2 = rearrange(p2, 'd0 d1 d2 d3 -> d0 (d2 d3)  d1')

        #print(p1.shape,p2.shape)

        p, _= self.attn(p2, p1, p1)
        p = rearrange(p, 'd0 (d2 d3)  d1 -> d0 d1 d2 d3',d2=shape[2],d3=shape[3])

        return p

# ------------------------------------------------------------------
# 'PSPNet4C':
# 
# This class is a variant designed to handle 4-channel input images
# (RGB-D images), using a specific feature extractor ('backend+"4Ch"').
# This specific feature extractor is the resnet18 which we can get from
# the extractors_attn.py file.
# ------------------------------------------------------------------

class PSPNet4C(nn.Module):
    def __init__(self, n_classes=21, sizes=(1, 2, 3, 6), psp_size=2048, deep_features_size=1024, backend='resnet18',
                 pretrained=False):
        #super(PSPNet, self).__init__()
        super(PSPNet4C, self).__init__()
        # The 'feats' attribute is set to an instace of extractors specific
        # to handling 4-channel input images. The exact feature extractor
        # is determined by appending "4Ch" to the backend parameter. 
        # In this case the resnet18.
        self.feats = getattr(extractors, backend+"_4Ch")(pretrained)
        self.psp = PSPModule(psp_size, 1024, sizes)
        self.drop_1 = nn.Dropout2d(p=0.3)

        self.up_1 = PSPUpsample(1024, 256)
        self.up_2 = PSPUpsample(256, 64)
        self.up_3 = PSPUpsample(64, 64)

        self.drop_2 = nn.Dropout2d(p=0.15)
        self.final = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=1),
            nn.LogSoftmax()
        )



        self.classifier = nn.Sequential(
            nn.Linear(deep_features_size, 256),
            nn.ReLU(),
            nn.Linear(256, n_classes)
        )

    def forward(self, x):
        f, class_f = self.feats(x) 
        

        p = self.psp(f)
        p = self.drop_1(p)

        p = self.up_1(p)
        p = self.drop_2(p)

        p = self.up_2(p)
        p = self.drop_2(p)

        p = self.up_3(p)
        
        return self.final(p)

