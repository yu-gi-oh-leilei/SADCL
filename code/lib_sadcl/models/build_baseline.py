from collections import OrderedDict
import os
import warnings

import torch
from torch.functional import Tensor
import torch.nn.functional as F
import torchvision
from torch import nn
from torchvision.models._utils import IntermediateLayerGetter
from typing import Dict, List

import models
from models.cls_cvt import build_CvT
from models.vision_transformer.swin_transformer import build_swin_transformer

from utils.misc import clean_state_dict, clean_body_state_dict

from .position_encoding import build_position_encoding



class FrozenBatchNorm2d(torch.nn.Module):
    """
    BatchNorm2d where the batch statistics and the affine parameters are fixed.

    Copy-paste from torchvision.misc.ops with added eps before rqsrt,
    without which any other models than torchvision.models.resnet[18,34,50,101]
    produce nans.
    """

    def __init__(self, n):
        super(FrozenBatchNorm2d, self).__init__()
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        num_batches_tracked_key = prefix + 'num_batches_tracked'
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]

        super(FrozenBatchNorm2d, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs)

    def forward(self, x):
        # move reshapes to the beginning
        # to make it fuser-friendly
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        eps = 1e-5
        scale = w * (rv + eps).rsqrt()
        bias = b - rm * scale
        return x * scale + bias


class resnet101_backbone(nn.Module):
    def __init__(self, pretrain=True):
        super(resnet101_backbone,self).__init__()

        # res101 = torchvision.models.resnet101(pretrained=True)

        name = 'resnet101'
        dilation = False
        res101 = getattr(torchvision.models, name)(
                        replace_stride_with_dilation=[False, False, dilation],
                        pretrained=True,
                        norm_layer=FrozenBatchNorm2d)

        train_backbone = True
        for name, parameter in res101.named_parameters():
            if not train_backbone or 'layer1' not in name and 'layer2' not in name and 'layer3' not in name and 'layer4' not in name:
                parameter.requires_grad_(False)
    
        numFit = res101.fc.in_features # 2048
        self.resnet_layer = nn.Sequential(*list(res101.children())[:-2])
        self.feat_dim = numFit

    def forward(self,x):
        feats = self.resnet_layer(x)
        return feats


class ResNet101_GAP(nn.Module):
    def __init__(self, num_classes):
        super(ResNet101_GAP, self).__init__()

        self.backbone = resnet101_backbone(pretrain=True)
        self.num_classes = num_classes
        self.pool= nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(2048, num_classes)

    def forward_feature(self, x):
        x = self.backbone(x)
        return x
   
    def forward(self, x):
        feature = self.forward_feature(x)
        feature = self.pool(feature).view(-1, 2048)
        out = self.fc(feature)
        
        return out

class ResNet101_GMP(nn.Module):
    def __init__(self, num_classes):
        super(ResNet101_GMP, self).__init__()

        self.backbone = resnet101_backbone(pretrain=True)
        self.num_classes = num_classes
        self.pool= nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(2048, num_classes)

    def forward_feature(self, x):
        x = self.backbone(x)
        return x
   
    def forward(self, x):
        feature = self.forward_feature(x)
        feature = self.pool(feature).view(-1, 2048)
        out = self.fc(feature)
        
        return out


def build_baseline(args):

    # model = ResNet101_GMP(
    #     num_classes=args.num_class
    # )

    model = ResNet101_GAP(
        num_classes=args.num_class
    )

    return model