# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
We borrow the positional encoding from Detr and adding some other backbones.
"""
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
from models.vision_transformer.vision_transformer import build_vision_transformer

from utils.misc import clean_state_dict, clean_body_state_dict

from .position_encoding import build_position_encoding


def get_model_path(modelname):
    """
        Config your pretrained model path here!
    """
    # raise NotImplementedError("Please config your pretrained modelpath!")
    pretrained_dir = '/media/data2/maleilei/MLIC_pretrained'
    PTDICT = {
        'CvT_w24': 'CvT-w24-384x384-IN-22k.pth',
        'swin_L_384_22k': 'swin_large_patch4_window12_384_22k.pth',
        'tresnetl': 'tresnet_l_448.pth',
        'tresnetxl': 'tresnet_xl_448.pth',
        'tresnetl_v2': 'tresnet_l_v2_miil_21k.pth',
        'tresnetl_v3': 'tresnet_l_pretrain_openimage.pth',
        'vit_L_16_224_22k': 'L_16-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.1-sd_0.1.npz'
    }
    return os.path.join(pretrained_dir, PTDICT[modelname]) 

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


class BackboneBase(nn.Module):

    def __init__(self, backbone: nn.Module, train_backbone: bool, num_channels: int, return_layers: Dict,
    # return_interm_layers: bool
    ):
        super().__init__()
        for name, parameter in backbone.named_parameters():
            if not train_backbone or 'layer2' not in name and 'layer3' not in name and 'layer4' not in name:
                parameter.requires_grad_(False)
        # if return_interm_layers:
        #     return_layers = {"layer1": "0", "layer2": "1", "layer3": "2", "layer4": "3"}
        # else:
        #     return_layers = {'layer4': "0"}
        self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)
        self.num_channels = num_channels

    def forward(self, input: Tensor):
        xs = self.body(input)
        out: Dict[str, Tensor] = {}
        for name, x in xs.items():
            out[name] = x
        return out


class Backbone(BackboneBase):
    """ResNet backbone with frozen BatchNorm."""
    def __init__(self, name: str,
                 train_backbone: bool,
                 return_interm_layers: bool,
                 dilation: bool,
                 pretrained: bool = True):
        if name in ['resnet18', 'resnet50', 'resnet34', 'resnet101']:
            if pretrained is True:
                WEIGHTDICT_V1 = {
                    'resnet18': 'ResNet18_Weights.IMAGENET1K_V1',
                    'resnet34': 'ResNet34_Weights.IMAGENET1K_V1',
                    'resnet50': 'ResNet50_Weights.IMAGENET1K_V1',
                    'resnet101': 'ResNet101_Weights.IMAGENET1K_V1',
                }
                WEIGHTDICT_V2 = {
                    'resnet18': 'ResNet18_Weights.IMAGENET1K_V2',
                    'resnet34': 'ResNet34_Weights.IMAGENET1K_V2',
                    'resnet50': 'ResNet50_Weights.IMAGENET1K_V2',
                    'resnet101': 'ResNet101_Weights.IMAGENET1K_V2',
                }
                pretrained = WEIGHTDICT_V1[name]
                # pretrained = WEIGHTDICT_V2[name]

                # print('===='*30)
                # print(pretrained)

            backbone = getattr(torchvision.models, name)(
                replace_stride_with_dilation=[False, False, dilation],
                weights=pretrained,
                norm_layer=FrozenBatchNorm2d)

            if return_interm_layers:
                return_layers = {"layer1": "0", "layer2": "1", "layer3": "2", "layer4": "3"}
            else:
                return_layers = {'layer4': "0"}
        elif name in ['tresnetl', 'tresnetxl', 'tresnetl_v2', 'tresnetl_v3']:
            backbone = getattr(models, name)(
                {'num_classes': 1}
            )
            # load pretrained model
            if pretrained:
                pretrainedpath = get_model_path(name)
                checkpoint = torch.load(pretrainedpath, map_location='cpu')
                from collections import OrderedDict
                if 'model' in checkpoint:
                    checkpoint = checkpoint['model']
                if 'state_dict' in checkpoint:
                    checkpoint = checkpoint['state_dict']
                _tmp_st = OrderedDict({k:v for k, v in clean_state_dict(checkpoint).items() if 'head.fc' not in k})

                backbone.load_state_dict(_tmp_st, strict=False)
            
            if return_interm_layers:
                raise NotImplementedError('return_interm_layers must be False in TResNet!')
            return_layers = {'body': "0"}
        else:
            raise NotImplementedError("Unknow name: %s" % name)
            
        NCDICT = {
            'resnet18': 512,
            'resnet34': 512,
            'resnet50': 2048,
            'resnet101': 2048,
            'tresnetl': 2432,
            'tresnetxl': 2656,
            'tresnetl_v2': 2048,
            'tresnetl_v3': 2048,
        }
        num_channels = NCDICT[name]
        super().__init__(backbone, train_backbone, num_channels, return_layers)


class Joiner(nn.Sequential):
    def __init__(self, backbone, position_embedding, args=None):
        super().__init__(backbone, position_embedding)
        # self.args = args
        if args is not None and 'interpotaion' in vars(args) and args.interpotaion:
            self.interpotaion = True
        else:
            self.interpotaion = False


    def forward(self, input: Tensor):
        xs = self[0](input)
        out: List[Tensor] = []
        pos = []
        if isinstance(xs, dict):
            for name, x in xs.items():
                out.append(x)
                # position encoding
                pos.append(self[1](x).to(x.dtype))
        else:
            # for swin Transformer
            out.append(xs)
            pos.append(self[1](xs).to(xs.dtype))
        return out, pos


def build_backbone(args):
    position_embedding = build_position_encoding(args)
    train_backbone = False if args.frozen_backbone==True else True
    if args.backbone in ['swin_B_224_22k', 'swin_B_384_22k', 'swin_L_224_22k', 'swin_L_384_22k']:
        imgsize = int(args.backbone.split('_')[-2])
        backbone = build_swin_transformer(args.backbone, imgsize)
        if args.pretrained:
            pretrainedpath = get_model_path(args.backbone)
            checkpoint = torch.load(pretrainedpath, map_location='cpu')['model']
            from collections import OrderedDict
            _tmp_st = OrderedDict({k:v for k, v in clean_state_dict(checkpoint).items() if 'head' not in k})
            _tmp_st_output = backbone.load_state_dict(_tmp_st, strict=False)
            print(str(_tmp_st_output))
        backbone.forward = backbone.forward_features
        bb_num_channels = backbone.embed_dim * 8
        del backbone.avgpool
        del backbone.head
    elif args.backbone in ['CvT_w24']:
        backbone = build_CvT(args.backbone, args.num_class)
        if args.pretrained:
            pretrainedpath = get_model_path(args.backbone)
            checkpoint = torch.load(pretrainedpath, map_location='cpu')
            from collections import OrderedDict
            _tmp_st = OrderedDict({k:v for k, v in clean_state_dict(checkpoint).items() if 'head' not in k})
            _tmp_st_output = backbone.load_state_dict(_tmp_st, strict=False)
            print(str(_tmp_st_output))
        bb_num_channels = backbone.dim_embed[-1]
        backbone.forward = backbone.forward_features
        backbone.cls_token = False
        del backbone.head
    elif args.backbone in ['vit_L_16_224_22k']:
        backbone = build_vision_transformer(args.backbone, args.num_class)
        if args.pretrained:
            pretrainedpath = get_model_path(args.backbone)
            backbone.load_pretrained(pretrainedpath)
        # bb_num_channels = backbone.dim_embed[-1]
        backbone.forward = backbone.forward_features
        backbone.cls_token = False
        del backbone.head
    else:
        return_interm_layers = False
        backbone = Backbone(args.backbone, train_backbone, return_interm_layers, False, args.pretrained)
        bb_num_channels = backbone.num_channels
    model = Joiner(backbone, position_embedding, args)
    model.num_channels = bb_num_channels
    return model
