# --------------------------------------------------------
# SADCL
# Written by Leilei Ma
# most borrow from Query2lable.
# https://github.com/SlongLiu/query2labels
# --------------------------------------------------------

import os, sys
import os.path as osp

import torch
import torch.nn as nn
import torch.distributed as dist
import torch.nn.functional as F
import numpy as np
import math

from models.backbone import build_backbone
from models.transformer import build_transformer
# from models.transformer_newattention import build_transformer
from models.ffn import MLP, FFNLayer, MLP1D
from utils.misc import clean_state_dict

class GroupWiseLinear(nn.Module):
    # could be changed to: 
    # output = torch.einsum('ijk,zjk->ij', x, self.W)
    # or output = torch.einsum('ijk,jk->ij', x, self.W[0])
    def __init__(self, num_class, hidden_dim, bias=True):
        super().__init__()
        self.num_class = num_class
        self.hidden_dim = hidden_dim
        self.bias = bias

        self.W = nn.Parameter(torch.Tensor(1, num_class, hidden_dim))
        if bias:
            self.b = nn.Parameter(torch.Tensor(1, num_class))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.W.size(2))
        for i in range(self.num_class):
            self.W[0][i].data.uniform_(-stdv, stdv)
        if self.bias:
            for i in range(self.num_class):
                self.b[0][i].data.uniform_(-stdv, stdv)

    def forward(self, x):
        # x: B,K,d
        x = (self.W * x).sum(-1)
        if self.bias:
            x = x + self.b
        return x


class SADCL(nn.Module):
    def __init__(self, backbone, transfomer, prototype, dataname, num_class, batch_size):
        """[summary]
    
        Args:
            backbone ([type]): backbone model.
            transfomer ([type]): transformer model.
            num_class ([type]): number of classes. (80 for MSCOCO).
        """
        super().__init__()
        self.backbone = backbone
        self.transformer = transfomer
        self.prototype_path = prototype
        self.dataname = dataname
        self.num_class = num_class
        self.mini_batch_size = int(batch_size / int(os.environ['WORLD_SIZE']))
        # self.mini_batch_size = 16
        self.momentum = 0.99
        self.proj_head_dim = 1024
        
        hidden_dim = transfomer.d_model
        self.input_proj = nn.Conv2d(backbone.num_channels, hidden_dim, kernel_size=1)
        self.query_embed = nn.Embedding(num_class, hidden_dim)
        self.fc = GroupWiseLinear(num_class, hidden_dim, bias=True)

        # vis_prototype and vis_memorybank
        self.select_prototype_path = self.select_Prototype()
        self.memorybank = self.load_MemoryBank()
        self.prototype  = self.load_Prototype()

        self.proj_head = MLP(input_dim=hidden_dim, hidden_dim=hidden_dim, output_dim=self.proj_head_dim, num_layers=2)

    
    def select_Prototype(self):
        if self.dataname == 'coco14':
            self.prototype_path = osp.join(self.prototype_path, 'vis_prototype_embed_file_select_resnet101_448_coco_085.npy') 
        elif self.dataname == 'nuswide':
            self.prototype_path = osp.join(self.prototype_path, 'vis_prototype_embed_file_select_resnet101_448_nuswide_085.npy') 
        elif self.dataname == 'vg500':
            self.prototype_path = osp.join(self.prototype_path, 'vis_prototype_embed_file_select_resnet101_576_vg500_085.npy') 
        elif self.dataname == 'voc2007':
            self.prototype_path = osp.join(self.prototype_path, 'vis_prototype_embed_file_select_resnet101_448_voc_085.npy')
        elif self.dataname == 'voc2012':
            self.prototype_path = osp.join(self.prototype_path, 'vis_prototype_embed_file_select_resnet101_448_voc_085.npy') 
        else:
            raise NotImplementedError
        return self.prototype_path

    def load_Prototype(self):
        prototype = torch.from_numpy(np.load(self.select_prototype_path)).unsqueeze(0)
        prototype = nn.Parameter(prototype, requires_grad=True)
        # prototype = nn.Parameter(prototype, requires_grad=False)
        return prototype

    def load_MemoryBank(self):
        memorybank = torch.from_numpy(np.load(self.select_prototype_path)).unsqueeze(0).repeat(self.mini_batch_size, 1, 1)
        memorybank = nn.Parameter(memorybank, requires_grad=False)
        return memorybank

    @torch.no_grad()
    def memorybank_update(self, embedding, target, output):
        """
        Prototype update of the network.
        """
        output = torch.sigmoid(output.unsqueeze(-1))
        output[output>0.8]=1; output[output<0.8]=0
        embedding = output * embedding

        # target = output * target
        target = target.nonzero()
        for label_info in target:
            batch_id, label = label_info
            batch_id, label = int(batch_id), int(label)
            if output[batch_id][label] == 1:
                self.memorybank.data[batch_id][label] = embedding.data[batch_id][label]

                # self.memorybank.data[batch_id][label] = \
                # (1-self.momentum)*self.memorybank.data[batch_id][label] + \
                # self.momentum*embedding.data[batch_id][label]

                # self.memorybank.data[batch_id][label] = \
                # self.momentum*self.memorybank.data[batch_id][label] + \
                # (1-self.momentum)*embedding.data[batch_id][label]

    def forward(self, input):
        src, pos = self.backbone(input)
        src, pos = src[-1], pos[-1]
        # import ipdb; ipdb.set_trace()

        query_input = self.query_embed.weight
        hs = self.transformer(self.input_proj(src), query_input, pos)[0] # B,K,d
        
        # combine feature and projected
        contrast_feature = torch.cat((hs[-1], self.memorybank.detach()), dim=0)#.permute(0, 2, 1)
        contrast_feature = self.proj_head(contrast_feature)#.permute(0, 2, 1)
        contrast_feature = torch.split(contrast_feature, self.mini_batch_size, dim=0)
        vis_embed, memorybank = contrast_feature[0], contrast_feature[1]

        # project prototype
        prototype = self.proj_head(self.prototype) #.permute(0, 2, 1)

        out = self.fc(hs[-1]) # hs[-1].shape [16, 80, 2048]

        return out, hs[-1].clone().detach(), vis_embed, memorybank, prototype

    def finetune_paras(self):
        from itertools import chain
        return chain(self.transformer.parameters(), self.fc.parameters(), self.input_proj.parameters(), self.query_embed.parameters())

    def load_backbone(self, path):
        print("=> loading checkpoint '{}'".format(path))
        checkpoint = torch.load(path, map_location=torch.device(dist.get_rank()))
        # import ipdb; ipdb.set_trace()
        self.backbone[0].body.load_state_dict(clean_state_dict(checkpoint['state_dict']), strict=False)
        print("=> loaded checkpoint '{}' (epoch {})"
                  .format(path, checkpoint['epoch']))


def build_sadcl(args):
    backbone = build_backbone(args)
    transformer = build_transformer(args)

    model = SADCL(
        backbone = backbone,
        transfomer = transformer,
        prototype = args.prototype,
        dataname = args.dataname,
        num_class = args.num_class,
        batch_size = args.batch_size
    )

    if not args.keep_input_proj:
        model.input_proj = nn.Identity()
        print("set model.input_proj to Indentify!")

    return model
     
        
