import torch
import torch.nn as nn

import torch.nn.functional as F

def cos_simi(embedded_fg, embedded_bg):
    embedded_fg = F.normalize(embedded_fg, dim=-1)
    embedded_bg = F.normalize(embedded_bg, dim=-1)
    sim = torch.matmul(embedded_fg, embedded_bg.permute(0, 2, 1))
    b, n, n = sim.size()
    sim = sim * torch.eye(n, n).repeat(b, 1, 1).to(sim.device)
    return torch.clamp(sim, min=0.0005, max=0.9995)


class SimMinLoss(nn.Module):
    def __init__(self, metric='cos', reduction='mean'):
        super(SimMinLoss, self).__init__()
        self.metric = metric
        self.reduction = reduction

    def forward(self, embedded_bg, embedded_fg, target):
        """
        :param embedded_fg: [bs, n, dim]
        :param embedded_bg: [bs, n, dim]
        :return:
        """
        target =  target.unsqueeze(-1)
        embedded_bg = target * embedded_bg
        embedded_fg = target * embedded_fg
        if self.metric == 'l2':
            raise NotImplementedError
        elif self.metric == 'cos':
            sim = cos_simi(embedded_bg, embedded_fg)
            loss = -torch.log(1 - sim)
        else:
            raise NotImplementedError

        if self.reduction == 'mean':
            return torch.sum(loss) / target.sum()
        elif self.reduction == 'sum':
            return torch.sum(loss)


def cos_simi_all(embedded_fg, embedded_bg):
    embedded_fg = F.normalize(embedded_fg, dim=-1).flatten(1)
    embedded_bg = F.normalize(embedded_bg, dim=-1).flatten(1)
    sim = torch.matmul(embedded_fg, embedded_bg.T)
    return torch.clamp(sim, min=0.0005, max=0.9995)


class SimMinLoss_all(nn.Module):
    def __init__(self, metric='cos', reduction='mean'):
        super(SimMinLoss_all, self).__init__()
        self.metric = metric
        self.reduction = reduction

    def forward(self, embedded_bg, embedded_fg, target):
        """
        :param embedded_fg: [bs, n, dim]
        :param embedded_bg: [bs, n, dim]
        :return:
        """
        target =  target.unsqueeze(-1)
        embedded_bg = target * embedded_bg
        embedded_fg = target * embedded_fg
        if self.metric == 'l2':
            raise NotImplementedError
        elif self.metric == 'cos':
            sim = cos_simi_all(embedded_bg, embedded_fg)
            loss = -torch.log(1 - sim)
        else:
            raise NotImplementedError

        if self.reduction == 'mean':
            return torch.sum(loss) / target.sum()
        elif self.reduction == 'sum':
            return torch.sum(loss)

class MultiSupConLoss(nn.Module):
    """
    Supervised Multi-Label Contrastive Learning.
    Author: Leilei Ma
    Date: Nov 4, 2022"""
    def __init__(self, temperature=0.2, contrast_mode='all',
                 base_temperature=0.2):
        super(MultiSupConLoss, self).__init__()

        self.contrast_mode = contrast_mode
        self.temperature = temperature
        self.base_temperature = base_temperature
        self.eps = 1e-5

    def forward(self, features, labels=None, mask=None, multi=True):

        device = features.device
        features = torch.nn.functional.normalize(features, dim=-1)           # [bs, views, nc, dim]

        batch_size = features.shape[0]
        num_class = features.shape[2]

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)   # [bs, nc, dim]

        if self.contrast_mode == 'all':
            anchor_feature = contrast_feature                                # [256, 80, 2432]  [bs*n_view, nc, dim]
            anchor_count = contrast_count                                    # [256, 80, 2432]  [bs*n_view, nc, dim]
        # intra-class and inter-class 类内和类间

        # 同类不同图片之间的对比
        anchor_dot_contrast_intra = torch.div(
            torch.matmul(anchor_feature.transpose(1, 0), contrast_feature.permute(1, 2, 0)),
            self.temperature)                                                # [80, 256, 256]   [nc, bs*n_view, bs*n_view]

        # 同类不同图片之间, 为了数值稳定
        logits_max, _ = torch.max(anchor_dot_contrast_intra.permute(1, 2, 0), dim=-1, keepdim=True)
        anchor_dot_contrast_intra = (anchor_dot_contrast_intra.permute(1, 2, 0) - logits_max.detach()).permute(2, 0, 1)

        # 生成同类不同图片的mask
        mask_intra = labels.transpose(1, 0).unsqueeze(-1).matmul(labels.transpose(1, 0).unsqueeze(1))            # [nc, bs, bs] 
        mask_intra = mask_intra.repeat(1, anchor_count, contrast_count)      # [nc, bs*n_view, bs*n_view] 

        # 所有的特征的mask
        all_mask = labels.view(-1, 1)*labels.view(1, -1)
        all_mask = all_mask.repeat(anchor_count, contrast_count)             # [bs*n_view*nc, bs*n_view*nc]


        # 同类不同图片之间，去掉自身的mask
        logits_mask_intra = torch.scatter(
            torch.ones(batch_size * anchor_count, batch_size * anchor_count).to(device),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0)                                                               # [80, 256, 256]  [bs*n_view, bs*n_view]
        logits_mask_intra = logits_mask_intra.repeat(num_class, 1, 1)
        mask_intra = mask_intra * logits_mask_intra

        logits_intra = mask_intra * anchor_dot_contrast_intra                # [80, 256, 256]   [nc, bs*n_view, bs*n_view]

        # 所有特征都相互计算相似度
        all_features = contrast_feature.view(-1, contrast_feature.shape[-1]) # [bs*n_view*nc, dim]
        all_anchor_dot_contrast = torch.div(
            torch.matmul(all_features, all_features.transpose(1, 0)),
            self.temperature)                                                # [20480, 20480]    [bs*n_view*nc, dim, bs*n_view*nc, dim]

        # 重新思考
        # 所有特征, 为了数值稳定
        # all_anchor_dot_contrast = all_anchor_dot_contrast.reshape(batch_size*contrast_count, num_class, num_class, batch_size*contrast_count)
        all_anchor_dot_contrast = all_anchor_dot_contrast.reshape(batch_size*contrast_count, num_class, batch_size*contrast_count, num_class)
        logits_max_all, _ = torch.max(all_anchor_dot_contrast, dim=-2, keepdim=True)
        # print(logits_max_all.shape)
        logits_all = all_anchor_dot_contrast - logits_max_all.detach()
        logits_all = logits_all.reshape(batch_size*contrast_count*num_class, batch_size*contrast_count*num_class)

        # logits_all = all_anchor_dot_contrast
        # import ipdb; ipdb.set_trace()
        # 所有的类，去掉自身
        self_mask = torch.eye(batch_size*num_class*contrast_count).to(device)
        self_mask = torch.ones_like(self_mask) - self_mask
        all_mask = all_mask * self_mask
        # logits_all = torch.exp(logits_all) * self_mask
        # print(all_mask.sum())
        logits_all = torch.exp(logits_all) * all_mask


        # 分母 [bs*n_view*nc, bs*n_view*nc]
        logits_all = logits_all.sum(1, keepdim=True)
        logits_all[logits_all<=0] = 1
        logits_all = all_mask*torch.log(logits_all)


        # 分子 [bs*n_view*nc, bs*n_view]
        logits_intra = (mask_intra*logits_intra).transpose(1, 0).reshape(batch_size*contrast_count*num_class, batch_size*contrast_count)

        # 计算对数似然除以正的均值
        log_prob = logits_intra - logits_all.sum(1, keepdim=True)            # [bs*n_view*nc, bs*n_view] - [bs*n_view*nc, 1]
        # print(logits_intra.shape,     logits_all.sum(1, keepdim=True).shape)
        mean_log_prob_pos = (log_prob).sum() / mask_intra.sum()


        # 计算loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        # loss = loss.view(batch_size, anchor_count, num_class).mean()
        return loss / batch_size



def main():
    import numpy as np
    batch_size = 128
    views = 2
    classes = 80
    criterion = MultiSupConLoss()

    labels = np.load('/media/data2/maleilei/MLIC/Query2Labels_base/contrastiveloss/target.npy')
    vis_embed = np.load('/media/data2/maleilei/MLIC/Query2Labels_base/contrastiveloss/vis_embed.npy')
    prototype_embed = np.load('/media/data2/maleilei/MLIC/Query2Labels_base/contrastiveloss/prototype_embed.npy')
    
    labels = torch.from_numpy(labels)
    vis_embed = torch.from_numpy(vis_embed)
    prototype_embed = torch.from_numpy(prototype_embed)

    features = torch.cat([vis_embed.unsqueeze(1), prototype_embed.unsqueeze(1)], dim=1)

    loss = criterion(features, labels)
    print(loss)

def sim():
    import numpy as np
    batch_size = 128
    views = 2
    classes = 80
    # criterion = SimMinLoss_all()
    criterion = SimMinLoss()
    
    labels = np.load('/media/data2/maleilei/MLIC/Query2Labels_base/contrastiveloss/target.npy')
    vis_embed = np.load('/media/data2/maleilei/MLIC/Query2Labels_base/contrastiveloss/vis_embed.npy')
    prototype_embed = np.load('/media/data2/maleilei/MLIC/Query2Labels_base/contrastiveloss/prototype_embed.npy')

    labels = torch.from_numpy(labels)
    vis_embed = torch.from_numpy(vis_embed)
    prototype_embed = torch.from_numpy(prototype_embed)

    loss = criterion(vis_embed, prototype_embed, labels)
    print(loss)



def intra_class_mask():
    batch_size = 3
    anchor_count = 2
    contrast_count = 2
    num_class = 5
    labels = [[1, 1, 0, 0, 1],
             [1, 0, 0, 1, 1],
             [0, 1, 0, 0, 1]
            ]
    # # [3, 5]
    labels = torch.tensor(labels, dtype=torch.float32)
    all_labels = torch.cat([labels, labels], dim=0)
    all_mask = all_labels.transpose(1, 0).unsqueeze(-1).matmul(all_labels.transpose(1, 0).unsqueeze(1))
    print(all_mask.shape)
    tmp_mask = labels.transpose(1, 0).unsqueeze(-1).matmul(labels.transpose(1, 0).unsqueeze(1))
    print(tmp_mask.shape)
    new_all_mask = tmp_mask.repeat(1, anchor_count, contrast_count)             # [bs*n_view*nc, bs*n_view*nc]
    print( (all_mask-new_all_mask).sum() )

def intra_class_mask_woself():
    batch_size = 3
    anchor_count = 2
    contrast_count = 2
    num_class = 5
    labels = [[1, 1, 0, 0, 1],
             [1, 0, 0, 1, 1],
             [0, 1, 0, 0, 1]
            ]
    # # [3, 5]
    labels = torch.tensor(labels, dtype=torch.float32)
    device = labels.device

    all_labels = torch.cat([labels, labels], dim=0)
    all_mask = all_labels.transpose(1, 0).unsqueeze(-1).matmul(all_labels.transpose(1, 0).unsqueeze(1))
    
    logits_mask = torch.scatter(
        torch.ones(batch_size * anchor_count, batch_size * anchor_count).to(device),
        1,
        torch.arange(batch_size * anchor_count).view(-1, 1),
        0
    )
    logits_mask = logits_mask.repeat(num_class, 1, 1)
    print(logits_mask)
    all_mask = all_mask * logits_mask


    tmp_mask = labels.transpose(1, 0).unsqueeze(-1).matmul(labels.transpose(1, 0).unsqueeze(1))
    new_all_mask = tmp_mask.repeat(1, anchor_count, contrast_count)             # [bs*n_view*nc, bs*n_view*nc]
    new_all_mask = new_all_mask * logits_mask
    print( (all_mask-new_all_mask).sum() )

def inter_class():
    
    num_class = 3
    batch_size = 2 
    anchor_count = 2
    contrast_count = 2

    labels = [
             [1, 1, 0],
             [1, 0, 1]
             ]
    labels = torch.tensor(labels, dtype=torch.float32)
    device = labels.device

    # index_label = labels.view(-1, 1)

    all_mask = labels.view(-1, 1)*labels.view(1, -1)
    # [1, 1, 0], [1, 0, 1]
    print(all_mask)
    all_mask = all_mask.repeat(anchor_count, contrast_count)             # [bs*n_view*nc, bs*n_view*nc]
    
    all_labels = torch.cat([labels, labels], dim=0)
    all_new_mask = all_labels.view(-1, 1)*all_labels.view(1, -1)

    print( (all_new_mask-all_mask).sum() )

if __name__ == '__main__':
    # sim()
    # main()

    # intra_class_mask()
    # intra_class_mask_woself()

    # inter_class()

    # self attention
    import torch.nn as nn
    self_attn = nn.MultiheadAttention(embed_dim=256, num_heads=4, dropout=0.1, kdim=256, vdim=256)

    q = torch.randn(80, 4, 256)
    k = torch.randn(80, 4, 256)
    v = torch.randn(80, 4, 256)

    out1 = self_attn(query=q, key=k, value=v)[0]
    
    # cross attention
    corss_attn = nn.MultiheadAttention(embed_dim=256, num_heads=4, dropout=0.1, kdim=512, vdim=256)

    q = torch.randn(80, 4, 512)
    k = torch.randn(80, 4, 512)
    v = torch.randn(80, 4, 256)

    out2 = corss_attn(query=q, key=k, value=v)[0]
    print(out2.shape)