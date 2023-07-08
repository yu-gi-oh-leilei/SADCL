import math
import torch
import torch.nn as nn
import torch.nn.functional as F

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
        '''
        P(i, j) = \left\{ Z_{k j} \in A(i,j) \mid y_{k j} = y_{i j} = 1 \right\}

        \sum_{z_{i,j} \in  I}
        \mathcal{L}_{L L C L}^{i j}=\frac{-1}{|P(i, j)|} \sum_{z_p \in P(i, j)} \log \frac{\exp \left(z_{i j} \cdot z_p / \tau\right)}{\sum_{z_a \in A(i, j)} \exp \left(z_{i j} \cdot z_a / \tau\right)} 
        '''
        # labels shape: [bs, n_view]

        device = features.device
        features = F.normalize(features, dim=-1)           # [bs, n_view, nc, dim]

        batch_size = features.shape[0]
        num_class = features.shape[2]

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)   # [n_view*bs, nc, dim]

        if self.contrast_mode == 'all':
            anchor_feature = contrast_feature                                # [256, 80, 2432]  [n_view*bs, nc, dim]
            anchor_count = contrast_count                                    # [256, 80, 2432]  [n_view*bs, nc, dim]
        # intra-class and inter-class 类内和类间

        # 同类不同图片之间的对比
        anchor_dot_contrast_intra = torch.div(
            torch.matmul(anchor_feature.transpose(1, 0), contrast_feature.permute(1, 2, 0)),
            self.temperature)                                                # [80, 256, 256]   [nc, n_view*bs, n_view*bs]


        # 所有特征都相互计算相似度
        # contrast_feature                                                   # [n_view*bs, nc, dim]
        all_features = contrast_feature.view(-1, contrast_feature.shape[-1]) # [n_view*bs*nc, dim]
        all_anchor_dot_contrast = torch.div(
            torch.matmul(all_features, all_features.transpose(1, 0)),
            self.temperature)                                                # [20480, 20480]    [n_view*bs*nc, dim, n_view*bs*nc, dim]

        # 生成同类不同图片的mask
        mask_intra = labels.transpose(1, 0).unsqueeze(-1).matmul(labels.transpose(1, 0).unsqueeze(1))    # [nc, bs, bs] 
        mask_intra = mask_intra.repeat(1, anchor_count, contrast_count)      # [nc, n_view*bs, n_view*bs] 


        # 所有的特征的mask
        all_mask = torch.matmul(labels.contiguous().view(-1, 1), labels.contiguous().view(1, -1))
        all_mask = all_mask.repeat(anchor_count, contrast_count)             # [n_view*bs*nc, n_view*bs*nc]


        # 同类不同图片之间，去掉自身的mask
        logits_mask_intra = torch.scatter(
            torch.ones(batch_size * anchor_count, batch_size * anchor_count).to(device),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0)                                                               # [80, 256, 256]  [n_view*bs, n_view*bs]
        logits_mask_intra = logits_mask_intra.repeat(num_class, 1, 1)
        mask_intra = mask_intra * logits_mask_intra                          # [n_view*bs*nc, n_view*bs*nc]
        logits_intra = mask_intra * anchor_dot_contrast_intra                # [80, 256, 256]   [nc, n_view*bs, n_view*bs]

        # 同类不同图片之间, 为了数值稳定
        logits_max, _ = torch.max(logits_intra.permute(1, 2, 0), dim=-1, keepdim=True)      # [n_view*bs, n_view*bs, 1]
        logits_intra = (logits_intra.permute(1, 2, 0) - logits_max.detach()).permute(2, 0, 1)

        # import ipdb; ipdb.set_trace()
        # 所有的类，去掉自身
        self_mask = torch.eye(batch_size*num_class*contrast_count).to(device)
        self_mask = torch.ones_like(self_mask) - self_mask
        all_mask = all_mask * self_mask


        all_anchor_dot_contrast = all_anchor_dot_contrast * all_mask


        # 所有特征, 为了数值稳定
        # all_anchor_dot_contrast 已经被mask
        logits_max_all, _ = torch.max(all_anchor_dot_contrast, dim=-1, keepdim=True)
        logits_all = all_anchor_dot_contrast - logits_max_all.detach()

    
        # label mask
        labels_mask = labels.unsqueeze(1).repeat(1, contrast_count, 1).contiguous().view(-1, 1) # [n_view*bs*nc, 1]

        # 分母 [n_view*bs*nc, n_view*bs*nc]
        logits_all = all_mask * torch.exp(logits_all) # [n_view*bs*nc, n_view*bs*nc] 
        logits_all = logits_all.sum(-1, keepdim=True) # [n_view*bs*nc, 1] 
        logits_all[logits_all<=0] = 1
        logits_all = labels_mask * torch.log(logits_all+self.eps) # [n_view*bs*nc, 1]


        # 分子 [n_view*bs*nc, n_view*bs]   [nc, n_view*bs, n_view*bs] => [n_view*bs*nc, n_view*bs]
        logits_intra = (logits_intra).permute(1, 0, 2).reshape(contrast_count*batch_size*num_class, contrast_count*batch_size)
        mask_intra = (mask_intra).permute(1, 0, 2).reshape(contrast_count*batch_size*num_class, contrast_count*batch_size)

        # 计算对数似然除以正的均值
        log_prob = logits_intra - logits_all            # [n_view*bs*nc, n_view*bs] - [n_view*bs*nc, 1] => [n_view*bs*nc, n_view*bs] 


        # mask_intra 对应论文中的 P(i, j) => [n_view*bs*nc, n_view*bs]
        log_prob = (mask_intra * log_prob).sum(-1) # [n_view*bs*nc, 1]
        mask_intra = mask_intra.sum(-1)
        mask_intra[mask_intra==0] = 1
        mean_log_prob_pos = log_prob / mask_intra

        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos # [n_view*bs*nc]


        # 所有存在的示例计算平均，计算 （\mathcal{L}_{L L C L}^{i j}).sum(i, j) / |count(i,j)|
        labels_mask = labels_mask.view(-1)
        if labels_mask.sum() == 0.0:
            loss = (loss * labels_mask).sum()
        else:
            loss = (loss * labels_mask).sum() / labels_mask.sum()
   
        return loss


def main():
    import numpy as np
    batch_size = 128
    views = 2
    classes = 80
    criterion = MultiSupConLoss()

    labels = np.load('/media/data2/maleilei/MLIC/Query2Labels_base/data/contrastiveloss/target.npy')
    # print(labels.shape)
    # print(labels.sum(dim=0).shape)
    # labels = labels.sum(dim=0)
    # labels[labels > 1] = 1
    # print(labels)
    vis_embed = np.load('/media/data2/maleilei/MLIC/Query2Labels_base/data/contrastiveloss/vis_embed.npy')
    prototype_embed = np.load('/media/data2/maleilei/MLIC/Query2Labels_base/data/contrastiveloss/prototype_embed.npy')
    
    labels = torch.from_numpy(labels)
    vis_embed = torch.from_numpy(vis_embed)
    prototype_embed = torch.from_numpy(prototype_embed)

    

    features = torch.cat([vis_embed.unsqueeze(1), prototype_embed.unsqueeze(1)], dim=1)

    loss = criterion(features, labels)
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
    print(tmp_mask.shape) # [n, bs, bs] [5, 3, 3]
    print(labels.view(1, -1))
    print(tmp_mask)
    print(tmp_mask.permute(1, 0, 2))
    print(tmp_mask.permute(1, 0, 2).reshape(batch_size*num_class, batch_size))
    print(tmp_mask.reshape(batch_size*num_class, batch_size))
    
    
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


    
    print(all_mask.shape) 
    
    all_labels = torch.cat([labels, labels], dim=0)
    all_new_mask = all_labels.view(-1, 1)*all_labels.view(1, -1)

    print( (all_new_mask-all_mask).sum() )


    tmp = torch.matmul(all_labels.view(-1, 1), all_labels.view(1, -1))
    print( (tmp-all_mask).sum() )




def test():
    print('==='*30)
    import numpy as np
    device = 'cuda:2'
    scaler = torch.cuda.amp.GradScaler(enabled=True) 
    criterion = MultiSupConLoss()

    labels = np.load('/media/data2/maleilei/MLIC/Query2Labels_base/debug/target.1.npy')
    vis_embed = np.load('/media/data2/maleilei/MLIC/Query2Labels_base/debug/vis_embed.1.npy')
    memorybank = np.load('/media/data2/maleilei/MLIC/Query2Labels_base/debug/memorybank.1.npy')
    
    labels = torch.from_numpy(labels)
    vis_embed = torch.from_numpy(vis_embed)
    memorybank = torch.from_numpy(memorybank)

    labels = labels.to(device)
    vis_embed = vis_embed.to(device)
    memorybank = memorybank.to(device)
    criterion = criterion.to(device)

    # print(vis_embed)
    # print(memorybank)
    # print(vis_embed)
    print(labels.sum())
    print(labels.shape)

    target = labels.clone().view(-1, 1)

    print(64 * 81)
    print(target.sum(dim=-1, keepdim=False))
    target = target.sum(dim=-1, keepdim=False)
    # print(target.sum(dim=-1, keepdim=False))
    print(target.shape)
    one_count = 0
    zero_count = 0
    for index in  range(64 * 81):
        if target[index] == 1:
            one_count = one_count + 1

        if target[index] == 0:
            zero_count = zero_count + 1
        
        if target[index] != 0 and target[index] != 1:
            print(target[index])

    print(one_count)
    print(zero_count)
        # print(target[index])
    
    print(target.sum())
    
    print('==========')


    # 

    assert torch.isnan(vis_embed).sum() == 0, print(vis_embed)
    assert torch.isnan(memorybank).sum() == 0, print(memorybank)

    assert not torch.any(torch.isnan(vis_embed))
    assert not torch.any(torch.isnan(memorybank))
    
    
    assert not torch.any(torch.isinf(vis_embed))
    assert not torch.any(torch.isinf(memorybank))


    features = torch.cat([vis_embed.unsqueeze(1), memorybank.unsqueeze(1)], dim=1)



    with torch.cuda.amp.autocast(enabled=True):
        loss = criterion(features, labels)
        print(loss)



if __name__ == '__main__':
    test()
    