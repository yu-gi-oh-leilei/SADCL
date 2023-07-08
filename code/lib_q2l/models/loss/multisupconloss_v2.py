import torch
import torch.nn as nn

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
        mask_intra = mask_intra.repeat(1, anchor_count, contrast_count)                  # [nc, bs*n_view, bs*n_view] 

        # 所有的特征的mask
        all_mask = labels.view(-1, 1)*labels.view(1, -1)
        all_mask = all_mask.repeat(anchor_count, contrast_count)             # [nc, bs*n_view, bs*n_view]


        # 同类不同图片之间，去掉自身
        logits_mask_intra = torch.scatter(
            torch.ones(batch_size * anchor_count, batch_size * anchor_count).to(device),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0) #
        logits_mask_intra = logits_mask_intra.repeat(num_class, 1, 1)
        mask_intra = mask_intra * logits_mask_intra

        logits_intra = mask_intra * anchor_dot_contrast_intra                # [80, 256, 256]   [nc, bs*n_view, bs*n_view]

        # 所有特征都相互计算相似度
        all_features = contrast_feature.view(-1, contrast_feature.shape[-1]) # [nc*bs*n_view, dim]
        all_anchor_dot_contrast = torch.div(
            torch.matmul(all_features, all_features.transpose(1, 0)),
            self.temperature)                                                # [20480, 20480]    [nc*bs*n_view]

        # 所有特征, 为了数值稳定
        all_anchor_dot_contrast = all_anchor_dot_contrast.reshape(batch_size*contrast_count, num_class, num_class, batch_size*contrast_count)
        logits_max_all, _ = torch.max(all_anchor_dot_contrast, dim=-1, keepdim=True)
        logits_all = all_anchor_dot_contrast - logits_max_all.detach()
        logits_all = logits_all.reshape(batch_size*contrast_count*num_class, batch_size*contrast_count*num_class)
        # import ipdb; ipdb.set_trace()

        # 所有的类，去掉自身
        self_mask = torch.eye(batch_size*num_class*contrast_count).to(device)
        self_mask = torch.ones_like(self_mask) - self_mask
        all_mask = all_mask * self_mask
        logits_all = torch.exp(logits_all) * self_mask

        # print(torch.log(logits_all.sum(1, keepdim=True)).shape)
        
        # 分母
        logits_all = all_mask*torch.log(logits_all.sum(1, keepdim=True))
        print(logits_all.shape)

        # 分子
        logits_intra = (mask_intra*logits_intra).transpose(1, 0).reshape(batch_size*contrast_count*num_class, batch_size*contrast_count)
        print(logits_intra.shape)


        log_prob = logits_intra - logits_all.sum(1, keepdim=True)

        mean_log_prob_pos = (log_prob).sum() / all_mask.sum()

        print(mean_log_prob_pos)

        # all_mask.sum()
        # temp = all_mask.sum(1 ,keepdim=True)
        # for index in range(0, 20480):
        #     if temp[index][0] == torch.zeros(1):
        #         print(temp[index][0])

        # print(log_prob.shape)


        # 计算loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(batch_size, anchor_count, num_class).mean()
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

if __name__ == '__main__':
    main()