import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


# class ContrastiveLoss(nn.Module):
#     """
#     Document: https://github.com/adambielski/siamese-triplet/blob/master/losses.py
#     """

#     def __init__(self, batchSize, reduce=None, size_average=None):
#         super(ContrastiveLoss, self).__init__()

#         self.batchSize = batchSize
#         self.concatIndex = self.getConcatIndex(batchSize)

#         self.reduce = reduce
#         self.size_average = size_average

#         self.cos = torch.nn.CosineSimilarity(dim=2, eps=1e-9)

#     def forward(self, input, target):
#         """
#         Shape of input: (BatchSize, classNum, featureDim)
#         Shape of target: (BatchSize, classNum), Value range of target: (-1, 0, 1)
#         postive=1, unknow=0, negtive=-1
#         """
            
#         target_ = target.detach().clone()
#         target_[target_ != 1] = 0
#         pos2posTarget = target_[self.concatIndex[0]] * target_[self.concatIndex[1]]

#         pos2negTarget = 1 - pos2posTarget
#         pos2negTarget[(target[self.concatIndex[0]] == 0) | (target[self.concatIndex[1]] == 0)] = 0
#         pos2negTarget[(target[self.concatIndex[0]] == -1) & (target[self.concatIndex[1]] == -1)] = 0

#         target_ = -1 * target.detach().clone()
#         target_[target_ != 1] = 0
#         neg2negTarget = target_[self.concatIndex[0]] * target_[self.concatIndex[1]]

#         distance = self.cos(input[self.concatIndex[0]], input[self.concatIndex[1]])

#         if self.reduce:
#             pos2pos_loss = (1 - distance)[pos2posTarget == 1]
#             pos2neg_loss = (1 + distance)[pos2negTarget == 1]
#             neg2neg_loss = (1 + distance)[neg2negTarget == 1]

#             if pos2pos_loss.size(0) != 0:
#                 if neg2neg_loss.size(0) != 0:
#                     neg2neg_loss = torch.cat((torch.index_select(neg2neg_loss, 0, torch.randperm(neg2neg_loss.size(0))[:2 * pos2pos_loss.size(0)].cuda()),
#                                               torch.sort(neg2neg_loss, descending=True)[0][:pos2pos_loss.size(0)]), 0)
#                 if pos2neg_loss.size(0) != 0:
#                     if pos2neg_loss.size(0) != 0:    
#                         pos2neg_loss = torch.cat((torch.index_select(pos2neg_loss, 0, torch.randperm(pos2neg_loss.size(0))[:2 * pos2pos_loss.size(0)].cuda()),
#                                                   torch.sort(pos2neg_loss, descending=True)[0][:pos2pos_loss.size(0)]), 0)

#             loss = torch.cat((pos2pos_loss, pos2neg_loss, neg2neg_loss), 0)

#             if self.size_average:
#                 return torch.mean(loss) if loss.size(0) != 0 else torch.mean(torch.zeros_like(loss).cuda())
#             return torch.sum(loss) if loss.size(0) != 0 else torch.sum(torch.zeros_like(loss).cuda())
 
#         return distance

#     def getConcatIndex(self, classNum):
#         res = [[], []]
#         for index in range(classNum - 1):
#             res[0] += [index for i in range(classNum - index - 1)]
#             res[1] += [i for i in range(index + 1, classNum)]
#         return 

class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf
        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss

if __name__ == '__main__':
    import numpy as np
    batch_size = 128
    views = 2
    classes = 80
    criterion = SupConLoss()

    # labels = np.load('/media/data2/maleilei/MLIC/Query2Labels_base/contrastiveloss/target.npy')
    # vis_embed = np.load('/media/data2/maleilei/MLIC/Query2Labels_base/contrastiveloss/vis_embed.npy')
    # prototype_embed = np.load('/media/data2/maleilei/MLIC/Query2Labels_base/contrastiveloss/prototype_embed.npy')
    
    # labels = torch.from_numpy(labels)
    # vis_embed = torch.from_numpy(vis_embed)
    # prototype_embed = torch.from_numpy(prototype_embed)
    # print(labels.shape)          # 128, 80
    # print(vis_embed.shape)       # 128, 80, 2432
    # print(prototype_embed.shape) # 128, 80, 2432

    # labels = labels.view(labels.shape[0], labels.shape[1], -1)
    # vis_embed = vis_embed.view(-1, vis_embed.shape[2])
    # prototype_embed = prototype_embed.view(-1, prototype_embed.shape[2])







    features = torch.cat([vis_embed.unsqueeze(1), prototype_embed.unsqueeze(1)], dim=1)

    loss = criterion(features, labels)
    print(loss)


