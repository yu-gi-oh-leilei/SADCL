import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# class Semantic2PrototypeConLoss(nn.Module):
#     def __init__(self, metric='cos', reduction='mean', temperature=0.07):
#         super(Semantic2PrototypeConLoss, self).__init__()
#         self.temperature = temperature
#         self.metric = metric
#         self.reduction = reduction
#         self.eps = 1e-5
        
#     def forward(self, semantic_embed, proto_embed, target):
#         """
#         :param semantic_embed: [bs, n, dim]
#         :param proto_embed:    [n, dim] or [1, n, dim]
#         :param target:         [bs, num]
#         :return:
#         """
#         if proto_embed.dim() != 3:
#             proto_embed = proto_embed.unsqueeze(0)

#         bs, num_c, dim = semantic_embed.shape                                        
#         semantic_embed = semantic_embed.permute(1, 0, 2)                                # [n, bs, dim]
#         proto_embed    = proto_embed.repeat(bs, 1, 1).permute(1, 0, 2)                  # [n, bs, dim] 
        
#         target         = target.permute(1, 0)                                           # [n, bs] 
#         pos_mask = target                                                               # [n, bs] 
#         neg_mask = 1- target                                                            # [n, bs] 


#         cos_sim = F.cosine_similarity(semantic_embed, proto_embed, dim=-1) / self.temperature # [n, bs]

#         # 为了数值稳定
#         # cos_sim_max, _ = torch.max(cos_sim, dim=-1, keepdim=True)                         # [n, bs]
#         # cos_sim = cos_sim - cos_sim_max.detach()                                          # [n, bs]

#         pos2proto = cos_sim * pos_mask                                       # 正类与原型对比  # [n, bs]
#         neg2proto = cos_sim * neg_mask                                       # 负类与原型对比  # [n, bs]


#         # 分母
#         neg2proto = torch.exp(neg2proto)                                                      # [n, bs]
#         neg2proto = neg2proto * neg_mask                                                      # [n, bs] # 保留负类与原型对比，mask掉正类  
#         neg2proto = neg2proto.sum(dim=-1, keepdim=True)                                       # [n, 1]

#         singlepos_neg2proto = pos_mask * (torch.exp(pos2proto) + neg2proto)
#         singlepos_neg2proto[singlepos_neg2proto<=0] = 1.0
#         singlepos_neg2proto = torch.log(singlepos_neg2proto)                                  # [n, bs]


#         # 分子减去分母
#         log_prob = pos_mask * (pos2proto - singlepos_neg2proto)
#         log_prob = log_prob.sum(dim=-1, keepdim=True)

#         # 计算对数似然除以正的均值
#         if pos_mask.sum() != 0:
#             loss = - log_prob.sum() / pos_mask.sum()
#         else:
#             loss = - log_prob.meam()

#         return loss


# v1 版本 
# contras_loss1 = 100*criterion['contrasloss'](features_vis_memorybank, target)
# contras_loss2 = 5*criterion['sem2pro'](vis_embed, prototype, target)
class Semantic2PrototypeConLoss(nn.Module):
    def __init__(self, metric='cos', reduction='mean', temperature=0.07):
        super(Semantic2PrototypeConLoss, self).__init__()
        self.temperature = temperature
        self.metric = metric
        self.reduction = reduction
        self.eps = 1e-5
        
    def forward(self, semantic_embed, proto_embed, target):
        """
        :param semantic_embed: [bs, n, dim]
        :param proto_embed:    [n, dim] or [1, n, dim]
        :param target:         [bs, num]
        :return:
        """
        if proto_embed.dim() != 3:
            proto_embed = proto_embed.unsqueeze(0)

        bs, num_c, dim = semantic_embed.shape                                        
        semantic_embed = semantic_embed.permute(1, 0, 2)                                # [n, bs, dim]
        proto_embed    = proto_embed.repeat(bs, 1, 1).permute(1, 0, 2)                  # [n, bs, dim] 
        
        target         = target.permute(1, 0)                                           # [n, bs] 
        pos_mask = target                                                               # [n, bs] 
        neg_mask = 1- target                                                            # [n, bs] 

        # target =  target.unsqueeze(-1)                                                      # [n, bs, 1]
        cos_sim = F.cosine_similarity(semantic_embed, proto_embed, dim=-1) / self.temperature # [n, bs]
        
        # 为了数值稳定
        cos_sim_max, _ = torch.max(cos_sim, dim=-1, keepdim=True)                         # [n, bs]
        cos_sim = cos_sim - cos_sim_max.detach()                                        # [n, bs]


        pos2proto = cos_sim * pos_mask                                       # 正类与原型对比  # [n, bs]
        # neg2proto = cos_sim * neg_mask                                       # 负类与原型对比  # [n, bs]
        neg2proto = cos_sim

        # 分母
        neg2proto = torch.exp(neg2proto)                                                      # [n, bs]
        # neg2proto = neg2proto * neg_mask                                                      # [n, bs] # 保留负类与原型对比，mask掉正类  

        neg2proto = neg2proto.sum(dim=-1, keepdim=True)                                       # [n, 1]
        neg2proto[neg2proto<=0] = 1.0                                                         # [n, 1]  # 优于被mask部分为0，消除求和后为0的部分
        neg2proto = torch.log(neg2proto)                                                      # [n, 1]
        
        # 分子
        pos2proto = torch.exp(pos2proto)                                                      # [n, bs]
        pos2proto = pos2proto * pos_mask                                                      # [n, bs] # 保留负类与原型对比，mask掉正类  

        pos2proto = pos2proto.sum(dim=-1, keepdim=True)                                       # [n, 1]
        pos2proto[pos2proto<=0] = 1.0                                                         # [n, 1]  # 优于被mask部分为0，消除求和后为0的部分
        pos2proto = torch.log(pos2proto)                                                      # [n, 1]
        
        
        # 分子减去分母
        log_prob = (pos2proto - neg2proto).sum(dim=-1, keepdim=True) 


        # 计算对数似然除以正的均值
        if pos_mask.sum() != 0:
            loss = - log_prob.sum() / pos_mask.sum()
        else:
            loss = - log_prob.meam()
        
        return loss
        




        # neg2proto = torch.log(neg2proto)                                                      # [n, 1]
        

        # # 分子
        # pos2proto = self.eps * torch.ones_like(pos2proto) + pos2proto
        
        
        # # 分子减去分母
        # log_prob = (pos2proto - neg2proto).sum(dim=-1, keepdim=True) 
        # # print(log_prob)

        # # 计算对数似然除以正的均值
        # loss =  - log_prob.mean()
        
        # return loss
        

        # # 分子减去分母，然后保留正类与原型对比
        # log_prob = (pos_mask * (pos2proto - neg2proto)).sum(dim=-1, keepdim=True)             # [n, 1]
        # pos_mask_count = pos_mask.sum(dim=-1, keepdim=True)
        # pos_mask_count[pos_mask_count<=0] = 1.0


        # # 计算对数似然除以正的均值
        # log_prob = log_prob / pos_mask_count

        # log_prob = log_prob.mean() 


        # print(log_prob.shape)
        # 每个类别对应的对比损失，然后求和
        # pos_mask = pos_mask.sum(dim=-1, keepdim=True)
        # pos_mask[pos_mask > 1] = 1                        # N个类中，被激活的数目为n个，以80个类别的COCO为例， n <= 80, 
        # # print(pos_mask.sum())    73
        # log_prob = pos_mask * log_prob
        # log_prob = log_prob.sum() / pos_mask.sum()

        # loss = - log_prob

        # return loss
# neg2proto[neg2proto<=0] = 1.0                                                         # [n, 1]  # 优于被mask部分为0，消除求和后为0的部分

def semantic_prototype():
    import numpy as np
    batch_size = 128
    views = 2
    classes = 80
    # criterion = SimMinLoss_all()
    criterion = Semantic2PrototypeConLoss()
    
    labels = np.load('/media/data2/maleilei/MLIC/Query2Labels_base/data/contrastiveloss/target.npy')
    vis_embed = np.load('/media/data2/maleilei/MLIC/Query2Labels_base/data/contrastiveloss/vis_embed.npy')
    prototype_embed = np.load('/media/data2/maleilei/MLIC/Query2Labels_base/data/prototype/vis_prototype_embed_file_select_tresnetv1_448_coco_085.npy')
    labels = torch.from_numpy(labels)
    vis_embed = torch.from_numpy(vis_embed)
    prototype_embed = torch.from_numpy(prototype_embed)
    # print(labels.shape)
    # print(vis_embed.shape)
    # print(prototype_embed.shape)


    loss = criterion(vis_embed, prototype_embed, labels)
    print(loss)


def test():
    import numpy as np
    scaler = torch.cuda.amp.GradScaler(enabled=True) 
    criterion = Semantic2PrototypeConLoss()
    
    device = 'cuda:2'

    # embedding = np.load('/media/data2/maleilei/MLIC/Query2Labels_base/debug/embedding.1.npy')
    # embedding = torch.from_numpy(embedding)
    # print(embedding)

    labels_0 = np.load('/media/data2/maleilei/MLIC/Query2Labels_base/debug2/target.0.npy')
    vis_embed_0 = np.load('/media/data2/maleilei/MLIC/Query2Labels_base/debug2/vis_embed.0.npy')
    prototype_embed_0 = np.load('/media/data2/maleilei/MLIC/Query2Labels_base/debug2/prototype.0.npy')

    labels_1 = np.load('/media/data2/maleilei/MLIC/Query2Labels_base/debug2/target.1.npy')
    vis_embed_1 = np.load('/media/data2/maleilei/MLIC/Query2Labels_base/debug2/vis_embed.1.npy')
    prototype_embed_1 = np.load('/media/data2/maleilei/MLIC/Query2Labels_base/debug2/prototype.1.npy')


    labels_0 = torch.from_numpy(labels_0)#.float()
    vis_embed_0 = torch.from_numpy(vis_embed_0)#.float()
    prototype_embed_0 = torch.from_numpy(prototype_embed_0)#.float()
    

    labels_1 = torch.from_numpy(labels_1)#.float()
    vis_embed_1 = torch.from_numpy(vis_embed_1)#.float()
    prototype_embed_1 = torch.from_numpy(prototype_embed_1)#.float()

    # print(labels_1.shape, vis_embed_1.shape, prototype_embed_1.shape)


    labels = torch.cat((labels_0, labels_1), dim=0)
    vis_embed = torch.cat((vis_embed_0, vis_embed_1), dim=0)
    # prototype_embed = torch.cat((prototype_embed_0, prototype_embed_1), dim=0)
    prototype_embed = prototype_embed_0
    labels = labels_0
    vis_embed = vis_embed_0
    # print(labels.shape, vis_embed.shape, prototype_embed.shape)

    # labels = torch.from_numpy(labels_1)#.float()
    # vis_embed = torch.from_numpy(vis_embed_1)#.float()
    # prototype_embed = torch.from_numpy(prototype_embed_1)#.float()

    # print(labels.shape)
    # print(vis_embed.shape)
    # print(prototype_embed.shape)



    labels = labels.to(device)
    vis_embed = vis_embed.to(device)
    prototype_embed = prototype_embed.to(device)


    # assert torch.isnan(vis_embed).sum() == 0, print('vis_embed')
    # assert torch.isnan(prototype_embed).sum() == 0, print('prototype_embed')

    # torch.isnan()
    
    # print()


    criterion = criterion.to(device)


    # print(vis_embed)
    # print(prototype_embed)
    
    # print(labels.shape)
    # print(vis_embed.shape)
    # print(prototype_embed.shape)

    with torch.cuda.amp.autocast(enabled=True):
        loss = criterion(vis_embed, prototype_embed, labels)
    # loss = criterion(vis_embed, prototype_embed, labels)
    print(loss)


    # labels = torch.from_numpy(labels).float()
    # vis_embed = torch.from_numpy(vis_embed).float()
    # prototype_embed = torch.from_numpy(prototype_embed).float()

if __name__ == '__main__':
    test()
    # semantic_prototype()
