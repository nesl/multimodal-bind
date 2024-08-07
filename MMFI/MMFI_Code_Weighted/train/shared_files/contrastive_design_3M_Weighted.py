"""
Author: Yonglong Tian (yonglong@mit.edu)
Date: May 07, 2020
"""
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def FeatureConstructor(f1, f2, f3, num_positive):

    fused_feature = []

    fused_feature.append(f1)
    fused_feature.append(f2)
    fused_feature.append(f3)
    
    fused_feature = torch.stack(fused_feature, dim = 1)

    return fused_feature


## contrastive loss with supervised format

class ConFusionLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(ConFusionLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature
        self.incomplete_weighted = nn.Parameter(torch.FloatTensor([0.1]))

    def forward(self, features, similarity, labels=None, mask=None):
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
       
        mask = torch.eye(batch_size, dtype=torch.float32).to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)# change to [n_views*bsz, 3168]
        contrast_feature = F.normalize(contrast_feature, dim = 1)

        # print(contrast_feature.shape)

        anchor_feature = contrast_feature
        anchor_count = contrast_count

        # compute logits, z_i * z_a / T
        similarity_matrix = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)

        # for numerical stability
        # similarity_matrix = F.normalize(similarity_matrix, p=2, dim = 1)

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)# positive index
        # print(mask.shape)#[1151, 1152] (btz*9)

        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )#dig to 0, others to 1 (negative samples)
        # print(logits_mask.shape)

        mask = mask * logits_mask#positive samples except itself

        # compute log_prob
        exp_logits = torch.exp(similarity_matrix) * logits_mask #exp(z_i * z_a / T)
        # all_log_prob = torch.log(exp_logits.sum(1))# log(sum(exp(z_i * z_a / T))), need change to I\{i} later

        # SupCon out
        log_prob = similarity_matrix - torch.log(exp_logits.sum(1, keepdim=True)) # 3b_size x 3bsize
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)#sup_out

        # SupCon in
        # log_prob =  torch.exp(similarity_matrix) / exp_logits.sum(1, keepdim=True)
        # mean_log_prob_pos = torch.log((mask * log_prob).sum(1) / mask.sum(1))
        indices = torch.where(similarity != 0.1)
        similarity[indices] = (similarity[indices] - 0.5) * 3
        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos * similarity.repeat(3)
        # print("loss:",loss)

        loss = loss.view(anchor_count, batch_size).mean()
        # print("mean loss:",loss)

        return loss


## contrastive center of negative samples

# class ConFusionLoss(nn.Module):

#     def __init__(self, temperature=0.07, contrast_mode='all',
#                  base_temperature=0.07):
#         super(ConFusionLoss, self).__init__()
#         self.temperature = temperature
#         self.contrast_mode = contrast_mode
#         self.base_temperature = base_temperature

#     def forward(self, features, labels=None, mask=None):

#         device = (torch.device('cuda')
#                   if features.is_cuda
#                   else torch.device('cpu'))

#         # print("original features:",features) 
#         batch_size = features.shape[0]

#         batch_mask = torch.eye(batch_size, dtype=torch.float32).to(device)

#         contrast_count = features.shape[1]#[bsz, n_views, 3168]
#         center_feature = features.mean(1).to(device)
#         center_feature = F.normalize(center_feature, dim = 1)
#         # print("center_feature shape:",center_feature.shape)


#         contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)# change to [n_views*bsz, 3168]
#         contrast_feature = F.normalize(contrast_feature, dim = 1)
#         # print(contrast_feature.shape)

#         anchor_feature = contrast_feature.to(device)
#         anchor_count = contrast_count

#         # compute logits, z_i * z_a / T
#         similarity_matrix = torch.div(
#             torch.matmul(anchor_feature, contrast_feature.T),
#             self.temperature)
#         # similarity_matrix = F.normalize(similarity_matrix, dim = 1)

#         # compute center logits, z_i * z_c / T
#         center_similarity_matrix = torch.div(
#             torch.matmul(anchor_feature, center_feature.T),
#             self.temperature)
#         # center_similarity_matrix = F.normalize(center_similarity_matrix, dim = 1)   

#         # tile mask
#         mask = batch_mask.repeat(anchor_count, contrast_count)# positive index
#         # print(mask.shape)#[1151, 1152] (btz*9)

#         # mask-out self-contrast cases
#         logits_mask = torch.scatter(
#             torch.ones_like(mask),
#             1,
#             torch.arange(batch_size * anchor_count).view(-1, 1),
#             0
#         )#dig to 0, others to 1
#         mask = mask * logits_mask#positive masks

#         zero_mask = torch.zeros_like(batch_mask).to(device)
#         one_mask = torch.ones_like(batch_mask).to(device)
#         negative_batch_mask = torch.where(batch_mask<0.5, one_mask, zero_mask).to(device)
#         # print("negative_batch_mask:", negative_batch_mask)

#         center_mask = negative_batch_mask.repeat(anchor_count, 1)# negative index
#         # print("center_mask.shape", center_mask.shape)#[1151, 128] (btz*9, btz)

#         # compute log_prob
#         exp_logits = torch.exp(center_similarity_matrix) * center_mask #exp(z_i * z_a / T)
#         log_prob = similarity_matrix - torch.log(exp_logits.sum(1))
#         mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)
#         # print("mean_log_prob_pos:",mean_log_prob_pos)

#         # loss
#         loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
#         # print("loss:",loss)

#         loss = loss.view(anchor_count, batch_size).mean()
#         # print("mean loss:",loss)

#         return loss


## contrastive loss with self-supervised format
# class ConFusionLoss(nn.Module):
#     """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
#     It also supports the unsupervised contrastive loss in SimCLR"""
#     def __init__(self, temperature=0.07, contrast_mode='all',
#                  base_temperature=0.07):
#         super(ConFusionLoss, self).__init__()
#         self.temperature = temperature
#         self.contrast_mode = contrast_mode
#         self.base_temperature = base_temperature
#         self.criterion = torch.nn.CrossEntropyLoss().cuda()

#     def forward(self, features):

#         device = (torch.device('cuda')
#                   if features.is_cuda
#                   else torch.device('cpu'))

#         batch_size = features.shape[0]
#         n_views = features.shape[1]

#         labels = torch.cat([torch.arange(batch_size) for i in range(n_views)], dim=0)
#         labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
#         labels = labels.to(device)

#         features = torch.cat(torch.unbind(features, dim=1), dim=0)# change to [n_views*bsz, 3168]
#         features = F.normalize(features, dim=1) # |features|=1

#         similarity_matrix = torch.matmul(features, features.T) # s
#         # assert similarity_matrix.shape == (
#         #     self.args.n_views * self.args.batch_size, self.args.n_views * self.args.batch_size)
#         # assert similarity_matrix.shape == labels.shape

#         # discard the main diagonal from both: labels and similarities matrix
#         mask = torch.eye(labels.shape[0], dtype=torch.bool).to(device)
#         labels = labels[~mask].view(labels.shape[0], -1) #labels[0][31]=labels[1][32]=1
#         similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)
#         # assert similarity_matrix.shape == labels.shape

#         # select and combine multiple positives
#         positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

#         # select only the negatives
#         negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

#         logits = torch.cat([positives, negatives], dim=1)
#         logits = logits / self.temperature

        # loss = 0
        # for view_id in range(n_views-1):
        #     labels = (torch.ones(logits.shape[0], dtype=torch.long) * view_id).to(device)
        #     # print("labels:", labels)#label = 0 (contrast with the first view)
        #     loss += criterion(logits, labels)
        #     # print("loss:", loss)
        # loss = loss / (n_views-1)
#         labels = torch.zeros(logits.shape[0], dtype=torch.long).to(device)
#         loss = self.criterion(logits, labels)

#         return loss

