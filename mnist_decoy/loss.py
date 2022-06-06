# def loss_gradient(self,gradients,log_soft_out,explanation2,margin):
#         s = .0
#         # print('len grad is - --',len(gradients))
#         for i in range(gradients.size()[0]):
#             y_hat = torch.sum(log_soft_out[i])
#             gr = gradients[i]
#             # grad_yhat = y_hat * gr
#             grad_mul =  gr * (1-explanation2)
#             # grad_mul_missing = (margin - grad_yhat) * explanation2
#             grad_mul = grad_mul **2
#             # grad_mul_missing = grad_mul_missing**2
#             grad_mul = torch.sum(grad_mul)
#             # grad_mul_missing = torch.sum(grad_mul_missing)
#             s+=grad_mul



import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

class OhemCELoss(nn.Module):
    def __init__(self, thresh, n_min, ignore_lb=255, *args, **kwargs):
        super(OhemCELoss, self).__init__()
        self.thresh = -torch.log(torch.tensor(thresh, dtype=torch.float)).cuda()
        self.n_min = n_min
        self.ignore_lb = ignore_lb
        self.criteria = nn.CrossEntropyLoss(ignore_index=ignore_lb, reduction='none')

    def forward(self, logits, labels):
        N, C, H, W = logits.size()
        loss = self.criteria(logits, labels).view(-1)
        loss, _ = torch.sort(loss, descending=True)
        if loss[self.n_min] > self.thresh:
            loss = loss[loss>self.thresh]
        else:
            loss = loss[:self.n_min]
        return torch.mean(loss)
class input_grad_boundry_loss(nn.Module):
    def __init__(self, thresh, n_min, ignore_lb=255, *args, **kwargs):
        super(OhemCELoss, self).__init__()
        self.thresh = -torch.log(torch.tensor(thresh, dtype=torch.float)).cuda()
        self.n_min = n_min
        self.ignore_lb = ignore_lb
        self.criteria = nn.CrossEntropyLoss(ignore_index=ignore_lb, reduction='none')

    def forward(self, logits, labels):
        N, C, H, W = logits.size()
        loss = self.criteria(logits, labels).view(-1)
        loss, _ = torch.sort(loss, descending=True)
        if loss[self.n_min] > self.thresh:
            loss = loss[loss>self.thresh]
        else:
            loss = loss[:self.n_min]
        return torch.mean(loss)
