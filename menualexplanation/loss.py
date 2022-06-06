# -*- coding: utf-8 -*-
"""
Created on Sun Dec 13 23:43:27 2020

@author: Dimo
"""

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


"""
Minimizing Mahalanobis distance between related pairs, and maximizing between negative pairs.

A loss typically used for creating a Euclidian embedding space for a wide variety of supervised learning problems.
The original implementation was by Davis King @ Dlib.

PyTorch Implementation: https://gist.github.com/bkj/565c5e145786cfd362cffdbd8c089cf4

Made changes such that accuracy is provided on a forward pass as well.
"""

import torch
import torch.nn.functional as F
from torch import nn

#This loss will focus on possitive part of gradcam
#pass to relu and sigmoid
class BCE_possitive_loss(nn.Module):

    def __init__(self,*args, **kwargs):

        super(BCE_possitive_loss, self).__init__()

        self.criteria = nn.BCELoss(reduction='none')

    def forward(self, sampels, explanations):

        explanation2 = F.interpolate(explanations,size=(14,14),mode='bilinear')
        explanation3 = explanation2.squeeze()

        # N, C, H, W = sampels.size()
        smpl = F.relu(sampels)
        smpl2 = F.sigmoid(smpl).view_as(sampels)

        loss = self.criteria(smpl2, explanation3)
        return torch.mean(loss)

class Omid_loss(nn.Module):

    def __init__(self,*args, **kwargs):

        super(Omid_loss, self).__init__()

        self.criteria = nn.BCELoss(reduction='none')

    def forward(self, samples, explanations):
        
        explanation2 = F.interpolate(explanations,size=(14,14),mode='bilinear')
        explanation3 = explanation2.squeeze()

        # N, C, H, W = samples.size()

        #####Two BCE Loss#########
        upsampled_attr_positive_relu = F.relu(samples)
        upsampled_attr_negetive_relu = F.relu(-samples)
        upsampled_attr_positive_relu = F.sigmoid(upsampled_attr_positive_relu).view_as(samples)
        upsampled_attr_negetive_relu = F.sigmoid(upsampled_attr_negetive_relu).view_as(samples)

        Two_BCE_loss = self.criteria(upsampled_attr_positive_relu,explanation3) + self.criteria(upsampled_attr_negetive_relu,1-explanation3)

        return torch.mean(Two_BCE_loss)



class Omid_loss_menual(nn.Module):

    def __init__(self,*args, **kwargs):

        super(Omid_loss_menual, self).__init__()

        self.criteria = nn.BCELoss(reduction='none')

    def forward(self, samples, explanations):

        N, C, H, W = samples.size()

        #####Two BCE Loss#########
        upsampled_attr_positive_relu = F.relu(samples)
        upsampled_attr_negetive_relu = F.relu(-samples)
        upsampled_attr_positive_relu = F.sigmoid(upsampled_attr_positive_relu).view_as(samples)
        upsampled_attr_positive_relu = torch.log(upsampled_attr_positive_relu)

        upsampled_attr_negetive_relu = F.sigmoid(upsampled_attr_negetive_relu).view_as(samples)
        upsampled_attr_negetive_relu = torch.log(upsampled_attr_negetive_relu)

        Two_BCE_loss = explanations*upsampled_attr_positive_relu + (1-explanations)*upsampled_attr_negetive_relu


        return torch.mean(Two_BCE_loss)

class Stefano_loss(nn.Module):

    def __init__(self,*args, **kwargs):

        super(Stefano_loss, self).__init__()

        self.criteria = nn.BCELoss(reduction='none')

    def forward(self, samples, explanations):

        N, C, H, W = samples.size()

        ###########stefano loss sugestion##############
        #for possitive
        upsampled_attr_positive_relu = F.relu(samples)
        upsampled_attr_positive_relu = F.sigmoid(upsampled_attr_positive_relu).view_as(samples)
        upsampled_attr_positive_relu = torch.log(upsampled_attr_positive_relu)
        #for negetive
        upsampled_attr_negetive_relu = torch.abs(samples)
        upsampled_attr_negetive_relu = 1 - F.sigmoid(upsampled_attr_negetive_relu).view_as(samples)
        upsampled_attr_negetive_relu = torch.log(upsampled_attr_negetive_relu)
        TWo_log_st = explanations*upsampled_attr_positive_relu + (1-explanations)*upsampled_attr_negetive_relu

        return torch.mean(TWo_log_st)

class emb_hinge_loss(nn.Module):

    def __init__(self,*args, **kwargs):

        super(emb_hinge_loss, self).__init__()

        self.criteria = nn.HingeEmbeddingLoss(reduction='mean')

    def forward(self, samples, explanations):

        N, C, H, W = samples.size()
        
        loss = self.criteria(samples,explanations)

        return loss

class BCE_LOGIT(nn.Module):

    def __init__(self,*args, **kwargs):

        super(BCE_LOGIT, self).__init__()

        self.criteria = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, samples, explanations):

        N, C, H, W = samples.size()
        
        samples = F.sigmoid(samples).view_as(samples)
        
        loss = self.criteria(samples,explanations)

        return torch.mean(loss)
        
class MahalanobisMetricLoss(nn.Module):
    def __init__(self, margin=0.6, extra_margin=0.04):
        super(MahalanobisMetricLoss, self).__init__()

        self.margin = margin
        self.extra_margin = extra_margin

    def forward(self, outputs, targets):
        """
        :param outputs: Outputs from a network. (sentence_batch size, # features)
        :param targets: Target labels. (sentence_batch size, 1)
        :param margin: Minimum distance margin between contrasting sample pairs.
        :param extra_margin: Extra acceptable margin.
        :return: Loss and accuracy. Loss is a variable which may have a backward pass performed.
        """

        loss = torch.zeros(1)
        if torch.cuda.is_available(): loss = loss.cuda()
        loss = torch.autograd.Variable(loss)

        batch_size = outputs.size(0)

        # Compute Mahalanobis distance matrix.
        magnitude = (outputs ** 2).sum(1).expand(batch_size, batch_size)
        squared_matrix = outputs.mm(torch.t(outputs))
        mahalanobis_distances = F.relu(magnitude + torch.t(magnitude) - 2 * squared_matrix).sqrt()

        # Determine number of positive + negative thresholds.
        neg_mask = targets.expand(batch_size, batch_size)
        neg_mask = (neg_mask - neg_mask.transpose(0, 1)) != 0

        num_pairs = (1 - neg_mask).sum()  # Number of pairs.
        num_pairs = (num_pairs - batch_size) / 2  # Number of pairs apart from diagonals.
        num_pairs = num_pairs.data[0]

        negative_threshold = mahalanobis_distances[neg_mask].sort()[0][num_pairs].data[0]

        num_right, num_wrong = 0, 0

        for row in range(batch_size):
            for column in range(batch_size):
                x_label = targets[row].data[0]
                y_label = targets[column].data[0]
                mahalanobis_distance = mahalanobis_distances[row, column]
                euclidian_distance = torch.dist(outputs[row], outputs[column])

                if x_label == y_label:
                    # Positive examples should be less than (margin - extra_margin).
                    if mahalanobis_distance.data[0] > self.margin - self.extra_margin:
                        loss += mahalanobis_distance - (self.margin - self.extra_margin)

                    # Compute accuracy w/ Euclidian distance.
                    if euclidian_distance.data[0] < self.margin:
                        num_right += 1
                    else:
                        num_wrong += 1
                else:
                    # Negative examples should be greater than (margin + extra_margin).
                    if (mahalanobis_distance.data[0] < self.margin + self.extra_margin) and (
                                mahalanobis_distance.data[0] < negative_threshold):
                        loss += (self.margin + self.extra_margin) - mahalanobis_distance

                    # Compute accuracy w/ Euclidian distance.
                    if euclidian_distance.data[0] < self.margin:
                        num_wrong += 1
                    else:
                        num_right += 1

        accuracy = num_right / (num_wrong + num_right)
        return loss / (2 * num_pairs), accuracy