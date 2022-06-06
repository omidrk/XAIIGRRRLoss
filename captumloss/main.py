# -*- coding: utf-8 -*-
"""
Created on Sun Dec 13 20:52:09 2020

@author: Dimo
"""
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

from resnet18 import Resnet18
from DS import LoadData
from ModelManager import ModelManager

if __name__ == "__main__":
    manager = ModelManager(ModelRoot='./model')

    #train for the known model
    manager.train_known(1,30,40)
    manager.validate()
    for i in range(3):
        #start unknown part
        print('Calculate sample probability...')
        probs = manager.predict_probability(10000) #budget of searching
        # print(len(temp[1]))
        logprob = torch.log(probs[1])
        E = logprob*probs[1]
        E = E.sum(1)
        probs1 = list(zip(probs[0],E))
    #     print(probs1)
        res = sorted(probs1, key = lambda x: x[1])
        indx = np.array(res[:100],dtype=np.int) #untill batch index
        indx = indx[:,0].astype('int32')
        print('Sample probability done.')
#         print(indx)


        #train unknown
        # print('Start training unkown ...')
        # manager.train_Unnown(indx,1,30,40)
        ###### start to explain the unknown...
        # manager.train_unknown_exp(indx,1,30,40)



        # img,a,b,c = manager.explanation(indx)
        # # img = img.squeeze().cpu().numpy()
        # img = img.squeeze()
        # img[img == -0.5] = 0
        # img[img != 0] = 1
        # # print(img)
        # # a = a.squeeze().detach().cpu().numpy()
        # #define loss
        # cross_entropy = nn.BCELoss()
        # print(b.size(),img.size())
        # b = F.relu(b, inplace=False)
        # print(b.size(),img.size())

        # newLoss = cross_entropy(b.cuda(),img.cuda())
        # print(newLoss)

        # plt.subplot(2,1,1)
        # plt.imshow(img.cpu().numpy())
        # # plt.subplot(2,1,2)
        # # plt.imshow(b)
        
        # # print(plt.show())


        
        #Move Items from unknown to known
        print("start moving from unknown to known ...")
        manager.move_unknown(indx)
        del indx
        print('Train started again...')
        manager.train_known_expl(1,30,40)
        manager.validate()
        # if(i%10 == 0):
        #     manager.validate()

        # len(E)
        # print(E)
        # temp2 = list([temp[0],E])