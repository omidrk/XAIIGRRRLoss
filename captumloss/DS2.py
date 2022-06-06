# -*- coding: utf-8 -*-
"""
Created on Sun Dec 13 20:56:43 2020

@author: Dimo
"""

import os
import torch
import pandas as pd
import numpy as np
import torch.nn as nn
from torch.autograd import Variable
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
from PIL import Image
from torch.utils.data import Dataset
from utils.tranforms2 import get_transforms
from utils.arguments import parse_arguments
import pywt
import matplotlib.pyplot as plt





## load covid dataset
use_cuda = torch.cuda.is_available()

class LoadData:
    def __init__(self,args, root, mode='train'):        
        self.root = root # example = './data'
        self.mode = mode
        self.args =args
        # if not os.path.exists(root):
        #     os.mkdir(root)
        
    
    def __call__(self):
        
    
        args = self.args
        # load data
        data = pd.read_pickle(os.path.join(args.dataset_root, 'dataset.pkl'))
        data = data[data.sensor.str.contains('|'.join(args.sensors))]  # filter sensors
        print("Data pickle loaded.")

        # load splits
        splits = pd.read_csv(os.path.join(args.dataset_root, 'train_test_split.csv'))
        train_patients = splits[splits.split.str.contains('train')].patient_hash.tolist()
        test_patients = splits[splits.split.str.contains('test')].patient_hash.tolist()
        print("splits loaded.")

        # get data accorting to patient split
        train_data = data[data.patient_hash.str.contains('|'.join(train_patients))]
        test_data = data[data.patient_hash.str.contains('|'.join(test_patients))]
        print('patients splited')

        # subset the dataset
        train_dataset = COVID19Dataset(args, train_data, get_transforms(args, 'train'))
        test_dataset = COVID19Dataset(args, test_data, get_transforms(args, 'test'))
        print('train test loaded.')

        # For unbalanced dataset we create a weighted sampler
        train_labels = [sum(l) for l in train_data.label.tolist()]
        self.train_labels = train_labels
        weights = self.get_weights_for_balanced_classes(train_labels, len(list(set(train_labels))))
        weights = torch.DoubleTensor(weights)
        sampler = torch.utils.data.sampler.WeightedRandomSampler(weights=weights, num_samples=len(weights))
        print('Sampler loaded.')

        nclasses = len(list(set(train_labels)))
        # dataloaders from subsets
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            sampler=sampler,
            num_workers=args.num_workers,
            drop_last=True,
            pin_memory=True)
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            sampler=None,
            num_workers=args.num_workers,
            drop_last=True,
            pin_memory=True)
        
        print('total trainning known batch number: {}'.format(len(train_dataset)))
        # print('total trainning unknown batch number: {}'.format(len(b)))
        print('total testing batch number: {}'.format(len(test_dataset)))
        
        # return train_known_loader,train_Unknown_loader,test_loader
        return train_loader,test_loader,nclasses

    def get_weights_for_balanced_classes(self,labels, nclasses):
        count = [0] * nclasses
        for item in labels:
            count[item] += 1
        weight_per_class = [0.] * nclasses
        N = float(sum(count))
        for i in range(nclasses):
            if count[i] != 0:
                weight_per_class[i] = N / float(count[i])
        weight = [0] * len(labels)
        for idx, val in enumerate(labels):
            weight[idx] = weight_per_class[val]
        return weight


class COVID19Dataset(Dataset):

    def __init__(self, args, data, transforms=None):
        self.dataset_root = args.dataset_root
        self.transforms = transforms
        self.data = data

    def __len__(self):
        return len(self.data.index)

    def __repr__(self):
        return repr(self.data)

    def __getitem__(self, idx):
        frame_file = self.data.iloc[idx].filename
        frame_path = os.path.join(self.dataset_root, 'frames', frame_file)
        frame = np.load(frame_path)
#         print('frame',frame)
        ##wavelet on the input picture
#         LL, (LH, HL, HH) = pywt.dwt2(frame, 'bior1.3')
#         np.stack(arrays, axis=2).shape
#         temp = np.stack([LL, LH, HL,HH], axis=2)
#         frame =temp
#         print(temp.shape)
#         temp = []
#         temp.append(LL)
#         temp.append(LH)
#         temp.append(HL)
#         temp.append(HH)
#         frame = temp
#         print(LL.size, LH.size, HL.size, HH.size)
#         frame = np.stack((LL,LH, HL, HH), axis=2)
        
        if self.transforms:
            frame = self.transforms(frame)
        label = torch.tensor(sum(self.data.iloc[idx].label), dtype=torch.long)
#         print('new frame',frame.size())
        return frame, label
    
if __name__ == "__main__":
    args = parse_arguments()
    print(args)
    myclass = LoadData(args,root='~/home/data/dataset')
    train_loader,test_loader,classes = myclass()
    print('done all')
    ###fft/wavlet testing samples
    for i,(data,target) in enumerate(test_loader):
        data = data[0]
#         Image.show(data.numpy())
        print(data.size())
        break
    
