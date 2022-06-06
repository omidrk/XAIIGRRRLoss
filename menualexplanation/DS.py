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
from utils.tranforms import get_transforms
from utils.arguments import parse_arguments
import matplotlib.pyplot as plt
import torchvision.transforms as transformsss




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
        
        #setup explanation dataset
        ##you should fix path later Omid
#         print('Start loadeing explanation...')
#         explanation_data = pd.read_csv('explanation_split.csv')
#         explanation_data['file_name'] = explanation_data['file_name'].apply(lambda x: '0'+x)
#         explanation_datasource = COVID19Dataset_expl(args, explanation_data,get_transforms(args, 'test'))
#         print('explanation loaded.')
        
        print('Start loadeing explanation...')
        #explanation_data = pd.read_pickle(os.path.join(args.dataset_root, 'activeset.pkl'))
        explanation_data = pd.read_pickle(os.path.join(args.dataset_root, 'cleared_activeset_final.pkl'))
        # explanation_data = pd.read_csv(os.path.join(args.dataset_root, 'cleared_activeset.csv'))
#         explanation_data['file_name'] = explanation_data['file_name'].apply(lambda x: '0'+x)
        explanation_datasource = COVID19Dataset_explB(args, explanation_data,get_transforms(args, 'test'))
        print('explanation loaded.')
        

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
        expl_loader = torch.utils.data.DataLoader(
            explanation_datasource,
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
        return train_loader,test_loader,nclasses,expl_loader

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
        if self.transforms:
            frame = self.transforms(frame)
        label = torch.tensor(sum(self.data.iloc[idx].label), dtype=torch.long)
        return frame, label
    
class COVID19Dataset_expl(Dataset):

    def __init__(self, args, data, transforms=None):
        self.dataset_root = args.dataset_root
        self.transforms = transforms
        self.data = data

    def __len__(self):
        return len(self.data.index)

    def __repr__(self):
        return repr(self.data)

    def __getitem__(self, idx):
        #get file name
        frame_file = self.data.iloc[idx].file_name
        path_gt = self.data.iloc[idx].path
#         print(path_gt)
        frame_gt = np.load(path_gt)
        
        mainfile_path = os.path.join(self.dataset_root, 'frames', frame_file)
#         print(mainfile_path)
        mainfile = np.load(mainfile_path)
        if self.transforms:
            frame_gt = self.transforms(frame_gt)
            mainfile = self.transforms(mainfile)
        return mainfile, frame_gt
    
class COVID19Dataset_explB(Dataset):

    def __init__(self, args, data, transforms=None):
        self.dataset_root = args.dataset_root
        self.transforms = transforms
        self.data = data
        self.args =args
        self.transforms_expl = transformsss.Compose([lambda x: Image.fromarray(x).convert('1'),
                                                     transformsss.Resize((args.img_size, args.img_size)),
                                                     transformsss.ToTensor()])

    def __len__(self):
        return len(self.data.index)

    def __repr__(self):
        return repr(self.data)

    def __getitem__(self, idx):
        #get file name
        frame_file_expl = self.data.iloc[idx].explanation
        path_expl= os.path.join(self.dataset_root,
                                'segmentation_frames',
                                self.data.iloc[idx].hospital,
                                frame_file_expl)
#         print(path_expl)
        frame_expl = np.load(path_expl)
        
#         os.path.join(self.dataset_root, 'frames', frame_file)
#         print(path_gt)
        frame_gt_path = os.path.join(self.dataset_root, 'frames', self.data.iloc[idx].filename)
#         print(frame_gt_path)
        frame_gt = np.load(frame_gt_path)
        
        
#         mainfile_path = os.path.join(self.dataset_root, 'frames', frame_file)
#         print(mainfile_path)
#         mainfile = np.load(mainfile_path)
        if self.transforms:
#             frame_expl = self.transforms(frame_expl)
            frame_expl = self.transforms_expl(frame_expl)
            frame_gt = self.transforms(frame_gt)
        label = torch.tensor(sum(self.data.iloc[idx].label), dtype=torch.long)
        return frame_gt, frame_expl,label
    
if __name__ == "__main__":
    args = parse_arguments()
    print(args)
    myclass = LoadData(args,root='~/home/data/dataset')
    train_loader,test_loader,classes,exp = myclass()
    print('loaded all')
    exp_iter = iter(exp)
    x,y,L = exp_iter.next()
    print(x.size())
    print('iterator works')
    plt.subplot(321)
    plt.imshow(x[4].numpy().transpose([1,2,0]))
    plt.imshow(y[4].numpy().transpose([1,2,0]),alpha=0.2)
    plt.title(str(L[4].numpy()))
    plt.subplot(322)
    plt.imshow(x[5].numpy().transpose([1,2,0]))
    plt.imshow(y[5].numpy().transpose([1,2,0]),alpha=0.2)
    plt.title(str(L[5].numpy()))
    plt.subplot(323)
    plt.imshow(x[6].numpy().transpose([1,2,0]))
    plt.imshow(y[6].numpy().transpose([1,2,0]),alpha=0.2)
    plt.title(str(L[6].numpy()))
    
#     plt.imshow(y[2].numpy().transpose([1,2,0]))
#     plt.subplot(321)
#     plt.imshow(x[0].numpy().transpose([1,2,0]))
#     plt.subplot(322)
#     plt.imshow(y[0].numpy().transpose([1,2,0]))
#     plt.subplot(323)
#     plt.imshow(x[1].numpy().transpose([1,2,0]))
#     plt.subplot(324)
#     plt.imshow(y[1].numpy().transpose([1,2,0]))
#     plt.subplot(325)
#     plt.imshow(x[2].numpy().transpose([1,2,0]))
#     plt.subplot(326)
#     plt.imshow(y[2].numpy().transpose([1,2,0]))
    plt.savefig('Hi.png')
        
    
    
