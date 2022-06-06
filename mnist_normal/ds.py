import os
from funcs import create_circular_mask
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
import argparse

import matplotlib.pyplot as plt
import torchvision.transforms as transformsss

def plot_me(img_arr):
    plt.figure(figsize=(10,18))
    for idx,i in enumerate(img_arr):
        # print(idx)
        plt.subplot(5,2,idx+1)
        plt.imshow(np.array(i).reshape(28,28))
    plt.show()

def plot_loader(dl):
    plt.figure(figsize=(10,18))
    it = iter(dl)
    a,b = next(it)
    circle = create_circular_mask(224,224,None,100)
    circle = np.array([circle])
    # print(circle.shape)
    #circle = torch.tensor(circle)
    #circle = circle.to(device)
    for idx,i in enumerate(a):
        # print(idx)
        plt.subplot(5,2,idx+1)
        i = i*circle
        plt.imshow(i.numpy().transpose(1,2,0))
    plt.show()

    
class MnistDecoy(Dataset):
    
    def __init__(self, args, transforms=None, mode = None):
        
        self.args = args
        if mode:
            self.args.mode = mode
        if args.mode:
            mode = self.args.mode
        # self.transforms = transforms
        self.transforms_expl = transformsss.Compose([lambda x: Image.fromarray(x).convert('L'),
                                                     transformsss.Resize((224,224),interpolation=transformsss.functional.InterpolationMode.NEAREST),
                                                     transformsss.ToTensor()])
        # Load the Decoy MNIST dataset
        cached = np.load(args.path)
        # self.data = cached
        arrays = [cached[f] for f in sorted(cached.files)]
        X_train, y_train, X_val, y_val, X_test, y_test = arrays

        # Verify we get 50000/10000/10000 x 784
        # flter to 8000 data only
        print(X_train.shape, X_val.shape, X_test.shape, ' --- all data loaded.') 
        print(mode,args.mode)
        if self.args.mode == 'test':
            self.data = [X_test[:1000], y_test[:1000]]
        if self.args.mode == 'train':
            self.data = [X_train[:8000], y_train[:8000]]
        if self.args.mode == 'val':
            self.data = [X_val[:6000], y_val[:6000]]
        else:
            print('please select mode train,test,val')
        ## Load whole data desabled for performance.

        # if self.args.mode == 'test':
        #     self.data = [X_test, y_test]
        # if self.args.mode == 'train':
        #     self.data = [X_train, y_train]
        # if self.args.mode == 'val':
        #     self.data = [X_val, y_val]
        # else:
        #     print('please select mode train,test,val')
        
        # self.data = data

    def __len__(self):
        return len(self.data[0])

    def __repr__(self):
        return repr(self.data[0])

    def __getitem__(self, idx):
        #get file name
            # X_test, y_test = self.data
        X = self.data[0][idx]
        Y = self.data[1][idx]
        Y = Y.astype(np.long)
        if self.transforms_expl:
            X = np.array(X).reshape(28,28)
            X = self.transforms_expl(X)
        return X,Y
        # else:
        #     # X_train, y_train, X_val, y_val = self.data
        #     X = self.data[0][idx]
        #     Y = self.data[1][idx]
        #     X_test = self.data[2][idx]
        #     Y_test = self.data[3][idx]
        #     if self.transforms_expl:
        #         X = np.array(X).reshape(28,28)
        #         X = self.transforms_expl(X)
        #         X_test = self.transforms_expl(X_test)
        #     return X,Y,X_test,Y_test


if __name__ == '__main__':
    # Load the Decoy MNIST dataset
    parser = argparse.ArgumentParser()

    parser.add_argument('--path', type=str,default='data/',
                        help='input images path')
    
    parser.add_argument('--mode', type=str,default='test',
                        help='train or test')
    # parser.add_argument('--start_frame', type=int,default='1',
    #                     help='num frame to start')
    # parser.add_argument('--end_frame', type=int,default='100',
    #                     help='the last frame')
    args = parser.parse_args()
    # cached = np.load('./data/decoy_mnist.npz')
    # arrays = [cached[f] for f in sorted(cached.files)]
    # X_train, y_train, X_val, y_val, X_test, y_test = arrays

    # # Verify we get 50000/10000/10000 x 784
    # print(X_train.shape, X_val.shape, X_test.shape)
    # plot_me(X_train[:10])

    ds = MnistDecoy(args)
    # it = iter(ds)
    # a,b,c,d = next(it)
    # print(a.size(),c.size())

    train_loader = torch.utils.data.DataLoader(
            ds,
            batch_size=10,
            shuffle=False,
            drop_last=True,
            pin_memory=False)
    test_loader = torch.utils.data.DataLoader(
        ds,
        batch_size=10,
        shuffle=False,
        drop_last=True,
        pin_memory=False)

    it = iter(test_loader)
    a,b = next(it)
    print(a.size(),b.size())
    plot_loader(train_loader)



    # print(X_train[1])