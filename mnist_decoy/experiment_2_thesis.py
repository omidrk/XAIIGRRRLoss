"""
Created on Sun September 2021

@author: Omid Jadidi
"""
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import grad
from torchvision.utils import make_grid
import seaborn as sns
from sklearn.metrics import confusion_matrix

import numpy as np
# import cv2
import os
from arguments import parse_arguments

import time
import pandas as pd
from PIL import Image
# import scikitplot as skplt
from funcs import plot_roc_curve,create_circular_mask
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, confusion_matrix,roc_auc_score,roc_curve,auc
from torch.utils.tensorboard import SummaryWriter
from random import shuffle

from ds import MnistDecoy
from net import Resnet18

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class experiment():
    def __init__(self,args,experiment = 'none'):
        self.experiment = experiment
        self.ds_train = MnistDecoy(args,mode='train')
        self.ds_test = MnistDecoy(args,mode='test')
        self.ds_val = MnistDecoy(args,mode='val')
        self.df_normal = pd.DataFrame([],columns=['epoch','per','rec','f1'])
        self.df_explanation = pd.DataFrame([],columns=['epoch','per','rec','f1'])
        self.df_val = pd.DataFrame([],columns=['epoch','per','rec','f1'])


        self.model = Resnet18()
        self.model = self.model.to(device)
        self.writer = SummaryWriter(os.path.join('logs',args.run_name))

        self.train_loader = torch.utils.data.DataLoader(
                self.ds_train,
                batch_size=args.batch,
                shuffle=False,
                drop_last=True,
                pin_memory=False)
        self.test_loader = torch.utils.data.DataLoader(
            self.ds_test,
            batch_size=args.batch,
            shuffle=False,
            drop_last=True,
            pin_memory=False)
        self.val_loader = torch.utils.data.DataLoader(
            self.ds_val,
            batch_size=args.batch,
            shuffle=False,
            drop_last=True,
            pin_memory=False)

        print(f'test size:{len(self.ds_train)}, val size: {len(self.ds_val)}')

        ###prepare directories
        # create directories
        args.weights_dir = os.path.join('logs', args.run_name, 'weights')
        os.makedirs(args.weights_dir, exist_ok=True)
        args.train_viz_dir = os.path.join('logs', args.run_name, 'viz_train')
        os.makedirs(args.train_viz_dir, exist_ok=True)
        args.test_viz_dir = os.path.join('logs', args.run_name, 'viz_test')
        os.makedirs(args.test_viz_dir, exist_ok=True)
        args.pic_viz_dir = os.path.join('logs', args.run_name, 'pic')
        os.makedirs(args.pic_viz_dir, exist_ok=True)
        args.grad_vis = os.path.join('logs', args.run_name, 'grad')
        os.makedirs(args.grad_vis, exist_ok=True)

        #init fixed sample visualization out: (img,label)
        self.fixed_vis_sample = self.make_Visualization_prepare(self.val_loader)

    def make_Visualization_prepare(self,data_loader):
        #This function prepare initial visualization samples.
        #get first batch of dataloader as fixed samples.
        loader = data_loader
        loader_it = iter(loader)
        return next(loader_it)

    def uncertainty_sample_visualization_prepare(self,model,data_loader):
        #This function prepare initial visualization samples after second epoch.
        # loader = data_loader
        # loader_it = iter(loader)
        # return next(loader_it)
        model.eval()
        probs = torch.zeros([20, 2])
        with torch.no_grad():
            for idx, (img, _) in enumerate(data_loader):
                img = img.to(device)
                class_prediction = model(img)
                prob = F.softmax(class_prediction, dim=0) 
                probs[idx] = prob.cpu()

    def vis_fixed_inputgrad(self,model):

        loss = nn.CrossEntropyLoss()
        X,y = self.fixed_vis_sample
        X = X.to(device)
        X = X.requires_grad_(True)
        y = y.to(device)
        out = model(X)
        soft_out = F.softmax(out,dim = 1)
        log_soft_out = torch.log(soft_out)
        loss_inputgrad = torch.sum(log_soft_out)

        input_gard = grad(outputs=loss_inputgrad,
                            inputs=X,
                            retain_graph=False,
                            create_graph=False)
        # input_gard_poss = torch.where(input_gard[0]>0,1,0)
        # input_gard_negative = torch.where(input_gard[0]<0,1,0)
        ## for any funny reason opsitive and negative is reversed :))
        input_gard_poss = F.relu(-input_gard[0].detach(), inplace=False)
        input_gard_negative = F.relu(input_gard[0].detach(), inplace=False)
        # print(len(input_gard[0]))
        # print('size is:',input_gard[0].size())
        #vis_img = torch.cat((X,input_gard[0]),dim = 0)
        vis_img = torch.cat(    
            (X,
            input_gard_poss,
            input_gard_negative),
            dim = 0)
        grid_img = make_grid(vis_img,
                                value_range=(0,1),
                                scale_each=True,
                                normalize = False,
                                nrow=10)
        del input_gard
        return grid_img
        # plt.imshow(grid_img.permute(1,2,0)*255)
        # self.writer.add_image('fixed_samples',grid_img*255,global_step = step)
        # plt.show()
        

    def normal_learn(self,args,model):
        #define loss
        optim = torch.optim.Adam(self.model.parameters(), 
                                          lr=0.001, 
                                          weight_decay=0)
        loss = nn.CrossEntropyLoss()

        for epoch in range(args.epoch):
            # epoch_counter = 0
            print('validate')
            # if epoch> 0:
            #   self.test_val(args,model,epoch)
            Y_full_pred = []
            Y_full = []
            
            for idx,(X,y) in enumerate(self.train_loader):
                X = X.to(device)
                y = y.long().to(device)
                out = model(X)
                soft_out = F.softmax(out,dim=0)
                predlb = torch.argmax(out,1)
                # print(soft_out)
                lss = loss(out,y)
                optim.zero_grad()
                lss.backward()
                optim.step()

                Y_full_pred.append(predlb)
                Y_full.append(y)

                ####for metric
                ##f1 percision recal

                
                
                ##roc_auc
                
                if idx% args.log_interval == 0:
                    Y_full_pred_temp = torch.cat(Y_full_pred,dim=0)
                    Y_full_temp = torch.cat(Y_full,dim=0)
                    (precision,recall,fscore,support) = precision_recall_fscore_support(Y_full_temp.cpu().numpy(),Y_full_pred_temp.cpu().numpy(),average='micro')
                    f_final = f'iter: {idx}, epoch: {epoch},normal_loss: {lss}, percision:{precision}, recall: {recall}, fscore:{fscore}'
                    print(f_final)                
                    self.writer.add_scalar('normal_loss_train',lss,idx+ epoch*idx)
                    self.writer.add_scalar('precision_train',precision,idx+ epoch*idx)
                    self.writer.add_scalar('recall_train',recall,idx+ epoch*idx)
                    self.writer.add_scalar('fscore_train',fscore,idx+ epoch*idx)

      


                    #start visualization and add to tensorboard
                    grd_img = self.vis_fixed_inputgrad(model)
                    self.writer.add_image('fixed_samples',grd_img,global_step = idx+ epoch*idx)
                    # self.writer.add_graph(model,X)
            Y_full_pred = torch.cat(Y_full_pred,dim=0)
            Y_full = torch.cat(Y_full,dim=0)
            (precision,recall,fscore,_) = precision_recall_fscore_support(Y_full.cpu().numpy(),Y_full_pred.cpu().numpy(),average='micro')
            self.df_normal.loc[epoch] = [epoch,precision,recall,fscore]
        # df.to_csv('normal.csv')
        self.test_val(args,model,epoch)


    def explanation_learn(self,args,model):
        #define loss
        optim = torch.optim.SGD(self.model.parameters(), 
                                          lr=0.01, 
                                          momentum = 0)
        loss = nn.CrossEntropyLoss()
        expl_loss =0 #to be implemented
        model.train()

        #explanation is masking digit and multiply it to a cirle with 
        # radios of 100px which will remove the corner extra pixel. This will
        # help us to gauid network avoid corner pixel.
        circle = create_circular_mask(224,224,None,100)
        circle = torch.tensor(circle)
        circle = circle.to(device)
        # df = pd.DataFrame([],columns=['epoch','per','rec','f1'])
        

        for epoch in range(args.epoch):
            # epoch_counter = 0
            print('validate')
            # if epoch> 0:
            #   self.test_val(args,model,epoch)
            Y_full_pred = []
            Y_full = []
            
            
            for idx,(X,y) in enumerate(self.train_loader):
                X = X.to(device)
                y = y.long().to(device)
                optim.zero_grad()
                #mask number as explanation with value bigger than 0
                ##main idea
                # expl = torch.where(X>0,1,0) * circle
                #second idea
                msk = torch.where(X>0,1,0)
                msk_rm = msk* circle
                expl = msk - msk_rm

                X = X.requires_grad_(True)

                out = model(X)
                soft_out = F.softmax(out,dim=0)
                predlb = torch.argmax(out,1)
                # print(soft_out)
                lss = loss(out,y)
                optim.zero_grad()

                #compute explanation loss based on label log of output
                log_soft_out = torch.log(soft_out)
                lss_expl = torch.sum(log_soft_out)
                test_grad = grad(outputs=lss_expl, inputs=X,retain_graph=True,create_graph=True)
                expl_loss = self.loss_gradient(test_grad[0],log_soft_out,expl,0.2)


                #final loss is mixed of two. based on the hyper params.
                # give some rest to network before learning and skip first epoch 
                # just train on the label. start mixing from second epoch.

                # final_loss = 0.002*expl_loss

                

                if epoch> 0:
                    final_loss = lss + 0.002*expl_loss
                else: 
                    final_loss = lss

                optim.zero_grad()
                final_loss.backward()
                optim.step()

                Y_full_pred.append(predlb)
                Y_full.append(y)            
                
                if idx% args.log_interval == 0:
                    Y_full_pred_temp = torch.cat(Y_full_pred,dim=0)
                    Y_full_temp = torch.cat(Y_full,dim=0)
                    (precision,recall,fscore,support) = precision_recall_fscore_support(Y_full_temp.cpu().numpy(),Y_full_pred_temp.cpu().numpy(),average='micro')
                    f_final = f'iter: {idx}, epoch: {epoch},normal_loss: {lss},explanation_loss:{expl_loss.detach().cpu().numpy()}, percision:{precision}, recall: {recall}, fscore:{fscore}'
                    print(f_final)                
                    self.writer.add_scalar('normal_loss_train',lss,idx+ epoch*idx)
                    self.writer.add_scalar('explanation_loss',expl_loss.detach().cpu().numpy(),idx+ epoch*idx)
                    # self.writer.add_scalar('normal_loss_train',lss,idx+ epoch*idx)
                    self.writer.add_scalar('precision_train',precision,idx+ epoch*idx)
                    self.writer.add_scalar('recall_train',recall,idx+ epoch*idx)
                    self.writer.add_scalar('fscore_train',fscore,idx+ epoch*idx)

                    #start visualization and add to tensorboard
                    grd_img = self.vis_fixed_inputgrad(model)
                    self.writer.add_image('fixed_samples',grd_img,global_step = idx+ epoch*idx)
                    # self.writer.add_graph(model,X)      
                    # 
            Y_full_pred = torch.cat(Y_full_pred,dim=0)
            Y_full = torch.cat(Y_full,dim=0)
            (precision,recall,fscore,_) = precision_recall_fscore_support(Y_full.cpu().numpy(),Y_full_pred.cpu().numpy(),average='micro')
            self.df_explanation.loc[epoch] = [epoch,precision,recall,fscore]
        # df.to_csv('explanation.csv')       
            self.test_val(args,model,epoch)     
    def test_val(self,args,model,epoch):
        #define loss
        print('validation mode started: ')
        # df = pd.DataFrame([],columns=['epoch','per','rec','f1'])
        optim = torch.optim.Adam(self.model.parameters(), 
                                          lr=0.001, 
                                          weight_decay=0)
        loss = nn.CrossEntropyLoss()
        Y_full_pred_T = []
        Y_full_T = []
        soft_out_final = []
        model.eval()

        #remove co_founder from val set
        circle = create_circular_mask(224,224,None,100)
        circle = torch.tensor(circle)
        circle = circle.to(device)

        with torch.no_grad():
            for idx,(X,y) in enumerate(self.val_loader):
                X = X.to(device)
                #remove co founder
                X= X*circle

                y = y.to(device)
                out = model(X)
                soft_out = F.softmax(out,dim=0)
                predlb = torch.argmax(soft_out,1)
                # print(soft_out)
                # lss = loss(out,y)
                # optim.zero_grad()
                # lss.backward()
                # optim.step()
                Y_full_pred_T.append(predlb.cpu())
                Y_full_T.append(y.cpu())
                soft_out_final.append(soft_out.cpu())

                ####for metric
                ##f1 percision recal
        #convert to flat 
        Y_full_pred_T = torch.cat(Y_full_pred_T,dim=0)
        Y_full_T = torch.cat(Y_full_T,dim=0)
        soft_out_final = torch.cat(soft_out_final,dim=0)
        (precision,recall,fscore,_) = precision_recall_fscore_support(Y_full_T.cpu().numpy(),Y_full_pred_T.cpu().numpy(),average='micro')
        self.df_val.loc[epoch] = [epoch,precision,recall,fscore]
        # df.to_csv('val.csv')            

        cf_matrix = confusion_matrix(Y_full_T.cpu().numpy(), Y_full_pred_T.cpu().numpy())
        figs = plt.figure()
        fig_conf = sns.heatmap(cf_matrix, annot=True)
        figs.add_axes(fig_conf)
       
        # ax = sns.heatmap(uniform_data)



        (precision,recall,fscore,support) = precision_recall_fscore_support(
                                                        Y_full_T.cpu().numpy(),
                                                        Y_full_pred_T.cpu().numpy(),
                                                        average='micro')
        f_final = f'train percision and recal: percision:{precision}, recall: {recall}, fscore:{fscore}'
        print('validation loss is: '+ f_final)


        ##ruc auc test and plot
        fig_roc,ax_roc = self.roc_auc_plot(Y_full_pred_T.numpy(),
                                                soft_out_final.numpy(),
                                                Y_full_T.cpu().numpy())

        self.writer.add_scalar('precision_val',precision)
        self.writer.add_scalar('recall_val',recall,epoch)
        self.writer.add_scalar('fscore_val',fscore,epoch)
        self.writer.add_figure('roc_auc_curve',fig_roc,global_step=epoch)

        self.writer.add_figure('confusion_matrix_val',figs,global_step=epoch)

        # self.writer.add_scalar('auc_score',auc_out,epoch,display_name='auc_score')
        # self.writer.add_scalar('support',support,idx+ epoch*idx,display_name='support')
        # self.writer2.add_scalar('expl_loss',expl_loss,batch_idx+ epoch*batch_idx,display_name='expl_LOSS')
             
    def roc_auc_plot(self,Y_pred,Y_prob,Y_label):
        #to comute roc and auc and plot in matplotlib
        #return figura and ax. plot_roc_curve() external lib for vis in funcs.py
        fig,ax = plot_roc_curve(Y_label,Y_prob,nclass = 10)
        return fig,ax
    def loss_gradient(self,gradients,log_soft_out,explanation2,margin):
        s = .0
        # print('len grad is - --',len(gradients))
        for i in range(gradients.size()[0]):
            y_hat = torch.sum(log_soft_out[i])
            gr = gradients[i]
            # grad_yhat = y_hat * gr
            # this is not true, we want gradient of co-founder to be zero. so the nxt one shoud be correct
            #not correct one cause explanation gt is  co-founder #grad_mul =  gr * (1-explanation2)
            grad_mul =  gr * (explanation2)
            # grad_mul_missing = (margin - grad_yhat) * explanation2
            grad_mul = grad_mul **2
            # grad_mul_missing = grad_mul_missing**2
            grad_mul = torch.sum(grad_mul)
            # grad_mul_missing = torch.sum(grad_mul_missing)
            s+=grad_mul
        return torch.sqrt(s)


    def debug_temp(self,model):
        circle = create_circular_mask(224,224,None,100)

        with torch.no_grad():
            loss = nn.CrossEntropyLoss()
            X,y = self.fixed_vis_sample
            X = X.to(device)
            expl = torch.where(X>0,1,0) * circle
            print(expl)
            # X = X.requires_grad_(True)
            y = y.to(device)
            out = model(X)
            soft_out = F.softmax(out,dim = 1)
            predlb = torch.argmax(soft_out,1)
            pix = expl[0].numpy().transpose([1,2,0])
            plt.imshow(pix)
            plt.show()
            # self.roc_auc_plot(predlb,soft_out.cpu().numpy(),y.cpu().numpy())
        # log_soft_out = torch.log(soft_out)
        # loss_inputgrad = torch.sum(log_soft_out)

        # input_gard = grad(outputs=loss_inputgrad, inputs=X,retain_graph=True,create_graph=True)
        # print(len(input_gard[0]))
        # print('size is:',input_gard[0].size())
        # vis_img = torch.cat((X,input_gard[0]),dim = 0)
        # grid_img = make_grid(vis_img,value_range=(0,254),scale_each=True,normalize = False,nrow=10)
        # return grid_img
        # plt.imshow(grid_img.permute(1,2,0)*255)
        # self.writer.add_image('fixed_samples',grid_img*255,global_step = step)
        # plt.show()

if __name__ == '__main__':
    args = parse_arguments()

    exe = experiment(args)
    #test vis
    # exe.normal_learn(args,exe.model)

    #for debug
    # exe.debug_temp(exe.model)

    #for run
    # exe.normal_learn(args,exe.model)
    
    exe.explanation_learn(args,exe.model)
    exe.df_normal.to_csv('normal.csv')
    exe.df_explanation.to_csv('expla.csv')
    exe.df_val.to_csv('val.csv')
    exe.writer.close()