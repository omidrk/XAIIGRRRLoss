# -*- coding: utf-8 -*-
"""
Created on Sun Dec 13 23:19:58 2021

@author: Omid Jadidi
"""
import warnings
warnings.filterwarnings("ignore")

import torch
import torchvision
from torchvision.utils import save_image, make_grid

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import grad
from torch.utils.tensorboard import SummaryWriter

import numpy as np
# import cv2
import os

import time
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib import cm
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, confusion_matrix
from funcs import plot_roc_curve

from random import shuffle

from DS import LoadData
from models.net import Resnet18

##############################
#load model.
from utils.arguments import parse_arguments
# from utils.gaussian import gaussian_blur
import kornia as K
import seaborn as sns

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



class experiment:
    def __init__(self,args):
        # args = parse_arguments()
        print(args)
        self.args =args
        self.explainVis = []
        myclass = LoadData(self.args,root='~/home/data/dataset')
        self.train_loader, self.test_loader, self.nclasses,self.expl_loader = myclass()
        self.train_labels = myclass.train_labels
        #loading resnet18
        self.model = Resnet18()
        self.model = self.model.to(device)

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

        #fixed samples for stn visualization
        (self.fixed_samples_train,
            self.fixed_y_train,
            self.fixed_samples_test, 
            self.fixed_y_test,
            self.fixed_samples_expl, 
            self.fixed_expl_segments,
            self.fixed_expl_label) = self.make_Visualization_prepare(self.train_loader,self.test_loader,self.expl_loader)
        

        
    
        ####prepare tensorboard logging
        self.writer = SummaryWriter(args.grad_vis+'/main_logger')
        self.writer2 = SummaryWriter(args.grad_vis+'/explanation_logger')
        
    def make_Visualization_prepare(self,train_loader,test_loader,expl_loader):
        #This function prepare initial visualization samples.
        #get first batch of dataloader as fixed samples.
        fixed_samples_iter = iter(train_loader)
        fixed_samples_train, fixed_y_train = fixed_samples_iter.next()
        fixed_samples_iter = iter(test_loader)
        fixed_samples_test, fixed_y_test = fixed_samples_iter.next()

        fixed_samples_iter = iter(expl_loader)
        fixed_samples_expl, fixed_expl_segments,fixed_expl_label = fixed_samples_iter.next()
        return (fixed_samples_train,
                    fixed_y_train,
                    fixed_samples_test,
                    fixed_y_test,
                    fixed_samples_expl,
                    fixed_expl_segments,
                    fixed_expl_label)


    def vis_fixed_inputgrad(self,model):

        num_sample = 10

        #X,y = self.fixed_vis_sample
        data,explanation, target = self.fixed_samples_expl.to(device), self.fixed_expl_segments.to(device), self.fixed_expl_label.long().to(device)
        data = data.requires_grad_(True)
        data,explanation, target = data[:num_sample],explanation[:num_sample], target[:num_sample]
        out = model(data)
        soft_out = F.softmax(out,dim = 1)
        log_soft_out = torch.log(soft_out)
        loss_inputgrad = torch.sum(log_soft_out)

        #mask overlay
        # mask_overlay = cv2.addWeighted(data, 1, explanation, 0.7, 128)
        mask_overlay = data * 1 + explanation * 0.7

        input_gard = grad(outputs=loss_inputgrad,
                            inputs=data,
                            retain_graph=False,
                            create_graph=False)
        input_gard_poss = F.relu(-input_gard[0].detach(), inplace=False)
        input_gard_negative = F.relu(input_gard[0].detach(), inplace=False)
        vis_img = torch.cat(    
            (data,
            mask_overlay,
            input_gard_poss,
            input_gard_negative),
            dim = 0)
        grid_img = make_grid(vis_img,
                                value_range=(0,2),
                                scale_each=True,
                                normalize = False,
                                nrow=10)

        #maik histogram
        # vis_grad = input_gard[0].detach()
        # h = torch.cat([torch.histc(torch.sum(x,dim=0), bins=50, min=0, max=10) for x in input_gard_poss], 0)

        
        del input_gard
        # return grid_img
        #ver 2 
        return grid_img
    def prepare_baseline_input_grad(self,data,n_step):
        pass
        
        num_sample = 10
        # Generate m_steps intervals for integral_approximation() below
        alphas = torch.linspace(start=0.0, end=1.0, steps=n_step).unsqueeze(1).unsqueeze(2).unsqueeze(3).unsqueeze(0)
        
        # base_line = gaussian_blur(data,(5,5), (0, 15))
        base_line = K.filters.gaussian_blur2d(data, (51,51), (50.0, 50.0))

        base_line = base_line.to(device)
        alphas = alphas.to(device)
        
        #interpolate now
        data = data.unsqueeze(dim=1)
        base_line = base_line.unsqueeze(dim=1)
        # base_line = torch.zeros_like(data)
        
        # print("baseline",base_line.size())
        # data,base_line,alphas = data.to(device), base_line.to(device), alphas.to(device)
        new_data = data - base_line
        # print(alphas.size(),base_line.size(),new_data.size())
        imgs = alphas * new_data+ base_line 
        # print(imgs.size())
        imgs = imgs.view(num_sample*n_step,data.size()[2],data.size()[3],data.size()[4])
        return imgs,new_data
        
    def vis_average_grad(self,avg_grad):

      fig = plt.figure(figsize=(20,21))
      avg_grad_array = avg_grad.detach().cpu().numpy()
      (x,y)=avg_grad_array.shape
      mn,mx = np.min(avg_grad_array,axis=1),np.max(avg_grad_array,axis=1)
      for i in range(x):
        ax = plt.subplot(10,5,i+1)
        # ax1.set_xlim([0, 5])
        # ax.set_ylim([mn[i],mx[i]])
        ax.plot(np.linspace(0,y,y),avg_grad_array[i,:]-0.1*i, c ='blue')
      return fig

    def vis_fixed_IntegratedGradient(self,model,n_step):

        num_sample = 10
        n_step = 30
        # Generate m_steps intervals for integral_approximation() below
        # alphas = torch.linspace(start=0.0, end=1.0, steps=n_step).unsqueeze(1).unsqueeze(2).unsqueeze(3).unsqueeze(0)

        # #X,y = self.fixed_vis_sample
        data,explanation, target = self.fixed_samples_expl.to(device), self.fixed_expl_segments.to(device), self.fixed_expl_label.long().to(device)
        # # data = data.requires_grad_(True)
        data,explanation, target = data[:num_sample],explanation[:num_sample], target[:num_sample]

        # #interpolate now
        # data = data.unsqueeze(dim=1)
        # base_line = torch.zeros_like(data)
        # # print("baseline",base_line.size())
        # data,base_line,alphas = data.to(device), base_line.to(device), alphas.to(device)
        # new_data = data - base_line
        # # print(alphas.size(),base_line.size(),new_data.size())
        # imgs = alphas * new_data+ base_line 
        # # print(imgs.size())
        # imgs = imgs.view(num_sample*n_step,data.size()[2],data.size()[3],data.size()[4])
        imgs,new_data = self.prepare_baseline_input_grad(data,n_step)
        imgs = imgs.requires_grad_(True)
        # print(imgs.size())



        out = model(imgs)
        soft_out = F.softmax(out,dim = 1)
        log_soft_out = torch.log(soft_out)
        loss_inputgrad = torch.sum(log_soft_out)

        #mask overlay
        # mask_overlay = cv2.addWeighted(data, 1, explanation, 0.7, 128)
        # mask_overlay = data * 1 + explanation * 0.7
        

        input_gard = grad(outputs=loss_inputgrad,
                            inputs=imgs,
                            retain_graph=False,
                            create_graph=False)

        grads = input_gard[0].detach()
        grads = grads.view(num_sample,n_step,data.size()[1],data.size()[2],data.size()[3])

        #average pixel gradient
        avg_grad = torch.mean(grads,dim=[2,3,4])
        print("size of avg grad is: ",avg_grad.size())
        avg_figure = self.vis_average_grad(avg_grad)
        # avg_grad = torch.sum(avg_grad,dim=1)
        # avg_grad = avg_grad.cpu().numpy()
        # avg_grad = avg_grad.permute()
        # print("size of avg grad is: ",avg_grad.size())
        
        # g = sns.PairGrid(avg_grad)
        # g.map(sns.scatterplot)
        ##for max grad visualization

        # print(grads.size())
        ing_grad = new_data.squeeze(dim=1) * torch.sum(grads,dim=1) / n_step
  
        # data = data.squeeze().detach()
        data = data.detach()
        ing_grad = ing_grad.detach()
        # input_gard_poss = F.relu(-ing_grad, inplace=False)
        # input_gard_negative = F.relu(ing_grad, inplace=False)
        # print(data.size(),ing_grad.size())
        # vis_img = torch.cat(    
        #     (
        #     input_gard_poss,
        #     input_gard_negative
        #     ),
        #     dim = 0)
        #clamp values
        vis_img = torch.clip(ing_grad,min=0,max=1)
        ##seaborn 
        # g = sns.PairGrid(vis_img)
        # g.map(sns.scatterplot)
        grid_img = make_grid(vis_img,
                                value_range=(0,1),
                                scale_each=True,
                                normalize = False,
                                nrow=10)

        
        del input_gard
        # self.writer2.add_image('fixed_samples',grid_img,global_step = 1)
        return grid_img,avg_figure
        #ver 2 
        # return grid_img

    def load_model(self):
        if os.path.exists('./models') and os.path.exists('./models/best_model.pth'):
            print('Model folder found...')
            self.model.load_state_dict(torch.load('./models/best_model.pth'))
            print('model successfully loaded.')
        else:
            print('no pre trianed model found')
        
        
            
    def train_normal_extractGrad(self):
        #train loop
        self.model.train()
        avg_loss = []
 
        optimizer = optim.Adam(self.model.parameters(), lr=self.args.lr, weight_decay=1e-4)
        exp_lr_scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[70], gamma=0.1) # 10, 50
        state_dict = {'best_f1': 0., 'precision': 0., 'recall': 0., 'accuracy': 0.}
#         self.state_dict = state_dict
        i = 0
        for epoch in range(self.args.epochs):
            model = self.train(self.args, self.model, self.train_loader, len(list(set(self.train_labels))), optimizer, epoch,
                        self.fixed_samples_train, self.fixed_y_train)
            # test(args, self.model, self.test_loader, len(list(set(self.train_labels))), epoch, state_dict, args.weights_dir,
            #     fixed_samples_test)
            exp_lr_scheduler.step()
            self.validate_test(state_dict)
            if epoch%10 == 0 and epoch> 0: 
                self.test(args, self.model, self.test_loader, len(list(set(self.train_labels))), epoch, state_dict, args.weights_dir,
                self.fixed_samples_test)
        self.writer.close()

    def train_co_learning(self):
        #train loop
        self.model.train()
        # self.train_loader, self.test_loader, self.nclasses,self.expl_loader


        avg_loss = []
        train_loss, preds, labels = [], [], []

        normal_it = iter(self.train_loader)
        expl_it = iter(self.expl_loader)
        state_dict = {'best_f1': 0., 'precision': 0., 'recall': 0., 'accuracy': 0.}
        # optimizer = optim.SGD(self.model.get_params()[0],lr = 0.01,momentum = 0)
        optimizer = optim.Adam(self.model.parameters(), lr=self.args.lr, weight_decay=1e-4)


        epoch = 0
        max_iter = 50000
        nclasses = 4
  
        for it in range(max_iter):
            optimizer.zero_grad()
            try:
                # print('In TRY OK.')
                data, target = next(normal_it)
                data, target = data.to(device), target.long().to(device)

                if not data.size()[0] == self.args.batch_size:
                    raise StopIteration

            except StopIteration:
                print('next main epoch started.')
                epoch += 1
                normal_it = iter(self.train_loader)
                data, target = next(normal_it)
                data, target = data.to(device), target.long().to(device)

            if it % args.log_interval ==0:
                try:

                    data2,explanation, target2 = next(expl_it)
                    data2,explanation, target2 = data2.to(device), explanation.to(device), target2.long().to(device)

                    if not data2.size()[0] == self.args.batch_size:
                        raise StopIteration
                        
                except StopIteration:
                    print('next second main epoch started.')
                    expl_it = iter(self.expl_loader)
                    data2,explanation, target2 = next(expl_it)
                    data2,explanation, target2 = data2.to(device), explanation.to(device), target2.long().to(device)


            output = self.model(data)
            optimizer.zero_grad()

            loss_normal_1 = self.sord_loss(logits=output, ground_truth=target, num_classes=nclasses, multiplier=args.multiplier)
            loss_normal_1.backward()
            optimizer.step()
            optimizer.zero_grad()
            if it % args.log_interval ==0:
                optimizer.zero_grad()

                data2.requires_grad_(True)
                output2 = self.model(data2)
                loss_normal_2 = self.sord_loss(logits=output2, ground_truth=target2, num_classes=nclasses, multiplier=args.multiplier)

                optimizer.zero_grad()
                 #co training explanation loss
                soft_out = F.softmax(output2,dim = 1)
                log_soft_out = torch.log(soft_out)
                lss_expl = torch.sum(log_soft_out)

                test_grad = grad(outputs=lss_expl, inputs=data2,retain_graph=True,create_graph=True)
                expl_loss = self.loss_gradient(test_grad[0],log_soft_out,explanation,0.2)
                # avg_grad = torch.mean(test_grad[0])
                self.writer2.add_scalar('normal_loss_2',loss_normal_2,it)
                # self.writer2.add_scalar('avg_grad_2',avg_grad,it)
                self.writer2.add_scalar('expl_loss_2',expl_loss,it)
                print(f'normal loss is: {loss_normal_2}')
                
                if it % args.log_interval*3 ==0:

                    grd_img = self.vis_fixed_inputgrad(self.model)
                    # id_image,fig_image = self.vis_fixed_IntegratedGradient(self.model,20)
                    self.writer2.add_image('fixed_samples',grd_img,global_step = it)
                    # self.writer2.add_image('fixed_samples_IG',id_image,global_step = it)
                    # self.writer2.add_figure('fixed_samples_IG',fig_image,global_step = it,close =True)

                if epoch> 0 and epoch% 2 == 0:
                    loss_all = loss_normal_2 + 0.0008*expl_loss
                else: 
                    loss_all = loss_normal_2

                # optimizer.zero_grad()
                loss_all.backward()
                optimizer.step()
                optimizer.zero_grad()
            
            # total_loss_normal = loss_normal_1+loss_normal_2
            
            if it % args.log_interval == 0 and it>0:        
                print(f'iter is :{it}')
                self.writer2.add_scalar('normal_loss',loss_normal_1,it)
                # self.writer2.add_scalar('normal_loss_2',loss_normal_2,it)

            if it % 2000 == 0 and it>0:        
                self.test(args, self.model, self.test_loader, len(list(set(self.train_labels))), epoch, state_dict, args.weights_dir,
                self.fixed_samples_test)
                self.model.train()

           

            
                #start visualization and add to tensorboard
                # grd_img = self.vis_fixed_inputgrad(model)
                # id_image = self.vis_fixed_IntegratedGradient(model,20)
                # self.writer2.add_image('fixed_samples',grd_img,global_step = batch_idx+ epoch*batch_idx)
                # self.writer2.add_image('fixed_samples_IG',id_image,global_step = batch_idx+ epoch*batch_idx)
            # optimizer.zero_grad()
            # optimizer.zero_grad()
            # loss.backward()
       
        # for epoch in range(self.args.epochs):
            
        #     # self.validate_test(state_dict)
        #     if epoch%10 == 0 and epoch> 0: 
        #         self.test(args, self.model, self.test_loader, len(list(set(self.train_labels))), epoch, state_dict, args.weights_dir,
        #         self.fixed_samples_test)
        # self.writer2.close()
        
    def train_explanation_extractGrad(self):
        #train loop
        self.model.train()
        avg_loss = []
 
        optimizer = optim.Adam(self.model.parameters(), lr=self.args.lr)
        ###only update 2d cov params
        optimizer2 = optim.SGD(self.model.get_params()[0],lr = 0.01,momentum = 0)
        exp_lr_scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[70], gamma=0.1) # 10, 50
        state_dict = {'best_f1': 0., 'precision': 0., 'recall': 0., 'accuracy': 0.}
        i = 0
        ##init eval 
        # self.test(args, self.model, self.test_loader, len(list(set(self.train_labels))), 100, state_dict, args.weights_dir,
        #         self.fixed_samples_test)

        for epoch in range(self.args.epochs):
            
            if epoch == 0:
                self.test(args, self.model, self.test_loader, len(list(set(self.train_labels))), epoch, state_dict, args.weights_dir,
                    self.fixed_samples_test)
            
            model = self.train_expl(self.args, self.model, self.expl_loader, len(list(set(self.train_labels))), optimizer,optimizer2, epoch,
                        self.fixed_samples_train, self.fixed_y_train)
            
            exp_lr_scheduler.step()
#             self.validate_test()
            if epoch%5 == 0 and epoch> 0: 
                self.test(args, self.model, self.test_loader, len(list(set(self.train_labels))), epoch, state_dict, args.weights_dir,
                self.fixed_samples_test)
        self.writer2.close()
        

    def train(self,args, model, train_loader, nclasses, optimizer, epoch, fixed_samples, fixed_y):
            model.train()
            correct = 0
            train_loss, mse_losses, stn_reg_losses, preds, labels = [], [], [], [], []
            confusion_matrix = torch.zeros(nclasses, nclasses)
#             layer_gc = LayerGradCam(self.model, self.model.layer2[1].conv2)
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(device), target.long().to(device)
                output = model(data)
                optimizer.zero_grad()

                loss = self.sord_loss(logits=output, ground_truth=target, num_classes=nclasses, multiplier=args.multiplier)
                train_loss.append(loss.item())


                optimizer.zero_grad()
                loss.backward()
                    
                ##########vis grad#############
                if batch_idx % args.log_interval == 0:
                    self.writer.add_scalar('normal_loss',loss,batch_idx+ epoch*batch_idx)

                    grd_img= self.vis_fixed_inputgrad(model)
                    self.writer.add_image('fixed_samples',grd_img,global_step = batch_idx+ epoch*batch_idx)
                    
                ###############################
                optimizer.step()
                pred = F.softmax(output, dim=1).max(1, keepdim=True)[1]
                correct += pred.eq(target.view_as(pred)).sum().item()
                # to compute metrics
                preds.append(pred.view(-1).cpu())
                labels.append(target.view(-1).cpu())
                
                for t, p in zip(target.view(-1), pred.view(-1)):
                    confusion_matrix[t.long(), p.long()] += 1

                if batch_idx % args.log_interval == 0:
                        print('Train epoch: {} [{}/{} ({:.0f}%)]\tCELoss: {:.6f}\tConLoss: {:.6f}'.format(epoch,
                                                                                                batch_idx * len(data),
                        len(train_loader.dataset), 100. * batch_idx / len(train_loader),
                        loss.item(), 0.))


            # compute the metrics
            precision, recall, fscore, _ = precision_recall_fscore_support(
                y_true=torch.cat(labels), y_pred=torch.cat(preds), average='micro')

            # print the logs
            per_class_accuracy = confusion_matrix.diag() / confusion_matrix.sum(1)
            print('\nTrain set: Accuracy: {}/{}({:.2f}%)'.format(correct,
            len(train_loader.dataset), 100 * correct / len(train_loader.dataset)))

            print('Classwise Accuracy:: Cl-0: {}/{}({:.2f}%),\
            Cl-1: {}/{}({:.2f}%), Cl-2: {}/{}({:.2f}%), Cl-3: {}/{}({:.2f}%); \
            Precision: {:.3f}, Recall: {:.3f}, F1: {:.3f}'.format(
            int(confusion_matrix.diag()[0].item()), int(confusion_matrix.sum(1)[0].item()), per_class_accuracy[0].item() * 100.,
            int(confusion_matrix.diag()[1].item()), int(confusion_matrix.sum(1)[1].item()), per_class_accuracy[1].item() * 100.,
            int(confusion_matrix.diag()[2].item()), int(confusion_matrix.sum(1)[2].item()), per_class_accuracy[2].item() * 100.,
            int(confusion_matrix.diag()[3].item()), int(confusion_matrix.sum(1)[3].item()), per_class_accuracy[3].item() * 100.,
            precision, recall, fscore))


            return model
    
    def train_expl(self,args, model, train_loader, nclasses, optimizer,optimizer2, epoch, fixed_samples, fixed_y):
        model.train()
        # print(model)
        ######################
#             ct = 0
#             for child in self.model.children():
#                 ct += 1
# #                 print(child)
#             if ct > 6:
#                 for param in child.parameters():
#                     param.requires_grad = False
        ######################
        correct = 0
        train_loss, mse_losses, stn_reg_losses, preds, labels = [], [], [], [], []
        confusion_matrix = torch.zeros(nclasses, nclasses)

        for batch_idx, (data,explanation, target) in enumerate(train_loader):
            if batch_idx == 0:
                continue
            data,explanation, target = data.to(device), explanation.to(device), target.long().to(device)
            data.requires_grad_(True)
            optimizer.zero_grad()


            output = self.model(data)        
            optimizer.zero_grad()               
            #############normal loss############
            # supervised loss
            loss = self.sord_loss(logits=output, ground_truth=target, num_classes=nclasses, multiplier=args.multiplier)
            normal_loss = loss

            ###planc#############
            optimizer.zero_grad()
            optimizer2.zero_grad()

            soft_out = F.softmax(output,dim = 1)
            log_soft_out = torch.log(soft_out)
            predlb = torch.argmax(output,1)

            
            optimizer.zero_grad()
            optimizer2.zero_grad()
            # arrr = []
            lss_expl = torch.sum(log_soft_out)

            test_grad = grad(outputs=lss_expl, inputs=data,retain_graph=True,create_graph=True)
            expl_loss = self.loss_gradient(test_grad[0],log_soft_out,explanation,0.2)

            #alpha = normal_loss*expl_loss/(expl_loss+normal_loss)
            #beta = 0.001
            # loss_all = normal_loss+expl_loss*alpha*beta
            #loss_all = 0.9 * normal_loss+ 0.1 * expl_loss
            if epoch> 0:
                loss_all = normal_loss + 0.0008*expl_loss
            else: 
                loss_all = normal_loss


            optimizer.zero_grad()
            optimizer2.zero_grad()
            loss_all.backward()
            
            optimizer2.step()
            optimizer.zero_grad()
            optimizer2.zero_grad()

            #####################

            if batch_idx % args.log_interval == 0:        
                self.writer2.add_scalar('normal_loss',normal_loss,batch_idx+ epoch*batch_idx)
                self.writer2.add_scalar('explanation_loss',expl_loss.detach().cpu().numpy(),batch_idx+ epoch*batch_idx)
                #start visualization and add to tensorboard
                grd_img = self.vis_fixed_inputgrad(model)
                id_image = self.vis_fixed_IntegratedGradient(model,20)
                self.writer2.add_image('fixed_samples',grd_img,global_step = batch_idx+ epoch*batch_idx)
                self.writer2.add_image('fixed_samples_IG',id_image,global_step = batch_idx+ epoch*batch_idx)
                # self.writer2.add_histogram('fixed_samples_hist',hst,bins='auto',max_bins=50,global_step = batch_idx+ epoch*batch_idx)
                        
            pred = F.softmax(output, dim=1).max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()
            # to compute metrics
            preds.append(pred.view(-1).cpu())
            labels.append(target.view(-1).cpu())
            
            for t, p in zip(target.view(-1), pred.view(-1)):
                confusion_matrix[t.long(), p.long()] += 1


        # compute the metrics
        precision, recall, fscore, _ = precision_recall_fscore_support(
            y_true=torch.cat(labels), y_pred=torch.cat(preds), average='micro')

        # print the logs
        per_class_accuracy = confusion_matrix.diag() / confusion_matrix.sum(1)
        print('\nTrain set: Accuracy: {}/{}({:.2f}%)'.format(correct,
        len(train_loader.dataset), 100 * correct / len(train_loader.dataset)))

        print('Classwise Accuracy:: Cl-0: {}/{}({:.2f}%),\
        Cl-1: {}/{}({:.2f}%), Cl-2: {}/{}({:.2f}%), Cl-3: {}/{}({:.2f}%); \
        Precision: {:.3f}, Recall: {:.3f}, F1: {:.3f}'.format(
        int(confusion_matrix.diag()[0].item()), int(confusion_matrix.sum(1)[0].item()), per_class_accuracy[0].item() * 100.,
        int(confusion_matrix.diag()[1].item()), int(confusion_matrix.sum(1)[1].item()), per_class_accuracy[1].item() * 100.,
        int(confusion_matrix.diag()[2].item()), int(confusion_matrix.sum(1)[2].item()), per_class_accuracy[2].item() * 100.,
        int(confusion_matrix.diag()[3].item()), int(confusion_matrix.sum(1)[3].item()), per_class_accuracy[3].item() * 100.,
        precision, recall, fscore))
            
        return model

    def loss_gradient(self,gradients,log_soft_out,explanation2,margin):
        s = .0
        # print('len grad is - --',len(gradients))
        for i in range(gradients.size()[0]):
            y_hat = torch.sum(log_soft_out[i])
            gr = gradients[i]
            # grad_yhat = y_hat * gr
            grad_mul =  gr * (1-explanation2)
            # grad_mul_missing = (margin - grad_yhat) * explanation2
            grad_mul = grad_mul **2
            # grad_mul_missing = grad_mul_missing**2
            grad_mul = torch.sum(grad_mul)
            # grad_mul_missing = torch.sum(grad_mul_missing)
            s+=grad_mul
        return torch.sqrt(s)

    def roc_auc_plot(self,Y_pred,Y_prob,Y_label):
        #to comute roc and auc and plot in matplotlib
        #return figura and ax. plot_roc_curve() external lib for vis in funcs.py
        fig,ax = plot_roc_curve(Y_label,Y_prob,nclass = 4)
        return fig,ax

    def validate_test(self,state_dict):
        self.test(self.args,self.model,self.test_loader, self.nclasses,1,state_dict,None,None)
    def test(self,args, model, test_loader, nclasses, epoch, state_dict, weights_path, fixed_samples):
        args.pic_viz_dir = os.path.join('logs', args.run_name)
        logger = open(args.pic_viz_dir+'/'+str(epoch)+'test_logger.txt', "w")
        
#         model.load_state_dict(torch.load('./models/best_model.pth'))
#         print('Model trained wights successfully loaded.')
        logger.write('Model test started.'+ '\n')
        model.eval()
        test_losses = []
        correct = 0
        preds, labels,soft_out_final = [], [],[]
        confusion_matrix = torch.zeros(nclasses, nclasses)
        
        # expl_preceion = 0
        # expl_recall = 0
        # layer_gc = LayerGradCam(self.model, self.model.layer2[1].conv2)

        with torch.no_grad():
            for i, (data, target) in enumerate(test_loader):
                
                data, target = data.to(device), target.long().to(device)
                output= model(data)

                loss = self.sord_loss(logits=output, ground_truth=target, num_classes=nclasses, multiplier=args.multiplier)
                
                test_losses.append(loss.item())
                pred_soft = F.softmax(output, dim=1)
                pred = pred_soft.max(1, keepdim=True)[1]
                correct += pred.eq(target.view_as(pred)).sum().item()

                # compute metrics
                preds.append(pred.view(-1).cpu())
                labels.append(target.view(-1).cpu())
                soft_out_final.append(pred_soft.cpu())
                

                for t, p in zip(target.view(-1), pred.view(-1)):
                    confusion_matrix[t.long(), p.long()] += 1
        test_loss = np.mean(np.asarray(test_losses))
        ###omid custom
        soft_out_final = torch.cat(soft_out_final,dim=0)
        y_pred=torch.cat(preds,dim=0)
        y_true=torch.cat(labels,dim=0)
        
        precision, recall, fscore, _ = precision_recall_fscore_support(
            y_true=torch.cat(labels), y_pred=torch.cat(preds), average='micro')
        per_class_accuracy = confusion_matrix.diag() / confusion_matrix.sum(1)
        ##ruc auc test and plot
        fig_roc,ax_roc = self.roc_auc_plot(y_pred.numpy(),
                                                soft_out_final.numpy(),
                                                y_true.numpy())

        self.writer.add_scalar('precision_val',precision)
        self.writer.add_scalar('recall_val',recall,epoch)
        self.writer.add_scalar('fscore_val',fscore,epoch)
        self.writer.add_figure('roc_auc_curve_val',fig_roc,global_step=epoch)
        # self.writer.add_figure('loss_val',fig_roc,global_step=epoch)


        temp = '\nTest Set: Average Loss: {:.4f}, Accuracy: {}/{} \
                    ({:.2f}%)'.format(test_loss, correct, len(test_loader.dataset), 100 *
                    correct / len(test_loader.dataset))
        logger.write(temp+'\n')
        print(temp)
        temp = '\nTest Set: Average Loss: {:.4f}, Accuracy: {}/{} \
                    ({:.2f}%)'.format(test_loss, correct, len(test_loader.dataset), 100 *
                    correct / len(test_loader.dataset))
        logger.write(temp+'\n')
        print(temp)

        temp ='Classwise Accuracy:: Cl-0: {}/{}({:.2f}%), Cl-1: {}/{}({:.2f}%) \
        Cl-2: {}/{}({:.2f}%), Cl-3: {}/{}({:.2f}%); \
        Precision: {:.3f}, Recall: {:.3f}, F1: {:.3f}'.format(
        int(confusion_matrix.diag()[0].item()), int(confusion_matrix.sum(1)[0].item()), per_class_accuracy[0].item() * 100.,
        int(confusion_matrix.diag()[1].item()), int(confusion_matrix.sum(1)[1].item()), per_class_accuracy[1].item() * 100.,
        int(confusion_matrix.diag()[2].item()), int(confusion_matrix.sum(1)[2].item()), per_class_accuracy[2].item() * 100.,
        int(confusion_matrix.diag()[3].item()), int(confusion_matrix.sum(1)[3].item()), per_class_accuracy[3].item() * 100.,
        precision, recall, fscore)
        logger.write(temp+'\n')
        print(temp)

        metrics = {'test/accuracy': correct / len(test_loader.dataset) * 100.,
                'test/precision': precision,
                'test/recall': recall,
                'test/F1': fscore,
                'test/loss': test_loss}
        logger.close()

        if(state_dict is not None):
            print('Saving weights...')
            self.save_weights(model, os.path.join(args.weights_dir, 'model.pth'))
            self.save_best_model(model, args.weights_dir, metrics, state_dict)

    def calculate_measures(self,gts, masks):
        final_prec = 0
        final_rec = 0
        final_corr = 0
        # print('size is : ', masks.size(),gts.size())

        for mask, gt in zip(masks, gts): 
            
            if torch.sum(mask) == 0:
                precision = 0
                correlation = 0
                recall = 0
            else:
                mask = mask.detach().cpu().numpy()
                gt = gt.cpu().numpy()
                max_ = np.max(mask)

                precision = np.sum(gt*mask) / (np.sum(gt*mask) + np.sum((1-gt)*mask))
                # correlation = (1 / (gt.shape[0]*gt.shape[1])) * np.sum(gt*mask)
                # correlation = 0
                recall = np.sum(gt*mask) / (np.sum(gt*mask) + np.sum(gt*(1-mask)))

            final_prec = final_prec + precision
            final_rec = final_rec + recall 
        return final_prec, final_rec

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

    def save_weights(self,model, path):
        torch.save(model.state_dict(), path)

    def load_weights(self,args, model, path):
        if args.arch == 'ResNet50':
            state_dict_ = torch.load(path)
            modified_state_dict = {}
            for key in state_dict_.keys():
                mod_key = key[7:]
                modified_state_dict.update({mod_key: state_dict_[key]})
        else:
            modified_state_dict = torch.load(path)
        model.load_state_dict(modified_state_dict, strict=True)
        return model

    def save_best_model(self,model, path, metrics, state_dict):
        if metrics['test/F1'] > state_dict['best_f1']:
            state_dict['best_f1'] = max(metrics['test/F1'], state_dict['best_f1'])
            state_dict['accuracy'] = metrics['test/accuracy']
            state_dict['precision'] = metrics['test/precision']
            state_dict['recall'] = metrics['test/recall']
            print('F1 score improved over the previous. Saving model...')
            self.save_weights(model=model, path=os.path.join(path, 'best_model.pth'))
        best_str = "Best Metrics:" + '; '.join(["%s - %s" % (k, v) for k, v in state_dict.items()])
        print(best_str)

    def sord_loss(self,logits, ground_truth, num_classes=4, multiplier=2, wide_gap_loss=False):
        batch_size = ground_truth.size(0)
        # Allocates sord probability vector
        labels_sord = np.zeros((batch_size, num_classes))
        for element_idx in range(batch_size):
            current_label = ground_truth[element_idx].item()
            # Foreach class compute the distance between ground truth label and current class
            for class_idx in range(num_classes):
                # Distance computation that increases the distance between negative patients and
                # positive patients in the sord loss.
                if wide_gap_loss:
                    wide_label = current_label
                    wide_class_idx = class_idx
                    # Increases the gap between positive and negative
                    if wide_label == 0:
                        wide_label = -0.5
                    if wide_class_idx == 0:
                        wide_class_idx = -0.5
                    labels_sord[element_idx][class_idx] = multiplier * abs(wide_label - wide_class_idx) ** 2
                # Standard computation distance = 2 * ((class label - ground truth label))^2
                else:
                    labels_sord[element_idx][class_idx] = multiplier * abs(current_label - class_idx) ** 2
        labels_sord = torch.from_numpy(labels_sord).cuda(non_blocking=True)
        labels_sord = F.softmax(-labels_sord, dim=1)
        # Uses log softmax for numerical stability
        log_predictions = F.log_softmax(logits, 1)
        # Computes cross entropy
        loss = (-labels_sord * log_predictions).sum(dim=1).mean()
        return loss


    def debug_me(self):
        # self.vis_fixed_IntegratedGradient(self.model,10)
        data = torch.random(10,3,224,224)
        self.prepare_baseline_input_grad(data, 30)
    def _compute_scores(self,y_true, y_pred):

            folder = "test"

            labels = list(range(10)) # 4 is the number of classes: {0,1,2,3}
            confusion = confusion_matrix(y_true, y_pred, labels=labels)
            precision, recall, fscore, _ = precision_recall_fscore_support(y_true=y_true, y_pred=y_pred, average='macro')
            accuracy = accuracy_score(y_true, y_pred)

            # print(confusion)

            scores = {}
            scores["{}/accuracy".format(folder)] = accuracy
            scores["{}/precision".format(folder)] = precision
            scores["{}/recall".format(folder)] = recall
            scores["{}/f1".format(folder)] = fscore

            precision, recall, fscore, _ = precision_recall_fscore_support(y_true=y_true, y_pred=y_pred, labels=labels, average=None)

            for i in range(len(labels)):
                prefix = "{}_{}/".format(folder, i)
                scores[prefix + "precision"] = precision[i]
                scores[prefix + "recall"] = recall[i]
                scores[prefix + "f1"] = fscore[i]

            return scores

if __name__ == '__main__':
#     cudnn.benchmark = True
    args = parse_arguments()
#     print('AAAAAARRRRRGGGGGG',args)
    
    manager = experiment(args)
    #just load trained model
    manager.load_model()
#     manager.visual_expl_stn()
    #train on one epoch and extract grads
    
    ###############################
    # manager.train_normal_extractGrad()
    # manager.debug_me()
    # manager.train_explanation_extractGrad()
    # manager.train_normal_extractGrad()
    manager.train_co_learning()
    
    ###############################
    
#   manager.train_known_expl()
#   manager.test_vis(args)
    # manager.validate_test(None)