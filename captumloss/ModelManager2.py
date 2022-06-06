# -*- coding: utf-8 -*-
"""
Created on Sun Dec 13 23:19:58 2020

@author: Dimo
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
import numpy as np
import cv2
import os

import time
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, confusion_matrix

from random import shuffle

from DS import LoadData
from resnet18 import Resnet18,SimpleCNN
from loss import OhemCELoss

#to test
##################
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from matplotlib import cm

########################
#captum library
from captum.attr import visualization as viz
from captum.attr import LayerGradCam, FeatureAblation, LayerActivation, LayerAttribution,LayerDeepLiftShap

from tensorboardX import SummaryWriter
##############################
#load model.
from models.network import CNNConStn
from utils.arguments import parse_arguments

from scipy import fftpack


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



class ModelManager:
    def __init__(self,args):
        # args = parse_arguments()
        print(args)
        self.args =args
        self.explainVis = []
        myclass = LoadData(self.args,root='~/home/data/dataset')
        self.train_loader, self.test_loader, self.nclasses,self.expl_loader = myclass()
        self.train_labels = myclass.train_labels
        # self.ModelRoot = ModelRoot
        #comment this for loading custom resnet18
#         self.model = CNNConStn(args.img_size, self.nclasses, args.fixed_scale)
#         self.model = self.model.to(device)
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
        fixed_samples_iter = iter(self.train_loader)
        self.fixed_samples_train, self.fixed_y_train = fixed_samples_iter.next()
        fixed_samples_iter = iter(self.test_loader)
        self.fixed_samples_test, self.fixed_y_test = fixed_samples_iter.next()

        fixed_samples_iter = iter(self.expl_loader)
        self.fixed_samples_expl, self.fixed_expl_segments,self.fixed_expl_label = fixed_samples_iter.next()
    
        ####prepare tensorboard logging
        self.writer = SummaryWriter(args.grad_vis+'/main_logger')
        self.writer2 = SummaryWriter(args.grad_vis+'/explanatin_logger')


    def load_model(self):
        if os.path.exists('./models'):
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
            if epoch%10 == 0: 
                self.test(args, self.model, self.test_loader, len(list(set(self.train_labels))), epoch, state_dict, args.weights_dir,
                self.fixed_samples_test)
        self.writer.close()
        
    def train_explanation_extractGrad(self):
        #train loop
        self.model.train()
        avg_loss = []
 
        optimizer = optim.Adam(self.model.parameters(), lr=self.args.lr)
        exp_lr_scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[70], gamma=0.1) # 10, 50
        state_dict = {'best_f1': 0., 'precision': 0., 'recall': 0., 'accuracy': 0.}
        i = 0
        for epoch in range(self.args.epochs):
            model = self.train_expl(self.args, self.model, self.expl_loader, len(list(set(self.train_labels))), optimizer, epoch,
                        self.fixed_samples_train, self.fixed_y_train)
            # test(args, self.model, self.test_loader, len(list(set(self.train_labels))), epoch, state_dict, args.weights_dir,
            #     fixed_samples_test)
            exp_lr_scheduler.step()
#             self.validate_test()
            if epoch%10 == 0: 
                self.test(args, self.model, self.test_loader, len(list(set(self.train_labels))), epoch, state_dict, args.weights_dir,
                self.fixed_samples_test)
        self.writer2.close()
        
        
    def vis_explanation_train(self,args,vis_samples,explanations,number,epoch,img,upsampled_attr,upsampled_attrB):
        
      args.pic_viz_dir = os.path.join('logs', args.run_name, 'pic')

      ####PLot################################################
      plotMe = viz.visualize_image_attr(upsampled_attr[7].detach().cpu().numpy().transpose([1,2,0]),
                            original_image=img[7].detach().cpu().numpy().transpose([1,2,0]),
                            method='heat_map',
                            sign='all', plt_fig_axis=None, outlier_perc=2,
                            cmap='inferno', alpha_overlay=0.2, show_colorbar=True,
#                             title=str(predlb[7]),
                            fig_size=(8, 10), use_pyplot=True)

      plotMe[0].savefig(args.pic_viz_dir+'/'+str(number+number*epoch)+'gradcam_pred.jpg')
        ################################################

      plotMe = viz.visualize_image_attr(upsampled_attrB[7].detach().cpu().numpy().transpose([1,2,0]),
                            original_image=img[7].detach().cpu().numpy().transpose([1,2,0]),
                            method='heat_map',
                            sign='all', plt_fig_axis=None, outlier_perc=2,
                            cmap='inferno', alpha_overlay=0.9, show_colorbar=True,
#                             title=str(lb[7].cpu()),
                            fig_size=(8, 10), use_pyplot=True)
                            
      plotMe[0].savefig(args.pic_viz_dir+'/'+str(number+number*epoch)+'gradcam_label.jpg')
        ################################################

      outImg = img[7].squeeze().detach().cpu().numpy().transpose([1,2,0])
      fig2 = plt.figure(figsize=(8,10))
      prImg = plt.imshow(outImg)
      plt.imshow(explanations[7].squeeze().detach().cpu().numpy().transpose([1,2,0]),alpha = 0.2)
      
      fig2.savefig(args.pic_viz_dir+'/'+str(number+number*epoch)+'original.jpg')
      ################################################
      fig = plt.figure(figsize=(8,10))
      ax = fig.add_subplot(111, projection='3d')

      z = upsampled_attr[7].squeeze().detach().cpu().numpy()
      x = np.arange(0,224,1)
      y = np.arange(0,224,1)
      X, Y = np.meshgrid(x, y)
          
      plll = ax.plot_surface(X, Y , z, cmap=cm.coolwarm)
      # Customize the z axis.
      ax.set_zlim(np.min(z)+0.1*np.min(z),np.max(z)+0.1*np.max(z))
      ax.zaxis.set_major_locator(LinearLocator(10))
      ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

      # Add a color bar which maps values to colors.
      fig.colorbar(plll, shrink=0.5, aspect=5)
      fig.savefig(args.pic_viz_dir+'/'+str(number+number*epoch)+'3dgradcam_pred.jpg')
    
        



    # def train_known_expl(self):
    #     self.model.train()
    #     avg_loss = []
 
    #     optimizer = optim.Adam(self.model.parameters(), lr=self.args.lr, weight_decay=1e-4)
    #     exp_lr_scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[70], gamma=0.1) # 10, 50
    #     state_dict = {'best_f1': 0., 'precision': 0., 'recall': 0., 'accuracy': 0.}
    #     i = 0
    #     for epoch in range(self.args.epochs):
    #         model = self.train(self.args, self.model, self.train_loader, len(list(set(self.train_labels))), optimizer, epoch,
    #                     self.fixed_samples_train, self.fixed_y_train)
    #         # test(args, self.model, self.test_loader, len(list(set(self.train_labels))), epoch, state_dict, args.weights_dir,
    #         #     fixed_samples_test)
    #         exp_lr_scheduler.step()
    #         if i%20 == 0: 
    #             print(i)
    #             #test dataset here

    def train(self,args, model, train_loader, nclasses, optimizer, epoch, fixed_samples, fixed_y):
            model.train()
            correct = 0
            train_loss, mse_losses, stn_reg_losses, preds, labels = [], [], [], [], []
            confusion_matrix = torch.zeros(nclasses, nclasses)
#             layer_gc = LayerGradCam(self.model, self.model.layer2[1].conv2)
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(device), target.long().to(device)
                feat8, feat16, feat32, output = model(data)
                optimizer.zero_grad()

                loss = self.sord_loss(logits=output, ground_truth=target, num_classes=nclasses, multiplier=args.multiplier)
                train_loss.append(loss.item())


                optimizer.zero_grad()
                loss.backward()
                    
                ##########vis grad#############
                if batch_idx % 50 == 0:
                    for name, param in self.model.named_parameters():
                      if param.requires_grad and param.grad is not None:
#                           print (name, param.grad.data.view(param.grad.data.size()[0],-1).sum(1))
                          sums = param.grad.data.view(param.grad.data.size()[0],-1).sum(1)
                          self.writer.add_histogram(name, sums, batch_idx+ epoch*batch_idx)
                if batch_idx % 5 == 0:
                    self.writer.add_scalar('SORD_loss',loss,batch_idx+ epoch*batch_idx,display_name='SORD_LOSS')
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
                ####visual explanation later
#                 if batch_idx%50 == 0:
#                     self.vis_explanation(self.args,batch_idx,epoch)

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
    
    def train_expl(self,args, model, train_loader, nclasses, optimizer, epoch, fixed_samples, fixed_y):
        model.train()
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
        def wrapper(img):
            _,_,_,output = self.model(img)
            return output
        # def not_forward_wrapper(model_output):
        #     return model_output

        expl_preceion = 0
        expl_recall = 0
        layer_gc = LayerGradCam(wrapper, self.model.layer3[1].conv2)

        criteria_BCE = nn.BCELoss()
        criteria_BCELog = nn.BCEWithLogitsLoss(reduction='mean')

        for batch_idx, (data,explanation, target) in enumerate(train_loader):
            data,explanation, target = data.to(device), explanation.to(device), target.long().to(device)
            # print(explanation.size())
            # target = torch.cat((target,target),dim = 0)
            #for visualization
            data_pic = data.clone()

            gc_attrC = layer_gc.attribute(data, target=target, relu_attributions=False)
            upsampled_attrC = LayerAttribution.interpolate(gc_attrC, (224, 224))
            upsampled_attrC_raw = upsampled_attrC.clone()

            ###prepare explanation for validation
            upsampled_attr_expl_validation = upsampled_attrC.clone()
            upsampled_attr_expl_validation = F.sigmoid(upsampled_attr_expl_validation).view_as(upsampled_attrC)

            feat8, feat16, feat32, output = model(data)        
            optimizer.zero_grad()               
            predlb = torch.argmax(output,1)

            #for percion and recal on explanation
            a,b = self.calculate_measures(explanation[:upsampled_attrC.size()[0],:,:,:],upsampled_attr_expl_validation)
            expl_preceion += a/upsampled_attrC.size()[0]
            expl_recall += b/upsampled_attrC.size()[0]
            

            #############normal loss############

            # supervised loss
            loss = self.sord_loss(logits=output, ground_truth=target, num_classes=nclasses, multiplier=args.multiplier)
            train_loss.append(loss.item())


            normal_loss = loss
            # loss.backward()

            #############start explanation loss##############
            upsampled_attrC_vis = upsampled_attrC.clone()

            upsampled_attrC = F.sigmoid(upsampled_attrC).view_as(upsampled_attrC)
            # print('comapre:  -- ',upsampled_attrC.size(),explanation.size())
            loss_semantic = criteria_BCELog(upsampled_attrC,explanation)
#                 print(explanation[0])
                # loss_semantic = criteria_BCE(upsampled_attrC,explanation)
            # loss_all = 0.3*normal_loss + 0.7*loss_semantic

            #####Two BCE Loss#########
            upsampled_attr_positive_relu = F.relu(upsampled_attrC_raw)
            upsampled_attr_negetive_relu = F.relu(-upsampled_attrC_raw)
            upsampled_attr_positive_relu = F.sigmoid(upsampled_attr_positive_relu).view_as(upsampled_attrC_raw)
            upsampled_attr_negetive_relu = F.sigmoid(upsampled_attr_negetive_relu).view_as(upsampled_attrC_raw)

            Two_BCE_loss = criteria_BCE(upsampled_attr_positive_relu,explanation) + criteria_BCE(upsampled_attr_negetive_relu,1-explanation)
            # ###########stefano loss sugestion##############
            # #for possitive
            # upsampled_attr_positive_relu = F.relu(upsampled_attrC_raw)
            # upsampled_attr_positive_relu = F.sigmoid(upsampled_attr_positive_relu).view_as(upsampled_attrC_raw)
            # upsampled_attr_positive_relu = torch.log(upsampled_attr_positive_relu)
            # #for negetive
            # upsampled_attr_negetive_relu = torch.abs(upsampled_attrC_raw)
            # upsampled_attr_negetive_relu = 1 - F.sigmoid(upsampled_attr_negetive_relu).view_as(upsampled_attrC_raw)
            # upsampled_attr_negetive_relu = torch.log(upsampled_attr_negetive_relu)
            # TWo_log_st = explanation*upsampled_attr_positive_relu + (1-explanation)*upsampled_attr_negetive_relu
            # TWo_log_st = torch.mean(TWo_log_st)
            # # TWo_log_st = criteria_BCE(upsampled_attr_positive_relu,explanation) + criteria_BCE(upsampled_attr_negetive_relu,1-explanation)
            # ##########################
            # loss_all = loss_semantic
            #This was winner loss
            loss_all = Two_BCE_loss*0.3+0.7*normal_loss
            #Stefano suggested loss
            # loss_all = TWo_log_st*0.3+0.7*normal_loss
            # loss_all = normal_loss
            optimizer.zero_grad()
            loss_all.backward()

            if batch_idx% 10 == 0:
                print('Normal - semantic - all: ',normal_loss,loss_semantic,loss_all)               
                self.writer2.add_scalar('SORD_loss',normal_loss,batch_idx+ epoch*batch_idx,display_name='SORD_LOSS')
                self.writer2.add_scalar('expl_loss',loss_semantic,batch_idx+ epoch*batch_idx,display_name='expl_LOSS')
                # upsampled_attrC_vis_possitive = F.relu(upsampled_attrC_vis)
                
                
                # upsampled_attrC_vis = F.sigmoid(upsampled_attrC_vis).view_as(upsampled_attrC_raw)
#                 plotMe = viz.visualize_image_attr(upsampled_attrC_vis[7].detach().cpu().numpy().transpose([1,2,0]),
#                             original_image=data_pic[7].detach().cpu().numpy().transpose([1,2,0]),
#                             method='heat_map',
#                             sign='all', plt_fig_axis=None, outlier_perc=2,
#                             cmap='inferno', alpha_overlay=0.3, show_colorbar=True,
# #                             title=str(lb[7].cpu()),
#                             fig_size=(8, 10), use_pyplot=True)
#                 plotMe[0].savefig(args.pic_viz_dir+'/pic/'+str(batch_idx+epoch*batch_idx)+'gradcam_possitive_label.jpg')
#                 outImg = data_pic[7].squeeze().detach().cpu().numpy().transpose([1,2,0])
#                 fig2 = plt.figure(figsize=(8,10))
#                 prImg = plt.imshow(outImg)
#                 plt.imshow(explanation[7].squeeze().detach().cpu().numpy(),alpha = 0.2)
#                 fig2.savefig(args.pic_viz_dir+'/pic/'+str(batch_idx+epoch*batch_idx)+'original.jpg')
        
                
                
                ##########vis gradients#############
#                 if batch_idx % 10 == 0:
                for name, param in self.model.named_parameters():
                  if param.requires_grad and param.grad is not None:
#                           print (name, param.grad.data.view(param.grad.data.size()[0],-1).sum(1))
                      sums = param.grad.data.view(param.grad.data.size()[0],-1).sum(1)
                      self.writer2.add_histogram(name, sums, batch_idx+ args.epochs*batch_idx)
                        
            optimizer.step()
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

        #calculate explanation validation final 
        # normalize_expl_val = len(train_loader)/args.batch_size
        normalize_expl_val = len(train_loader)
        expl_preceion = expl_preceion/normalize_expl_val
        expl_recall = expl_recall/normalize_expl_val  
        print("- explanation precision: {:.6f} , recal {:.6f}".format(expl_preceion,expl_recall))
            
        return model
    
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
        preds, labels = [], []
        confusion_matrix = torch.zeros(nclasses, nclasses)
        
        # expl_preceion = 0
        # expl_recall = 0
        # layer_gc = LayerGradCam(self.model, self.model.layer2[1].conv2)

        with torch.no_grad():
            for i, (data, target) in enumerate(test_loader):
                if i%5000 ==0 and i>100:
                    print('Test Iterator is on :',i)
#                     break
                data, target = data.to(device), target.long().to(device)
                feat8,feat16,feat32,output= model(data)
#                 output_1, output_2 = torch.split(output, split_size_or_sections=output.shape[0] // 2)
#                 output = output_1
                loss = self.sord_loss(logits=output, ground_truth=target, num_classes=nclasses, multiplier=args.multiplier)
                test_losses.append(loss.item())
                pred = F.softmax(output, dim=1).max(1, keepdim=True)[1]
                correct += pred.eq(target.view_as(pred)).sum().item()

                # compute metrics
                preds.append(pred.view(-1).cpu())
                labels.append(target.view(-1).cpu())

                # gc_attr = layer_gc.attribute(data, target=preds, relu_attributions=False)
                # upsampled_attr = LayerAttribution.interpolate(gc_attr, (224, 224))
                # upsampled_attr = F.sigmoid(upsampled_attr).view_as(upsampled_attr)
                # a,b = self.calculate_measures(self.sintetic[:upsampled_attr.size()[0],:,:,:],upsampled_attr)

                # expl_preceion += a/upsampled_attr.size()[0]
                # expl_recall += b/upsampled_attr.size()[0]

                for t, p in zip(target.view(-1), pred.view(-1)):
                    confusion_matrix[t.long(), p.long()] += 1
        test_loss = np.mean(np.asarray(test_losses))

        # expl_preceion = expl_preceion/len(self.test_loader)
        # expl_recall = expl_recall/len(self.test_loader)

        precision, recall, fscore, _ = precision_recall_fscore_support(
            y_true=torch.cat(labels), y_pred=torch.cat(preds), average='micro')
        per_class_accuracy = confusion_matrix.diag() / confusion_matrix.sum(1)
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
                # with np.printoptions(threshold=np.inf):
                #   print(mask[0],gt[0])
                # break
                # print(mask,truncate=False)
                
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
    def vis_explanation(self,args,number,epoch):
      args.pic_viz_dir = os.path.join('logs', args.run_name, 'pic')
    
      if len(self.explainVis) == 0:
        for i, batch in enumerate(self.test_loader):
          self.explainVis = batch
          break


      # oldIndices = self.test_loader.indices.copy()
      # self.test_loader.indices = self.test_loader.indices[:2]

      # datasetLoader = self.test_loader 
      def wrapper(img):
        output, _ = self.model(img)

        output_1, output_2 = torch.split(output, split_size_or_sections=output.shape[0] // 2)
        pred = output_1
#         predlb = torch.argmax(pred,1)

        return pred
        
      layer_gc = LayerGradCam(wrapper, self.model.block3[3])

      # for i, batch in enumerate(datasetLoader):

      lb = self.explainVis[1].to(device)
      print(len(lb))
      img = self.explainVis[0].to(device)
      # plt.subplot(2,1,1)
      # plt.imshow(img.squeeze().cpu().numpy())
      
#       pred,_ = self.model(img)
      output, _ = self.model(img)
                
      output_1, output_2 = torch.split(output, split_size_or_sections=output.shape[0] // 2)
      pred = output_1
      predlb = torch.argmax(pred,1)

      print('Prediction label is :',predlb.cpu().numpy())
      print('Ground Truth label is: ',lb.cpu().numpy())
      ##explain to me :
      gc_attr = layer_gc.attribute(img, target=predlb, relu_attributions=False)
      upsampled_attr = LayerAttribution.interpolate(gc_attr, (64, 64))

      gc_attr = layer_gc.attribute(img, target=lb, relu_attributions=False)
      upsampled_attrB = LayerAttribution.interpolate(gc_attr, (64, 64))
#       if not os.path.exists('./pic'):
#         os.mkdir('./pic')
#       stn_out = F.interpolate(self.model.stn(upsampled_attr.detach())[7].cpu(), size=(args.img_size, args.img_size))
#       stn_out_B = F.interpolate(self.model.stn(upsampled_attrB.detach())[7].cpu(), size=(args.img_size, args.img_size))
        
#       stn_out_1, stn_out_2 = torch.split(stn_out, split_size_or_sections=stn_out.shape[0] // 2)
#       stn_out_1_B, stn_out_2_B = torch.split(stn_out_B, split_size_or_sections=stn_out_B.shape[0] // 2)
        
#       viz_tensor = torch.cat([upsampled_attr[7].detach(), stn_out_1, stn_out_2], dim=3)
#       viz_tensor_B = torch.cat([upsampled_attrB[7].detach(), stn_out_1_B, stn_out_2_B], dim=3)
        
        
      ####PLot################################################
#       plotMe = viz.visualize_image_attr(upsampled_attr[7].detach().cpu().numpy().transpose([1,2,0]),
#                             original_image=img[7].detach().cpu().numpy().transpose([1,2,0]),
#                             method='heat_map',
#                             sign='all', plt_fig_axis=None, outlier_perc=2,
#                             cmap='inferno', alpha_overlay=0.2, show_colorbar=True,
#                             title=str(predlb[7]),
#                             fig_size=(8, 10), use_pyplot=True)
        
# #       pil_img = Image.fromarray(plotMe[0]).convert('RGB')
# #       pil_img = torchvision.transforms.ToTensor(pil_img)
        
#       stn_out = F.interpolate(self.model.stn(plotMe[0]), size=(args.img_size, args.img_size))
#       stn_out_1, stn_out_2 = torch.split(stn_out, split_size_or_sections=stn_out.shape[0] // 2)
#       viz_tensor = torch.cat([plotMe[0], stn_out_1, stn_out_2], dim=3)

#       plotMe[0].savefig(args.pic_viz_dir+'/'+str(number+number*epoch)+'STNNotEQPred.jpg')

      ####PLot################################################
      plotMe = viz.visualize_image_attr(upsampled_attr[7].detach().cpu().numpy().transpose([1,2,0]),
                            original_image=img[7].detach().cpu().numpy().transpose([1,2,0]),
                            method='heat_map',
                            sign='all', plt_fig_axis=None, outlier_perc=2,
                            cmap='inferno', alpha_overlay=0.2, show_colorbar=True,
                            title=str(predlb[7]),
                            fig_size=(8, 10), use_pyplot=True)

      plotMe[0].savefig(args.pic_viz_dir+'/'+str(number+number*epoch)+'NotEQPred.jpg')
        ################################################

      plotMe = viz.visualize_image_attr(upsampled_attrB[7].detach().cpu().numpy().transpose([1,2,0]),
                            original_image=img[7].detach().cpu().numpy().transpose([1,2,0]),
                            method='heat_map',
                            sign='all', plt_fig_axis=None, outlier_perc=2,
                            cmap='inferno', alpha_overlay=0.9, show_colorbar=True,
                            title=str(lb[7].cpu()),
                            fig_size=(8, 10), use_pyplot=True)
                            
      plotMe[0].savefig(args.pic_viz_dir+'/'+str(number+number*epoch)+'NotEQLabel.jpg')
        ################################################

      outImg = img[7].squeeze().detach().cpu().numpy().transpose([1,2,0])
      fig2 = plt.figure(figsize=(12,12))
      prImg = plt.imshow(outImg)
      fig2.savefig(args.pic_viz_dir+'/'+str(number+number*epoch)+'NotEQOrig.jpg')
      ################################################
      fig = plt.figure(figsize=(15,10))
      ax = fig.add_subplot(111, projection='3d')

      z = upsampled_attr[7].squeeze().detach().cpu().numpy()
      x = np.arange(0,64,1)
      y = np.arange(0,64,1)
      X, Y = np.meshgrid(x, y)
          
      plll = ax.plot_surface(X, Y , z, cmap=cm.coolwarm)
      # Customize the z axis.
      ax.set_zlim(np.min(z)+0.1*np.min(z),np.max(z)+0.1*np.max(z))
      ax.zaxis.set_major_locator(LinearLocator(10))
      ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

      # Add a color bar which maps values to colors.
      fig.colorbar(plll, shrink=0.5, aspect=5)
      fig.savefig(args.pic_viz_dir+'/'+str(number+number*epoch)+'NotEQ3D.jpg')


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
    def filter_list(self,full_list, excludes):
        s = set(excludes)
        return list(x for x in full_list if x not in s)

    def move_unknown(self,indx):
        # print(self.unknown.__dict__.keys())
        # print(len(self.unknown.indices))
        for i in indx:
            self.known.indices.append(i)
            self.unknown.indices.remove(i)
            
        self.unknown.indices = self.filter_list(self.unknown.indices,indx)
        print('moved all new samples from unknown to known.')
        

if __name__ == '__main__':
#     cudnn.benchmark = True
    args = parse_arguments()
#     print('AAAAAARRRRRGGGGGG',args)
    
    manager = ModelManager(args)
    #just load trained model
    manager.load_model()
#     manager.visual_expl_stn()
    #train on one epoch and extract grads
    
    ###############################
    # manager.train_normal_extractGrad()
    manager.train_explanation_extractGrad()
    # manager.train_normal_extractGrad()
    
    ###############################
    
#   manager.train_known_expl()
#   manager.test_vis(args)
    # manager.validate_test(None)