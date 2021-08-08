from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import datasets, transforms, utils
from torch.autograd import Variable
import torch.utils.data as tud
import numpy as np
from torchvision import models
import matplotlib.pyplot as plt
import os
import sys
import time
import pandas as pd
from pandas import DataFrame
from PIL import Image
import random
import logging
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from tripletloss import TripletLoss
from FusionNet import FusionNet
from MyDataset import MyDataset
from dcnn import run_node_classification3, A_to_diffusion_kernel
from scatterplot import ScatterPlot, ScatterGroup


class Logger(object):
    def __init__(self, fileN="Default.log"):
        self.terminal = sys.stdout
        self.log = open(fileN, "w")
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.flush()
    def flush(self):
        self.log.flush()
 
sys.stdout = Logger("log_filefusion.txt")

USE_CUDA = torch.cuda.is_available()
if torch.cuda.is_available():
    print('cuda!')
else:
    print('false')

# Fixed random seed to ensure program recurrence
seed_numder = 1 
random.seed(seed_numder)
np.random.seed(seed_numder)
torch.manual_seed(seed_numder)
if USE_CUDA:
    torch.cuda.manual_seed(seed_numder)
    torch.cuda.manual_seed_all(seed_numder)
if seed_numder == 1:
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

batch_size = 32
learning_rate = 0.0001
EPOCH = 100  # number of train epoch
num_features = 75  # number of diffusion convolution feature
num_hops = 3  # number of diffusion convolution hop

train_transforms = transforms.Compose([
    transforms.RandomResizedCrop(224), #avoid overfitting
    # transforms.Resize(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((.5, .5, .5), (.5, .5, .5))
])
eval_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomResizedCrop(224),
    transforms.ToTensor(),
    transforms.Normalize((.5, .5, .5), (.5, .5, .5))
])


def GetData():
    '''
        get the DOM and DSM data
    '''
    # labelpath = 'labelnew\\label12345.txt'  # label directory
    labelpath = 'labelnew\\label12345_12345.txt'  # label directory
    fh = open(labelpath, 'r')
    imgs = []
    for line in fh:
        line = line.strip('\n')
        line = line.rstrip() # 删除 string 字符串末尾的指定字符（默认为空格）
        words = line.split()
        imgs.append((words[0],int(words[1])))
    A, X = run_node_classification3()
    Apow = A_to_diffusion_kernel(A, num_hops)
    Apow.tolist()
    return imgs, Apow

start_tratime = time.time()
gettime = time.time()
imgs, Apow = GetData()
getdata_time = time.time() - gettime
print('>>getdata time:{:.3f}s'.format(getdata_time))

eval_pre_list = []
eval_rec_list = []
eval_F1_list = []

for ndata in range(0, 1):
    # Fixed random number seed to ensure program recurrence
    seed_numder = 1
    random.seed(seed_numder)
    np.random.seed(seed_numder)
    torch.manual_seed(seed_numder)
    if USE_CUDA:
        torch.cuda.manual_seed(seed_numder)
        torch.cuda.manual_seed_all(seed_numder)
    if seed_numder == 1:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    print('------------- {} --------------'.format(ndata+1))
    testimgs = imgs[432*5:]  #
    trainimgs = imgs[:432*5]  # =imgs[:]
    trainApow = Apow[:432*5, :, :]
    testApow = Apow[432*5:, :, :]

    # testimgs = imgs[432*ndata:432*(ndata+1)]
    # trainimgs = imgs.copy() # =imgs[:]
    # del trainimgs[432*ndata:432*(ndata+1)] #去掉测试数据
    # trainApow = np.vstack((Apow[:18 * 24 * ndata, :, :], Apow[18 * 24 * (ndata + 1):, :, :]))
    # testApow = Apow[432 * ndata:432 * (ndata + 1), :, :]  # Apow data, ndarray

    train_dataset = MyDataset(trainimgs, trainApow,transform = train_transforms)
    train_dataloader = tud.DataLoader(train_dataset, batch_size = batch_size, shuffle=True)
    eval_dataset = MyDataset(testimgs, testApow, transform = eval_transforms)
    eval_dataloader = tud.DataLoader(eval_dataset, batch_size = batch_size)

    #--------------------training network---------------------------------
    
    model = FusionNet()
    logging.info('model:{}'.format(model))
    if torch.cuda.is_available():
        # print('cuda!')
        model.cuda()
    dcnn_net_param = list(map(id, model.dcnn_net.parameters()))
    dense_param = list(map(id, model.myclassifier.parameters()))

    base_param = filter(lambda p: id(p) not in dcnn_net_param, model.parameters())
    base_param = filter(lambda p: id(p) not in dense_param, base_param)
    params = [{'params': md.parameters()} for md in model.children()
            if md in [model]]
    optimizer = optim.Adam([{'params': model.dcnn_net.parameters(), 'lr': learning_rate*500},
                            {'params': model.myclassifier.parameters(), 'lr': learning_rate*100},
                            {'params': base_param}

    ], lr=learning_rate)

    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=8, gamma=0.1) #每8个epoch调整lr*gamma
    loss_func2 = nn.CrossEntropyLoss()
    # loss_func1 = TripletLoss()
    Loss_list = []
    F1_list = []
    for epoch in range(EPOCH):
        print('epoch {}'.format(epoch))
        # training-----------------------------
        model.train()
        train_loss, train_loss1 = 0., 0.
        train_acc, train_pre, train_rec = 0., 0., 0.
        count = 0
        tra_conmat = np.zeros((2,2)) #混淆矩阵
        tra_zvis, tra_zdsm, tra_zmean = 0, 0, 0.

        for batch_x, batch_y, Apow_n in train_dataloader:
            batch_x, batch_y = Variable(batch_x).cuda(), Variable(batch_y).cuda()
            Apow_n = Variable(Apow_n.float()).cuda()
            lasize=batch_y.squeeze(0).size()
            out = model(batch_x,  Apow_n)

            loss2 = loss_func2(out, batch_y) #cross-entropy
            loss = loss2
            train_loss += loss.item()
            pred = torch.max(out, 1)[1]
            train_correct = (pred == batch_y).sum()

            y_pred = pred.cpu().numpy()
            y_true = batch_y.cpu().numpy()
            tra_conmat += confusion_matrix(y_true, y_pred, labels=[0,1])
            train_acc += train_correct.item()        
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            count += 1

        exp_lr_scheduler.step()
        print('lr:{:.6f}'.format(optimizer.param_groups[0]['lr']))
        t_mat = np.array(tra_conmat)
        train_pre = float(t_mat[1][1]) / (t_mat[:, 1].sum() + 1e-4)
        train_rec = float(t_mat[1][1]) / (t_mat[1, :].sum() + 1e-4)
        train_acc = train_acc / (len(train_dataset)) # 精确率
        train_loss = train_loss / count

        if (train_rec + train_pre) != 0:
            train_F1 = 2 * train_rec * train_pre / (train_rec + train_pre)  #micro
        else:
            train_F1 = 0
        print('Train Loss: {:.5f}, Acc: {:.5f}, Pre: {:.5f}, Rec: {:.5f}, F1:{:.5f}'.\
            format(train_loss, train_acc, train_pre, train_rec, train_F1))

        Loss_list.append(train_loss)
        F1_list.append(train_F1)
        if epoch>10 and abs(Loss_list[epoch-1]-Loss_list[epoch])<0.008 and abs(Loss_list[epoch-1]-Loss_list[epoch-2])<0.01:  # Convergence judgment
            break
        if epoch > 20:
            break
    train_time = time.time() - start_tratime
    # evaluation--------------------------------
    start_testtime = time.time()
    model.eval()
    eval_loss = 0.
    eval_acc, eval_pre, eval_rec = 0., 0., 0.
    neweval_pre, neweval_rec = 0., 0.
    eval_conmat = np.zeros((2,2))  # confusion matrix
    neweval_conmat = np.zeros((2,2))
    zvis, zdsm = 0, 0
    zvis0,zvis1,zdsm0,zdsm1 = 0,0,0,0
    count = 0
    y_predall = []
    y_trueall = []
    dsmnewall = []
    contribute = np.zeros(len(eval_dataset), dtype=float) # 要改！！！
    group0 = np.zeros(200, dtype=float)
    group1 = np.zeros(200, dtype=float)
    x = 0
    for batch_x, batch_y, Apow_n in eval_dataloader:
        with torch.no_grad():
            batch_x, batch_y = Variable(batch_x).cuda(), Variable(batch_y).cuda()
            # dsm_f = Variable(dsm_f.float()).cuda()
            Apow_n = Variable(Apow_n.float()).cuda()
            # batch_x, batch_y = Variable(batch_x), Variable(batch_y)

        out = model(batch_x, Apow_n)
        loss = loss_func2(out, batch_y) #交叉熵
        targets = batch_y.unsqueeze(1)  #1：升维的位数

        targets = targets.cpu().detach().numpy()
        eval_loss += loss.item()
        pred = torch.max(out, 1)[1]
        num_correct = (pred == batch_y).sum()

        y_pred = pred.cpu().numpy()
        y_true = batch_y.cpu().numpy()
        y_predall.extend(y_pred)
        y_trueall.extend(y_true)
        eval_conmat += confusion_matrix(y_true, y_pred, labels=[0,1])
        eval_acc += num_correct.item()
        count += 1

    del model, params, optimizer #delete model
    # with torch.cuda.device('cuda:0'):
    torch.cuda.empty_cache()
    time.sleep(3)
    group0 = group0/count
    group1 = group1/count
    # print(group0, group1)
    # ScatterPlot(group0, group1, dsmnewall, y_trueall, y_predall)  # draw scatterplot to analyse
    e_mat = np.array(eval_conmat)
    eval_pre = float(e_mat[1][1]) / (e_mat[:,1].sum() + 1e-4)
    eval_rec = float(e_mat[1][1]) / (e_mat[1,:].sum() + 1e-4)
    eval_acc = eval_acc / (len(eval_dataset)) # 精确率
    eval_loss = eval_loss / (count)
    eval_F1 = 0
    print(eval_F1)
    print(2 * eval_rec * eval_pre)
    print(eval_rec + eval_pre)
    if (eval_rec + eval_pre) != 0:
        eval_F1 = 2 * eval_rec * eval_pre / (eval_rec + eval_pre)
    else:
        eval_F1 = 0

    print('Test Loss: {:.5f}, Acc: {:.5f}, Pre: {:.5f}, Rec: {:.5f}, F1:{:.5f}'.\
        format(eval_loss, eval_acc, eval_pre, eval_rec, eval_F1))  
    eval_pre_list.append(eval_pre)
    eval_rec_list.append(eval_rec)
    eval_F1_list.append(eval_F1)
    print(eval_conmat)

    print(classification_report(y_trueall, y_predall, target_names=['0','1'], digits=3))

    x1 = range(0, len(Loss_list))
    x2 = range(0, len(Loss_list))
    y1 = F1_list
    y2 = Loss_list
    plt.subplot(2, 1, 1)
    plt.plot(x1, y2, 'o-')
    # plt.subplot(2, 1, 1)
    # plt.plot(x1, y1, 'o-')
    plt.xlabel('Train loss vs. epoches')
    plt.ylabel('Train loss')
    plt.subplot(2, 1, 2)
    plt.plot(x2, y1, 'o-')
    plt.xlabel('Train F1 vs. epoches')
    plt.ylabel('Train F1')
    # plt.savefig("vggtest4.jpg") #在show之前
    with open('test.txt', 'a') as file0:
        print('{} 5Dataset: pre:{:.3f}, rec:{:.3f}, F1:{:.3f}'.format(ndata + 1, eval_pre, eval_rec, eval_F1), file=file0)

test_time = time.time() - start_testtime
all_time = time.time() - start_tratime
print('-'*20)
print('>>train time:{:.0f}m{:.0f}s'.format(train_time//60, train_time%60))
print('>>test time:{:.0f}m{:.0f}s'.format(test_time//60, test_time%60))
print('>>all time:{:.0f}m{:.0f}s'.format(all_time//60, all_time%60))
pre = np.mean(eval_pre_list)
rec = np.mean(eval_rec_list)
F1 = np.mean(eval_F1_list)
eval_pre_list = [round(i, 3) for i in eval_pre_list]
eval_rec_list = [round(i, 3) for i in eval_rec_list]
eval_F1_list = [round(i, 3) for i in eval_F1_list]
print('pre:{}'.format(eval_pre_list))
print('rec:{}'.format(eval_rec_list))
print('f1:{}'.format(eval_F1_list))
plt.show()
