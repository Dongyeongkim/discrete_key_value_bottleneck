import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim

from dataset import clustered2D
from torch.utils.data import DataLoader

from model import MLP

from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt



### Testing MLP distillation 


# Setting two MLP 

targetMLP = MLP(feature_num=32).cuda()
studentMLP = MLP(feature_num=32).cuda()

# temp of target and online 

temp_t = 1
temp_o = 1

# Set the parameters of all MLP as freeze state

for p in targetMLP.parameters():
    p.requires_grad = False

for p in studentMLP.parameters():
    p.requires_grad = False

# make random value vectors to represent or to be represented.

targetD_num = 8
studentD_num = 8

targetD = [torch.randn((1,32), device="cuda", requires_grad=True) for _ in range(targetD_num)]
studentD = [torch.randn((1,32), device="cuda", requires_grad=True) for _ in range(studentD_num)]

# pretrain the targetD

pretrain_numepochs = 1

for i in range(targetD_num):
    celoss = nn.CrossEntropyLoss()
    targetoptim = optim.Adam([targetD[i]], lr=3e-3)
    
    loss_set = []
    for j in tqdm(range(pretrain_numepochs)):
        target_set = []
        plt.ylim(0,1)
        for _ in range(1000):
            targetoptim.zero_grad()
            target = targetMLP(targetD[i])
            lbls = torch.Tensor([i]).long().cuda()
            loss = celoss(temp_t*target, lbls)
            loss_set.append(loss.data.cpu())
            loss.backward()
            targetoptim.step()
        print(loss.data)
    targetsoftmax = target.softmax(dim=1).cpu().detach()
    plt.plot(np.arange(32), targetsoftmax.squeeze())
    plt.savefig('a1/pretrain/mlp'+str(i)+'.png')
    plt.close()

# freeze targetD

for targetDval in targetD:
    targetDval.requires_grad = False

MLP_numepochs = 1

for i in range(targetD_num):
    cedistillloss = nn.CrossEntropyLoss()
    optimizer = optim.Adam([studentD[i]], lr=3e-3)
    
    # Calculating target and inp and update / check the performance
    for j in tqdm(range(MLP_numepochs)):
        plt.ylim(0,1)
        pred_list = []
        for _ in range(1000):
            optimizer.zero_grad()
            target = targetMLP(targetD[i]).softmax(dim=1)
            inp = temp_o*studentMLP(studentD[i])
            loss = cedistillloss(inp, temp_t*target)
            loss.backward()
            optimizer.step()
        for _ in range(100):
            inp = temp_o*studentMLP(studentD[i])
            if torch.argmax(inp, dim=1) == i:
                pred_list.append(1)
            else:
                pred_list.append(0)
            
        print(loss.data)
        print(sum(pred_list)/len(pred_list))
    plt.plot(np.arange(32), target.cpu().detach().squeeze())
    inpsoftmax = inp.softmax(dim=1).cpu().detach()
    plt.plot(np.arange(32), inpsoftmax.squeeze())
    plt.savefig('a1/distill/mlp'+str(i)+'.png')
    plt.close()

for studentDval in studentD:
    studentDval.requires_grad = False

loss_avg = []
for i in range(targetD_num):

    output_student = studentMLP(studentD[i])
    output_target = targetMLP(targetD[i]).detach()

    testloss = nn.CrossEntropyLoss()

    loss = testloss(output_student, output_target.softmax(dim=1))

    print("Simple MLP distillation: "+str(loss.data))
    loss_avg.append(loss.data)

print("Average Loss: "+str(sum(loss_avg)/len(loss_avg)))

