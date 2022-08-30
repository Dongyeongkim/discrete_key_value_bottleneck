import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim

from dataset import clustered2D
from torch.utils.data import DataLoader

from model import MLPb, VQMLP

from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt



### Testing MLP distillation 


# Setting two MLP 

targetMLP = MLPb(feature_num=10).cuda()
studentMLP = MLPb(feature_num=20).cuda()

# Set the parameters of all MLP as freeze state

for p in targetMLP.parameters():
    p.requires_grad = False

for p in studentMLP.parameters():
    p.requires_grad = False

# make random value vectors to represent or to be represented.

targetD = torch.randn((1,10), device="cuda", requires_grad=True)
studentD = torch.randn((1,20), device="cuda", requires_grad=True)

# pretrain the targetD

celoss = nn.CrossEntropyLoss()
targetoptim = optim.Adam([targetD], lr=3e-4)

pretrain_numepochs = 25

loss_set = []
for i in tqdm(range(pretrain_numepochs)):
    target_set = []
    plt.ylim(0,0.3)
    for j in range(10000):
        targetoptim.zero_grad()
        target = targetMLP(targetD)
        lbls = torch.Tensor([5]).long().cuda()
        loss = celoss(target, lbls)
        loss_set.append(loss.data.cpu())
        loss.backward()
        targetoptim.step()
    print(loss.data)
    
    targetsoftmax = target.softmax(dim=1).cpu().detach()
    plt.plot(np.arange(32), targetsoftmax.squeeze())
    plt.savefig('MLPpretrainDistribution'+str(i)+'.png')
    plt.close()

# freeze targetD

targetD.requires_grad = False

# Set the loss function

cedistillloss = nn.CrossEntropyLoss()

# set the optimizer 

optimizer = optim.Adam([studentD], lr=3e-4)
# Calculating target and inp and update / check the performance


MLP_numepochs = 25

for i in tqdm(range(MLP_numepochs)):
    plt.ylim(0,0.3)
    pred_list = []
    for j in range(10000):
        optimizer.zero_grad()
        target = targetMLP(targetD).softmax(dim=1)
        inp = 4*studentMLP(studentD)
        if torch.argmax(inp, dim=1) == 5:
            pred_list.append(1)
        else:
            pred_list.append(0)
        loss = cedistillloss(inp, target)
        loss.backward()
        optimizer.step()
    print(loss.data)
    print(sum(pred_list)/len(pred_list))
    plt.plot(np.arange(32), target.cpu().detach().squeeze())
    inpsoftmax = inp.softmax(dim=1).cpu().detach()
    plt.plot(np.arange(32), inpsoftmax.squeeze())
    plt.savefig('MLPDistillDistribution'+str(i)+'.png')
    plt.close()

studentD.requires_grad_ = False

output_student = studentMLP(studentD)
output_target = targetMLP(targetD).detach()

testloss = nn.CrossEntropyLoss()

loss = testloss(output_student, output_target.softmax(dim=1))

print("Simple MLP distillation: "+str(loss.data))


