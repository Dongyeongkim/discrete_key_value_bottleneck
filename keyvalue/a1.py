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
targetoptim = optim.Adam([targetD], lr=1e-5)

pretrain_numepochs = 100

loss_set = []
for i in tqdm(range(pretrain_numepochs)):
    for j in range(10000):
        targetoptim.zero_grad()
        target = targetMLP(targetD)
        lbls = torch.Tensor([1]).long().cuda()
        loss = celoss(target, lbls)
        loss_set.append(loss.data.cpu())
        loss.backward()
        targetoptim.step()
    print(loss.data)

plt.plot(np.arange(start=0, stop = len(loss_set)), loss_set)
plt.savefig('MLPpretrainloss.png')
plt.close()

# freeze targetD

targetD.requires_grad = False

# Set the loss function

cedistillloss = nn.CrossEntropyLoss()

# set the optimizer 

optimizer = optim.Adam([studentD], lr=1e-4)
# Calculating target and inp and update / check the performance

MLP_numepochs = 100

loss_set = []
for i in tqdm(range(MLP_numepochs)):
    for j in range(10000):
        optimizer.zero_grad()
        target = targetMLP(targetD).softmax(dim=1)
        inp = studentMLP(studentD)
        loss = cedistillloss(inp, target)
        loss.backward()
        loss_set.append(loss.detach().cpu())
        optimizer.step()

plt.plot(np.arange(start=0, stop = len(loss_set)), loss_set)
plt.savefig('MLPdistillloss.png')
plt.close()

studentD.requires_grad_ = False

output_student = studentMLP(studentD)
output_target = targetMLP(targetD).detach()

testloss = nn.CrossEntropyLoss()

loss = testloss(output_student, output_target.softmax(dim=1))

print("Simple MLP distillation: "+str(loss.data))


