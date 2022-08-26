import torch
import torch.nn as nn
import torch.optim as optim

from model import MLPb

from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt


# Setting two MLP 

targetMLP = MLPb(feature_num=10).cuda()
studentMLP = MLPb(feature_num=20).cuda()

# Set student MLP as freeze state

for p in targetMLP.parameters():
    p.requires_grad = False

for p in studentMLP.parameters():
    p.requires_grad = False


# make random value vectors to represent or to be represented.

targetD = torch.randn((1,10)).cuda()
studentD = torch.randn((1,20), device="cuda", requires_grad=True)

print(studentD)

# Set the loss function

celoss = nn.CrossEntropyLoss()

# set the optimizer 

optimizer = optim.Adam([studentD], lr=3e-3)
# Calculating target and inp and update / check the performance

loss_set = []
for i in tqdm(range(100000)):
    optimizer.zero_grad()
    target = targetMLP(targetD).softmax(dim=1)
    inp = studentMLP(studentD)
    loss = celoss(inp, target)
    loss.backward()
    loss_set.append(loss.detach().cpu())
    optimizer.step()

plt.plot(np.arange(start=0, stop = len(loss_set)), loss_set)
plt.show()
plt.savefig('lossgraph.png')

studentD.requires_grad_ = False

output_student = studentMLP(studentD)
output_target = targetMLP(targetD).detach()

testloss = nn.CrossEntropyLoss()

loss = testloss(output_student, output_target.softmax(dim=1))
print(output_student)
print(output_target)
print(loss)