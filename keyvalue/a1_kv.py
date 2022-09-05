import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim

from dataset import clustered2D, diffclustered2D
from torch.utils.data import DataLoader

from model import KVMLP

from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from visualizer import visualizer, visualizer_without_scat, visualizer_VQ, visualizer_without_scat_VQ, visualizer_KV, visualizer_without_scat_KV




dataset = clustered2D()
traindataloader = DataLoader(dataset, batch_size=100, shuffle=False)
testdataloader = DataLoader(dataset, batch_size=800, shuffle=True)

distilled_dataset = diffclustered2D()
distill_traindataloader = DataLoader(distilled_dataset, batch_size=100, shuffle=False)
distill_testdataloader = DataLoader(distilled_dataset, batch_size=800, shuffle=True)

num_epochs = 10000

h = .02  # step size in the mesh
x_min, x_max = 0, 8
y_min, y_max = 0, 8
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
positions = torch.Tensor(np.vstack(list(zip(xx.ravel(), yy.ravel())))).cuda()

targetKV = KVMLP(feature_num=32, key_num_embeddings=100, key_embeddings_dim=2, value_embeddings_dim=32).cuda()
studentKV = KVMLP(feature_num=32, key_num_embeddings=256, key_embeddings_dim=2, value_embeddings_dim=32).cuda()

def traindataload(classnum):
    for x, (imgs, lbls) in enumerate(traindataloader):
        if x == classnum:
            return imgs.cuda()

for i in tqdm(range(num_epochs), desc="Initializing Keys in Key-Value Bottleneck"):
    targetKV(positions)

targetKV.keyvalmem.vq._ema_w.requires_grad = False

for x, (imgs, lbls) in enumerate(testdataloader):
    imgs = imgs.cuda()
    lbls = lbls.cuda()

visualizer_without_scat_KV(imgs, lbls, targetKV, 'targetKV0', 8)



for p in targetKV.parameters():
    p.requires_grad = False

targetKV.keyvalmem.values.requires_grad = True

optimizer = optim.Adam([targetKV.keyvalmem.values], lr=3e-3)

criterion = nn.CrossEntropyLoss()

## Training


for x, (imgs, lbls) in enumerate(traindataloader):
    imgs = imgs.cuda()
    lbls = lbls.cuda()
    for j in range(1000):
        optimizer.zero_grad()
        output = targetKV(imgs)
        loss = criterion(output, lbls)
        loss.backward()
        optimizer.step()
    print("targetKV: Batch %d, Loss %f" %(x+1, loss.data))
    visualizer_KV(imgs, lbls, targetKV, 'targetKV'+str(x+1), 8)

targetKV.keyvalmem.values.requires_grad = False 

for x, (imgs, lbls) in enumerate(testdataloader):
    imgs = imgs.cuda()
    output = targetKV(imgs)
    lbls = lbls.cuda()
    precision_set = []
    output = torch.argmax(targetKV(imgs), dim=1)
    for i, j in enumerate(lbls):
        if j == output[i]:
            precision_set.append(1)
        else:
            precision_set.append(0)
    
    print(sum(precision_set)/len(precision_set))
    
    visualizer_KV(imgs, lbls, targetKV, 'targetKV5', 8)




for i in tqdm(range(num_epochs), desc="Initializing Keys in Key-Value Bottleneck"):
    studentKV(positions)

studentKV.keyvalmem.vq._ema_w.requires_grad = False
visualizer_without_scat_KV(imgs, lbls, studentKV, 'studentKV0', 9)

for p in studentKV.parameters():
    p.requires_grad = False

studentKV.keyvalmem.values.requires_grad = True

optimizer = optim.Adam([studentKV.keyvalmem.values], lr=3e-3)

criterion = nn.CrossEntropyLoss()


for x, (imgs, ans) in enumerate(distill_traindataloader):
    imgs = imgs.cuda()
    pslbls = traindataload(x)
    for j in range(1000):
        optimizer.zero_grad()
        output = studentKV(imgs)
        lbls = targetKV(pslbls)
        loss = criterion(output, lbls.softmax(dim=1).detach())
        loss.backward()
        optimizer.step()
    print("studentKV: Batch %d, Loss %f" %(x+1, loss.data))
    visualizer_KV(imgs, torch.argmax(lbls, dim=1).cpu().detach(), studentKV, 'studentKV'+str(x+1), 9)

targetKV.keyvalmem.values.requires_grad = False 

for x, (imgs, lbls) in enumerate(distill_testdataloader):
    imgs = imgs.cuda()
    lbls = lbls.cuda()
    output = torch.argmax(studentKV(imgs), dim=1)
    precision_set = []
    for i, j in enumerate(lbls):
        if j == output[i]:
            precision_set.append(1)
        else:
            precision_set.append(0)
    
    print(sum(precision_set)/len(precision_set))
    visualizer_KV(imgs, lbls, studentKV, 'studentKV5', 9)
