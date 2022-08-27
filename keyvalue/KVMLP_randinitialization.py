import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch import optim
from model import MLPb, VQMLP, VQEMAMLP, KVMLP
from dataset import clustered2D
from visualizer import visualizer, visualizer_without_scat, visualizer_VQ, visualizer_without_scat_VQ, visualizer_KV, visualizer_without_scat_KV
from tqdm import tqdm
import numpy as np
# configuration

num_epochs = 10000 # key initializing epochs

# Data generation

dataset = clustered2D()
traindataloader = DataLoader(dataset, batch_size=100, shuffle=False)
testdataloader = DataLoader(dataset, batch_size=800, shuffle=True)

# Random imitialization in Unit Sqaure

h = .02  # step size in the mesh
x_min, x_max = 0, 8
y_min, y_max = 0, 8
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
positions = torch.Tensor(np.vstack(list(zip(xx.ravel(), yy.ravel())))).cuda()
# KV+MLP test of two groups

## Random initialization

KVMLP = KVMLP(feature_num=32, key_num_embeddings=100, key_embeddings_dim=2, value_embeddings_dim=32).cuda()


for i in tqdm(range(num_epochs)):
    KVMLP(positions)

for x, (imgs, lbls) in enumerate(testdataloader):
    visualizer_without_scat_KV(imgs, lbls, KVMLP, 'KVMLP0', 5)

KVMLP.keyvalmem.vq._ema_w.requires_grad = False


for p in KVMLP.parameters():
    p.requires_grad = False

KVMLP.keyvalmem.values.requires_grad = True

optimizer = optim.Adam([KVMLP.keyvalmem.values], lr=3e-3)

criterion = nn.CrossEntropyLoss()

## Training

print(KVMLP.MLP.linear1.weight)

for x, (imgs, lbls) in enumerate(traindataloader):
    imgs = imgs.cuda()
    lbls = lbls.cuda()
    for j in range(1000):
        optimizer.zero_grad()
        output = KVMLP(imgs)
        loss = criterion(output, lbls)
        loss.backward()
        optimizer.step()
    print(loss.data)
    visualizer_KV(imgs, lbls, KVMLP, 'KVMLP'+str(x+1), 5)

KVMLP.keyvalmem.values.requires_grad = False 

for x, (imgs, lbls) in enumerate(testdataloader):
    imgs = imgs.cuda()
    lbls = lbls.cuda()
    output = KVMLP(imgs)
    loss = criterion(output, lbls)
    print(loss.data)
    visualizer_KV(imgs, lbls, KVMLP, 'KVMLPpretrained0', 15)

KVMLP.MLP.linear1.reset_parameters()
print(KVMLP.MLP.linear1.weight)

for x, (imgs, lbls) in enumerate(testdataloader):
    imgs = imgs.cuda()
    lbls = lbls.cuda()
    output = KVMLP(imgs)
    loss = criterion(output, lbls)
    print(loss.data)
    visualizer_KV(imgs, lbls, KVMLP, 'KVMLPrandinitialized0', 20)

print(KVMLP.MLP.linear1.weight)

KVMLP.keyvalmem.values.requires_grad = True

for x, (imgs, lbls) in enumerate(traindataloader):
    imgs = imgs.cuda()
    lbls = lbls.cuda()
    for j in range(1000):
        optimizer.zero_grad()
        output = KVMLP(imgs)
        loss = criterion(output, lbls)
        loss.backward()
        optimizer.step()
    print(loss.data)
    visualizer_KV(imgs, lbls, KVMLP, 'KVMLPRein'+str(x+1), 9)

KVMLP.keyvalmem.values.requires_grad = False
