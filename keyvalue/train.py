import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch import optim
from model import MLPb, VQMLP, VQEMAMLP, KVMLP
from dataset import clustered2D
from visualizer import visualizer, visualizer_without_scat, visualizer_VQ, visualizer_without_scat_VQ, visualizer_KV, visualizer_without_scat_KV

# Data generation

dataset = clustered2D()
traindataloader = DataLoader(dataset, batch_size=100, shuffle=False)
testdataloader = DataLoader(dataset, batch_size=800, shuffle=True)


# MLP test of two groups

## Random initialization

MLP_baseline = MLPb(feature_num=2).cuda()

for x, (imgs, lbls) in enumerate(testdataloader):
    imgs = imgs.cuda()
    lbls = lbls.cuda()
visualizer_without_scat(imgs, lbls, MLP_baseline, 'MLP0', 3)

optimizer = optim.Adam(MLP_baseline.parameters(), lr=5e-3)

criterion = nn.CrossEntropyLoss()

## Training 

for x, (imgs, lbls) in enumerate(traindataloader):
    imgs = imgs.cuda()
    lbls = lbls.cuda()
    for j in range(1000):
        optimizer.zero_grad()
        output = MLP_baseline(imgs)
        loss = criterion(output, lbls)
        loss.backward()
        optimizer.step()
    print(loss)
    visualizer(imgs, lbls, MLP_baseline, 'MLP'+str(x+1), 3)

for p in MLP_baseline.parameters():
    p.requires_grad = False

for x, (imgs, lbls) in enumerate(testdataloader):
    imgs = imgs.cuda()
    lbls = lbls.cuda()
    visualizer(imgs, lbls, MLP_baseline, 'MLP5', 3)

# VQ+MLP test of two groups

## Random initialization

VQMLP = VQMLP(feature_num=2, codebook_num_embeddings=100, codebook_embeddings_dim=2).cuda()
trainbefore = VQMLP.vq._embedding.weight
for x, (imgs, lbls) in enumerate(testdataloader):
    imgs = imgs.cuda()
    lbls = lbls.cuda()
visualizer_without_scat_VQ(imgs, lbls, VQMLP, 'VQMLP0', 5)

optimizer = optim.Adam(VQMLP.parameters(), lr=7e-3)

criterion = nn.CrossEntropyLoss()

## Training 

for x, (imgs, lbls) in enumerate(traindataloader):
    imgs = imgs.cuda()
    lbls = lbls.cuda()
    for j in range(1000):
        optimizer.zero_grad()
        q_latent_loss, output = VQMLP(imgs)
        loss = criterion(output, lbls) + q_latent_loss
        loss.backward()
        optimizer.step()
    print(loss)
    visualizer_VQ(imgs, lbls, VQMLP, 'VQMLP'+str(x+1), 5)


for p in VQMLP.parameters():
    p.requires_grad = False

for x, (imgs, lbls) in enumerate(testdataloader):
    imgs = imgs.cuda()
    lbls = lbls.cuda()
    visualizer_VQ(imgs, lbls, VQMLP, 'VQMLP5', 5)


# VQ(EMA)+MLP test of two groups

## Random initialization

VQEMAMLP = VQEMAMLP(feature_num=2, codebook_num_embeddings=100, codebook_embeddings_dim=2).cuda()
for x, (imgs, lbls) in enumerate(testdataloader):
    imgs = imgs.cuda()
    lbls = lbls.cuda()
visualizer_without_scat_VQ(imgs, lbls, VQEMAMLP, 'VQEMAMLP0', 8)

optimizer = optim.Adam(VQEMAMLP.parameters(), lr=7e-3)

criterion = nn.CrossEntropyLoss()

## Training 

for x, (imgs, lbls) in enumerate(traindataloader):
    imgs = imgs.cuda()
    lbls = lbls.cuda()
    for j in range(1000):
        optimizer.zero_grad()
        q_latent_loss, output = VQEMAMLP(imgs)
        loss = criterion(output, lbls) + q_latent_loss
        loss.backward()
        optimizer.step()
    print(loss)
    visualizer_VQ(imgs, lbls, VQEMAMLP, 'VQEMAMLP'+str(x+1), 8)


for p in VQEMAMLP.parameters():
    p.requires_grad = False

for x, (imgs, lbls) in enumerate(testdataloader):
    imgs = imgs.cuda()
    lbls = lbls.cuda()
    visualizer_VQ(imgs, lbls, VQEMAMLP, 'VQEMAMLP5', 8)


# KV+MLP test of two groups

## Random initialization

KVMLP = KVMLP(feature_num=32, key_num_embeddings=100, key_embeddings_dim=2, value_embeddings_dim=32).cuda()
for x, (imgs, lbls) in enumerate(testdataloader):
    imgs = imgs.cuda()
    lbls = lbls.cuda()
visualizer_without_scat_KV(imgs, lbls, KVMLP, 'KVMLP0', 5)

for p in KVMLP.parameters():
    p.requires_grad = False
optimizer = optim.Adam([KVMLP.keyvalmem.values], lr=3e-3)

criterion = nn.CrossEntropyLoss()

## Training 

for x, (imgs, lbls) in enumerate(traindataloader):
    imgs = imgs.cuda()
    lbls = lbls.cuda()
    for j in range(1000):
        optimizer.zero_grad()
        output = KVMLP(imgs)
        loss = criterion(output, lbls)
        loss.requires_grad = True
        loss.backward()
        optimizer.step()
    print(loss)
    visualizer_KV(imgs, lbls, KVMLP, 'KVMLP'+str(x+1), 5)




for x, (imgs, lbls) in enumerate(testdataloader):
    imgs = imgs.cuda()
    lbls = lbls.cuda()
    visualizer_KV(imgs, lbls, KVMLP, 'KVMLP5', 5)
