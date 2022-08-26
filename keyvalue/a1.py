import torch
import torch.nn as nn
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

# Set student MLP as freeze state

for p in targetMLP.parameters():
    p.requires_grad = False

for p in studentMLP.parameters():
    p.requires_grad = False


# make random value vectors to represent or to be represented.

targetD = torch.randn((1,10)).cuda()
studentD = torch.randn((1,20), device="cuda", requires_grad=True)

# Set the loss function

celoss = nn.CrossEntropyLoss()

# set the optimizer 

optimizer = optim.Adam([studentD], lr=3e-3)
# Calculating target and inp and update / check the performance

loss_set = []
for i in tqdm(range(10000)):
    optimizer.zero_grad()
    target = targetMLP(targetD).softmax(dim=1)
    inp = studentMLP(studentD)
    loss = celoss(inp, target)
    loss.backward()
    loss_set.append(loss.detach().cpu())
    optimizer.step()

plt.plot(np.arange(start=0, stop = len(loss_set)), loss_set)
plt.show()
plt.savefig('MLPloss.png')

studentD.requires_grad_ = False

output_student = studentMLP(studentD)
output_target = targetMLP(targetD).detach()

testloss = nn.CrossEntropyLoss()

loss = testloss(output_student, output_target.softmax(dim=1))

print("Simple MLP distillation: "+str(loss))


### Testing VQMLP(Without EMA, VQ is updated with VQloss(q_latent_loss) but MLP has freezed)


# pretraining VQMLP to 2D datapoint cluster dataset (custom made)

targetVQMLP = VQMLP(feature_num=2, codebook_num_embeddings=100, codebook_embeddings_dim=2).cuda()

dataset = clustered2D()
pretraindataloader = DataLoader(dataset, batch_size=800, shuffle=True)
distilldataloader = DataLoader(dataset, batch_size=100, shuffle=True)

num_epochs = 10

optimizer = optim.Adam(targetVQMLP.parameters(), lr=3e-4)
celoss = nn.CrossEntropyLoss()

losslist = []
for i in tqdm(range(num_epochs)):
    for j, (imgs, lbls) in enumerate(pretraindataloader):
        imgs = imgs.cuda()
        lbls = lbls.cuda()
        for k in range(10000):
            optimizer.zero_grad()
            q_latent_loss, output = targetVQMLP(imgs)
            loss = celoss(output, lbls) + q_latent_loss
            losslist.append(loss.cpu().detach().numpy())
            loss.backward()
            optimizer.step()
plt.close()
plt.plot(np.arange(start=0, stop = len(losslist)), losslist)
plt.savefig('VQMLPPretrainloss.png')

# Setting Student VQMLP - makes only VQ code can be trained

StudentVQMLP = VQMLP(feature_num=2, codebook_num_embeddings=100, codebook_embeddings_dim=2).cuda()
StudentVQMLP.MLP.requires_grad = False

# Setting Target VQMLP - Freeze all things

for p in targetVQMLP.parameters():
    p.requires_grad = False

# Procedure of distilation

distil_epochs = 10

optimizer = optim.Adam([StudentVQMLP.vq._embedding.weight], lr=3e-4)
distill_loss = nn.CrossEntropyLoss()

distill_losslist = []

for i in tqdm(range(distil_epochs)):
    for j, (imgs, lbls) in enumerate(distilldataloader):
        imgs = imgs.cuda()
        _, target = targetVQMLP(imgs)
        for k in range(10000):
            optimizer.zero_grad()
            q_latent_loss, inp = StudentVQMLP(imgs)
            loss = distill_loss(inp, target.softmax(dim=1)) + q_latent_loss
            distill_losslist.append(loss.cpu().detach().numpy())
            loss.backward()
            optimizer.step()

plt.close()
plt.plot(np.arange(start=0, stop = len(losslist)), losslist)
plt.savefig('VQMLPDistillloss.png')
        





