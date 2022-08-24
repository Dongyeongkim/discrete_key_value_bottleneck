# visualizing of the figure 2 in Discrete Key-Value Bottleneck paper(https://arxiv.org/pdf/2207.11240.pdf)

import traceback 

import numpy as np


import torch
import torch.nn as nn
from torch import einsum 
from torch.nn import functional as F
from torch.autograd import Function, Variable

from einops import rearrange, repeat


# VectorQuantizer - referenced from https://colab.research.google.com/github/zalandoresearch/pytorch-vq-vae/blob/master/vq-vae.ipynb#scrollTo=wkzUw2JW09P7


class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim):
        super().__init__()
        
        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings
        
        self._embedding = nn.Embedding(self._num_embeddings, self._embedding_dim)
        self._embedding.weight.data.uniform_(-1/self._num_embeddings, 1/self._num_embeddings)

    def forward(self, inputs):
        # Calculate distances
        distances = (torch.sum(inputs**2, dim=1, keepdim=True) 
                    + torch.sum(self._embedding.weight**2, dim=1)
                    - 2 * torch.matmul(inputs, self._embedding.weight.t()))
            
        # Encoding
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        
        encodings = torch.zeros(encoding_indices.shape[0], self._num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)
        
        # Quantize and unflatten
        quantized = torch.matmul(encodings, self._embedding.weight).view(-1, 2)
        # Loss
        q_latent_loss = F.mse_loss(quantized, inputs.detach())
        loss = q_latent_loss

        quantized = inputs + (quantized - inputs).detach()
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        return quantized, perplexity, q_latent_loss

# VectorQuantizerEMA class - referenced from https://colab.research.google.com/github/zalandoresearch/pytorch-vq-vae/blob/master/vq-vae.ipynb#scrollTo=wkzUw2JW09P7


class VectorQuantizerEMA(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, decay, epsilon=1e-5):
        super(VectorQuantizerEMA, self).__init__()
        
        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings
        
        self._embedding = nn.Embedding(self._num_embeddings, self._embedding_dim)
        self._embedding.weight.data.normal_()
        
        self.register_buffer('_ema_cluster_size', torch.zeros(num_embeddings))
        self._ema_w = nn.Parameter(torch.Tensor(num_embeddings, self._embedding_dim))
        self._ema_w.data.normal_()
        
        self._decay = decay
        self._epsilon = epsilon

    def forward(self, inputs):
        
        # Calculate distances
        distances = (torch.sum(inputs**2, dim=1, keepdim=True) 
                    + torch.sum(self._embedding.weight**2, dim=1)
                    - 2 * torch.matmul(inputs, self._embedding.weight.t()))
            
        # Encoding
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self._num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)
        
        # Quantize and unflatten
        quantized = torch.matmul(encodings, self._embedding.weight).view(-1,2)
        
        # Use EMA to update the embedding vectors
        if self.training:
            self._ema_cluster_size = self._ema_cluster_size * self._decay + \
                                     (1 - self._decay) * torch.sum(encodings, 0)
            
            # Laplace smoothing of the cluster size
            n = torch.sum(self._ema_cluster_size.data)
            self._ema_cluster_size = (
                (self._ema_cluster_size + self._epsilon)
                / (n + self._num_embeddings * self._epsilon) * n)
            
            dw = torch.matmul(encodings.t(), inputs)
            self._ema_w = nn.Parameter(self._ema_w * self._decay + (1 - self._decay) * dw)
            
            self._embedding.weight = nn.Parameter(self._ema_w / self._ema_cluster_size.unsqueeze(1))
        
        # Straight Through Estimator
        quantized = inputs + (quantized - inputs).detach()
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        # Loss
        q_latent_loss = F.mse_loss(quantized, inputs.detach())
        loss = q_latent_loss
        
        # convert quantized from BHWC -> BCHW
        return quantized, perplexity, q_latent_loss


# Discrete Key-Value Memory - referenced from https://github.com/lucidrains/discrete-key-value-bottleneck-pytorch

class KeyValueMem(nn.Module):
    def __init__(self, key_num_embeddings, key_embeddings_dim, value_embeddings_dim):
        super(KeyValueMem, self).__init__()
        self.vq = VectorQuantizerEMA(num_embeddings=key_num_embeddings, embedding_dim=key_embeddings_dim, decay=0.99)
        self.values = nn.Parameter(torch.randn(1, key_num_embeddings, value_embeddings_dim))
    
    def forward(self, x):
        output = self.vq(x)
        quantized, perplexity, encoding = output

        memory_indices = encoding

        if memory_indices.ndim == 2:
            memory_indices = rearrange(memory_indices, '...->... 1')
        
        memory_indices = rearrange(memory_indices, 'b n h -> b h n')

        values = repeat(self.values, 'h n d -> b h n d', b = memory_indices.shape[0])
        memory_indices = repeat(memory_indices, 'b h n -> b h n d', d = values.shape[-1])
        memory_indices = memory_indices.long()
        memories = values.gather(2, memory_indices)
        flattened_memories = rearrange(memories, 'b h n d -> b n (h d)')
        return flattened_memories



# Simple MLP (32, 32)

class MLP(nn.Module):
    def __init__(self, feature_num):
        super().__init__()
        self.linear1 = nn.Linear(in_features=feature_num, out_features=32)
        
    
    def forward(self, x):
        x = self.linear1(x)
        return x


# MLP Baseline in Figure 2

class MLPb(nn.Module):
    def __init__(self, feature_num):
        super().__init__()
        self.linear = nn.Linear(in_features=feature_num, out_features=32)
        self.activation = nn.Sigmoid()
        self.MLP = MLP(feature_num=32)
    
    def forward(self, x):
        x = self.linear(x)
        x = self.activation(x)
        x = self.MLP(x)
        return x


# VQ + MLP
# Need Codebook loss but not a commitment loss (because input is 2D space points)

class VQMLP(nn.Module):
    def __init__(self, feature_num, codebook_num_embeddings, codebook_embeddings_dim):
        super(VQMLP, self).__init__()
        self.vq = VectorQuantizer(num_embeddings=codebook_num_embeddings, embedding_dim=codebook_embeddings_dim) 
        self.MLP = MLP(feature_num=feature_num)
        
    
    def forward(self, x):
        x = self.vq(x)
        quantized, perplexity, q_latent_loss = x
        x = self.MLP(quantized)
        return q_latent_loss, x

# VQ(EMA) + MLP in figure 2
# Need Codebook loss but not a commitment loss (because input is 2D space points)

class VQELAMLP(nn.Module):
    def __init__(self, feature_num, codebook_num_embeddings, codebook_embeddings_dim):
        super(VQELAMLP, self).__init__()
        self.vq = VectorQuantizerEMA(num_embeddings=codebook_num_embeddings, embedding_dim=codebook_embeddings_dim, decay=0.99) 
        self.MLP = MLP(feature_num=feature_num)
        
    
    def forward(self, x):
        x = self.vq(x)
        quantized, perplexity, q_latent_loss = x
        x = self.MLP(quantized)
        return q_latent_loss, x


# key-value bottleneck + MLP in figure 2

class KVMLP(nn.Module):
    def __init__(self, key_num_embeddings, key_embeddings_dim, value_embeddings_dim):
        super(KVMLP, self).__init__()
        self.keyvalmem = KeyValueMem(key_num_embeddings = key_num_embeddings, key_embeddings_dim = key_embeddings_dim, value_embeddings_dim = value_embeddings_dim)
        self.MLP = MLP()

    
    def forward(self, x):
        x = self.keyvalmem(x)
        x = self.MLP(x)
        return x


# For testing the modules

if __name__ == '__main__':

    if torch.cuda.is_available():
        a = torch.randn((1,1,1,2)).cuda()
        MLP_baseline = MLPb(feature_num=2).cuda()
        VQ_MLP = VQMLP(feature_num=2, codebook_num_embeddings=100, codebook_embeddings_dim=2).cuda()
        KV_MLP = KVMLP(key_num_embeddings=100, key_embeddings_dim=2, value_embeddings_dim=32).cuda()
    
    else:
        a = torch.randn((1,1,1,2))
        MLP_baseline = MLPb(feature_num=2)
        VQ_MLP = VQMLP(feature_num=2, codebook_num_embeddings=100, codebook_embeddings_dim=2)
        KV_MLP = KeyValueMLP(key_num_embeddings=100, key_embeddings_dim=2, value_embeddings_dim=32)
    
    try:
        print("MLP Baseline ...")
        print("output: "+str(MLP_baseline(a)))
        print("size: "+str(MLP_baseline(a).size()))
        print("\u2713 MLP Basline - Checked")
        print("VQ+MLP Baseline ...")
        print("ouput: "+str(VQ_MLP(a)))
        print("size: "+str(VQ_MLP(a).size()))
        print("\u2713 VQ+MLP Basline - Checked")
        print("KV+MLP Baseline ...")
        print("output: "+str(KV_MLP(a)))
        print("size: "+str(KV_MLP(a).size()))
        print("\u2713 KV+MLP Basline - Checked")
        print("All baseline works well")
    
    except:
        print("Test has failed. Please check the Traceback")
        print(traceback.format_exc())