import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from einops import rearrange
import numpy as np
import random
import math

def sinusoidalEmbeddings(time_steps, embed_dim):
    position = torch.arange(time_steps).unsqueeze(1).float()
    div = torch.exp(torch.arange(0, embed_dim, 2).float() * -(math.log(10000.0) / embed_dim))
    embeddings = torch.zeros(time_steps, embed_dim)
    embeddings[:, 0::2] = torch.sin(position * div)
    embeddings[:, 1::2] = torch.cos(position * div)
    return embeddings

class ddpm_simple(nn.Module):
    def __init__(self, time_steps=1000, b1=1e-4, bT=0.02, embed_dim=784):
        super().__init__()
        self.beta = torch.linspace(b1, bT, time_steps).tolist()
        self.alpha = [1-self.beta[0]]
        for i in range(1,time_steps):
            self.alpha.append(self.alpha[i-1]*(1-self.beta[i]))
        self.embeddings = sinusoidalEmbeddings(time_steps, embed_dim).requires_grad_(False)
        self.relu = nn.ReLU(inplace=True)
        self.l1 = nn.Linear(784, 512)
        self.l2 = nn.Linear(512, 256)
        self.l3 = nn.Linear(256, 256)
        self.l4 = nn.Linear(256, 512)
        self.l5 = nn.Linear(512, 784)

    def forward(self, x):
        t = random.randint(0, 999)
        e = torch.randn_like(x, requires_grad=False)
        x = x*math.sqrt(self.alpha[t])+torch.sqrt(1-self.alpha[t]*e) 
        x = rearrange(x, 'b c h w -> b (c h w)') + self.embeddings[t].to(device=x.device)
        x = self.relu(self.l1(x))
        r = self.relu(self.l2(x))
        r = self.relu(self.l3(r))
        x = self.relu(self.l4(r)) + x
        x = self.l5(x)
        x = rearrange(x, 'b (c h w) -> b c h w', c=1, h=28, w=28)
        return x, e
    
def main():
    model = ddpm_simple().cuda()
    x = torch.randn(64, 1, 28, 28).cuda()
    out, e = model(x)
    print(out.shape)
    print(e.shape)

if __name__ == '__main__':
    main()
    
