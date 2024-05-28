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
    def __init__(self, time_steps, embed_dim):
        super().__init__()
        self.embeddings = sinusoidalEmbeddings(time_steps, embed_dim).requires_grad_(False)
        self.relu = nn.ReLU(inplace=True)
        self.l1 = nn.Linear(784, 512)
        self.l2 = nn.Linear(512, 256)
        self.l3 = nn.Linear(256, 256)
        self.l4 = nn.Linear(256, 512)
        self.l5 = nn.Linear(512, 784)

    def forward(self, x, t):
        embeddings = self.embeddings[t].to(device=x.device)
        x = rearrange(x, 'b c h w -> b (c h w)') + embeddings
        x = self.relu(self.l1(x)) 
        x = self.relu(self.l2(x)) 
        x = self.relu(self.l3(x))
        x = self.relu(self.l4(x)) 
        x = self.l5(x)
        x = rearrange(x, 'b (c h w) -> b c h w', c=1, h=28, w=28)
        return x
    
def main():
    model = ddpm_simple().cuda()

if __name__ == '__main__':
    main()
    
