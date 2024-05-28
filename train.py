import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torch.optim as optim
from models.ddpm_basic import ddpm_simple
from timm.utils import ModelEmaV3
import numpy as np
import random
import math

def main():
    batch_size = 32
    num_time_steps = 100
    train_dataset = datasets.MNIST(root='./data', train=True, download=False, transform=transforms.ToTensor())
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    beta = torch.linspace(0.001, 0.01, num_time_steps).cuda()
    alpha = 1 - beta
    alpha = torch.cumprod(alpha, dim=0).cuda()

    model = ddpm_simple(time_steps=num_time_steps, embed_dim=784).cuda()
    criterion = nn.MSELoss(reduction='mean')
    optimizer = optim.Adam(model.parameters(), lr=2e-5)
    model = torch.load('test_ddpm').cuda()
    ema = ModelEmaV3(model, decay=0.9999)

    fig, axes = plt.subplots(1, 10, figsize=(10,1))
    t = [0,100,200,300,400,500,600,700,800,999]
    for i, ax in enumerate(axes.flat):
        x = train_dataset[0][0]
        e = torch.randn_like(x, requires_grad=False)
        x = (x*math.sqrt(alpha[t[i]]))+(math.sqrt(1-alpha[t[i]])*e) 
        x = rearrange(x, 'c h w -> h w c')
        x = x.numpy()
        ax.imshow(x)
        ax.axis('off')
    plt.show()

    for i in range(200):
        total_loss = 0
        for bidx, (x,_) in enumerate(train_loader):
            x = x.cuda()
            t = [random.randint(0, num_time_steps-1) for _ in range(batch_size)]
            e = torch.randn_like(x, requires_grad=False)
            a = alpha[t].view(batch_size,1,1,1)
            x = (torch.sqrt(a)*x) + (torch.sqrt(1-a)*e)
            output = model(x, t)
            optimizer.zero_grad()
            loss = criterion(output, e)
            total_loss += loss.item()
            loss.backward()
            optimizer.step()
            ema.update(model)
        print(f'Epoch {i+1} | Loss {total_loss / (60000/32)}')
    torch.save(model, 'test_ddpm')

    with torch.no_grad():
        model = ema.module.eval()
        z = torch.randn(1, 1, 28, 28).cuda()
        for t in reversed(range(1, num_time_steps)):
            z = (1/torch.sqrt(1-beta[t]))*z - (beta[t]/( torch.sqrt(1-alpha[t])*torch.sqrt(1-beta[t]) ))*model(z,t)
            e = torch.randn(1, 1, 28, 28).to(z.device)
            z = z + (e*torch.sqrt(beta[t]))
        x = (1/torch.sqrt(1-beta[0]))*z - (beta[0]/( torch.sqrt(1-alpha[0])*torch.sqrt(1-beta[0]) ))*model(z,0)
        x = rearrange(x.squeeze(0), 'c h w -> h w c').cpu().detach()
        x = x.numpy()
        plt.imshow(x)
        plt.show()
    

if __name__ == '__main__':
    main()
