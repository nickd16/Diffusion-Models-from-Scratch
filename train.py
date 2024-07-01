import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torch.optim as optim
from models.ddpm_basic import ddpm_simple
from models.unet import UNET
from models.utils import DDPM_Scheduler
from timm.utils import ModelEmaV3
import numpy as np
import random
import math
import pdb

def main():
    batch_size = 32
    num_time_steps = 1000
    train_dataset = datasets.MNIST(root='./data', train=True, download=False, transform=transforms.ToTensor())
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    scheduler = DDPM_Scheduler(num_time_steps=num_time_steps)
    #model = ddpm_simple(time_steps=num_time_steps, embed_dim=784).cuda()
    model = torch.load('test_ddpm').cuda()
    #model = UNET().cuda()
    criterion = nn.MSELoss(reduction='mean')
    optimizer = optim.Adam(model.parameters(), lr=2e-5)
    ema = ModelEmaV3(model, decay=0.9999)

    # fig, axes = plt.subplots(1, 10, figsize=(10,1))
    # t = [0,100,200,300,400,500,600,700,800,999]
    # for i, ax in enumerate(axes.flat):
    #     x = F.pad(train_dataset[0][0], (2,2,2,2))
    #     e = torch.randn_like(x, requires_grad=False)
    #     x = (x*math.sqrt(scheduler(t[i])[1]))+(math.sqrt(1-scheduler(t[i])[1])*e) 
    #     x = rearrange(x, 'c h w -> h w c')
    #     x = x.numpy()
    #     ax.imshow(x)
    #     ax.axis('off')
    # plt.show()

    # for i in range(5):
    #     total_loss = 0
    #     for bidx, (x,_) in enumerate(train_loader):
    #         x = x.cuda()
    #         x = F.pad(x, (2,2,2,2))
    #         t = [random.randint(0, num_time_steps-1) for _ in range(batch_size)]
    #         e = torch.randn_like(x, requires_grad=False)
    #         a = scheduler.alpha[t].view(batch_size,1,1,1).cuda()
    #         x = (torch.sqrt(a)*x) + (torch.sqrt(1-a)*e)
    #         output = model(x, t)
    #         optimizer.zero_grad()
    #         loss = criterion(output, e)
    #         total_loss += loss.item()
    #         loss.backward()
    #         optimizer.step()
    #         ema.update(model)
    #     print(f'Epoch {i+1} | Loss {total_loss / (60000/32)}')
    # torch.save(model, 'test_ddpm')

    with torch.no_grad():
        model = ema.module.eval()
        z = torch.randn(1, 1, 32, 32)
        for t in reversed(range(1, num_time_steps)):
            t = [t]
            temp = (scheduler.beta[t]/( (torch.sqrt(1-scheduler.alpha[t]))*(torch.sqrt(1-scheduler.beta[t])) ))
            z = (1/(torch.sqrt(1-scheduler.beta[t])))*z - (temp*model(z.cuda(),t).cpu())
            e = torch.randn(1, 1, 32, 32)
            z = z + (e*torch.sqrt(scheduler.beta[t]))
        temp = scheduler.beta[0]/( (torch.sqrt(1-scheduler.alpha[0]))*(torch.sqrt(1-scheduler.beta[0])) )
        x = (1/(torch.sqrt(1-scheduler.beta[0])))*z - (temp*model(z.cuda(),[0]).cpu())
        x = rearrange(x.squeeze(0), 'c h w -> h w c').detach()
        x = x.numpy()
        plt.imshow(x)
        plt.show()
    

if __name__ == '__main__':
    main()
