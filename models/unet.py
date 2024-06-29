import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

class ResBlock(nn.Module):
    def __init__(self, C, num_groups, dropout_prob):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        self.gnorm1 = nn.GroupNorm(num_groups=num_groups, num_channels=C)
        self.gnorm2 = nn.GroupNorm(num_groups=num_groups, num_channels=C)
        self.conv1 = nn.Conv2d(C, C, padding=1)
        self.conv2 = nn.Conv2d(C, C, padding=1)
        self.dropout = nn.Dropout(p=dropout_prob, inplace=True)

    def forward(self, x):
        r = self.conv1(self.relu(self.gnorm1(x)))
        if self.drop:
            r = self.dropout(r)
        r = self.conv2(self.relu(self.gnorm2(r)))
        return r + x
    
class Attention(nn.Module):
    def __init__(self, C, num_heads, dropout_prob):
        super().__init__()
        self.proj1 = nn.Linear(C, C*3)
        self.proj2 = nn.Linear(C, C)
        self.num_heads = num_heads
        self.dropout_prob = dropout_prob

    def forward(self, x):
        h, w = x.shape[2:]
        x = rearrange(x, 'b c h w -> b (h w) c')
        x = self.proj1(x)
        x = rearrange(x, 'b L (C H K) -> K b H L C', K=3, H=self.num_heads)
        q,k,v = x[0], x[1], x[2]
        x = F.scaled_dot_product_attention(q,k,v, is_causal=False, dropout_p=self.dropout_prob)
        x = rearrange(x, 'b H (h w) C -> b h w (C H)', h=h, w=w)
        x = self.proj2(x)
        return rearrange(x, 'b h w C -> b C h w')

def main():
    att = Attention(512, 8, 0).cuda()
    x = torch.randn(16, 512, 16, 16).cuda()
    print(att(x).shape)

if __name__ == '__main__':
    main()


         
