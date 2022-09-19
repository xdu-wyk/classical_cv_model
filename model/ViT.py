import math

import torch
from torch import nn
import numpy as np


def pair(t):
    return t if isinstance(t, tuple) else t


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dropout=0.2):
        super(Attention, self).__init__()
        assert dim % heads == 0
        self.heads = heads
        self.head_dim = dim // heads
        self.Qm = nn.Linear(dim, dim)
        self.Km = nn.Linear(dim, dim)
        self.Vm = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        nBatch = x.size(0)
        q = self.Qm(x).view(nBatch, -1, self.heads, self.head_dim).transpose(1, 2)
        k = self.Km(x).view(nBatch, -1, self.heads, self.head_dim).transpose(1, 2)
        v = self.Vm(x).view(nBatch, -1, self.heads, self.head_dim).transpose(1, 2)

        # Attention Calculate
        attn = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn = nn.functional.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v).transpose(1, 2). \
            contiguous().view(nBatch, -1, self.heads * self.head_dim)
        return out, attn


class TransformerEncoderLayer(nn.Module):
    def __init__(self, layers=6, heads=8, dim=2048):
        super(TransformerEncoderLayer, self).__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.ff = FeedForward(dim, dim // 4)
        self.attn = Attention(dim, heads)

    def forward(self, x):
        x, _ = self.attn(x)
        x = self.norm1(x) + x
        return self.norm2(self.ff(x)) + x


class TransformerEncoder(nn.Module):
    def __init__(self, layers=6, heads=8, dim=2048):
        super(TransformerEncoder, self).__init__()
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(layers, heads, dim)
            for _ in range(layers)])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class VisionTransformer(nn.Module):
    def __init__(self, img_size, patch_size, num_classes, in_channel=3, layers=6, heads=8, dim=2048):
        """
        ViT Model
        """
        super(VisionTransformer, self).__init__()
        img_size = pair(img_size)
        patch_size = pair(patch_size)
        assert img_size % patch_size == 0

        self.cls_token = nn.Parameter(torch.rand(1, 1, dim))
        self.dim = dim
        self.patch_proj = nn.Conv2d(in_channels=in_channel, out_channels=dim, kernel_size=(patch_size, patch_size),
                                    stride=(patch_size, patch_size))

        self.transformer = TransformerEncoder(layers, heads, dim)
        self.mlp = torch.nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, x):
        """
        input format : N C H W
        """
        # Linear Projection
        x = self.patch_proj(x)
        nBatch = x.size(0)
        x = x.view(nBatch, self.dim, -1).transpose(1, 2)

        # Add CLS Token
        cls_token = torch.cat([self.cls_token for _ in range(nBatch)], dim=0)
        x = torch.cat([x, cls_token], dim=1)

        x = self.transformer(x)
        return self.mlp(x[:, 0])


if __name__ == '__main__':
    # test case
    model = VisionTransformer(img_size=256, patch_size=16, num_classes=100)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(model(torch.rand(10, 3, 256, 256)).shape)
