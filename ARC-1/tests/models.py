import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, d=2048):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, d, bias=False),
            nn.ReLU(),
            nn.Linear(d, d, bias=False),
            nn.ReLU(),
            nn.Linear(d, d, bias=False),
        )

    def forward(self, x):
        return self.net(x)

class TinyAttention(nn.Module):
    def __init__(self, d=512, nheads=8):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=d, num_heads=nheads, batch_first=True)
        self.ff = nn.Sequential(nn.Linear(d, 4*d), nn.ReLU(), nn.Linear(4*d, d))

    def forward(self, x):
        y, _ = self.attn(x, x, x, need_weights=False)
        return self.ff(y)

class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 32, 1),
        )

    def forward(self, x):
        return self.net(x)
