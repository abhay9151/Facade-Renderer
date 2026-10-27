import torch
import torch.nn as nn

class Down(nn.Module):
    def __init__(self, cin, cout, norm=True):
        super().__init__()
        L = [nn.Conv2d(cin, cout, 4, 2, 1, bias=False)]
        if norm: L.append(nn.BatchNorm2d(cout))
        L.append(nn.LeakyReLU(0.2, True))
        self.b = nn.Sequential(*L)
    def forward(self, x): return self.b(x)

class Up(nn.Module):
    def __init__(self, cin, cout, dropout=False):
        super().__init__()
        L = [nn.ConvTranspose2d(cin, cout, 4, 2, 1, bias=False), nn.BatchNorm2d(cout)]
        if dropout: L.append(nn.Dropout(0.5))
        L.append(nn.ReLU(True))
        self.b = nn.Sequential(*L)
    def forward(self, x, skip): return torch.cat([self.b(x), skip], 1)

class UNetGenerator(nn.Module):
    def __init__(self, in_ch=3, out_ch=3, nf=64):
        super().__init__()
        self.d1 = Down(in_ch, nf, norm=False); self.d2 = Down(nf, nf*2); self.d3 = Down(nf*2, nf*4)
        self.d4 = Down(nf*4, nf*8); self.d5 = Down(nf*8, nf*8); self.d6 = Down(nf*8, nf*8)
        self.d7 = Down(nf*8, nf*8); self.d8 = Down(nf*8, nf*8, norm=False)
        self.u1 = Up(nf*8, nf*8, True); self.u2 = Up(nf*16, nf*8, True); self.u3 = Up(nf*16, nf*8, True)
        self.u4 = Up(nf*16, nf*8); self.u5 = Up(nf*16, nf*4); self.u6 = Up(nf*8, nf*2); self.u7 = Up(nf*4, nf)
        self.final = nn.Sequential(nn.ConvTranspose2d(nf*2, out_ch, 4, 2, 1), nn.Tanh())
    def forward(self, x):
        d1 = self.d1(x); d2 = self.d2(d1); d3 = self.d3(d2); d4 = self.d4(d3)
        d5 = self.d5(d4); d6 = self.d6(d5); d7 = self.d7(d6); d8 = self.d8(d7)
        u = self.u1(d8, d7); u = self.u2(u, d6); u = self.u3(u, d5); u = self.u4(u, d4)
        u = self.u5(u, d3); u = self.u6(u, d2); u = self.u7(u, d1)
        return self.final(u)
