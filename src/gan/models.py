"""pix2pix (Isola et al., 2017): generador U-Net y discriminador PatchGAN condicional. Necesita PyTorch.

El generador recibe el layout (2 canales) y produce la imagen RGB en [-1, 1]. Es totalmente convolucional: entrenado con
recortes de 256 px, puede aplicarse a lienzos mayores (el lado debe ser múltiplo de 2**num_downs).
"""
from __future__ import annotations

import torch
from torch import nn


class Down(nn.Module):
    def __init__(self, cin: int, cout: int, act: bool, norm: bool):
        super().__init__()
        layers = [nn.LeakyReLU(0.2)] if act else []                 # sin inplace: la salida previa se reutiliza como skip
        layers.append(nn.Conv2d(cin, cout, 4, 2, 1, bias=not norm))
        if norm:
            layers.append(nn.BatchNorm2d(cout))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class Up(nn.Module):
    def __init__(self, cin: int, cout: int, dropout: bool):
        super().__init__()
        layers = [nn.ReLU(), nn.ConvTranspose2d(cin, cout, 4, 2, 1, bias=False), nn.BatchNorm2d(cout)]
        if dropout:
            layers.append(nn.Dropout(0.5))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class UNetGenerator(nn.Module):
    def __init__(self, in_ch: int = 2, out_ch: int = 3, ngf: int = 64, num_downs: int = 7):
        super().__init__()
        if num_downs < 3:
            raise ValueError("num_downs debe ser >= 3")
        self.num_downs = num_downs
        ch = ([ngf, ngf * 2, ngf * 4, ngf * 8] + [ngf * 8] * num_downs)[:num_downs]
        self.downs = nn.ModuleList()
        prev = in_ch
        for i, c in enumerate(ch):
            self.downs.append(Down(prev, c, act=i > 0, norm=0 < i < num_downs - 1))
            prev = c
        self.ups = nn.ModuleList()
        for j, i in enumerate(range(num_downs - 1, 0, -1)):
            self.ups.append(Up(ch[i] if j == 0 else ch[i] * 2, ch[i - 1], dropout=j < 3))
        self.final = nn.Sequential(nn.ReLU(), nn.ConvTranspose2d(ch[0] * 2, out_ch, 4, 2, 1), nn.Tanh())

    def forward(self, x):
        skips, h = [], x
        for d in self.downs:
            h = d(h)
            skips.append(h)
        h = self.ups[0](h)
        for j in range(1, len(self.ups)):
            h = self.ups[j](torch.cat([h, skips[-1 - j]], 1))
        return self.final(torch.cat([h, skips[0]], 1))


class PatchDiscriminator(nn.Module):
    """PatchGAN 70x70 condicional: entrada = layout (2) + imagen (3)."""

    def __init__(self, in_ch: int = 5, ndf: int = 64):
        super().__init__()

        def blk(cin, cout, stride, norm=True):
            layers = [nn.Conv2d(cin, cout, 4, stride, 1, bias=not norm)]
            if norm:
                layers.append(nn.BatchNorm2d(cout))
            layers.append(nn.LeakyReLU(0.2))
            return layers

        self.net = nn.Sequential(*blk(in_ch, ndf, 2, norm=False), *blk(ndf, ndf * 2, 2), *blk(ndf * 2, ndf * 4, 2),
                                 *blk(ndf * 4, ndf * 8, 1), nn.Conv2d(ndf * 8, 1, 4, 1, 1))

    def forward(self, cond, img):
        return self.net(torch.cat([cond, img], 1))


def init_weights(m: nn.Module) -> None:
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.normal_(m.weight, 0.0, 0.02)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)


def generation_mode(g: nn.Module) -> nn.Module:
    """BatchNorm en evaluación (estadísticas acumuladas) y Dropout activo, como en pix2pix: el ruido del dropout es la
    única fuente de variación para un mismo layout."""
    g.eval()
    for m in g.modules():
        if isinstance(m, nn.Dropout):
            m.train()
    return g
